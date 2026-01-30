
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# =============================================================================
# 1. WEIGHT GENERATION (Dale's Law)
# =============================================================================

def generate_dale_weights(N, density, spectral_radius, excitatory_ratio=0.8, random_seed=None):
    """
    Generates a weight matrix respecting Dale's Law:
    - distinct excitatory (weights > 0) and inhibitory (weights < 0) neurons.
    - Excitatory neurons only project positive weights.
    - Inhibitory neurons only project negative weights.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    N_E = int(N * excitatory_ratio)
    N_I = N - N_E
    
    # Initialize weights (Uniform 0-1)
    W_abs = np.random.uniform(0, 1, (N, N))
    
    # Apply Sparsity
    mask = np.random.rand(N, N) < density
    W_sparse = W_abs * mask
    
    # Apply Signs (Columns)
    signs = np.ones(N)
    signs[N_E:] = -1.0 # Last N_I neurons are inhibitory
    W_signed = W_sparse * signs[np.newaxis, :]
    
    # Rescale Spectral Radius
    eigenvalues = np.linalg.eigvals(W_signed)
    current_rho = np.max(np.abs(eigenvalues))
    if current_rho == 0: current_rho = 1e-8
        
    W_final = W_signed * (spectral_radius / current_rho)
    return W_final

# =============================================================================
# 2. POISSON UTILS & FILTERING
# =============================================================================

def poisson_spike_train(rate_hz, dt_s, rng):
    """u(t) -> s(t) (Poisson)"""
    p = np.clip(rate_hz * dt_s, 0.0, 1.0)
    return (rng.random(rate_hz.shape) < p).astype(np.float32)

def exp_trace(spikes, dt_s, tau_s):
    """Filter: x[t+1] = x[t]*exp(-dt/tau) + spikes[t]"""
    decay = np.exp(-dt_s / tau_s)
    x = np.zeros_like(spikes, dtype=np.float32)
    # Handle both (T,) and (T,N)
    if len(spikes.shape) == 1:
        for t in range(1, len(spikes)):
            x[t] = x[t-1] * decay + spikes[t-1]
    else:
        for t in range(1, spikes.shape[0]):
            x[t, :] = x[t-1, :] * decay + spikes[t-1, :]
    return x

def window_features(traces, win_len):
    """traces: (T,N) -> average over window -> (K,N)"""
    T, N = traces.shape
    K = T // win_len
    out = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        a = k * win_len
        b = a + win_len
        out[k] = traces[a:b].mean(axis=0)
    return out

def ridge_fit(X, Y, lam):
    """W = (X.T X + lam I)^-1 X.T Y"""
    D = X.shape[1]
    A = X.T @ X + lam * np.eye(D)
    B = X.T @ Y
    return np.linalg.solve(A, B)

# =============================================================================
# 3. HENON TASK
# =============================================================================

class HenonTask:
    def __init__(self, a=1.4, b=0.3, random_seed=None):
        self.a = a
        self.b = b
        self.rng = np.random.RandomState(random_seed)
        
    def generate_data(self, timesteps, wash_out_gen=1000):
        total_steps = timesteps + wash_out_gen + 1
        x, y = np.zeros(total_steps), np.zeros(total_steps)
        x[0], y[0] = self.rng.uniform(-0.5, 0.5), self.rng.uniform(-0.5, 0.5)
        
        for t in range(total_steps - 1):
            x[t+1] = 1 - self.a * x[t]**2 + y[t]
            y[t+1] = self.b * x[t]
            if abs(x[t+1]) > 5.0: # Stability check
                 return self.generate_data(timesteps, wash_out_gen) # Retry
                 
        x_seq = x[wash_out_gen:]
        u_raw = x_seq[:timesteps]
        target = x_seq[1:timesteps+1]
        
        # Normalize to [0, 1] for rate coding
        u_norm = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min())
        return u_norm, target

# =============================================================================
# 4. HODGKIN-HUXLEY SIMULATION (with Poisson Input)
# =============================================================================

def simulate_poisson_hh(weights, u_input_norm, seed=42):
    # Params
    N = weights.shape[0]
    dt = 0.05 # ms
    dt_s = dt / 1000.0
    task_step_ms = 20.0
    steps_per_v = int(task_step_ms / dt)
    T_total = len(u_input_norm) * steps_per_v
    
    # Rate coding: [5, 200] Hz
    r_min, r_max = 5.0, 200.0
    u_upsampled = np.repeat(u_input_norm, steps_per_v)
    rates = r_min + u_upsampled * (r_max - r_min)
    
    # Input Spike Train
    rng = np.random.default_rng(seed)
    spikes_in = (rng.random(rates.shape) < (rates * dt_s)).astype(np.float32)
    
    # Recurrent HH
    V = np.random.uniform(-70, -60, N)
    V_prev = V.copy()
    m = np.zeros(N); h = np.zeros(N); n = np.zeros(N); b_gate = np.zeros(N)
    s_trace = np.zeros(N)
    
    # Input synaptic filter
    tau_in = 10.0 # ms
    decay_in = np.exp(-dt / tau_in)
    s_in_trace = 0.0
    
    # Matrices for results
    res_spikes = np.zeros((T_total, N), dtype=np.float32)
    
    # Weights for input injection (e.g. only 20% neurons get input)
    W_in = (rng.random(N) < 0.2).astype(float)
    
    # HH Constants
    C=1.0; gNa=120.0; ENa=50.0; gK=36.0; EK=-77.0; gL=0.3; EL=-54.4
    gA=20.0; EA=-80.0; tauA=20.0; Eexc=0.0; Einh=-80.0; tau_syn=5.0
    decay_syn = np.exp(-dt / tau_syn)
    
    print("Starting HH simulation (Refined)...")
    for t in range(T_total):
        # 0. Store state BEFORE update
        V_old = V.copy()
        
        # 1. Input Current (Poisson -> Trace -> I_in)
        s_in = spikes_in[t]
        s_in_trace = s_in_trace * decay_in + s_in
        I_pulse = (5.0 * s_in_trace) * W_in + 6.0 # 5nA scale + 6nA bias
        
        # 2. Gates update
        a_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0))
        b_inf = 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))
        b_gate += (dt / tauA) * (b_inf - b_gate)
        
        alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
        m += dt * (alpha_m * (1 - m) - beta_m * m)
        alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
        h += dt * (alpha_h * (1 - h) - beta_h * h)
        alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
        n += dt * (alpha_n * (1 - n) - beta_n * n)
        
        # 3. Conductance-based Recurrent Current
        syn_activity = weights @ s_trace
        g_exc = np.maximum(0, syn_activity)
        g_inh = np.maximum(0, -syn_activity)
        I_syn = g_exc * (V - Eexc) + g_inh * (V - Einh)
        
        # 4. HH prądy + update V
        I_Na = gNa * (m**3) * h * (V - ENa)
        I_K = gK * (n**4) * (V - EK)
        I_L = gL * (V - EL)
        I_A_curr = gA * (a_inf**3) * b_gate * (V - EA)
        
        dV = (-I_Na - I_K - I_L - I_A_curr - I_syn + I_pulse) / C
        V += dt * dV
        
        if np.any(np.abs(V) > 150): V = np.clip(V, -100, 100)

        # 5. Rising-Edge Spike Detection (Correct Logic)
        spikes = ((V > -20.0) & (V_old <= -20.0)).astype(float)
        res_spikes[t] = spikes
        
        # 6. Update Synaptic Traces (Post-Spike)
        s_trace = s_trace * decay_syn + spikes

    print("Simulation finished.")
    return res_spikes

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    N_NEURONS = 100
    RHO = 0.15
    DENSITY = 0.2
    STEPS = 500 # Short for demo
    
    # 1. Data
    task = HenonTask(random_seed=42)
    u_norm, target = task.generate_data(STEPS)
    
    # 2. Weights
    W = generate_dale_weights(N_NEURONS, DENSITY, RHO, random_seed=42)
    
    # 3. Simulate
    S_res = simulate_poisson_hh(W, u_norm) # (T_total, N)
    
    # 4. Readout Processing (Poisson style)
    dt_s = 5e-5
    win_len = int(20.0 / 0.05) # 20ms windows
    
    # Filter spikes with 20ms tau
    Z_res = exp_trace(S_res, dt_s, tau_s=0.02)
    # Downsample
    Phi = window_features(Z_res, win_len) # (STEPS, N)
    
    # 5. Ridge Fit
    # Bias term
    X = np.concatenate([np.ones((Phi.shape[0], 1)), Phi], axis=1)
    # Target alignment
    Y = target[:len(Phi)]
    
    # Split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    
    W_readout = ridge_fit(X_train, Y_train.reshape(-1,1), lam=1e-3)
    Y_pred = (X_test @ W_readout).flatten()
    
    # Metric
    rmse = np.sqrt(np.mean((Y_pred - Y_test)**2))
    nrmse = rmse / np.std(Y_test)
    print(f"\nFINAL NRMSE (Poisson Encoding): {nrmse:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(Y_test[:100], 'k-', label='Target')
    plt.plot(Y_pred[:100], 'r--', label='Pred')
    plt.legend()
    plt.title(f"Poisson HH Reservoir - Henon Map Prediction (rho={RHO})")
    plt.show()
