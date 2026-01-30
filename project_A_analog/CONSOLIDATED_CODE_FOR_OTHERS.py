
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# =============================================================================
# 1. WEIGHT GENERATION (Dale's Law)
# =============================================================================

def generate_dale_weights(N, density, spectral_radius, excitatory_ratio=0.8, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        
    N_E = int(N * excitatory_ratio)
    W_abs = np.random.uniform(0, 1, (N, N))
    mask = np.random.rand(N, N) < density
    W_sparse = W_abs * mask
    signs = np.ones(N)
    signs[N_E:] = -1.0
    W_signed = W_sparse * signs[np.newaxis, :]
    
    current_rho = np.max(np.abs(np.linalg.eigvals(W_signed)))
    if current_rho == 0: current_rho = 1e-8
    return W_signed * (spectral_radius / current_rho)

# =============================================================================
# 2. HENON TASK
# =============================================================================

class HenonTask:
    def __init__(self, a=1.4, b=0.3, random_seed=None):
        self.a, self.b = a, b
        self.rng = np.random.RandomState(random_seed)
        
    def generate_data(self, timesteps, wash_out_gen=1000):
        total_steps = timesteps + wash_out_gen + 1
        x, y = np.zeros(total_steps), np.zeros(total_steps)
        x[0], y[0] = self.rng.uniform(-0.5, 0.5), self.rng.uniform(-0.5, 0.5)
        for t in range(total_steps - 1):
            x[t+1] = 1 - self.a * x[t]**2 + y[t]
            y[t+1] = self.b * x[t]
            if abs(x[t+1]) > 5.0: return self.generate_data(timesteps, wash_out_gen)
        x_seq = x[wash_out_gen:]
        return x_seq[:timesteps], x_seq[1:timesteps+1]

# =============================================================================
# 3. HODGKIN-HUXLEY SIMULATION (Analog Input)
# =============================================================================

def simulate_analog_hh(weights, u_input, seed=42):
    N = weights.shape[0]
    RAW_DT = 0.05
    STEPS_PER_TASK = int(20.0 / RAW_DT)
    TASK_STEPS = len(u_input)
    
    np.random.seed(seed)
    V = np.random.uniform(-70, -60, N)
    m = np.zeros(N); h = np.zeros(N); n = np.zeros(N); b_gate = np.zeros(N)
    s_trace = np.zeros(N)
    W_in = np.random.uniform(-1, 1, N)
    
    state_matrix = np.zeros((TASK_STEPS, N))
    decay_syn = np.exp(-RAW_DT / 5.0) # tau_syn = 5ms
    
    # Constants
    C=1.0; gNa=120.0; ENa=50.0; gK=36.0; EK=-77.0; gL=0.3; EL=-54.4
    gA=20.0; EA=-80.0; tauA=20.0; Eexc=0.0; Einh=-80.0
    
    print("Starting HH simulation (Analog)...")
    for t_task in range(TASK_STEPS):
        curr_u = u_input[t_task]
        for _ in range(STEPS_PER_TASK):
            V_old = V.copy()
            
            # HH Dynamics
            a_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))
            b_gate += (RAW_DT / tauA) * (b_inf - b_gate)
            
            alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
            m += RAW_DT * (alpha_m * (1 - m) - beta_m * m)
            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            h += RAW_DT * (alpha_h * (1 - h) - beta_h * h)
            alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
            n += RAW_DT * (alpha_n * (1 - n) - beta_n * n)
            
            syn_activity = weights @ s_trace
            g_exc = np.maximum(0, syn_activity)
            g_inh = np.maximum(0, -syn_activity)
            I_syn = g_exc * (V - Eexc) + g_inh * (V - Einh)
            
            I_inj = 6.0 + curr_u * W_in # Analog input
            
            I_Na = gNa * (m**3) * h * (V - ENa)
            I_K = gK * (n**4) * (V - EK)
            I_L = gL * (V - EL)
            I_A = gA * a_inf**3 * b_gate * (V - EA)
            
            dV = (-I_Na - I_K - I_L - I_A - I_syn + I_inj) / C
            V += RAW_DT * dV
            if np.any(np.abs(V) > 150): V = np.clip(V, -100, 100)
            
            spikes = ((V > -20.0) & (V_old <= -20.0)).astype(float)
            s_trace = s_trace * decay_syn + spikes
            
        state_matrix[t_task, :] = s_trace
    return state_matrix

# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == "__main__":
    task = HenonTask(random_seed=42)
    u, target = task.generate_data(500)
    W = generate_dale_weights(100, 0.2, 0.15, random_seed=42)
    
    Phi = simulate_analog_hh(W, u)
    
    # Readout
    X = np.concatenate([np.ones((Phi.shape[0], 1)), Phi], axis=1)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = target[:split], target[split:]
    
    ridge = Ridge(alpha=1e-3)
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_test)
    
    nrmse = np.sqrt(np.mean((Y_pred - Y_test)**2)) / np.std(Y_test)
    print(f"\nFINAL NRMSE (Analog): {nrmse:.4f}")
