
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# =============================================================================
# 1. UTILS & WEIGHT GENERATION (Dale's Law)
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
    eigenvalues = np.linalg.eigvals(W_signed)
    current_rho = np.max(np.abs(eigenvalues))
    if current_rho == 0: current_rho = 1e-8
    return W_signed * (spectral_radius / current_rho)

def exp_trace(spikes, dt_s, tau_s):
    decay = np.exp(-dt_s / tau_s)
    x = np.zeros_like(spikes, dtype=np.float32)
    if len(spikes.shape) == 1:
        for t in range(1, len(spikes)):
            x[t] = x[t-1] * decay + spikes[t-1]
    else:
        for t in range(1, spikes.shape[0]):
            x[t, :] = x[t-1, :] * decay + spikes[t-1, :]
    return x

def window_features(traces, win_len):
    T, N = traces.shape
    K = T // win_len
    out = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        a = k * win_len
        b = a + win_len
        out[k] = traces[a:b].mean(axis=0)
    return out

def ridge_fit(X, Y, lam):
    D = X.shape[1]
    A = X.T @ X + lam * np.eye(D)
    B = X.T @ Y
    return np.linalg.solve(A, B)

# =============================================================================
# 2. HODGKIN-HUXLEY GATES (STABLE, WITH STEADY-STATE INIT)
# =============================================================================

def alpha_m_safe(V):
    num = 0.1 * (V + 40.0)
    denom = 1.0 - np.exp(-(V + 40.0) / 10.0)
    mask = np.abs(V + 40.0) < 1e-6
    res = np.zeros_like(V)
    res[~mask] = num[~mask] / denom[~mask]
    res[mask] = 1.0 
    return res

def alpha_n_safe(V):
    num = 0.01 * (V + 55.0)
    denom = 1.0 - np.exp(-(V + 55.0) / 10.0)
    mask = np.abs(V + 55.0) < 1e-6
    res = np.zeros_like(V)
    res[~mask] = num[~mask] / denom[~mask]
    res[mask] = 0.1
    return res

def beta_m_safe(V): return 4.0 * np.exp(-(V + 65.0) / 18.0)
def alpha_h_safe(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h_safe(V): return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
def beta_n_safe(V): return 0.125 * np.exp(-(V + 65.0) / 80.0)

# =============================================================================
# 3. TASKS
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
            if abs(x[t+1]) > 5.0:
                 return self.generate_data(timesteps, wash_out_gen)
        x_seq = x[wash_out_gen:]
        u_raw = x_seq[:timesteps]
        target = x_seq[1:timesteps+1]
        u_norm = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min())
        return u_norm, target

class MemoryCapacityTask:
    def __init__(self, max_lag=20, random_seed=None):
        self.max_lag = max_lag
        self.rng = np.random.RandomState(random_seed)
        
    def generate_data(self, timesteps):
        u = self.rng.uniform(-1, 1, size=timesteps)
        return u
        
    def compute_memory_capacity(self, states, u_signal, wash_out=100):
        """
        states: (T, D) after windowing
        u_signal: (T,) aligned input
        """
        assert len(states) == len(u_signal)
        T, D = states.shape
        ridge = Ridge(alpha=1e-4)
        r2_scores = []
        lags = range(1, self.max_lag + 1)
        
        for k in lags:
            t_start = wash_out + k
            if t_start >= T:
                r2_scores.append(0.0)
                continue
            
            X = states[t_start:, :] 
            y = u_signal[t_start - k : T - k]        
            
            if len(X) < 20: 
                r2_scores.append(0.0)
                continue
                
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            ridge.fit(X_train, y_train)
            score = ridge.score(X_test, y_test)
            r2_scores.append(max(0.0, float(score)))
        
        return {"total_mc": float(sum(r2_scores)), "lags": list(lags), "r2_scores": r2_scores}

# =============================================================================
# 4. HODGKIN-HUXLEY SIMULATION (SCIENTIFICALLY REFINED)
# =============================================================================

def generate_poisson_input(u_input_norm, dt_ms=0.05, task_step_ms=20.0, r_min=5.0, r_max=200.0, seed=42):
    dt_s = dt_ms / 1000.0
    steps_per_v = int(task_step_ms / dt_ms)
    u_upsampled = np.repeat(u_input_norm, steps_per_v)
    rates = r_min + u_upsampled * (r_max - r_min)
    rng = np.random.default_rng(seed)
    spikes_in = (rng.random(rates.shape) < (rates * dt_s)).astype(np.float32)
    return spikes_in, steps_per_v

def simulate_hh_with_spikes_in(weights, spikes_in, W_in, bias_current=6.0, seed=42, V_init=None, trim_steps=0):
    N = weights.shape[0]
    dt = 0.05
    
    # Precise V Initialization
    if V_init is None:
        rng = np.random.default_rng(seed)
        V = rng.uniform(-70, -60, N)
    else:
        V = V_init.copy()
    
    # Steady-state gate initialization (Avoids long transients)
    am, bm = alpha_m_safe(V), beta_m_safe(V)
    ah, bh = alpha_h_safe(V), beta_h_safe(V)
    an, bn = alpha_n_safe(V), beta_n_safe(V)
    m, h, n = am/(am+bm), ah/(ah+bh), an/(an+bn)
    b_gate = np.zeros(N) 
    
    s_trace = np.zeros(N); s_in_trace = 0.0
    tau_in = 10.0; decay_in = np.exp(-dt / tau_in); decay_syn = np.exp(-dt / 5.0)
    
    C=1.0; gNa=120.0; ENa=50.0; gK=36.0; EK=-77.0; gL=0.3; EL=-54.4
    gA=20.0; EA=-80.0; tauA=20.0; Eexc=0.0; Einh=-80.0
    
    res_spikes = np.zeros((len(spikes_in), N), dtype=np.float32)
    
    for t in range(len(spikes_in)):
        V_old = V.copy()
        s_in_trace = s_in_trace * decay_in + spikes_in[t]
        I_pulse = (5.0 * s_in_trace) * W_in + bias_current
        
        a_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0)); b_inf = 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))
        b_gate += (dt / tauA) * (b_inf - b_gate)
        
        am, bm = alpha_m_safe(V), beta_m_safe(V); m += dt * (am * (1 - m) - bm * m)
        ah, bh = alpha_h_safe(V), beta_h_safe(V); h += dt * (ah * (1 - h) - bh * h)
        an, bn = alpha_n_safe(V), beta_n_safe(V); n += dt * (an * (1 - n) - bn * n)
        
        syn = weights @ s_trace; g_e, g_i = np.maximum(0, syn), np.maximum(0, -syn)
        I_syn = g_e * (V - Eexc) + g_i * (V - Einh)
        
        I_Na, I_K, I_L = gNa*(m**3)*h*(V-ENa), gK*(n**4)*(V-EK), gL*(V-EL)
        I_A = gA*(a_inf**3)*b_gate*(V-EA)
        
        dV = (-I_Na - I_K - I_L - I_A - I_syn + I_pulse) / C
        V += dt * dV; V = np.clip(V, -100, 100)
        
        spks = ((V > -20.0) & (V_old <= -20.0)).astype(np.float32)
        res_spikes[t] = spks
        s_trace = s_trace * decay_syn + spks
    
    if trim_steps > 0:
        return res_spikes[trim_steps:]
    return res_spikes

def conditional_lyapunov_check(weights, W_in, u_norm, bias_current, seed_in=42, seed_net=42, eps=1e-6):
    spikes_in, steps_v = generate_poisson_input(u_norm, seed=seed_in)
    rng = np.random.default_rng(seed_net)
    V0 = rng.uniform(-70, -60, weights.shape[0])
    
    S1 = simulate_hh_with_spikes_in(weights, spikes_in, W_in, bias_current=bias_current, V_init=V0)
    S2 = simulate_hh_with_spikes_in(weights, spikes_in, W_in, bias_current=bias_current, V_init=V0 + eps)
    
    Z1, Z2 = exp_trace(S1, 5e-5, 0.02), exp_trace(S2, 5e-5, 0.02)
    P1, P2 = window_features(Z1, steps_v), window_features(Z2, steps_v)
    
    logd = np.log(np.linalg.norm(P2 - P1, axis=1) + 1e-12)
    w_start, w_end = 100, 400
    if w_start >= len(logd):
        return 0.0, logd
    if len(logd) < w_end: 
        w_end = len(logd)
    
    if w_end > w_start:
        coeffs = np.polyfit(np.arange(w_start, w_end), logd[w_start:w_end], 1)
        return coeffs[0], logd
    return 0.0, logd

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    N_NEURONS = 100; RHO = 0.21; DENSITY = 0.2; WASH_OUT = 200
    print(f"--- FINAL SCIENTIFIC VALIDATION (PROJECT B - POISSON) ---")
    
    W = generate_dale_weights(N_NEURONS, DENSITY, RHO, random_seed=42)
    rng_main = np.random.default_rng(42)
    W_in = (rng_main.random(N_NEURONS) < 0.2).astype(float)
    
    # 1. HENON
    print("\n[TASK 1] Henon Map Prediction")
    task1 = HenonTask(random_seed=42); u_norm, target = task1.generate_data(3000)
    sp_in, steps_v = generate_poisson_input(u_norm, seed=42)
    S_h = simulate_hh_with_spikes_in(W, sp_in, W_in, 6.0, trim_steps=WASH_OUT*steps_v)
    P_h = window_features(exp_trace(S_h, 5e-5, 0.02), steps_v)
    
    Y = target[WASH_OUT:WASH_OUT+len(P_h)]
    X = np.concatenate([np.ones((P_h.shape[0], 1)), P_h], axis=1)
    split = int(0.7 * len(X))
    X_tr, X_te, Y_tr, Y_te = X[:split], X[split:], Y[:split], Y[split:]
    
    base_nrmse = np.sqrt(np.mean((np.mean(Y_te) - Y_te)**2)) / np.std(Y_te)
    W_rd = ridge_fit(X_tr, Y_tr.reshape(-1,1), 1e-3)
    Y_pd = (X_te @ W_rd).flatten()
    nrmse = np.sqrt(np.mean((Y_pd - Y_te)**2)) / np.std(Y_te)
    print(f"   >> Base NRMSE: {base_nrmse:.4f}, Model NRMSE: {nrmse:.4f}")
    
    # 2. LYAPUNOV
    print("\n[ANALYSIS] Conditional Lyapunov")
    lambda_c, log_d = conditional_lyapunov_check(W, W_in, u_norm[:600], bias_current=6.0)
    print(f"   >> Lambda: {lambda_c:.6f}")
    
    # 3. MC
    print("\n[TASK 2] Memory Capacity")
    task2 = MemoryCapacityTask(random_seed=42); u_mc = task2.generate_data(3000)
    umc_norm = (u_mc - u_mc.min()) / (u_mc.max() - u_mc.min())
    sp_mc, steps_v_mc = generate_poisson_input(umc_norm, seed=42)
    S_mc = simulate_hh_with_spikes_in(W, sp_mc, W_in, 0.0, trim_steps=WASH_OUT*steps_v_mc)
    P_mc = window_features(exp_trace(S_mc, 5e-5, 0.02), steps_v_mc)
    
    # Set wash_out=50 for MC readout since we already trimmed transient
    u_aligned_mc = umc_norm[WASH_OUT:WASH_OUT+len(P_mc)]
    mc_res = task2.compute_memory_capacity(P_mc, u_aligned_mc, wash_out=50) # Reduced to 50
    print(f"   >> Total MC: {mc_res['total_mc']:.4f}")
    
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1); plt.plot(Y_te[:100], 'k'); plt.plot(Y_pd[:100], 'r--'); plt.title(f"Henon (NRMSE={nrmse:.3f})")
    plt.subplot(3, 1, 2); plt.plot(log_d, 'b'); plt.axvspan(100, 400, color='orange', alpha=0.2); plt.title(f"Lyapunov (λ={lambda_c:.5f})")
    plt.subplot(3, 1, 3); plt.bar(mc_res['lags'], mc_res['r2_scores']); plt.title(f"MC Profile (Total={mc_res['total_mc']:.2f})")
    plt.tight_layout(); plt.show()
