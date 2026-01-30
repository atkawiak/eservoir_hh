
import numpy as np
import matplotlib.pyplot as plt

def generate_dale_weights(N, density, spectral_radius, excitatory_ratio=0.8, seed=42):
    np.random.seed(seed)
    N_E = int(N * excitatory_ratio)
    W = np.random.uniform(0, 1, (N, N)) * (np.random.rand(N, N) < density)
    signs = np.ones(N)
    signs[N_E:] = -1.0
    W = W * signs[np.newaxis, :]
    current_rho = np.max(np.abs(np.linalg.eigvals(W)))
    return W * (spectral_radius / max(current_rho, 1e-8))

def compute_step(V, m, h, n, b_gate, s_trace, I_pulse, weights, dt):
    # HH Constants
    C=1.0; gNa=120.0; ENa=50.0; gK=36.0; EK=-77.0; gL=0.3; EL=-54.4
    gA=20.0; EA=-80.0; tauA=20.0; Eexc=0.0; Einh=-80.0; tau_syn=5.0
    decay_syn = np.exp(-dt / tau_syn)
    
    # Store V_old for spike detection
    V_old = V.copy()
    
    # 1. Gates update
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

    # 2. Conductance-based Currents
    syn_activity = weights @ s_trace
    g_exc = np.maximum(0, syn_activity)
    g_inh = np.maximum(0, -syn_activity)
    I_syn = g_exc * (V - Eexc) + g_inh * (V - Einh)
    
    I_Na = gNa * (m**3) * h * (V - ENa)
    I_K = gK * (n**4) * (V - EK)
    I_L = gL * (V - EL)
    I_A = gA * a_inf**3 * b_gate * (V - EA)
    
    dV = (-I_Na - I_K - I_L - I_A - I_syn + I_pulse) / C
    V += dt * dV
    
    # 3. Rising-Edge Spike Detection
    spikes = ((V > -20.0) & (V_old <= -20.0)).astype(float)
    
    # 4. Synapse Update
    s_trace = s_trace * decay_syn + spikes
    
    return V, m, h, n, b_gate, s_trace

def compute_mle(rho, N=50, T_ms=1000, dt=0.05, eps=1e-6):
    """
    Computes Maximal Lyapunov Exponent by evolving two trajectories.
    """
    W = generate_dale_weights(N, 0.2, rho)
    steps = int(T_ms / dt)
    
    # Init trajs
    V1 = np.random.uniform(-70, -60, N)
    V2 = V1.copy()
    V2[0] += eps # Perturb one neuron
    
    m1 = np.zeros(N); h1 = np.zeros(N); n1 = np.zeros(N); b1 = np.zeros(N); s1 = np.zeros(N)
    m2 = np.zeros(N); h2 = np.zeros(N); n2 = np.zeros(N); b2 = np.zeros(N); s2 = np.zeros(N)
    
    I_pulse = 6.0 # Constant bias
    
    lyap = 0
    washout = int(200 / dt)
    
    for t in range(steps):
        V1, m1, h1, n1, b1, s1 = compute_step(V1, m1, h1, n1, b1, s1, I_pulse, W, dt)
        V2, m2, h2, n2, b2, s2 = compute_step(V2, m2, h2, n2, b2, s2, I_pulse, W, dt)
        
        # Stability clip
        V1 = np.clip(V1, -100, 100); V2 = np.clip(V2, -100, 100)
        
        if t > washout:
            # Distance in state space (Voltage only for simplicity)
            dist = np.linalg.norm(V1 - V2)
            if dist > 0:
                lyap += np.log(dist / eps)
                # Renormalize V2 to be eps away from V1
                V2 = V1 + (V2 - V1) * (eps / dist)
                # Keep other variables same or renormalize them too (simplified: just voltage)

    return (lyap / (steps - washout)) / dt # bit per ms approx

if __name__ == "__main__":
    rhos = [0.05, 0.15, 0.5, 1.0, 1.5]
    print("Computing MLE for different spectral radii...")
    for r in rhos:
        mle = compute_mle(r)
        status = "STABLE" if mle < 0 else "CHAOTIC"
        print(f"rho = {r:.2f} -> MLE = {mle:.5f} ({status})")
