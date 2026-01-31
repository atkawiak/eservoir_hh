
import numpy as np
import pandas as pd
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spectral_utils import generate_dale_weights
from poisson_utils import exp_trace, window_features

# --- Stable Functions ---
def alpha_m_safe(V):
    num = 0.1 * (V + 40.0); denom = 1.0 - np.exp(-(V + 40.0) / 10.0)
    mask = np.abs(V + 40.0) < 1e-6
    res = np.zeros_like(V); res[~mask] = num[~mask] / denom[~mask]; res[mask] = 1.0
    return res

def alpha_n_safe(V):
    num = 0.01 * (V + 55.0); denom = 1.0 - np.exp(-(V + 55.0) / 10.0)
    mask = np.abs(V + 55.0) < 1e-6
    res = np.zeros_like(V); res[~mask] = num[~mask] / denom[~mask]; res[mask] = 0.1
    return res

def beta_m_safe(V): return 4.0 * np.exp(-(V + 65.0) / 18.0)
def alpha_h_safe(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h_safe(V): return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
def beta_n_safe(V): return 0.125 * np.exp(-(V + 65.0) / 80.0)

def simulate_hh_core(weights, spikes_in, W_in, bias_current, dt=0.05, V_init=None):
    N = weights.shape[0]
    if V_init is None:
        V = np.random.uniform(-70, -60, N)
    else:
        V = V_init.copy()
        
    am, bm = alpha_m_safe(V), beta_m_safe(V)
    ah, bh = alpha_h_safe(V), beta_h_safe(V)
    an, bn = alpha_n_safe(V), beta_n_safe(V)
    m, h, n = am/(am+bm), ah/(ah+bh), an/(an+bn)
    b_gate = np.zeros(N)
    
    s_trace = np.zeros(N); s_in_trace = 0.0
    tau_in = 10.0; decay_in = np.exp(-dt / tau_in); decay_syn = np.exp(-dt / 5.0)
    C=1.0; gNa=120.0; ENa=50.0; gK=36.0; EK=-77.0; gL=0.3; EL=-54.4; gA=20.0; EA=-80.0; tauA=20.0; Eexc=0.0; Einh=-80.0
    
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
        syn = weights @ s_trace; I_syn = np.maximum(0, syn)*(V-Eexc) + np.maximum(0, -syn)*(V-Einh)
        dV = (-gNa*(m**3)*h*(V-ENa) - gK*(n**4)*(V-EK) - gL*(V-EL) - gA*(a_inf**3)*b_gate*(V-EA) - I_syn + I_pulse) / C
        V += dt * dV; V = np.clip(V, -100, 100)
        spks = ((V > -20.0) & (V_old <= -20.0)).astype(float); res_spikes[t] = spks; s_trace = s_trace * decay_syn + spks
    return res_spikes

def run_single_lyapunov(params):
    rho, bias, seed = params
    N = 100; dt = 0.05; win_ms = 20.0; steps_v = int(win_ms / dt)
    eps = 1e-6
    
    W = generate_dale_weights(N, 0.2, rho, random_seed=seed)
    rng = np.random.default_rng(seed); W_in = (rng.random(N) < 0.2).astype(float)
    V0 = rng.uniform(-70, -60, N)
    
    # Input signal (random in [0,1] for Lyapunov calculation)
    u = rng.uniform(0, 1, size=500)
    upsampled = np.repeat(u, steps_v)
    rates = 5.0 + upsampled * 195.0
    spikes_in = (rng.random(rates.shape) < (rates * dt / 1000.0)).astype(np.float32)
    
    S1 = simulate_hh_core(W, spikes_in, W_in, bias, V_init=V0)
    S2 = simulate_hh_core(W, spikes_in, W_in, bias, V_init=V0 + eps)
    
    Z1 = exp_trace(S1, 5e-5, 0.02); Z2 = exp_trace(S2, 5e-5, 0.02)
    P1 = window_features(Z1, steps_v); P2 = window_features(Z2, steps_v)
    
    logd = np.log(np.linalg.norm(P2 - P1, axis=1) + 1e-12)
    w_start, w_end = 50, 250
    if len(logd) < w_end: w_end = len(logd)
    
    if w_end > w_start + 10:
        coeffs = np.polyfit(np.arange(w_start, w_end), logd[w_start:w_end], 1)
        return {'rho': rho, 'bias': bias, 'lambda_cond': coeffs[0]}
    return {'rho': rho, 'bias': bias, 'lambda_cond': 0.0}

def main():
    spectral_radii = np.linspace(0.01, 1.2, 25); seeds = range(700, 705); biases = [0.0, 6.0]
    tasks = [(r, b, s) for b in biases for r in spectral_radii for s in seeds]
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {ex.submit(run_single_lyapunov, t): t for t in tasks}
        for f in as_completed(futures):
            results.append(f.result())
            if len(results) % 20 == 0: print(f"Progress: {len(results)}/{len(tasks)}")
    
    pd.DataFrame(results).groupby(['bias', 'rho'])['lambda_cond'].mean().unstack(level=0).to_csv(os.path.join(os.path.dirname(__file__), '../results/poisson_lyapunov_parallel_final.csv'))

if __name__ == "__main__":
    main()
