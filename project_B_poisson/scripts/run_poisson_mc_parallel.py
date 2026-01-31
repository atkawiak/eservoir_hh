
import numpy as np
import pandas as pd
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import Ridge

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spectral_utils import generate_dale_weights
from memory_capacity_task import MemoryCapacityTask
from poisson_utils import exp_trace, window_features

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

def simulate_hh_core(weights, spikes_in, W_in, bias_current, dt=0.05, trim_steps=0):
    N = weights.shape[0]
    V = np.random.uniform(-70, -60, N)
    am, bm = alpha_m_safe(V), beta_m_safe(V)
    ah, bh = alpha_h_safe(V), beta_h_safe(V)
    an, bn = alpha_n_safe(V), beta_n_safe(V)
    m = am / (am + bm); h = ah / (ah + bh); n = an / (an + bn)
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
        
    if trim_steps > 0:
        return res_spikes[trim_steps:]
    return res_spikes

def run_single_mc(params):
    rho, bias, seed = params
    N = 100; dt = 0.05; win_ms = 20.0; steps_v = int(win_ms / dt)
    W = generate_dale_weights(N, 0.2, rho, random_seed=seed)
    rng = np.random.default_rng(seed); W_in = (rng.random(N) < 0.2).astype(float)
    
    # Generate Data
    mc_task = MemoryCapacityTask(max_lag=20, random_seed=seed)
    u = mc_task.generate_data(2500) # Increased simulation time
    u_norm = (u - u.min()) / (u.max() - u.min())
    
    # Generate Spikes (Independent)
    upsampled = np.repeat(u_norm, steps_v)
    rates = 5.0 + upsampled * 195.0
    spikes_in = (rng.random(rates.shape) < (rates * dt / 1000.0)).astype(np.float32)
    
    # Transient discard in HH (200 steps)
    WASH_OUT = 200
    trim_steps = WASH_OUT * steps_v 
    S = simulate_hh_core(W, spikes_in, W_in, bias, trim_steps=trim_steps)
    
    # Window Features
    Phi = window_features(exp_trace(S, dt/1000.0, 0.02), steps_v)
    
    # Align Input: We discarded first 200 steps of output, so alignment starts from 200
    u_aligned = u_norm[WASH_OUT:WASH_OUT+len(Phi)]
    
    # Compute MC (Small readout washout of 50 steps)
    res = mc_task.compute_memory_capacity(Phi, u_aligned, wash_out=50) 
    return {'rho': rho, 'bias': bias, 'mc': res['total_mc']}

def main():
    spectral_radii = np.linspace(0.01, 1.2, 25); seeds = range(500, 510); biases = [0.0, 6.0]
    tasks = [(r, b, s) for b in biases for r in spectral_radii for s in seeds]
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {ex.submit(run_single_mc, t): t for t in tasks}
        for f in as_completed(futures):
            results.append(f.result())
            if len(results) % 20 == 0: print(f"Progress: {len(results)}/{len(tasks)}")
    
    df = pd.DataFrame(results)
    out_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(out_dir, exist_ok=True)
    for b in biases:
        label = "dead" if b < 1.0 else "alive"
        df[df['bias'] == b].groupby('rho')['mc'].mean().to_csv(os.path.join(out_dir, f'poisson_mc_{label}_parallel_final.csv'))

if __name__ == "__main__":
    main()
