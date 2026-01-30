
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spectral_utils import generate_dale_weights
from henon_task import HenonTask
from poisson_utils import poisson_spike_train, exp_trace, window_features, ridge_fit

# --- Simulation Parameters ---
TASK_STEPS = 1000 
N_NEURONS = 100
DENSITY = 0.2
dt = 0.05  # ms
dt_s = dt / 1000.0

# Task window
TASK_WINDOW_MS = 20.0
STEPS_PER_TASK = int(TASK_WINDOW_MS / dt)

# Biological Params
tau_syn = 5.0 # ms
decay_syn = np.exp(-dt / tau_syn)
E_exc = 0.0
E_inh = -80.0

# Input Encoding (Poisson)
R_MIN, R_MAX = 5.0, 200.0
TAU_IN = 10.0 # ms
DECAY_IN = np.exp(-dt / TAU_IN)

def run_simulation(weights, spectral_radius, bias_current=6.0, sim_seed=42):
    """
    Runs HH simulation with Poisson encoding for Henon Task.
    """
    # 1. Data Generation
    # HenonTask returns u in [0, 5], but for Poisson we normalize as in tmp_P_poission.py
    henon = HenonTask(input_scale_min=0, input_scale_max=5.0, random_seed=sim_seed)
    u_analog, target = henon.generate_data(TASK_STEPS)
    
    # Normalize for rate coding
    u_norm = (u_analog - u_analog.min()) / (u_analog.max() - u_analog.min())
    
    # 2. Input Spiking
    u_upsampled = np.repeat(u_norm, STEPS_PER_TASK)
    rates = R_MIN + u_upsampled * (R_MAX - R_MIN)
    rng = np.random.default_rng(sim_seed)
    spikes_in = (rng.random(rates.shape) < (rates * dt_s)).astype(np.float32)
    
    # 3. Initialization
    V = np.random.uniform(-70, -60, N_NEURONS)
    V_prev = V.copy()
    m = np.zeros(N_NEURONS); h = np.zeros(N_NEURONS); n = np.zeros(N_NEURONS)
    b_gate = np.zeros(N_NEURONS)
    s_trace = np.zeros(N_NEURONS)
    s_in_trace = 0.0
    
    # Input projections (20%)
    W_in = (rng.random(N_NEURONS) < 0.2).astype(float)
    
    res_spikes = np.zeros((TASK_STEPS * STEPS_PER_TASK, N_NEURONS), dtype=np.float32)
    
    # Constants
    C=1.0; gNa=120.0; ENa=50.0; gK=36.0; EK=-77.0; gL=0.3; EL=-54.4
    gA=20.0; EA=-80.0; tauA=20.0
    
    # 4. Simulation Loop
    total_steps = TASK_STEPS * STEPS_PER_TASK
    for t in range(total_steps):
        # Store state BEFORE update
        V_old = V.copy()
        
        # Input/Pulse update
        s_in_trace = s_in_trace * DECAY_IN + spikes_in[t]
        I_pulse = (5.0 * s_in_trace) * W_in + bias_current
        
        # HH Dynamics (Gates)
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
        
        # Conductance currents
        syn_activity = weights @ s_trace
        g_exc = np.maximum(0, syn_activity)
        g_inh = np.maximum(0, -syn_activity)
        I_syn = g_exc * (V - E_exc) + g_inh * (V - E_inh)
        
        # V update
        I_Na = gNa * (m**3) * h * (V - ENa)
        I_K = gK * (n**4) * (V - EK)
        I_L = gL * (V - EL)
        I_A = gA * a_inf**3 * b_gate * (V - EA)
        dV = (-I_Na - I_K - I_L - I_A - I_syn + I_pulse) / C
        V += dt * dV
        
        # Clip
        if np.any(np.abs(V) > 150): V = np.clip(V, -100, 100)
        
        # Rising-edge spikes (Correct Logic)
        spikes = ((V > -20.0) & (V_old <= -20.0)).astype(float)
        res_spikes[t] = spikes
        
        # Synapse update (Post-Spike)
        s_trace = s_trace * decay_syn + spikes

    # 5. Readout Processing (Poisson style)
    # Filter spikes with 20ms tau
    Z_res = exp_trace(res_spikes, dt / 1000.0, tau_s=0.02)
    # Downsample
    Phi = window_features(Z_res, STEPS_PER_TASK)
    
    # 6. Performance
    return henon.compute_performance(Phi, target, wash_out_reservoir=200)['nrmse']

def main():
    spectral_radii = np.linspace(0.05, 1.2, 12)
    N_SEEDS = 3
    bias = 6.0
    
    all_results = []
    print(f"Starting POISSON HENON SWEEP...")
    
    for rho in spectral_radii:
        print(f"rho = {rho:.2f}: ", end="", flush=True)
        scores = []
        for s_idx in range(N_SEEDS):
            seed = 300 + s_idx
            W = generate_dale_weights(N_NEURONS, DENSITY, rho, random_seed=seed)
            nrmse = run_simulation(W, rho, bias_current=bias, sim_seed=seed)
            scores.append(nrmse)
            print(f"[{nrmse:.3f}]", end="", flush=True)
            
        avg_nrmse = np.mean(scores)
        print(f" -> Avg NRMSE: {avg_nrmse:.4f}")
        all_results.append({'rho': rho, 'henon_nrmse': avg_nrmse})
            
    df = pd.DataFrame(all_results)
    out_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'poisson_henon_results.csv'), index=False)
    
    plt.figure()
    plt.plot(df['rho'], df['henon_nrmse'], 'o-', color='purple')
    plt.xlabel('rho')
    plt.ylabel('NRMSE')
    plt.title('Project B: Poisson Henon Sweep')
    plt.savefig(os.path.join(out_dir, 'poisson_henon_plot.png'))

if __name__ == "__main__":
    main()
