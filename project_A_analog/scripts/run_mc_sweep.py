
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spectral_utils import generate_echo_state_weights, generate_dale_weights
from memory_capacity_task import MemoryCapacityTask

# --- Simulation Parameters ---
TIMESTEPS = 3000
WASHOUT = 500
N_NEURONS = 100
DENSITY = 0.2
dt = 0.1  # ms

# HH Parameters
C = 1.0
g_Na = 120.0; E_Na = 50.0
g_K = 36.0; E_K = -77.0
# g_L will be variable or fixed
g_L_default = 0.3; E_L = -54.4

# Shriki A-current params
g_A = 20.0; E_A = -80.0
tau_A = 20.0

# Synaptic params
E_exc = 0.0
E_inh = -80.0
tau_syn = 5.0

def run_simulation(weights, spectral_radius, leak_conductance, sim_seed=42):
    """
    Runs the HH network simulation.
    """
    # Task Parameters (Optimized for speed)
    TASK_STEPS = 600 
    STEP_DURATION = 20.0 
    RAW_DT = 0.05 
    STEPS_PER_TASK = int(STEP_DURATION / RAW_DT)
    WASH_OUT_STEPS = 100
    
    mc_task = MemoryCapacityTask(max_lag=20, random_seed=sim_seed) 
    u_input = mc_task.generate_data(TASK_STEPS)
    
    # SEEDED initialization for state variables
    np.random.seed(sim_seed)
    V = np.random.uniform(-70, -60, N_NEURONS)
    m = np.zeros(N_NEURONS); h = np.zeros(N_NEURONS); n = np.zeros(N_NEURONS); b = np.zeros(N_NEURONS)
    s_trace = np.zeros(N_NEURONS)
    
    W_in = np.random.uniform(-1, 1, N_NEURONS) 
    
    state_matrix = np.zeros((TASK_STEPS, N_NEURONS))
    decay_syn = np.exp(-RAW_DT / tau_syn)
    
    for t_task in range(TASK_STEPS):
        curr_u = u_input[t_task]
        for _ in range(STEPS_PER_TASK):
            # Store V_old
            V_old = V.copy()
            
            # HH Dynamics (Gates)
            a_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))
            b = b + (RAW_DT / tau_A) * (b_inf - b)
            
            # Standard gates (simplified update for speed in multi-run)
            alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
            m = m + RAW_DT * (alpha_m * (1 - m) - beta_m * m)
            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            h = h + RAW_DT * (alpha_h * (1 - h) - beta_h * h)
            alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
            n = n + RAW_DT * (alpha_n * (1 - n) - beta_n * n)
            
            # Conductance currents
            syn_activity = weights @ s_trace
            g_exc = np.maximum(0, syn_activity)
            g_inh = np.maximum(0, -syn_activity)
            I_syn = g_exc * (V - E_exc) + g_inh * (V - E_inh)
            
            # Input Injection
            I_inj = 6.0 + 10.0 * curr_u * W_in 
            
            # V update
            I_Na = g_Na * (m**3) * h * (V - E_Na)
            I_K = g_K * (n**4) * (V - E_K)
            I_L = leak_conductance * (V - E_L)
            I_A_curr = g_A * (a_inf**3) * b * (V - E_A)
            
            dV = (-I_Na - I_K - I_L - I_A_curr - I_syn + I_inj) / C
            V = V + RAW_DT * dV
            
            if np.any(np.abs(V) > 150): V = np.clip(V, -100, 100)
            
            # Rising-Edge Spike Detection (Correct)
            spikes = ((V > -20.0) & (V_old <= -20.0)).astype(float)
            
            # Update Synaptic Trace (Post-Spike)
            s_trace = s_trace * decay_syn + spikes
            
        state_matrix[t_task, :] = s_trace
        
    results = mc_task.compute_memory_capacity(state_matrix, u_input, wash_out=WASH_OUT_STEPS)
    return results['total_mc']

def main():
    spectral_radii = np.linspace(0.05, 1.2, 12)
    N_SEEDS = 3
    leak = 0.3 
    
    all_results = []
    print(f"Starting ROBUST sweep over {len(spectral_radii)} radii, averaging {N_SEEDS} seeds...")
    
    for rho in spectral_radii:
        print(f"Testing rho = {rho:.2f}: ", end="", flush=True)
        mcs = []
        for s_idx in range(N_SEEDS):
            seed = 100 + s_idx
            # Use Dale's Weights
            W = generate_dale_weights(N_NEURONS, DENSITY, rho, excitatory_ratio=0.8, random_seed=seed)
            try:
                mc = run_simulation(W, rho, leak, sim_seed=seed)
                mcs.append(mc)
                print(f"[{mc:.2f}]", end="", flush=True)
            except:
                print("x", end="", flush=True)
        
        avg_mc = np.mean(mcs) if mcs else 0
        print(f" -> Avg MC: {avg_mc:.4f}")
        all_results.append({'rho': rho, 'mc': avg_mc})
            
    df = pd.DataFrame(all_results)
    out_path = os.path.join(os.path.dirname(__file__), '../results/mc_robust_results.csv')
    df.to_csv(out_path, index=False)
    
    plt.figure()
    plt.plot(df['rho'], df['mc'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Spectral Radius (rho)')
    plt.ylabel('Memory Capacity')
    plt.title('HH Reservoir: Memory Capacity vs Edge of Chaos')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(os.path.dirname(__file__), '../results/mc_plot.png'))
    print(f"\nDone! Results saved to results/mc_robust_results.csv")

if __name__ == "__main__":
    main()
