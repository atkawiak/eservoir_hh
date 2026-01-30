
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spectral_utils import generate_dale_weights
from henon_task import HenonTask

# HH Params (Shared)
# ... duplicating minimal params for standalone script
N = 100
DENSITY = 0.2
C = 1.0
g_Na = 120.0; E_Na = 50.0
g_K = 36.0; E_K = -77.0
g_L_default = 0.3; E_L = -54.4
g_A = 20.0; E_A = -80.0
tau_A = 20.0
E_exc = 0.0; E_inh = -80.0
tau_syn = 5.0

def run_raster(rho, seed=42):
    print(f"Generating Raster for rho={rho}...")
    np.random.seed(seed)
    
    # Task
    TASK_STEPS = 200 # Short run for visualization
    henon = HenonTask(input_scale_min=0, input_scale_max=5.0, random_seed=seed)
    u_input, _ = henon.generate_data(TASK_STEPS)
    
    # Network (Dale's Law)
    W = generate_dale_weights(N, DENSITY, rho, excitatory_ratio=0.8, random_seed=seed)
    W_in = np.random.uniform(-1, 1, N)
    
    # Init
    V = np.random.uniform(-70, -60, N)
    m = np.zeros(N); h = np.zeros(N); n = np.zeros(N); b = np.zeros(N); s = np.zeros(N)
    
    STEP_DURATION = 20.0
    RAW_DT = 0.05
    STEPS_PER_TASK = int(STEP_DURATION / RAW_DT)
    
    spikes_t = []
    spikes_id = []
    
    leak_conductance = 0.3
    
    for t_task in range(TASK_STEPS):
        curr_u = u_input[t_task]
        for step in range(STEPS_PER_TASK):
            abs_t = t_task * STEPS_PER_TASK + step
            t_ms = abs_t * RAW_DT
            
            I_inj = 6.0 + curr_u * W_in
            
            # HH Update
            a_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))
            b = b + (RAW_DT / tau_A) * (b_inf - b)
            
            alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
            m = m + RAW_DT * (alpha_m * (1 - m) - beta_m * m)
            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            h = h + RAW_DT * (alpha_h * (1 - h) - beta_h * h)
            alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
            n = n + RAW_DT * (alpha_n * (1 - n) - beta_n * n)
            
            I_Na = g_Na * (m**3) * h * (V - E_Na)
            I_K = g_K * (n**4) * (V - E_K)
            I_L = leak_conductance * (V - E_L)
            I_A_curr = g_A * (a_inf**3) * b * (V - E_A)
            syn_inputs = W @ s
            I_syn = np.maximum(0, syn_inputs) * (V - E_exc) + np.maximum(0, -syn_inputs) * (V - E_inh)
            
            dV = (-I_Na - I_K - I_L - I_A_curr - I_syn + I_inj) / C
            V = V + RAW_DT * dV
            
            # Spike Detection (Threshold crossing)
            # Simple threshold check for raster (e.g., > 0mV)
            # Or use fire_rate proxy. Let's strictly check V > 0 and previous V <= 0?
            # For simplicity, just check V > 0 but we need to avoid multiple logs per spike.
            # Using fire_rate variable for synaptic update anyway.
            
            fire_rate = 1.0 / (1.0 + np.exp(-(V - -20.0)*2.0))
            s = s + RAW_DT * (-s/tau_syn + fire_rate)
            
            if np.any(np.abs(V) > 150): V = np.clip(V, -100, 100)
            
            # Store spikes for raster
            # Let's say spike if V > 0. To thin out, only record if specific condition met?
            # Actually, standard raster needs exact times.
            # Let's verify spike by finding peaks or crossing threshold -20mV upwards.
            # Implementing simple threshold check.
            active_indices = np.where(V > -20.0)[0]
            # This will record 'spike' for entire duration of AP. Raster plot points will look like lines. Acceptable.
            if len(active_indices) > 0:
                 # To avoid dense points, maybe subsample?
                 if step % 10 == 0:
                     spikes_t.extend([t_ms] * len(active_indices))
                     spikes_id.extend(active_indices)
                     
    return np.array(spikes_t), np.array(spikes_id)

def main():
    rhos = [0.05, 0.9, 1.2]
    plt.figure(figsize=(15, 5))
    
    for i, rho in enumerate(rhos):
        t, ids = run_raster(rho)
        plt.subplot(1, 3, i+1)
        if len(t) > 0:
            # Color excitatory (0-79) blue, inhibitory (80-99) red
            colors = ['blue' if idx < 80 else 'red' for idx in ids]
            plt.scatter(t, ids, s=1, c=colors, alpha=0.5)
        plt.title(f'Rho = {rho}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        plt.ylim(-1, 100)
        
    out_file = os.path.join(os.path.dirname(__file__), '../results/raster_plots.png')
    plt.savefig(out_file)
    print(f"Raster plot saved to {out_file}")

if __name__ == "__main__":
    main()
