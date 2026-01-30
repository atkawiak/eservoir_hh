
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spectral_utils import generate_echo_state_weights, generate_dale_weights
from henon_task import HenonTask

# --- Simulation Parameters ---
# Increased steps for Henon task to ensure sufficient training data
TASK_STEPS = 1000 
WASHOUT = 200
N_NEURONS = 100
DENSITY = 0.2
dt = 0.1  # ms

# HH Parameters
C = 1.0
g_Na = 120.0; E_Na = 50.0
g_K = 36.0; E_K = -77.0
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
    Runs the HH network simulation for Henon Task.
    """
    STEP_DURATION = 20.0 
    RAW_DT = 0.05 
    STEPS_PER_TASK = int(STEP_DURATION / RAW_DT)
    
    # Initialize Henon Task
    # Scale input to [0, 5.0] nA to potentially drive neurons but not saturate too much
    henon = HenonTask(input_scale_min=0, input_scale_max=5.0, random_seed=sim_seed)
    
    # Generate Data
    # Only requesting TASK_STEPS. The task internal logic handles generation washout.
    # u_input is the input current sequence. target is the value to predict.
    u_input, target = henon.generate_data(TASK_STEPS)
    
    # Check lengths
    assert len(u_input) == TASK_STEPS
    # Target length might be slightly different depending on implementation? 
    # generate_data returns input_seq and target_seq corresponding to same time steps if aligned.
    # Let's verify HenonTask.generate_data implementation:
    # input_seq = x_sequence[:timesteps]
    # target_seq = x_sequence[1:timesteps+1]
    # So they are equal length. Good.
    
    # SEEDED initialization
    np.random.seed(sim_seed)
    V = np.random.uniform(-70, -60, N_NEURONS)
    V_prev = V.copy()
    m = np.zeros(N_NEURONS); h = np.zeros(N_NEURONS); n = np.zeros(N_NEURONS)
    b_gate = np.zeros(N_NEURONS) # A-current gate
    s_trace = np.zeros(N_NEURONS) # Synaptic activation
    
    W_in = np.random.uniform(-1, 1, N_NEURONS) 
    
    state_matrix = np.zeros((TASK_STEPS, N_NEURONS))
    decay_syn = np.exp(-RAW_DT / tau_syn)
    
    # Simulation Loop
    for t_task in range(TASK_STEPS):
        curr_u = u_input[t_task]
        
        for _ in range(STEPS_PER_TASK):
            # Store V_old
            V_old = V.copy()
            
            # HH Dynamics (Gates)
            a_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))
            b_gate += (RAW_DT / tau_A) * (b_inf - b_gate)
            
            alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
            m += RAW_DT * (alpha_m * (1 - m) - beta_m * m)
            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            h += RAW_DT * (alpha_h * (1 - h) - beta_h * h)
            alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
            n += RAW_DT * (alpha_n * (1 - n) - beta_n * n)
            
            # Conductance currents
            syn_activity = weights @ s_trace
            g_exc = np.maximum(0, syn_activity)
            g_inh = np.maximum(0, -syn_activity)
            I_syn = g_exc * (V - E_exc) + g_inh * (V - E_inh)
            
            # Input Injection
            I_inj = 6.0 + curr_u * W_in 
            
            # V update
            I_Na = g_Na * (m**3) * h * (V - E_Na)
            I_K = g_K * (n**4) * (V - E_K)
            I_L = leak_conductance * (V - E_L)
            I_A_curr = g_A * (a_inf**3) * b_gate * (V - E_A)
            
            dV = (-I_Na - I_K - I_L - I_A_curr - I_syn + I_inj) / C
            V += RAW_DT * dV
            
            if np.any(np.abs(V) > 150): V = np.clip(V, -100, 100)
            
            # Rising-Edge Spike Detection (Correct)
            spikes = ((V > -20.0) & (V_old <= -20.0)).astype(float)
            
            # Update Synaptic Trace (Post-Spike)
            s_trace = s_trace * decay_syn + spikes
            
        # Sample state at the end of the 20ms window
        state_matrix[t_task, :] = s_trace
        
    # Evaluate Performance
    # wash_out_reservoir should be passed to exclude initial transients
    results = henon.compute_performance(state_matrix, target, wash_out_reservoir=WASHOUT)
    return results['nrmse']

def main():
    spectral_radii = np.linspace(0.05, 1.2, 12)
    N_SEEDS = 3
    leak = 0.3 
    
    all_results = []
    print(f"Starting HENON SWEEP over {len(spectral_radii)} radii, averaging {N_SEEDS} seeds...")
    
    output_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(output_dir, exist_ok=True)
    
    for rho in spectral_radii:
        print(f"Testing rho = {rho:.2f}: ", end="", flush=True)
        scores = []
        for s_idx in range(N_SEEDS):
            seed = 200 + s_idx # Different seeds than MC to be independent, or same if we want direct comparison? Let's use different for now.
            # Use Dale's Law Weights
            W = generate_dale_weights(N_NEURONS, DENSITY, rho, excitatory_ratio=0.8, random_seed=seed)
            try:
                nrmse = run_simulation(W, rho, leak, sim_seed=seed)
                scores.append(nrmse)
                print(f"[{nrmse:.3f}]", end="", flush=True)
            except Exception as e:
                print(f"x({e})", end="", flush=True)
        
        avg_nrmse = np.mean(scores) if scores else np.nan
        print(f" -> Avg NRMSE: {avg_nrmse:.4f}")
        all_results.append({'rho': rho, 'henon_nrmse': avg_nrmse})
            
    df = pd.DataFrame(all_results)
    out_path = os.path.join(output_dir, 'henon_sweep_results.csv')
    df.to_csv(out_path, index=False)
    
    plt.figure()
    plt.plot(df['rho'], df['henon_nrmse'], 's-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Spectral Radius (rho)')
    plt.ylabel('Henon Prediction NRMSE')
    plt.title('HH Reservoir: Nonlinear Prediction vs Edge of Chaos')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'henon_plot.png'))
    print(f"\nDone! Results saved to {out_path}")

if __name__ == "__main__":
    main()
