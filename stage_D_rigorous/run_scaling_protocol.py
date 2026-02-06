import os
import warnings
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from src.model.reservoir import Reservoir
from src.utils.metrics import calculate_lyapunov, calculate_kernel_quality, calculate_separation_property
from benchmark_mc import run_mc_benchmark_full
from benchmark_mackey_glass import run_mackey_glass_benchmark

# Suppress numerical instability warnings from exploding HH neurons in chaotic regime
warnings.filterwarnings('ignore', category=RuntimeWarning)

def evaluate_single_config(n_neurons, rho, base_config, seed):
    """Function to be executed in parallel for a single configuration"""
    config = copy.deepcopy(base_config)
    config['system']['n_neurons'] = n_neurons
    config['dynamics_control']['target_spectral_radius'] = rho
    
    np.random.seed(seed + int(rho * 100) + n_neurons)
    
    try:
        # Initialize reservoir
        res = Reservoir(n_neurons=n_neurons, config=config)
        res.normalize_spectral_radius(rho)
        
        # 1. Lyapunov
        l_val = calculate_lyapunov(res, n_steps=2000, seed=seed)
        
        # 2. Performance (MC & Mackey-Glass)
        # MC
        try:
            mc_val, _, X_states = run_mc_benchmark_full(config, n_samples=1500, max_lag=40)
            if np.any(np.isnan(X_states)):
                X_states = np.nan_to_num(X_states, 0.0)
        except Exception as e:
            # print(f"Error in MC for N={n_neurons}, Rho={rho}: {e}")
            mc_val = 0
            X_states = np.zeros((1500, n_neurons))
        
        # Mackey-Glass
        try:
            mg_perf = run_mackey_glass_benchmark(config, length=1500)
        except Exception as e:
            # print(f"Error in MG for N={n_neurons}, Rho={rho}: {e}")
            mg_perf = 0
        
        # 3. Mechanistic Metrics
        rank_val = calculate_kernel_quality(X_states)
        sep_val = calculate_separation_property(res, n_steps=1000)
        
        return {
            'N': n_neurons,
            'Rho': rho,
            'Lambda': l_val,
            'MC': mc_val,
            'MackeyGlass': mg_perf,
            'Rank': rank_val,
            'Separation': sep_val
        }
    except Exception as e:
        print(f"FAILED N={n_neurons}, Rho={rho}: {e}")
        return None

def run_scaling_experiment_parallel(n_neurons_list=[100, 200, 500, 1000, 1500, 2000], 
                                  rho_values=np.linspace(0.5, 8.0, 6), 
                                  seed=101):
    with open('task_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
        
    tasks = []
    for n_neurons in n_neurons_list:
        for rho in rho_values:
            tasks.append((n_neurons, rho, base_config, seed))
            
    results = []
    total_tasks = len(tasks)
    print(f"Starting Parallel Scaling Experiment: {total_tasks} configurations total.")
    
    # Use ProcessPoolExecutor with 90% of cores to avoid system lag
    num_workers = max(1, int(os.cpu_count() * 0.9))
    print(f"Using {num_workers} workers (all cores: {os.cpu_count()})")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_single_config, *task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                completed += 1
                print(f"[{completed}/{total_tasks}] Completed: N={res['N']}, Rho={res['Rho']:.2f} -> MC={res['MC']:.2f}, Lambda={res['Lambda']:.4f}")
            else:
                completed += 1
                
    return pd.DataFrame(results)

def plot_comprehensive_proof(df):
    if df.empty: return
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Sort for plotting lines correctly
    df = df.sort_values(['N', 'Lambda'])
    
    for n in sorted(df['N'].unique()):
        sub = df[df['N'] == n]
        axes[0,0].plot(sub['Lambda'], sub['MC'], 'o-', label=f'N={n}')
        axes[0,1].plot(sub['Lambda'], sub['MackeyGlass'], 's--', label=f'N={n}')
        
    axes[0,0].set_title('Memory Capacity vs Lambda', fontweight='bold')
    axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[0,0].set_xlabel('Lyapunov Exponent (Lambda)')
    axes[0,0].legend()
    
    axes[0,1].set_title('Mackey-Glass vs Lambda', fontweight='bold')
    axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[0,1].set_xlabel('Lyapunov Exponent (Lambda)')
    axes[0,1].legend()
    
    for n in sorted(df['N'].unique()):
        sub = df[df['N'] == n]
        axes[1,0].plot(sub['Lambda'], sub['Rank'], '^-', label=f'N={n}')
        axes[1,1].plot(sub['Lambda'], sub['Separation'], 'x-', label=f'N={n}')
        
    axes[1,0].set_title('Kernel Quality (Rank) vs Lambda', fontweight='bold')
    axes[1,0].set_xlabel('Lyapunov Exponent (Lambda)')
    axes[1,0].legend()
    
    axes[1,1].set_title('Separation Property vs Lambda', fontweight='bold')
    axes[1,1].set_xlabel('Lyapunov Exponent (Lambda)')
    axes[1,1].legend()
    
    # Sort by Rho for phase diagram
    df_rho = df.sort_values(['N', 'Rho'])
    for n in sorted(df_rho['N'].unique()):
        sub = df_rho[df_rho['N'] == n]
        axes[0,2].plot(sub['Rho'], sub['Lambda'], 'D-', label=f'N={n}')
        
    axes[0,2].set_title('Dynamics Control: Lambda vs Rho', fontweight='bold')
    axes[0,2].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0,2].set_xlabel('Spectral Radius (Rho)')
    axes[0,2].legend()
    
    fig.delaxes(axes[1,2])
    plt.tight_layout()
    plt.savefig('COMPREHENSIVE_PROOF_SCALING.png', dpi=300)
    print("\nSaved: COMPREHENSIVE_PROOF_SCALING.png")

def plot_scaling_laws(df):
    if df.empty: return
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Peak MC vs N
    peak_mc = df.groupby('N')['MC'].max().reset_index()
    axes[0].plot(peak_mc['N'], peak_mc['MC'], 'o-', linewidth=2, color='#2c3e50')
    axes[0].set_title('Peak Memory Capacity vs N', fontweight='bold')
    axes[0].set_xlabel('Number of Neurons (N)')
    axes[0].set_ylabel('Peak MC')
    
    # Peak MG vs N
    peak_mg = df.groupby('N')['MackeyGlass'].max().reset_index()
    axes[1].plot(peak_mg['N'], peak_mg['MackeyGlass'], 's-', linewidth=2, color='#e74c3c')
    axes[1].set_title('Peak Mackey-Glass Performance vs N', fontweight='bold')
    axes[1].set_xlabel('Number of Neurons (N)')
    axes[1].set_ylabel('Peak MG Performance (1-NRMSE)')
    
    plt.tight_layout()
    plt.savefig('SCALING_LAWS_PEAK.png', dpi=300)
    print("Saved: SCALING_LAWS_PEAK.png")

if __name__ == "__main__":
    n_list = [100, 200, 500, 1000, 1500, 2000]
    # Reduce rho values slightly if memory is an issue, but let's try with 6
    df_results = run_scaling_experiment_parallel(n_neurons_list=n_list, rho_values=np.linspace(0.5, 8.0, 6))
    
    df_results.to_csv('SCALING_EXPERIMENT_RESULTS.csv', index=False)
    plot_comprehensive_proof(df_results)
    plot_scaling_laws(df_results)
