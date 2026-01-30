
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def main():
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    mc_path = os.path.join(results_dir, 'mc_robust_results.csv')
    henon_path = os.path.join(results_dir, 'henon_sweep_results.csv')
    
    if not os.path.exists(mc_path):
        print(f"Error: MC results not found at {mc_path}")
        return
    
    if not os.path.exists(henon_path):
        print(f"Error: Henon results not found at {henon_path}")
        return
        
    df_mc = pd.read_csv(mc_path)
    df_henon = pd.read_csv(henon_path)
    
    # Merge on rho (assuming approx equal floats, better to round)
    df_mc['rho_rounded'] = df_mc['rho'].round(3)
    df_henon['rho_rounded'] = df_henon['rho'].round(3)
    
    merged = pd.merge(df_mc, df_henon, on='rho_rounded', suffixes=('_mc', '_henon'))
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Spectral Radius (rho)')
    ax1.set_ylabel('Memory Capacity (MC)', color=color)
    ax1.plot(merged['rho_mc'], merged['mc'], color=color, marker='o', label='Memory Capacity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:orange'
    ax2.set_ylabel('Henon Prediction NRMSE (Lower is Better)', color=color)
    ax2.plot(merged['rho_henon'], merged['henon_nrmse'], color=color, marker='s', linestyle='--', label='Henon NRMSE')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Invert NRMSE axis to align "better" direction? 
    # Usually people prefer separate axes. Let's keep it standard but add arrow/text?
    # Or just keep it as is. Lower NRMSE = Better. Higher MC = Better.
    # If curves cross, that's interesting.
    
    plt.title('Approaching the Edge of Chaos: Memory vs Nonlinearity')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    out_file = os.path.join(results_dir, 'comparison_plot.png')
    plt.savefig(out_file)
    print(f"Comparison plot saved to {out_file}")
    
    # Identify Optima
    best_mc_idx = merged['mc'].idxmax()
    best_henon_idx = merged['henon_nrmse'].idxmin()
    
    print("\n--- Results Summary ---")
    print(f"Max Memory Capacity: {merged.loc[best_mc_idx, 'mc']:.2f} at rho={merged.loc[best_mc_idx, 'rho_mc']:.2f}")
    print(f"Min Henon NRMSE:     {merged.loc[best_henon_idx, 'henon_nrmse']:.3f} at rho={merged.loc[best_henon_idx, 'rho_henon']:.2f}")
    
    # Check for trade-off
    print("\nData Table:")
    print(merged[['rho_mc', 'mc', 'henon_nrmse']].to_string(index=False))

if __name__ == "__main__":
    main()
