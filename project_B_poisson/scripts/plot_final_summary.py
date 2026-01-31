
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results
results_dir = "/home/atkaw/Dokumenty/hailitacja/eservoir_hh/project_B_poisson/results"
henon_df = pd.read_csv(os.path.join(results_dir, "poisson_henon_parallel_final.csv"))
mc_dead_df = pd.read_csv(os.path.join(results_dir, "poisson_mc_dead_parallel_final.csv"))
mc_alive_df = pd.read_csv(os.path.join(results_dir, "poisson_mc_alive_parallel_final.csv"))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Henon Plot
ax1.plot(henon_df['rho'], henon_df['model_nrmse'], 'o-', label='HH Model NRMSE', color='red')
ax1.axhline(1.0, color='black', linestyle='--', label='Naive Baseline (Mean)')
ax1.set_title("Henon Task Performance vs Spectral Radius")
ax1.set_xlabel("Spectral Radius (rho)")
ax1.set_ylabel("NRMSE (Lower is better)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# MC Plot
ax2.plot(mc_dead_df['rho'], mc_dead_df['mc'], 's-', label='Dead Regime (Bias 0.0)', color='blue')
ax2.plot(mc_alive_df['rho'], mc_alive_df['mc'], '^-', label='Alive Regime (Bias 6.0)', color='green')
ax2.set_title("Memory Capacity vs Spectral Radius")
ax2.set_xlabel("Spectral Radius (rho)")
ax2.set_ylabel("Total MC (Higher is better)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "FINAL_SCIENTIFIC_SUMMARY.png"))
plt.show()
