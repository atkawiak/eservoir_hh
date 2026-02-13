
import numpy as np
import yaml
import logging
from src.reservoir.build_reservoir import build_reservoir
from src.reservoir.lyapunov import estimate_lyapunov

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def search_edge_of_chaos():
    # Load baseline config
    with open("configs/scientific/FULL_mc_maass.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    # Use N=150 for better statistical properties at criticality
    cfg['network']['N'] = 150
    # Higher Inhibitory Scaling for Balanced State (Legenstein 2007)
    inh_scaling = 6.0 

    # Spectral Radius Sweep Range (Spiking networks often need much higher rho)
    rho_range = np.linspace(1.0, 10.0, 19) # 1.0, 1.5, ..., 10.0

    results = []

    print("\n" + "="*50)
    print("SEARCHING FOR THE EDGE OF CHAOS (N=150, inh_scaling=6.0)")
    print("="*50)
    print(f"{'Rho':<10} | {'Lambda (ms^-1)':<20} | {'State':<15}")
    print("-" * 50)

    rhos = []
    lambdas = []

    for rho in rho_range:
        cfg['network']['spectral_radius'] = float(rho)
        
        # Build reservoir
        res = build_reservoir(cfg, seed=42)
        
        # Estimate MLE
        res_lyap = estimate_lyapunov(res, cfg, inh_scaling=inh_scaling, seed_lyapunov=100, seed_input=200)
        
        lam = res_lyap.lambda_estimate
        state = "STABLE" if lam < -0.01 else "CHAOTIC" if lam > 0.01 else "EDGE"
        
        print(f"{rho:<10.2f} | {lam:<20.4f} | {state:<15}")
        
        rhos.append(rho)
        lambdas.append(lam)
        results.append({'rho': rho, 'lambda': lam})

    # Find the critical rho (closest to zero)
    idx_crit = np.argmin(np.abs(lambdas))
    rho_crit = rhos[idx_crit]
    lam_crit = lambdas[idx_crit]

    print("\n" + "="*50)
    print(f"IDENTIFIED EDGE OF CHAOS: rho = {rho_crit:.2f} (Lambda = {lam_crit:.4f})")
    print("="*50)

    # Save data
    np.savez("edge_of_chaos_data.npz", rhos=rhos, lambdas=lambdas)
    
    # Save a config for the EoC
    cfg['network']['spectral_radius'] = float(rho_crit)
    with open("configs/scientific/EDGE_OF_CHAOS_SPEC.yaml", 'w') as f:
        yaml.dump(cfg, f)
    
    print(f"Saved Edge of Chaos spec to: configs/scientific/EDGE_OF_CHAOS_SPEC.yaml")

if __name__ == "__main__":
    search_edge_of_chaos()
