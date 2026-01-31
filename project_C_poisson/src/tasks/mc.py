
import numpy as np

class MemoryCapacity:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate_data(self, length: int) -> np.ndarray:
        return self.rng.uniform(0, 1, length)

    def run_mc_analysis(self, phi: np.ndarray, u: np.ndarray, 
                       readout_module, max_lag: int = 20) -> dict:
        
        mc_total = 0.0
        r2_per_lag = []
        valid_len = len(phi) - max_lag
        phi_valid = phi[max_lag:]
        
        for k in range(1, max_lag + 1):
            y_tgt = u[max_lag-k : -k] if k < max_lag else u[: -max_lag]
            if len(y_tgt) != len(phi_valid):
                 min_len = min(len(y_tgt), len(phi_valid))
                 y_tgt = y_tgt[:min_len]
                 phi_curr = phi_valid[:min_len]
            else:
                 phi_curr = phi_valid

            res = readout_module.train_ridge_cv(phi_curr, y_tgt, task_type="regression")
            r2 = 1.0 - (res['nrmse']**2)
            r2 = max(0.0, r2)
            
            mc_total += r2
            r2_per_lag.append(r2)
            
        return {'mc': mc_total, 'r2_by_lag': r2_per_lag}

    def compute_baseline(self, u: np.ndarray, phi: np.ndarray, 
                         readout_module, rng_shuffle: np.random.Generator, max_lag: int = 20) -> float:
        # Shuffle phi across time to destroy temporal structure
        # Use explicit RNG for reproducibility
        phi_shuffled = phi.copy()
        rng_shuffle.shuffle(phi_shuffled) 
        
        res = self.run_mc_analysis(phi_shuffled, u, readout_module, max_lag)
        return res['mc']
