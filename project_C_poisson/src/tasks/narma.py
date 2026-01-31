
import numpy as np
from typing import List

class NARMA:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        
    def generate_data(self, length: int, order: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Generates NARMA sequence. Uniform [0, 0.5] input per Journal-Grade spec."""
        u = self.rng.uniform(0, 0.5, length)
        y = np.zeros(length)
        
        for t in range(order, length):
            sum_y = np.sum(y[t-order:t])
            y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * sum_y + 1.5 * u[t-order] * u[t-1] + 0.1
            if np.abs(y[t]) > 1e10: # Numerical stability clip
                y[t] = np.sign(y[t]) * 1e10
            
        return u, y

    def compute_baseline(self, y_tgt: np.ndarray, 
                         readout_module, order: int,
                         alphas: List[float] = [1e-6, 1e-4, 1e-2, 1.0]) -> float:
        """
        Computes AR(k) baseline strictly on target y(t).
        y_tgt must be the SAME array used for readout target (downsampled).
        """
        n = len(y_tgt)
        if n <= order: return 0.0
        
        X_ar = np.zeros((n - order, order))
        y_ar = y_tgt[order:] # Shifted target for AR
        
        for i in range(order):
            X_ar[:, i] = y_tgt[order-(i+1) : n-(i+1)]  
            
        metrics = readout_module.train_ridge_cv(
            X_ar, y_ar, task_type="regression", alphas=alphas
        )
        return metrics['nrmse']
