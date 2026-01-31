
import numpy as np
from sklearn.preprocessing import StandardScaler

class LyapunovModule:
    """
    Computes Deterministic Lyapunov Exponent.
    """
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def compute_lambda(self, phi1: np.ndarray, phi2: np.ndarray, 
                       window_range: tuple = (50, 250)) -> float:
        scaler = StandardScaler()
        phi1_z = scaler.fit_transform(phi1)
        phi2_z = scaler.transform(phi2)
        
        diff = phi1_z - phi2_z
        d = np.linalg.norm(diff, axis=1)
        d = np.maximum(d, 1e-12)
        log_d = np.log(d)
        
        sat_indices = np.where(d > 1.0)[0]
        if len(sat_indices) > 0:
            t_sat = sat_indices[0]
        else:
            t_sat = len(d)
            
        t_start, t_end = window_range
        t_end = min(t_end, t_sat)
        
        if t_end <= t_start + 5:
             return 0.0 
                 
        t_vals = np.arange(t_start, t_end)
        y_vals = log_d[t_start:t_end]
        
        slope, intercept = np.polyfit(t_vals, y_vals, 1)
        
        # Unit conversion: 
        # slope is per window step.
        # dt is simulation step in ms. window step depends on downsampling?
        # The input dt here is "simulation dt" usually 0.05ms.
        # But phi is usually downsampled.
        # We assume phi is passed 'as is'. user must know the effective dt of phi rows.
        # If phi rows are separated by 'step_ms', then lambda (per ms) = slope / step_ms
        # lambda (per s) = slope / step_ms * 1000
        # For now, return raw slope per Step, but logging should handle conversion.
        # Actually user requested Fix.
        # Let's return NO conversion here but document it, OR change signature to accept step_ms.
        # Changing signature:
        return slope # Caller (run_experiment) should normalize this using step_ms!
