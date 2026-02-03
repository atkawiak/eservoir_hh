
import numpy as np
import subprocess

def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except:
        return "unknown"

def filter_and_downsample(spikes: np.ndarray, steps_per_symbol: int, dt: float = 0.05, tau: float = 20.0) -> np.ndarray:
    """
    Memory-efficient Exponential filtering + Downsampling.
    Avoids O(T*N) intermediate storage.
    """
    T, N = spikes.shape
    alpha = np.exp(-dt / tau)
    r = np.zeros(N)
    
    indices = np.arange(steps_per_symbol - 1, T, steps_per_symbol)
    res = np.zeros((len(indices), N))
    
    idx_ptr = 0
    for t in range(T):
        r = r * alpha + spikes[t]
        if idx_ptr < len(indices) and t == indices[idx_ptr]:
            res[idx_ptr] = r
            idx_ptr += 1
            
    return res
