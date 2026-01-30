
import numpy as np

def poisson_spike_train(rate_hz: np.ndarray, dt_s: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generates Poisson spike train from instantaneous rate.
    
    Args:
        rate_hz: (T,) array of instantaneous rates in Hz
        dt_s: Simulation time step in seconds
        rng: Random number generator
        
    Returns:
        (T,) spikes in {0,1}
    """
    p = np.clip(rate_hz * dt_s, 0.0, 1.0)
    return (rng.random(rate_hz.shape) < p).astype(np.float32)

def exp_trace(spikes: np.ndarray, dt_s: float, tau_s: float) -> np.ndarray:
    """
    Applies exponential filter to spike train (synaptic filter).
    x[t+1] = x[t]*exp(-dt/tau) + spikes[t]
    
    Args:
        spikes: (T,) or (T,N)
        dt_s: Simulation time step in seconds
        tau_s: Decay time constant in seconds
        
    Returns:
        same shape as spikes: exponentially filtered trace
    """
    decay = np.exp(-dt_s / tau_s)
    # If 1D array, make it 2D (T, 1) for uniform handling if needed, 
    # but let's keep it simple as requested. 
    # Logic: simple iterative filter.
    
    x = np.zeros_like(spikes, dtype=np.float32)
    
    # Use simple Python loop accelerated by numba if possible, but standard numpy loop here.
    # For speed in pure python, we might use infinite impulse response filter from scipy, 
    # but the manual loop is clearest for "minimal example".
    # Note: Loop in python is slow for large T. 
    # Optimization: scipy.signal.lfilter
    # Transfer function: H(z) = 1 / (1 - decay * z^-1)
    # y[n] = decay * y[n-1] + x[n-1] ... wait, the formula is x[t+1] = x[t]*decay + spikes[t]
    # This corresponds to: y[n] = decay * y[n-1] + x[n] (if spikes[t] is current input)
    # The snippet says: x[t+1] = x[t]*decay + spikes[t-1] implies delay?
    # User provided snippet: "x[t+1] = x[t]*exp(-dt/tau) + spikes[t-1]" -> Yes, delay 1 step.
    # Let's stick strictly to user snippet loop.
    
    for t in range(1, spikes.shape[0]):
        x[t] = x[t-1] * decay + spikes[t-1]
        
    return x

def window_features(traces: np.ndarray, win_len: int) -> np.ndarray:
    """
    Downsamples traces by averaging over windows.
    
    Args:
        traces: (T,N) array of filtered traces
        win_len: number of steps in window
        
    Returns:
        (K,N) array where K = T//win_len
    """
    T, N = traces.shape
    if N == 0: # Handle 1D case if passed as (T,)
        pass 
        
    K = T // win_len
    out = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        a = k * win_len
        b = a + win_len
        out[k] = traces[a:b].mean(axis=0)
    return out

def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solves Ridge Regression: W = (X^T X + lam*I)^-1 X^T Y
    
    Args:
        X: (K,D) features
        Y: (K,C) targets
        lam: regularization parameter
        
    Returns:
        W: (D,C) weights
    """
    D = X.shape[1]
    A = X.T @ X + lam * np.eye(D)
    B = X.T @ Y
    W = np.linalg.solve(A, B)
    return W
