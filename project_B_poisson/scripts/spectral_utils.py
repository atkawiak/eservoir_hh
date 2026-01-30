
import numpy as np

def generate_echo_state_weights(N, density, spectral_radius, random_seed=None):
    """
    Generates a sparse recurrent weight matrix with a specific spectral radius.
    
    Args:
        N (int): Number of neurons (size of matrix NxN).
        density (float): Connectivity probability (0 < density <= 1).
        spectral_radius (float): Target spectral radius (largest absolute eigenvalue).
        random_seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.array: NxN weight matrix.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # 1. Generate sparse mask
    # We use a uniform distribution for weights initially in [-1, 1]
    W = (np.random.rand(N, N) - 0.5) * 2
    
    # Apply sparsity
    mask = np.random.rand(N, N) < density
    W *= mask
    
    # 2. Compute current spectral radius
    # Use eigvals to find max absolute eigenvalue
    eigenvalues = np.linalg.eigvals(W)
    current_rho = np.max(np.abs(eigenvalues))
    
    # Avoid division by zero if matrix is all zeros (unlikely but possible with low density/N)
    if current_rho == 0:
        current_rho = 1e-8
        
    # 3. Rescale
    W_scaled = W * (spectral_radius / current_rho)
    
    return W_scaled

def generate_dale_weights(N, density, spectral_radius, excitatory_ratio=0.8, random_seed=None):
    """
    Generates a weight matrix respecting Dale's Law:
    - distinct excitatory (weights > 0) and inhibitory (weights < 0) neurons.
    - Excitatory neurons only project positive weights.
    - Inhibitory neurons only project negative weights.
    
    Args:
        N (int): Number of neurons.
        density (float): Connectivity probability.
        spectral_radius (float): Target spectral radius.
        excitatory_ratio (float): Fraction of excitatory neurons (default 0.8).
        random_seed (int): Random seed.
        
    Returns:
        np.array: NxN weight matrix respecting Dale's Law.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    N_E = int(N * excitatory_ratio)
    N_I = N - N_E
    
    # 1. Initialize weights
    # Use uniform distribution like in the baseline to avoid introduce too many variables
    W_abs = np.random.uniform(0, 1, (N, N))
    
    # 2. Apply Sparsity
    mask = np.random.rand(N, N) < density
    W_sparse = W_abs * mask
    
    # 3. Apply Signs based on pre-synaptic neuron type (Columns)
    signs = np.ones(N)
    signs[N_E:] = -1.0 # Last N_I neurons are inhibitory
    
    W_signed = W_sparse * signs[np.newaxis, :]
    
    # 4. Rescale Spectral Radius
    
    eigenvalues = np.linalg.eigvals(W_signed)
    current_rho = np.max(np.abs(eigenvalues))
    
    if current_rho == 0:
        current_rho = 1e-8
        
    W_final = W_signed * (spectral_radius / current_rho)
    
    return W_final
