
import numpy as np
import scipy.sparse as sp

class EchoStateNetwork:
    """
    Standard ESN Baseline for Journal Comparison.
    Uses tanh activation, sparse weights, and spectral radius scaling.
    """
    def __init__(self, N: int, spectral_radius: float, 
                 input_scale: float, density: float = 0.1, 
                 leaking_rate: float = 1.0, seed: int = 42):
        self.N = N
        self.rho = spectral_radius
        self.input_scale = input_scale
        self.alpha = leaking_rate # Leaking rate (1 = no leak, standard ESN)
        self.rng = np.random.default_rng(seed)
        
        # Init Weights
        self.W = self._init_weights(density)
        self.W_in = self.rng.uniform(-1, 1, (N, 1)) * input_scale
        
    def _init_weights(self, density):
        # Sparse random weights [-1, 1]
        mask = sp.random(self.N, self.N, density=density, random_state=self.rng).toarray() > 0
        W = np.zeros((self.N, self.N))
        W[mask] = self.rng.uniform(-1, 1, size=np.sum(mask))
        
        # Spectral Radius
        eigenvals = np.linalg.eigvals(W)
        current_rho = np.max(np.abs(eigenvals))
        if current_rho > 0:
            W *= (self.rho / current_rho)
        return W

    def simulate(self, u: np.ndarray, washout: int = 100) -> np.ndarray:
        """
        Runs ESN simulation on input u.
        Returns state matrix X (T, N).
        """
        T = len(u)
        x = np.zeros(self.N)
        X_all = np.zeros((T, self.N))
        
        for t in range(T):
            # u[t] is scalar or vector? Assuming scalar for now comparable to HH tasks
            curr_u = u[t]
            
            # ESN Update
            # x(t) = (1-a)x(t-1) + a * tanh( Win*u(t) + W*x(t-1) )
            pre_act = (self.W_in.flatten() * curr_u) + (self.W @ x)
            x = (1.0 - self.alpha) * x + self.alpha * np.tanh(pre_act)
            X_all[t] = x
            
        return X_all[washout:]
