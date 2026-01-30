
import numpy as np
from sklearn.linear_model import Ridge

class MemoryCapacityTask:
    def __init__(self, max_lag=50, random_seed=None):
        """
        Args:
            max_lag (int): Maximum delay k to test memory for.
            random_seed (int): Seed for input signal generation.
        """
        self.max_lag = max_lag
        self.rng = np.random.RandomState(random_seed)
        
    def generate_data(self, timesteps):
        """
        Generates random input signal u(t).
        Input is drawn uniformly from [-1, 1].
        """
        u = self.rng.uniform(-1, 1, size=timesteps)
        return u
        
    def compute_memory_capacity(self, states, u_signal, wash_out=100):
        """
        Trains readout weights to predict u(t-k) from reservoir states x(t).
        
        Args:
            states (np.array): Reservoir states of shape (time, N).
            u_signal (np.array): Input signal of shape (time,).
            wash_out (int): Number of initial steps to discard.
            
        Returns:
            dict: {
                'total_mc': sum(R^2),
                'lags': list of lags k,
                'r2_scores': list of R^2 scores for each k
            }
        """
        # Discard washout
        X = states[wash_out:, :]
        targets_base = u_signal[wash_out:]
        
        # Determine valid length after accommodating max_lag lookback
        # We need u(t-k). So for the current X[t], we need u[t-k + washout]
        # BUT: u_signal and states are aligned in time indices. 
        # states[t] corresponds to input u[t]. 
        # Task: predict u[t-k] using states[t].
        
        T_valid = len(X)
        
        r2_scores = []
        lags = range(1, self.max_lag + 1)
        
        # Rigid regression for stability
        ridge = Ridge(alpha=1e-4) # Small regularization
        
        for k in lags:
            # Target is u shifted by k
            # X_train: X[k:] (we can't predict first k steps because history doesn't exist)
            # y_train: targets_base[:-k]
            
            # Actually, to keep X consistent size, let's just truncate everything by max_lag
            # Or do it per lag to maximize data usage.
            # Let's do per lag.
            
            # X at time t (relative to washout start)
            # y at time t should be u(t-k)
            # Valid range: t starts at k
            
            X_k = X[k:]
            y_k = targets_base[:-k]
            
            # Train - Test split (e.g. 50/50 or 80/20)
            # For MC calculation, usually we use a good chunk for training
            split_idx = int(len(X_k) * 0.8)
            
            X_train, X_test = X_k[:split_idx], X_k[split_idx:]
            y_train, y_test = y_k[:split_idx], y_k[split_idx:]
            
            if len(X_train) == 0:
                r2_scores.append(0)
                continue
                
            ridge.fit(X_train, y_train)
            score = ridge.score(X_test, y_test)
            
            # MC definition sums R^2, only positive ones count
            # Use max(0, score) because negative R^2 means worse than mean, effectively 0 memory
            r2_scores.append(max(0, score))
            
        total_mc = sum(r2_scores)
        
        return {
            'total_mc': total_mc,
            'lags': list(lags),
            'r2_scores': r2_scores
        }
