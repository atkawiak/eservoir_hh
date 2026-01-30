
import numpy as np
from sklearn.linear_model import Ridge

class HenonTask:
    def __init__(self, a=1.4, b=0.3, input_scale_min=0, input_scale_max=20, random_seed=None):
        """
        Args:
            a, b (float): Henon map parameters.
            input_scale_min/max (float): Range to scale the input current for the HH neurons.
            random_seed (int): Seed for initialization.
        """
        self.a = a
        self.b = b
        self.input_scale_min = input_scale_min
        self.input_scale_max = input_scale_max
        self.rng = np.random.RandomState(random_seed)
        
    def generate_data(self, timesteps, wash_out_gen=1000):
        """
        Generates Henon map time series x(t).
        The reservoir will be driven by x(t) and trained to predict x(t+1).
        
        Args:
            timesteps (int): Total steps required (including internal washout).
            wash_out_gen (int): Steps to run the map before recording (to land on attractor).
            
        Returns:
            scaled_input (np.array): Input current for reservoir (scaled).
            raw_target (np.array): Target signal (x shifted by 1).
        """
        total_steps = timesteps + wash_out_gen + 1 # +1 for target shift
        
        # Try generating standard Henon map. If it diverges, retry.
        max_retries = 100
        for attempt in range(max_retries):
            # Reset arrays
            x = np.zeros(total_steps)
            y = np.zeros(total_steps)
            
            # Random initial condition within reasonable bounds for attractor
            x[0] = self.rng.uniform(-0.5, 0.5)
            y[0] = self.rng.uniform(-0.5, 0.5)
            
            diverged = False
            for t in range(total_steps - 1):
                x[t+1] = 1 - self.a * x[t]**2 + y[t]
                y[t+1] = self.b * x[t]
                
                # Check divergence
                if abs(x[t+1]) > 10.0:
                    diverged = True
                    break
            
            if not diverged:
                break
        else:
            # If loop finished without break (all diverged)
            print("Warning: Henon map generator failed to find stable orbit after retries. Returning zeros.")
            return np.zeros(timesteps), np.zeros(timesteps)

        # Discard generation washout
        x_sequence = x[wash_out_gen:]
        
        # Prepare Input vs Target
        # Input: x[t], Target: x[t+1]
        # We need 'timesteps' length for input
        input_seq = x_sequence[:timesteps]
        target_seq = x_sequence[1:timesteps+1]
        
        # Scale Input to Reservoir Range (e.g., 0 to 20 nA)
        # Min/Max of Henon attractor is approx x in [-1.28, 1.28] for standard parameters
        # But we compute actual min/max for robust scaling
        d_min, d_max = input_seq.min(), input_seq.max()
        
        # Avoid division by zero
        if d_max == d_min:
            scaled_input = np.ones_like(input_seq) * self.input_scale_min
        else:
            scaled_input = self.input_scale_min + (input_seq - d_min) * (self.input_scale_max - self.input_scale_min) / (d_max - d_min)
            
        return scaled_input, target_seq
        
    def compute_performance(self, states, target_signal, wash_out_reservoir=100):
        """
        Trains readout to predict next step of Henon map.
        
        Args:
            states (np.array): Reservoir states (time, N).
            target_signal (np.array): Target values (time,).
            wash_out_reservoir (int): Steps to discard from reservoir dynamics.
            
        Returns:
            dict: {
                'nrmse': Normalized Root Mean Square Error,
                'r2_score': R^2 score
            }
        """
        # Discard reservoir washout
        X = states[wash_out_reservoir:]
        y = target_signal[wash_out_reservoir:]
        
        T = len(X)
        if T < 10:
            return {'nrmse': 1.0, 'r2_score': 0.0}
            
        # Split Train/Test (e.g. 70% Train, 30% Test)
        split_idx = int(T * 0.7)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Ridge Regression
        ridge = Ridge(alpha=1e-8) # Lower alpha for cleaner signal, but prevent singularity
        ridge.fit(X_train, y_train)
        
        y_pred = ridge.predict(X_test)
        
        # Compute NRMSE
        rmse = np.sqrt(np.mean((y_pred - y_test)**2))
        sigma = np.std(y_test)
        
        if sigma < 1e-9:
            nrmse = 0.0 # Constant signal predicted perfectly? Or infinite error?
        else:
            nrmse = rmse / sigma
            
        score = ridge.score(X_test, y_test)
            
        return {
            'nrmse': nrmse,
            'r2_score': score,
            'prediction_sample': y_pred[:50].tolist(), # For visualization if needed
            'target_sample': y_test[:50].tolist()
        }
