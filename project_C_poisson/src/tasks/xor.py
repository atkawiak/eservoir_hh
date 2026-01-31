
import numpy as np
from cv import BlockedCV

class DelayedXOR:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate_data(self, length: int, delay: int = 2) -> tuple[np.ndarray, np.ndarray]:
        u = self.rng.integers(0, 2, length) 
        y = np.zeros(length)
        
        for t in range(delay, length):
            y[t] = int(u[t] ^ u[t-delay])
            
        return u, y

    def compute_baseline(self, y: np.ndarray, readout_module) -> float:
        outer_cv = BlockedCV(readout_module.folds, readout_module.gap)
        accuracies = []
        
        for train_idx, test_idx in outer_cv.split(len(y)):
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            mean_val = np.mean(y_train)
            majority_class = 1 if mean_val >= 0.5 else 0
            
            y_pred = np.full_like(y_test, majority_class)
            acc = np.mean(y_test == y_pred)
            accuracies.append(acc)
            
        return np.mean(accuracies)
