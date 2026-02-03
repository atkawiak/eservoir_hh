
import numpy as np
from typing import Iterator, Tuple

class BlockedCV:
    """
    Rigorous Blocked Cross-Validation with temporal gap.
    Prevents leakage of autocorrelation from Train to Test.
    """
    def __init__(self, n_folds: int = 5, gap: int = 0):
        self.n_folds = n_folds
        self.gap = gap

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices for Blocked CV.
        
        Args:
            n_samples: Total number of time steps.
            
        Returns:
            train_idx, test_idx
        """
        # Strict integer division, remainder goes to last fold (standard)
        # Or better: distribute remainder? For simplicity and reproducibility:
        # fold_size = n // k. Last fold gets extra.
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_folds
        
        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_folds - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Gap Logic: Remove 'gap' samples adjacent to test block
            train_mask = np.ones(n_samples, dtype=bool)
            
            # Mask Test
            train_mask[test_start:test_end] = False
            
            # Mask Gap Before
            gap_before = max(0, test_start - self.gap)
            train_mask[gap_before:test_start] = False
            
            # Mask Gap After
            gap_after = min(n_samples, test_end + self.gap)
            train_mask[test_end:gap_after] = False
            
            train_indices = indices[train_mask]
            
            yield train_indices, test_indices
