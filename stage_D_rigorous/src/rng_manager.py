
import numpy as np
import hashlib

class RNGManager:
    """
    Manages 4 independent RNG streams for scientific rigor:
    - rec: Reservoir topology (W_rec)
    - inmask: Input projection (W_in)
    - in: Signal u(t) and Poisson spikes
    - readout: CV splits and ridge selection
    
    Supports N=20 independent trials.
    """
    def __init__(self, base_seed: int):
        self.base_seed = base_seed
        
    def _derive_seed(self, trial_idx: int, stream_id: str) -> int:
        """Derives a deterministic seed for a specific trial and stream."""
        s = f"{self.base_seed}_{trial_idx}_{stream_id}"
        return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

    def get_trial_generators(self, trial_idx: int):
        """Returns 4 Generators for a given trial."""
        return {
            'rec': np.random.default_rng(self._derive_seed(trial_idx, 'rec')),
            'inmask': np.random.default_rng(self._derive_seed(trial_idx, 'inmask')),
            'in': np.random.default_rng(self._derive_seed(trial_idx, 'in')),
            'readout': np.random.default_rng(self._derive_seed(trial_idx, 'readout'))
        }
        
    def get_trial_seeds_tuple(self, trial_idx: int):
        """Returns raw seeds for logging in Parquet."""
        return (
            self._derive_seed(trial_idx, 'rec'),
            self._derive_seed(trial_idx, 'inmask'),
            self._derive_seed(trial_idx, 'in'),
            self._derive_seed(trial_idx, 'readout')
        )
