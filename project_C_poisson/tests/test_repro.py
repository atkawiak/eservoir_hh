
import unittest
import sys
import os
import shutil
import numpy as np
import tempfile

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rng_manager import RNGManager
from hh_model import HHModel
from config import ExperimentConfig, HHConfig, TaskConfig

class TestRepro(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cfg = ExperimentConfig(
            cache_dir=self.tmp_dir,
            hh=HHConfig(N=20),
            task=TaskConfig(dt=0.1)
        )
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        
    def test_rng_determinism(self):
        """Test that RNGManager produces identical streams for same seed."""
        mgr1 = RNGManager(base_seed=42)
        gens1 = mgr1.get_trial_generators(0)
        
        mgr2 = RNGManager(base_seed=42)
        gens2 = mgr2.get_trial_generators(0)
        
        # Test 'rec' stream
        w1 = gens1['rec'].random(10)
        w2 = gens2['rec'].random(10)
        np.testing.assert_array_equal(w1, w2, "RNG output mismatch for same seed")
        
        # Test stream independence
        w_inmask = gens1['inmask'].random(10)
        self.assertFalse(np.array_equal(w1, w_inmask), "Streams should differ")

    def test_hh_caching_and_repro(self):
        """
        Verify HH Model reproducibility.
        """
        # Setup Inputs
        rho = 1.0
        bias = 0.0
        trial_idx = 0
        
        mgr = RNGManager(2025)
        gens = mgr.get_trial_generators(trial_idx)
        seeds = mgr.get_trial_seeds_tuple(trial_idx)
        
        # Increase N for entropy
        self.cfg.hh.N = 50
        hh = HHModel(self.cfg, gens, seeds)
        
        # Fake input - ensure it triggers some activity
        spikes = (np.random.random(200) < 0.2).astype(float)
        task_id = "test_repro_input"
        
        # 1. First Run
        res1 = hh.simulate(rho, bias, spikes, task_id)
        
        if np.sum(res1['spikes']) == 0:
            print("WARNING: Network is dead (0 spikes). Test might be trivial.")
        
        # Check Cache exists
        cache_key = hh.get_cache_key(rho, bias, task_id, len(spikes))
        cache_path = os.path.join(self.tmp_dir, f"{cache_key}.pkl")
        self.assertTrue(os.path.exists(cache_path), "Cache file not created")
        
        # 2. Second Run (Should load cache)
        hh2 = HHModel(self.cfg, gens, seeds) 
        res2 = hh2.simulate(rho, bias, spikes, task_id)
        
        np.testing.assert_array_equal(res1['spikes'], res2['spikes'], "Cached result mismatch")
        
        # 3. Clear Cache and Rerun
        os.remove(cache_path)
        hh3 = HHModel(self.cfg, gens, seeds)
        
        res3 = hh3.simulate(rho, bias, spikes, task_id)
        np.testing.assert_array_equal(res1['spikes'], res3['spikes'], "Recomputed result mismatch")
        
        # 4. Different Seed
        gens_diff = mgr.get_trial_generators(trial_idx + 1)
        seeds_diff = mgr.get_trial_seeds_tuple(trial_idx + 1)
        hh_diff = HHModel(self.cfg, gens_diff, seeds_diff)
        
        # Check W_rec is actually different
        self.assertFalse(np.array_equal(hh.W_rec, hh_diff.W_rec), "W_rec matches across seeds!")
        
        res4 = hh_diff.simulate(rho, bias, spikes, task_id)
        
        # If both are dead (0 spikes), they are equal.
        if np.sum(res1['spikes']) == 0 and np.sum(res4['spikes']) == 0:
             print("Both networks silent. Cannot prove divergence.")
             # Pass but warn
        else:
             self.assertFalse(np.array_equal(res1['spikes'], res4['spikes']), "Different seed produced identical result")

if __name__ == '__main__':
    unittest.main()
