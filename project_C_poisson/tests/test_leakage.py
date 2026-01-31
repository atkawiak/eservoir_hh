
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from cv import BlockedCV

class TestBlockedCV(unittest.TestCase):
    def test_no_overlap(self):
        cv = BlockedCV(n_folds=5, gap=0)
        n_samples = 100
        for train_idx, test_idx in cv.split(n_samples):
            intersect = np.intersect1d(train_idx, test_idx)
            self.assertEqual(len(intersect), 0, "Train and Test overlap!")
            
    def test_gap_integrity(self):
        gap = 10
        cv = BlockedCV(n_folds=5, gap=gap)
        n_samples = 200
        
        for i, (train_idx, test_idx) in enumerate(cv.split(n_samples)):
            min_test = np.min(test_idx)
            max_test = np.max(test_idx)
            
            # Check gap before test
            train_before = train_idx[train_idx < min_test]
            if len(train_before) > 0:
                max_train_pre = np.max(train_before)
                dist = min_test - max_train_pre
                self.assertGreater(dist, gap, f"Gap Violation Pre-Test in fold {i}")
                
            # Check gap after test
            train_after = train_idx[train_idx > max_test]
            if len(train_after) > 0:
                min_train_post = np.min(train_after)
                dist = min_train_post - max_test
                self.assertGreater(dist, gap, f"Gap Violation Post-Test in fold {i}")

    def test_coverage(self):
        cv = BlockedCV(n_folds=5, gap=0)
        n_samples = 100
        test_mask = np.zeros(n_samples, dtype=bool)
        
        for _, test_idx in cv.split(n_samples):
            test_mask[test_idx] = True
            
        self.assertTrue(np.all(test_mask), "Not all samples were tested!")

if __name__ == '__main__':
    unittest.main()
