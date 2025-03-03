import torch
import unittest
import sys
import os
sys.path.append(os.path.abspath('..'))

from weatherflow.solvers.langevin import langevin_dynamics

class TestLangevin(unittest.TestCase):
    def setUp(self):
        # Create a simple score function that points toward the origin
        def score_fn(x, t):
            return -x
        
        self.score_fn = score_fn
        
        # Create test data
        self.x0 = torch.randn(10, 4, 8, 8)  # Smaller size for faster testing
        
    def test_langevin_dynamics(self):
        # Run langevin dynamics with small number of steps
        result = langevin_dynamics(
            self.score_fn,
            self.x0,
            n_steps=10,
            step_size=0.01,
            sigma=0.01
        )
        
        # Check shape
        self.assertEqual(result.shape, self.x0.shape)
        
        # Since our score function points to origin, the result should be closer to zero
        self.assertLess(torch.norm(result), torch.norm(self.x0))

if __name__ == '__main__':
    unittest.main()
