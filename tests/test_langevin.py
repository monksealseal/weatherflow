import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import unittest

from weatherflow.solvers.langevin import langevin_dynamics

class TestLangevin(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
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
