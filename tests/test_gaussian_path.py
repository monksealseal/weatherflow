import torch
import unittest
import sys
import os

# Add the parent directory to the path so we can import weatherflow
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from weatherflow.path import GaussianProbPath, CondOTPath

class TestGaussianPath(unittest.TestCase):
    # Rest of the code remains the same
    def setUp(self):
        # Define simple schedules for testing
        self.alpha_schedule = lambda t: t
        self.beta_schedule = lambda t: 1 - t
        self.gaussian_path = GaussianProbPath(self.alpha_schedule, self.beta_schedule)
        self.condot_path = CondOTPath()
        
        # Create test data
        self.z = torch.randn(10, 4, 32, 32)  # Batch of 10, 4 channels, 32x32 grid
        self.t = torch.rand(10)  # Time values between 0 and 1
        
    def test_sample_path(self):
        # Test sampling from gaussian path
        x = self.gaussian_path.sample_conditional(self.z, self.t)
        self.assertEqual(x.shape, self.z.shape)
        
        # Test sampling from CondOT path
        x_condot = self.condot_path.sample_conditional(self.z, self.t)
        self.assertEqual(x_condot.shape, self.z.shape)
    
    def test_conditional_score(self):
        # Sample a point
        x = self.gaussian_path.sample_conditional(self.z, self.t)
        
        # Test score computation
        score = self.gaussian_path.get_conditional_score(x, self.z, self.t)
        self.assertEqual(score.shape, self.z.shape)
        
        # Test score computation for CondOT
        score_condot = self.condot_path.get_conditional_score(x, self.z, self.t)
        self.assertEqual(score_condot.shape, self.z.shape)
    
    def test_conditional_vector_field(self):
        # Sample a point
        x = self.gaussian_path.sample_conditional(self.z, self.t)
        
        # Test vector field computation
        v = self.gaussian_path.get_conditional_vector_field(x, self.z, self.t)
        self.assertEqual(v.shape, self.z.shape)
        
        # Test vector field computation for CondOT (should be more efficient)
        v_condot = self.condot_path.get_conditional_vector_field(x, self.z, self.t)
        self.assertEqual(v_condot.shape, self.z.shape)
        
        # For CondOT, the vector field should be z - epsilon
        epsilon = (x - self.t.view(-1, 1, 1, 1) * self.z) / (1 - self.t.view(-1, 1, 1, 1) + 1e-8)
        expected_v = self.z - epsilon
        self.assertTrue(torch.allclose(v_condot, expected_v, rtol=1e-5, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
