
import unittest
import torch
from weatherflow.manifolds.sphere import Sphere

class TestSphere(unittest.TestCase):
    def setUp(self):
        self.sphere = Sphere()
        self.batch_size = 10
        self.points = torch.randn(self.batch_size, 3)
        self.points = self.points / torch.norm(self.points, dim=-1, keepdim=True) * self.sphere.radius
    
    def test_exp_map(self):
        v = torch.randn_like(self.points) * 0.1  # Small tangent vectors
        result = self.sphere.exp_map(self.points, v)
        # Check points are on sphere
        norms = torch.norm(result, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, self.sphere.radius), atol=1e-5))
    
    def test_log_map(self):
        y = self.sphere.exp_map(self.points, torch.randn_like(self.points) * 0.1)
        v = self.sphere.log_map(self.points, y)
        # Check vectors are tangent
        dot_products = torch.sum(v * self.points, dim=-1)
        self.assertTrue(torch.allclose(dot_products, torch.zeros_like(dot_products), atol=1e-5))
