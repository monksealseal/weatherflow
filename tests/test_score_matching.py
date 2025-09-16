import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import unittest

from weatherflow.models.conversion import score_to_vector_field, vector_field_to_score
from weatherflow.models.score_matching import ScoreMatchingModel
from weatherflow.path import CondOTPath, GaussianProbPath

# Rest of the code remains the same
class SimpleEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = torch.nn.Linear(1, 16)
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 4, 3, padding=1)
        )
        
    def forward(self, x, t):
        # Process time embedding
        t_emb = self.time_embed(t.view(-1, 1))
        t_emb = t_emb.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        
        # The network expects 4 channels, so we don't concatenate time embedding
        return self.net(x)

class TestScoreMatching(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Create path
        self.path = CondOTPath()
        
        # Create simple encoder
        self.encoder = SimpleEncoder()
        
        # Create score matching model
        self.score_model = ScoreMatchingModel(
            encoder=self.encoder,
            path=self.path,
            parameterization="score"
        )
        
        # Create test data
        self.z = torch.randn(10, 4, 32, 32)  # Batch of 10, 4 channels, 32x32 grid
        self.t = torch.rand(10)  # Time values between 0 and 1
        
    def test_score_matching_loss(self):
        # Sample a point
        x = self.path.sample_conditional(self.z, self.t)
        
        # Compute loss
        loss = self.score_model.compute_score_matching_loss(x, self.z, self.t)
        
        # Check that loss is scalar and requires grad
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(loss.requires_grad)
    
    def test_vector_field_conversion(self):
        # Sample a point
        x = self.path.sample_conditional(self.z, self.t)
        
        # Create a simple vector field function
        def vector_field_fn(x, t):
            return torch.ones_like(x)
        
        # Convert to score
        score = vector_field_to_score(vector_field_fn, self.path, x, self.t)
        
        # Check shape
        self.assertEqual(score.shape, x.shape)
        
        # Convert back to vector field
        def score_fn(x, t):
            return score
        
        v = score_to_vector_field(score_fn, self.path, x, self.t)
        
        # Check shape
        self.assertEqual(v.shape, x.shape)
