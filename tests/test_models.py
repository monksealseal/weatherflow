import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import torch

from weatherflow.models import PhysicsGuidedAttention, StochasticFlowModel


def test_physics_guided_attention():
    model = PhysicsGuidedAttention(
        input_channels=1,
        hidden_dim=64,
        num_layers=1,
        num_heads=2,
        grid_size=(8, 8),
    )
    batch_size = 2
    x = torch.randn(batch_size, 1, 8, 8)

    output = model(x)
    assert output.shape == x.shape

    assert torch.isfinite(output).all()
    assert torch.norm(output) > 0


def test_stochastic_flow():
    model = StochasticFlowModel(input_channels=4, latent_dim=8)
    batch_size = 3
    x = torch.randn(batch_size, 4)

    output = model(x, num_samples=5)
    assert output.shape == (5, batch_size, 4)
