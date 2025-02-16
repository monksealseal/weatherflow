import pytest
import torch
import numpy as np
from weatherflow.models import PhysicsGuidedAttention, StochasticFlowModel

def test_physics_guided_attention():
    model = PhysicsGuidedAttention(channels=1)
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 64)
    
    # Test forward pass
    output = model(x)
    assert output.shape == x.shape
    
    # Test physics constraints - normalize the input and output for better comparison
    with torch.no_grad():
        x_norm = x / torch.norm(x)
        output_norm = output / torch.norm(output)
        energy_before = torch.sum(x_norm ** 2)
        energy_after = torch.sum(output_norm ** 2)
        rel_diff = abs(energy_after - energy_before) / energy_before
        assert rel_diff < 0.1, f"Energy not conserved: relative difference = {rel_diff}"

def test_stochastic_flow():
    model = StochasticFlowModel(channels=1)
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 64)
    t = torch.rand(batch_size)
    
    # Test forward pass
    output = model(x, t)
    assert output.shape == x.shape
