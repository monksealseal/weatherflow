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
    
    # Test physics constraints
    with torch.no_grad():
        energy_before = torch.sum(x ** 2)
        energy_after = torch.sum(output ** 2)
        assert abs(energy_after - energy_before) / energy_before < 0.1

def test_stochastic_flow():
    model = StochasticFlowModel(channels=1)
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 64)
    t = torch.rand(batch_size)
    
    # Test forward pass
    output = model(x, t)
    assert output.shape == x.shape
    
    # Test time conditioning
    t1 = torch.zeros(batch_size)
    t2 = torch.ones(batch_size)
    out1 = model(x, t1)
    out2 = model(x, t2)
    assert not torch.allclose(out1, out2)
