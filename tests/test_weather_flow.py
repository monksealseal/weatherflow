import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from weatherflow.models.weather_flow import WeatherFlowModel


def test_weather_flow_model_forward_pass() -> None:
    model = WeatherFlowModel(hidden_dim=64, n_layers=2)

    batch_size = 3
    n_lat, n_lon = 16, 32
    features = 4

    x = torch.randn(batch_size, n_lat, n_lon, features)
    t = torch.rand(batch_size)

    output = model(x, t)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, torch.zeros_like(output))
