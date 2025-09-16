
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from weatherflow.manifolds.sphere import Sphere
from weatherflow.models.weather_flow import WeatherFlowModel


def test_package_imports() -> None:
    model = WeatherFlowModel(hidden_dim=64, n_layers=2)
    assert isinstance(model, WeatherFlowModel)
    assert hasattr(model, "solver")

    sphere = Sphere()
    assert sphere.radius > 0

    x = torch.randn(1, 16, 32, 4)
    t = torch.tensor([0.5])
    output = model(x, t)
    assert output.shape == x.shape
