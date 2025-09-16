"""Tests for flow-matching models and utilities."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE


def test_weather_flow_match_loss_components() -> None:
    model = WeatherFlowMatch(
        input_channels=2,
        hidden_dim=32,
        n_layers=1,
        use_attention=False,
        physics_informed=True,
    )

    batch_size = 3
    grid_shape = (8, 12)
    x0 = torch.randn(batch_size, 2, *grid_shape)
    x1 = torch.randn(batch_size, 2, *grid_shape)
    t = torch.full((batch_size,), 0.5)

    losses = model.compute_flow_loss(x0, x1, t)

    assert set(losses) == {"flow_loss", "div_loss", "energy_diff", "total_loss"}
    assert losses["total_loss"].requires_grad
    assert torch.isfinite(losses["total_loss"])
    assert losses["total_loss"] >= losses["flow_loss"]


def test_weather_flow_ode_prediction_shapes() -> None:
    class _LinearDrift(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return -x

    flow_model = _LinearDrift()
    ode_model = WeatherFlowODE(flow_model, solver_method="rk4", rtol=1e-3, atol=1e-3)

    x0 = torch.ones(2, 1, 4, 4)
    times = torch.linspace(0, 1, 4)

    predictions = ode_model(x0, times)

    assert predictions.shape == (len(times), *x0.shape)
    assert torch.allclose(predictions[0], x0)
