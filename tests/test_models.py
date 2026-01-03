import pytest

torch = pytest.importorskip("torch")

from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE


def test_weather_flow_match_forward_shapes():
    model = WeatherFlowMatch(
        input_channels=4,
        hidden_dim=32,
        n_layers=2,
        use_attention=True,
        window_size=4,
        physics_informed=True,
        use_graph_mp=False,
        use_spectral_mixer=True,
        spectral_modes=4,
    )
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    out = model(x, t)
    assert out.shape == x.shape


def test_weather_flow_ode_fast_and_standard():
    class IdentityFlow(torch.nn.Module):
        def forward(self, x, t, static=None, forcing=None):
            return torch.ones_like(x)

    flow = IdentityFlow()
    x0 = torch.zeros(1, 2, 4, 4)
    times = torch.tensor([0.0, 0.5, 1.0])

    # Fast Heun path
    ode_fast = WeatherFlowODE(flow_model=flow, fast_mode=True)
    preds_fast = ode_fast(x0, times)
    assert preds_fast.shape == (3, 1, 2, 4, 4)

    # Ensemble perturbation
    ensemble = ode_fast.ensemble_forecast(x0, times, num_members=3, noise_std=0.1)
    assert ensemble.shape == (3, 3, 1, 2, 4, 4)
