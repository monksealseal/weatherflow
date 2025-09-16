"""Tests for the graduate atmospheric dynamics educational toolkit."""

import numpy as np
import pytest

from weatherflow.education import GraduateAtmosphericDynamicsTool
from weatherflow.education.graduate_tool import GRAVITY, OMEGA, R_EARTH

try:  # pragma: no cover - exercised in CI environments with full deps
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - skip visualization tests when missing
    go = None  # type: ignore[assignment]
    HAS_PLOTLY = False
else:  # pragma: no cover - simple boolean assignment
    HAS_PLOTLY = True


def test_coriolis_parameter_matches_theory() -> None:
    """Ensure the Coriolis parameter matches the analytic expression."""

    tool = GraduateAtmosphericDynamicsTool()
    latitudes = np.array([-60.0, 0.0, 45.0])
    computed = tool.coriolis_parameter(latitudes)
    expected = 2.0 * OMEGA * np.sin(np.deg2rad(latitudes))
    assert np.allclose(computed, expected)


def test_geostrophic_wind_linear_height_field() -> None:
    """The geostrophic calculator should recover analytic solutions."""

    tool = GraduateAtmosphericDynamicsTool()
    latitudes = np.linspace(30.0, 45.0, 9)
    longitudes = np.linspace(-5.0, 5.0, 11)

    y_coords = R_EARTH * np.deg2rad(latitudes)
    gradient = 4.0e-5  # m m^-1 north-south height gradient
    height_field = gradient * (y_coords[:, None] - y_coords.mean())
    height_field = np.repeat(height_field, longitudes.size, axis=1)

    u_g, v_g = tool.compute_geostrophic_wind(height_field, latitudes, longitudes)
    f = tool.coriolis_parameter(latitudes)[:, None]
    expected_u = -GRAVITY * gradient / f

    assert np.allclose(u_g, expected_u, rtol=1e-3, atol=1e-3)
    assert np.allclose(v_g, 0.0, atol=1e-6)


def test_quasigeostrophic_pv_zero_streamfunction() -> None:
    """A zero streamfunction should yield only the beta*y contribution."""

    tool = GraduateAtmosphericDynamicsTool()
    z = np.linspace(0.0, 9000.0, 4)
    y = np.linspace(-600000.0, 600000.0, 6)
    x = np.linspace(0.0, 800000.0, 5)

    psi = np.zeros((z.size, y.size, x.size))
    f0 = float(tool.coriolis_parameter(45.0))
    beta = float(tool.beta_parameter(45.0))
    stratification = np.full(z.size, 0.01)

    pv = tool.compute_quasigeostrophic_pv(psi, z, y, x, f0, beta, stratification)
    y_grid = np.meshgrid(z, y, x, indexing="ij")[1]
    assert np.allclose(pv, beta * y_grid)


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly is required for visualization tests")
def test_balanced_flow_dashboard_returns_plotly_figure() -> None:
    """The balanced flow explorer should create a 3-D Plotly figure."""

    tool = GraduateAtmosphericDynamicsTool()
    latitudes = np.linspace(35.0, 45.0, 10)
    longitudes = np.linspace(-15.0, 15.0, 12)

    y_coords = R_EARTH * np.deg2rad(latitudes)
    x_coords = R_EARTH * np.cos(np.deg2rad(latitudes.mean())) * np.deg2rad(longitudes)
    height_field = (
        5600.0
        + 5.0e-5 * (y_coords[:, None] - y_coords.mean())
        + 2.0e-5 * (x_coords[None, :] - x_coords.mean())
    )

    fig = tool.create_balanced_flow_dashboard(height_field, latitudes, longitudes)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly is required for visualization tests")
def test_rossby_wave_lab_structure() -> None:
    """Rossby wave lab should contain dispersion and diagnostic panels."""

    tool = GraduateAtmosphericDynamicsTool()
    fig = tool.create_rossby_wave_lab(k_range=(1.0e-7, 2.0e-7, 10), l_range=(0.0, 1.0e-7, 10))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly is required for visualization tests")
def test_pv_atelier_generates_volume_and_slice() -> None:
    """The PV atelier should overlay a volume rendering and a surface slice."""

    tool = GraduateAtmosphericDynamicsTool()
    z = np.linspace(0.0, 9000.0, 4)
    y = np.linspace(-500000.0, 500000.0, 5)
    x = np.linspace(0.0, 600000.0, 6)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    psi = np.cos(np.pi * zz / z[-1]) * np.sin(np.pi * yy / y[-1]) * np.cos(2.0 * np.pi * xx / x[-1])

    fig = tool.create_pv_atelier(psi, z, y, x, stratification=np.full(z.size, 0.012))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_problem_scenarios_include_worked_steps() -> None:
    """Generated problems must include descriptive solutions."""

    tool = GraduateAtmosphericDynamicsTool()
    scenarios = tool.generate_problem_scenarios()
    assert len(scenarios) >= 3
    for scenario in scenarios:
        assert scenario.solution_steps
        assert all(step.description for step in scenario.solution_steps)
        assert scenario.answer

