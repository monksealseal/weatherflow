import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "cartopy",
    reason="Weather visualisation utilities require the optional cartopy dependency.",
)
pytest.importorskip(
    "PIL",
    reason="Weather visualisation utilities rely on Pillow for image handling.",
)

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from weatherflow.utils import WeatherVisualizer

@pytest.fixture
def sample_data():
    grid_size = (32, 64)
    true_state = {
        'temperature': np.random.randn(*grid_size),
        'pressure': np.random.randn(*grid_size)
    }
    pred_state = {k: v + np.random.randn(*v.shape) * 0.1 
                  for k, v in true_state.items()}
    return true_state, pred_state

def test_visualizer_creation():
    vis = WeatherVisualizer()
    assert vis.figsize == (12, 8)

def test_prediction_comparison(sample_data):
    true_state, pred_state = sample_data
    vis = WeatherVisualizer()
    fig, axes = vis.plot_comparison(true_state, pred_state, var_name='temperature')
    assert len(axes) == 3
    plt.close(fig)

def test_error_distribution(sample_data):
    true_state, pred_state = sample_data
    vis = WeatherVisualizer()
    fig, axes = vis.plot_error_metrics(true_state, pred_state, var_names=['pressure'])
    assert axes.size == 2
    plt.close(fig)

def test_global_forecast(sample_data):
    _, pred_state = sample_data
    vis = WeatherVisualizer()
    fig, ax = vis.plot_field(pred_state['pressure'], title='Forecast')
    assert ax is not None
    plt.close(fig)
