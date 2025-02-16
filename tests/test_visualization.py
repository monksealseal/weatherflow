import pytest
import numpy as np
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
    assert vis.figsize == (15, 10)

def test_prediction_comparison(sample_data):
    true_state, pred_state = sample_data
    vis = WeatherVisualizer()
    fig = vis.plot_prediction_comparison(true_state, pred_state)
    assert fig is not None

def test_error_distribution(sample_data):
    true_state, pred_state = sample_data
    vis = WeatherVisualizer()
    fig = vis.plot_error_distribution(true_state, pred_state)
    assert fig is not None

def test_global_forecast(sample_data):
    _, pred_state = sample_data
    vis = WeatherVisualizer()
    fig = vis.plot_global_forecast(pred_state)
    assert fig is not None
