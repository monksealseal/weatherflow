# WeatherFlow: Flow Matching for Weather Prediction

<div align="center">
<img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+"/>
<img src="https://img.shields.io/badge/PyTorch-1.9%2B-orange" alt="PyTorch 1.9+"/>
<img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT"/>
<img src="https://img.shields.io/badge/Version-0.3.0-brightgreen" alt="Version 0.3.0"/>
</div>

WeatherFlow is a Python library built on PyTorch that provides a flexible and extensible framework for developing weather prediction models using flow matching techniques. It integrates seamlessly with ERA5 reanalysis data and incorporates physics-guided neural network architectures.

## Key Features

* **Flow Matching Models:** Implementation of continuous normalizing flows for weather prediction, inspired by Meta AI's approach
* **Physics-Guided Architectures:** Neural networks that respect physical constraints
* **ERA5 Data Integration:** Robust loading of ERA5 reanalysis data from multiple sources
* **Spherical Geometry:** Proper handling of Earth's spherical surface for global weather modeling
* **Visualization Tools:** Comprehensive utilities for visualizing predictions and flow fields

## Installation

```bash
# Clone the repository
git clone https://github.com/monksealseal/weatherflow.git
cd weatherflow

# Install in development mode
pip install -e .

# Install extra dependencies for development
pip install -r requirements-dev.txt
```

## Quick Start

Here's a minimal example to get started:

```python
from weatherflow.data import ERA5Dataset, create_data_loaders
from weatherflow.models import WeatherFlowMatch
from weatherflow.utils import WeatherVisualizer
import torch

# Load data
train_loader, val_loader = create_data_loaders(
    variables=['z', 't'],             # Geopotential and temperature
    pressure_levels=[500],            # Single pressure level
    train_slice=('2015', '2016'),     # Training years
    val_slice=('2017', '2017'),       # Validation year
    batch_size=32
)

# Create model
model = WeatherFlowMatch(
    input_channels=2,                 # Number of variables
    hidden_dim=128,                   # Hidden dimension
    n_layers=4,                       # Number of layers
    physics_informed=True             # Use physics constraints
)

# Train model (simple example)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Train for one epoch
model.train()
for batch in train_loader:
    x0, x1 = batch['input'].to(device), batch['target'].to(device)
    t = torch.rand(x0.size(0), device=device)
    loss = model.compute_flow_loss(x0, x1, t)['total_loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate predictions
from weatherflow.models import WeatherFlowODE

ode_model = WeatherFlowODE(model)
x0 = next(iter(val_loader))['input'].to(device)
times = torch.linspace(0, 1, 5, device=device)  # 5 time steps
with torch.no_grad():
    predictions = ode_model(x0, times)

# Visualize results
visualizer = WeatherVisualizer()
vis_var = 'z'  # Geopotential
var_idx = 0
visualizer.plot_comparison(
    true_data={vis_var: x0[0, var_idx].cpu()},
    pred_data={vis_var: predictions[-1, 0, var_idx].cpu()},
    var_name=vis_var,
    title="Prediction vs Truth"
)
```

## Comprehensive Example

For a more comprehensive example, see the `examples/weather_prediction.py` script, which demonstrates:

1. Loading ERA5 data
2. Training a flow matching model with physics constraints
3. Generating predictions for different lead times
4. Visualizing results

Run the example script:

```bash
python examples/weather_prediction.py --variables z t --pressure-levels 500 \
    --train-years 2015 2016 --val-years 2017 --epochs 20 \
    --use-attention --physics-informed --save-model --save-results
```

## Key Components

### Data Loading

```python
from weatherflow.data import ERA5Dataset

# Load data directly from WeatherBench2
dataset = ERA5Dataset(
    variables=['z', 't', 'u', 'v'],        # Variables to load
    pressure_levels=[850, 500, 250],       # Pressure levels (hPa)
    time_slice=('2015', '2016'),           # Time period
    normalize=True                         # Apply normalization
)

# Load from local netCDF file
local_dataset = ERA5Dataset(
    data_path='/path/to/era5_data.nc',
    variables=['z', 't'],
    pressure_levels=[500]
)
```

### Flow Matching Models

```python
from weatherflow.models import WeatherFlowMatch

# Simple model
model = WeatherFlowMatch(
    input_channels=4,                  # Number of variables
    hidden_dim=256,                    # Hidden dimension
    n_layers=4                         # Number of layers
)

# Advanced model with physics constraints
advanced_model = WeatherFlowMatch(
    input_channels=4,
    hidden_dim=256,
    n_layers=6,
    use_attention=True,                # Use attention mechanism
    physics_informed=True,             # Apply physics constraints
    grid_size=(32, 64)                 # Latitude/longitude grid size
)
```

### ODE Solver for Prediction

```python
from weatherflow.models import WeatherFlowODE

# Create ODE solver with the trained flow model
ode_model = WeatherFlowODE(
    flow_model=model,
    solver_method='dopri5',           # ODE solver method
    rtol=1e-4,                        # Relative tolerance
    atol=1e-4                         # Absolute tolerance
)

# Generate predictions
x0 = initial_weather_state            # Initial state
times = torch.linspace(0, 1, 5)       # 5 time steps
predictions = ode_model(x0, times)    # Shape: [time, batch, channels, lat, lon]
```

### Visualization

```python
from weatherflow.utils import WeatherVisualizer

visualizer = WeatherVisualizer()

# Compare prediction with ground truth
visualizer.plot_comparison(
    true_data={'temperature': true_temp},
    pred_data={'temperature': pred_temp},
    var_name='temperature'
)

# Visualize flow field
visualizer.plot_flow_vectors(
    u=u_wind,                           # U-component of wind
    v=v_wind,                           # V-component of wind
    background=geopotential,            # Background field
    var_name='geopotential'
)

# Create animation
visualizer.create_prediction_animation(
    predictions=predictions[:, 0, 0],   # Time evolution of first variable
    var_name='temperature',
    interval=200,                       # Animation speed (ms)
    save_path='animation.gif'
)
```

## Advanced Usage

### Custom Flow Matching Models

You can create custom flow matching models by extending the base classes:

```python
import torch.nn as nn
from weatherflow.models import WeatherFlowMatch

class MyFlowModel(WeatherFlowMatch):
    def __init__(self, input_channels, hidden_dim=256):
        super().__init__(input_channels, hidden_dim)
        # Add custom layers
        self.extra_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, t):
        # Override forward method
        h = super().forward(x, t)
        # Add custom processing
        h = self.extra_layer(h)
        return h
```

### Physics-Informed Constraints

You can add custom physics constraints:

```python
def custom_physics_constraint(v, x):
    """Apply custom physics constraint to velocity field."""
    # Implement your physics constraint
    return v_constrained

# Use in model
model = WeatherFlowMatch(physics_informed=True)
model._apply_physics_constraints = custom_physics_constraint
```

## Contributing

We welcome contributions to WeatherFlow! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

See `CONTRIBUTING.md` for more details.

## License

WeatherFlow is released under the MIT License. See `LICENSE` for details.

## Citation

If you use WeatherFlow in your research, please cite:

```
@software{weatherflow2023,
  author = {Siman, Eduardo},
  title = {WeatherFlow: Flow Matching for Weather Prediction},
  url = {https://github.com/monksealseal/weatherflow},
  year = {2023}
}
```

## Acknowledgments

This project builds upon flow matching techniques introduced by Meta AI and is inspired by approaches from the weather and climate modeling community.