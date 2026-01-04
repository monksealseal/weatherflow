# WeatherFlow: Flow Matching for Weather Prediction

<div align="center">
<img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+"/>
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch 2.0+"/>
<img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT"/>
<img src="https://img.shields.io/badge/Version-0.4.2-brightgreen" alt="Version 0.4.2"/>
</div>

WeatherFlow is a Python library built on PyTorch that provides a flexible and extensible framework for developing weather prediction models using flow matching techniques. It integrates seamlessly with ERA5 reanalysis data and incorporates physics-guided neural network architectures.

## 🌐 Try It Online

**Visit our interactive web interface:** https://monksealseal.github.io/weatherflow/

The web interface provides access to all WeatherFlow functionality through your browser:
- Run ML training experiments
- Configure datasets and models
- Visualize predictions and results
- Track experiment history
- No installation required!

> **Note:** The web interface requires a backend server for training. See [BACKEND_QUICKSTART.md](BACKEND_QUICKSTART.md) for deployment instructions.

## Key Features

* **Flow Matching Models:** Implementation of continuous normalizing flows for weather prediction, inspired by Meta AI's approach
* **Physics-Guided Architectures:** Neural networks that respect physical constraints
* **ERA5 Data Integration:** Robust loading of ERA5 reanalysis data from multiple sources
* **Spherical Geometry:** Proper handling of Earth's spherical surface for global weather modeling
* **Visualization Tools:** Comprehensive utilities for visualizing predictions and flow fields
* **Graduate Learning Studio:** Interactive, physics-rich dashboards for atmospheric dynamics education

## 🎯 New: Model Zoo & Applications Gallery

WeatherFlow now includes **model zoo infrastructure** for hosting pre-trained models and **real-world applications**!

### 🏛️ Model Zoo

Infrastructure ready for hosting validated models for common forecasting tasks:

```python
from weatherflow.model_zoo import load_model

# Load a pre-trained model (once available)
model, metadata = load_model('wf_z500_3day_v1')
print(metadata.summary())

# Generate predictions
prediction = model.predict(initial_conditions)
```

**Planned Model Categories:**
- **Global Forecasting**: Z500 3-day, T850 weekly, Multi-variable 5-day
- **Extreme Events**: Tropical cyclone tracking, Atmospheric river detection
- **Climate**: Seasonal forecasting

**Note:** Pre-trained weights not yet available. See [Model Zoo](model_zoo/README.md) for training scripts.

[**→ Browse Model Zoo**](model_zoo/README.md) | [**→ View Gallery**](docs/gallery/index.md)

### 🚀 Applications Gallery

Complete, runnable examples for specific domains:

1. **Renewable Energy Forecasting**
   - Wind and solar power prediction
   - Uncertainty quantification
   - Grid integration tools
   - [View Application →](applications/renewable_energy/README.md)

2. **Extreme Event Analysis**
   - Heatwave and atmospheric river detection
   - Event-based model evaluation
   - Impact assessment tools
   - [View Application →](applications/extreme_event_analysis/README.md)

3. **Educational Laboratory**
   - Graduate-level teaching materials
   - Interactive Jupyter notebooks
   - Guided exercises with solutions
   - [View Educational Resources →](applications/educational/README.md)

[**→ Explore All Applications**](applications/README.md)

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

For managed notebook platforms (e.g., Google Colab or Lambda Labs) where editable
installs can be flaky, install the pinned runtime stack directly:

```bash
pip install -r requirements.txt
```

## Training with Sophisticated Methods and Real Data

For complete step-by-step instructions on training flow matching models with the most sophisticated methods and real ERA5 data, see [**TRAINING_INSTRUCTIONS.txt**](TRAINING_INSTRUCTIONS.txt).

This guide covers:
- Simple and advanced flow matching approaches
- Using the FlowTrainer API with physics constraints
- Foundation model pre-training (FlowAtmosphere)
- Common troubleshooting solutions

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

## Documentation

The `docs/` directory contains an extensive MkDocs site covering installation,
data ingestion, model APIs, advanced usage patterns, and tutorials. Build it
locally with:

```bash
pip install -e .[docs]
mkdocs serve
```

Then open `http://localhost:8000` to browse the rendered documentation.

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

## Interactive Web Dashboard

WeatherFlow now ships with a lightweight FastAPI service and React-based
dashboard that let you explore the library without writing code. The dashboard
walks you through dataset synthesis, model configuration, training, and flow
visualisation using the core `WeatherFlowMatch` and `WeatherFlowODE`
components.

### 1. Start the API service

```bash
uvicorn weatherflow.server.app:app --reload --port 8000
```

The server exposes `/api/options` for configuration metadata and
`/api/experiments` to launch a synthetic training run that exercises the
weather flow models and ODE solver.

### 2. Install and run the React app

```bash
cd frontend
npm install
npm run dev
```

Open the printed URL (typically http://localhost:5173) in your browser to
interact with the dashboard. Use the panels on the left to configure data,
model, and training parameters, then run an experiment to inspect loss curves,
channel statistics, and generated trajectories on the right-hand side.

To produce a production build and run the component tests:

```bash
npm run build
npm test
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

## Running Jupyter Notebooks

We provide several Jupyter notebooks to help you learn and work with WeatherFlow. 

### Setup Notebook Environment

For the easiest experience running the notebooks, use our setup script:

```bash
# Create a dedicated environment and fix notebook imports
python setup_notebook_env.py
```

This script:
1. Creates a virtual environment with all required dependencies
2. Installs the WeatherFlow package in development mode
3. Registers a Jupyter kernel
4. Fixes import paths in notebooks

### Alternative Manual Setup

If you prefer to set up manually:

1. Install notebook dependencies:
   ```bash
   pip install -r notebooks/notebook_requirements.txt
   ```

2. Fix notebook imports:
   ```bash
   python notebooks/fix_notebook_imports.py
   ```

3. Run Jupyter Lab or Notebook:
   ```bash
   jupyter lab
   ```

See [notebooks/README.md](notebooks/README.md) for more details.

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
## Graduate Learning Studio

WeatherFlow now ships with an advanced educational toolkit tailored for graduate-level
atmospheric dynamics and physics.  The `GraduateAtmosphericDynamicsTool`
combines interactive Plotly dashboards with step-by-step problem solvers so students can
experiment with balanced flows, Rossby-wave dispersion, and potential vorticity structures.

```python
from weatherflow.education import GraduateAtmosphericDynamicsTool
import numpy as np

tool = GraduateAtmosphericDynamicsTool(reference_latitude=45.0)

# 1. Build a balanced flow visualization from a synthetic jet streak
latitudes = np.linspace(35.0, 55.0, 30)
longitudes = np.linspace(-30.0, 30.0, 40)
y_metric = tool.R_EARTH * np.deg2rad(latitudes)
x_metric = tool.R_EARTH * np.cos(np.deg2rad(latitudes.mean())) * np.deg2rad(longitudes)
height = (
    5600.0
    + 5.0e-5 * (y_metric[:, None] - y_metric.mean())
    + 2.5e-5 * (x_metric[None, :] - x_metric.mean())
)
balanced_fig = tool.create_balanced_flow_dashboard(height, latitudes, longitudes)
balanced_fig.show()

# 2. Explore Rossby-wave dispersion characteristics interactively
rossby_fig = tool.create_rossby_wave_lab(mean_flow=18.0)
rossby_fig.show()

# 3. Generate curated practice problems with step-by-step solutions
for scenario in tool.generate_problem_scenarios():
    print(scenario.title)
    for step in scenario.solution_steps:
        print(f" - {step.description}: {step.value:.3f} {step.units}")
    print(scenario.answer)
```

The toolkit produces volumetric potential vorticity renderings, Rossby-wave dispersion
laboratories, and automated geostrophic/thermal-wind calculators that help students bridge
conceptual understanding with concrete numerical problem solving.

---

## 📊 Comprehensive Repository Report

A detailed, comprehensive report covering all aspects of this repository is available:

**[View Full Repository Report →](COMPREHENSIVE_REPOSITORY_REPORT.md)**

This 1,000+ line report includes:
- Complete repository statistics and analysis
- Detailed architecture documentation
- Scientific validation results (99.7% of IFS HRES skill)
- Full feature catalog and capabilities
- Development infrastructure overview
- Deployment and distribution guides
- Research roadmap and future directions

