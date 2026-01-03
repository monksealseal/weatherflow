# WeatherFlow Model Zoo

Infrastructure and templates for hosting pre-trained WeatherFlow models for weather prediction, climate analysis, and atmospheric dynamics research.

## Overview

The Model Zoo provides the infrastructure to load and use pre-trained models for common weather forecasting tasks. The framework is ready to host models with comprehensive documentation, performance metrics, and example usage scripts.

**Note:** Pre-trained model weights are not yet available. The infrastructure is in place and models can be trained using the provided scripts. See `train_model.py` for examples of how to train models that can be added to the zoo.

## Directory Structure

```
model_zoo/
├── global_forecasting/      # Global weather prediction models
│   ├── z500_3day/          # 500 hPa geopotential 3-day forecast
│   ├── t850_weekly/        # 850 hPa temperature weekly forecast
│   └── multi_variable/     # Multi-variable comprehensive models
├── regional_forecasting/    # Regional/domain-specific models
│   ├── north_america/      # North American regional models
│   ├── europe/             # European regional models
│   └── tropics/            # Tropical region models
├── extreme_events/          # Models specialized for extreme events
│   ├── tropical_cyclones/  # Tropical cyclone track/intensity
│   ├── atmospheric_rivers/ # Atmospheric river detection
│   └── heatwaves/          # Heatwave prediction
└── climate_analysis/        # Climate-scale models
    ├── seasonal/           # Seasonal forecasting
    └── subseasonal/        # Subseasonal-to-seasonal (S2S)
```

## Quick Start

### Loading a Pre-trained Model

```python
from weatherflow.model_zoo import load_model

# Load a pre-trained model
model, metadata = load_model('wf_z500_3day_v1')

# View model information
print(metadata.summary())

# Run inference
import torch
from weatherflow.data import ERA5Dataset

# Load initial conditions
dataset = ERA5Dataset(
    variables=['z'],
    pressure_levels=[500],
    time_slice=('2020-01-01', '2020-01-01')
)
x0 = dataset[0]['input']

# Generate 3-day forecast
from weatherflow.models import WeatherFlowODE
ode_solver = WeatherFlowODE(model)
times = torch.linspace(0, 1, 13)  # 3 days at 6-hour intervals
prediction = ode_solver(x0.unsqueeze(0), times)
```

### Exploring Available Models

```python
from weatherflow.model_zoo import list_models, get_model_info

# List all available models
models = list_models()
for model_id in models:
    info = get_model_info(model_id)
    print(f"{model_id}: {info['description']}")
    print(f"  Variables: {info['variables']}")
    print(f"  Performance: ACC={info['metrics']['acc']:.3f}")
```

## Model Metadata

Each model includes a `model_card.json` file with:

- **Model ID**: Unique identifier (e.g., `wf_z500_3day_v1`)
- **Description**: What the model predicts
- **Training Data**: Variables, pressure levels, temporal coverage, spatial resolution
- **Architecture**: Model type, hyperparameters, parameter count
- **Performance Metrics**: RMSE, ACC, bias, skill scores
- **Citation**: How to cite the model
- **License**: Usage terms

## Model Performance Standards

All models in the zoo meet these criteria:

1. **Validation**: Tested on held-out data (typically 2018-2019)
2. **Benchmarking**: Compared against climatology and persistence baselines
3. **Reproducibility**: Training scripts and configurations provided
4. **Documentation**: Complete metadata and usage examples
5. **Physics**: Evaluated for physical consistency (mass conservation, energy balance)

## Contributing Models

To contribute a model to the zoo:

1. Train and validate your model using WeatherFlow
2. Create a model card following our template
3. Provide a usage example script
4. Submit a pull request with your model

See `CONTRIBUTING_MODELS.md` for detailed guidelines.

## Model Hosting

### Current Models (Git LFS)

Small models (< 100 MB) are hosted directly in this repository using Git LFS.

### Large Models (External Storage)

Larger models are hosted on cloud storage with automatic download scripts:

```bash
# Download a large model
python model_zoo/download_model.py wf_global_multivariable_v2
```

## Planned Model Categories

The following are example model types that can be trained and added to the zoo:

### Global Forecasting (Planned)

| Model ID | Description | Variables | Lead Time | Target ACC |
|----------|-------------|-----------|-----------|------------|
| `wf_z500_3day_v1` | 500 hPa geopotential 3-day | Z500 | 72 hours | 0.92 |
| `wf_t850_weekly_v1` | 850 hPa temperature weekly | T850 | 7 days | 0.85 |
| `wf_multivariable_v1` | Multi-variable comprehensive | Z500, T850, U/V850 | 5 days | 0.89 |

### Extreme Events (Planned)

| Model ID | Description | Variables | Target Metric |
|----------|-------------|-----------|---------------|
| `wf_tropical_cyclone_v1` | TC track prediction | Z, T, U, V | Track Error < 200 km (72h) |
| `wf_atmospheric_river_v1` | AR detection/prediction | IVT, Z, T | F1 Score > 0.85 |

### Climate Analysis (Planned)

| Model ID | Description | Variables | Lead Time | Target Skill |
|----------|-------------|-----------|-----------|--------------|
| `wf_seasonal_v1` | Seasonal mean forecasting | T2m, Precip | 1-3 months | 0.65 |

## Usage Examples

See the `applications/` directory for complete example projects using these models:

- **Renewable Energy**: Wind/solar power forecasting
- **Extreme Events**: Event detection and analysis
- **Educational**: Teaching materials for atmospheric dynamics

## References

If you use models from the WeatherFlow Model Zoo, please cite:

```bibtex
@software{weatherflow_model_zoo2024,
  title={WeatherFlow Model Zoo: Pre-trained Models for Weather Prediction},
  author={WeatherFlow Contributors},
  year={2024},
  url={https://github.com/monksealseal/weatherflow}
}
```

## License

Models in this zoo are released under MIT License unless otherwise specified in individual model cards.
