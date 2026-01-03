# Renewable Energy Forecasting with WeatherFlow

Predict wind and solar power generation using weather forecasts from WeatherFlow models.

## Overview

This application demonstrates how to:
1. Generate weather forecasts using WeatherFlow
2. Convert meteorological predictions to power output
3. Quantify forecast uncertainty for energy trading
4. Optimize grid operations using ensemble forecasts

## Use Cases

- **Wind Farm Operators**: Predict power generation 1-7 days ahead
- **Solar Installers**: Forecast PV output for grid integration
- **Energy Traders**: Quantify uncertainty for bidding strategies
- **Grid Operators**: Multi-site forecasting for load balancing

## Features

- **Wind Power Models**: Convert wind speed/direction to turbine power
- **Solar Power Models**: Convert irradiance/temperature to PV output
- **Uncertainty Quantification**: Probabilistic forecasts with confidence intervals
- **Multi-Site Aggregation**: Portfolio-level forecasting
- **Visualization**: Publication-ready plots and dashboards

## Quick Start

### Wind Power Forecasting

```python
from weatherflow.model_zoo import load_model
from applications.renewable_energy import WindPowerConverter

# Load a pre-trained weather model
weather_model, _ = load_model('wf_multivariable_v1')

# Create wind power converter
wind_farm = WindPowerConverter(
    turbine_type='IEA-3.4MW',
    hub_height=100,  # meters
    num_turbines=50,
    farm_location={'lat': 45.0, 'lon': -95.0}
)

# Generate power forecast
weather_forecast = weather_model.predict(initial_conditions, lead_times=[24, 48, 72])
power_forecast = wind_farm.convert(weather_forecast)

print(f"Expected power generation (24h): {power_forecast['24h'].mean():.1f} MW")
print(f"Uncertainty (95% CI): ±{power_forecast['24h'].std() * 1.96:.1f} MW")
```

### Solar Power Forecasting

```python
from applications.renewable_energy import SolarPowerConverter

# Create solar farm model
solar_farm = SolarPowerConverter(
    panel_type='mono-Si',
    capacity=100,  # MW
    tilt_angle=30,  # degrees
    azimuth=180,   # south-facing
    farm_location={'lat': 35.0, 'lon': -110.0}
)

# Convert weather forecast to power
power_forecast = solar_farm.convert(weather_forecast)
```

## Installation

### Prerequisites

```bash
# Install WeatherFlow
pip install -e .

# Install renewable energy dependencies
pip install -r applications/renewable_energy/requirements.txt
```

Additional requirements:
- `pvlib`: Solar position and irradiance calculations
- `windpowerlib`: Wind turbine power curves
- `pandas`: Time series handling

## Components

### 1. Wind Power Converter

**File:** `wind_power.py`

Converts wind forecasts to power output using:
- Standard power curves (IEA, NREL turbines)
- Custom power curves
- Wake effects and array losses
- Temperature and density corrections

### 2. Solar Power Converter

**File:** `solar_power.py`

Converts irradiance/temperature to PV output:
- PVWatts model
- Sandia Performance Model
- Custom panel specifications
- Inverter efficiency curves

### 3. Uncertainty Quantification

**File:** `uncertainty.py`

Probabilistic forecasting methods:
- Ensemble spread
- Quantile regression
- Analog ensembles
- Scenario generation

### 4. Portfolio Optimization

**File:** `portfolio.py`

Multi-site forecasting:
- Spatial aggregation
- Correlation modeling
- Portfolio optimization
- Risk assessment

## Examples

### Example 1: Single Wind Farm Forecast

```bash
python scripts/wind_farm_forecast.py \
    --location 45.0,-95.0 \
    --turbine IEA-3.4MW \
    --num-turbines 50 \
    --lead-days 3
```

### Example 2: Solar Farm with Uncertainty

```bash
python scripts/solar_farm_forecast.py \
    --location 35.0,-110.0 \
    --capacity 100 \
    --ensemble-size 20 \
    --output-dir ./results
```

### Example 3: Portfolio Forecasting

```bash
python scripts/portfolio_forecast.py \
    --config config/portfolio_config.yaml \
    --start-date 2020-01-01 \
    --end-date 2020-12-31
```

## Jupyter Notebooks

### 1. Wind Power Forecasting Tutorial
**File:** `notebooks/wind_power_tutorial.ipynb`

Step-by-step guide to wind power forecasting.

### 2. Solar Power Case Study
**File:** `notebooks/solar_power_case_study.ipynb`

Real-world example with validation against observations.

### 3. Portfolio Optimization
**File:** `notebooks/portfolio_optimization.ipynb`

Multi-site forecasting and risk management.

## Configuration

Example configuration file (`config/wind_farm_config.yaml`):

```yaml
farm:
  name: "Example Wind Farm"
  location:
    lat: 45.0
    lon: -95.0
  turbines:
    type: "IEA-3.4MW"
    count: 50
    hub_height: 100
    array_efficiency: 0.95

weather_model:
  model_id: "wf_multivariable_v1"
  variables: ["u10", "v10", "t2m", "sp"]

forecast:
  lead_times: [6, 12, 24, 48, 72, 120, 168]  # hours
  ensemble_size: 20
  update_frequency: 6  # hours

output:
  format: "csv"
  include_uncertainty: true
  save_plots: true
```

## Validation

Validate forecasts against observed power generation:

```python
from applications.renewable_energy import ForecastValidator

validator = ForecastValidator()
metrics = validator.evaluate(
    forecasts=power_forecast,
    observations=actual_power,
    metrics=['mae', 'rmse', 'skill_score']
)

print(f"MAE: {metrics['mae']:.2f} MW")
print(f"Skill Score: {metrics['skill_score']:.3f}")
```

## Real-World Data

This application can integrate with:

- **SCADA Data**: Turbine-level measurements
- **NREL WIND Toolkit**: Synthetic wind resource data
- **NSRDB**: National Solar Radiation Database
- **Open Power System Data**: European renewable generation

## Performance

Typical forecast skill:

| Lead Time | Wind MAE | Solar MAE |
|-----------|----------|-----------|
| 6 hours   | 3-5% capacity | 2-4% capacity |
| 24 hours  | 8-12% capacity | 6-9% capacity |
| 72 hours  | 15-20% capacity | 12-16% capacity |

## Citation

If you use this application in your research, please cite:

```bibtex
@software{weatherflow_renewable_energy,
  title={Renewable Energy Forecasting with WeatherFlow},
  author={WeatherFlow Contributors},
  year={2024},
  url={https://github.com/monksealseal/weatherflow}
}
```

## References

1. Jung, J., & Broadwater, R. P. (2014). Current status and future advances for wind speed and power forecasting. *Renewable and Sustainable Energy Reviews*, 31, 762-777.

2. Inman, R. H., Pedro, H. T., & Coimbra, C. F. (2013). Solar forecasting methods for renewable energy integration. *Progress in Energy and Combustion Science*, 39(6), 535-576.

3. IEA Wind Task 36: Forecasting for Wind Energy

## License

MIT License
