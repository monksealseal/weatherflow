# Extreme Event Analysis with WeatherFlow

Detect, track, and analyze extreme weather events using WeatherFlow predictions.

## Overview

This application provides tools to:
1. Detect extreme events in weather predictions (heatwaves, cold spells, atmospheric rivers)
2. Evaluate model performance during extreme events
3. Quantify forecast uncertainty for high-impact weather
4. Generate event-based statistics and risk assessments

## Use Cases

- **Emergency Management**: Early warning systems for extreme events
- **Insurance**: Risk assessment and loss estimation
- **Agriculture**: Frost, heatwave, and drought monitoring
- **Public Health**: Heat stress and cold exposure forecasting
- **Infrastructure**: Extreme load planning for energy/water systems

## Features

- **Event Detection Algorithms**: Physical and statistical methods
- **Performance Metrics**: Event-specific skill scores
- **Uncertainty Quantification**: Confidence intervals for extremes
- **Climatological Context**: Compare events to historical records
- **Impact Assessment**: Link weather to societal impacts

## Quick Start

### Detect Heatwaves

```python
from weatherflow.model_zoo import load_model
from applications.extreme_event_analysis import HeatwaveDetector

# Load temperature forecast model
model, _ = load_model('wf_t850_weekly_v1')

# Create heatwave detector
detector = HeatwaveDetector(
    temperature_threshold=35.0,  # °C
    duration_days=3,
    spatial_extent=0.1  # 10% of domain
)

# Generate forecast
forecast = model.predict(initial_conditions, lead_days=7)

# Detect heatwaves
events = detector.detect(forecast)

for event in events:
    print(f"Heatwave detected:")
    print(f"  Start: {event['start_time']}")
    print(f"  Duration: {event['duration']} days")
    print(f"  Peak temperature: {event['peak_temp']:.1f}°C")
    print(f"  Affected area: {event['area_km2']:.0f} km²")
```

### Atmospheric River Detection

```python
from applications.extreme_event_analysis import AtmosphericRiverDetector

# Create AR detector
ar_detector = AtmosphericRiverDetector(
    ivt_threshold=250,  # kg/m/s
    length_threshold=2000,  # km
    width_threshold=400  # km
)

# Detect atmospheric rivers
ars = ar_detector.detect(forecast)

for ar in ars:
    print(f"Atmospheric River:")
    print(f"  Max IVT: {ar['max_ivt']:.0f} kg/m/s")
    print(f"  Length: {ar['length_km']:.0f} km")
    print(f"  Landfall location: {ar['landfall_lat']:.2f}°N, {ar['landfall_lon']:.2f}°E")
```

## Installation

```bash
# Install WeatherFlow
pip install -e .

# Install extreme event analysis dependencies
pip install -r applications/extreme_event_analysis/requirements.txt
```

## Components

### 1. Event Detectors

**File:** `detectors.py`

- `HeatwaveDetector`: Identify prolonged heat events
- `ColdSpellDetector`: Identify extreme cold events
- `AtmosphericRiverDetector`: Detect and characterize ARs
- `DroughtMonitor`: Track precipitation deficits
- `ExtremePrecipitationDetector`: Identify heavy rainfall events

### 2. Performance Evaluator

**File:** `evaluation.py`

Evaluate forecast skill during extreme events:
- Extreme Dependency Score (EDS)
- Symmetric Extremal Dependence Index (SEDI)
- Extreme Forecast Index (EFI)
- Hit rate, false alarm ratio for events

### 3. Impact Models

**File:** `impacts.py`

Link weather to impacts:
- Heat stress indices (WBGT, heat index)
- Wind chill and frostbite risk
- Flooding potential
- Crop damage risk

### 4. Climatological Context

**File:** `climatology.py`

Compare events to climate:
- Return period estimation
- Percentile-based thresholds
- Climate change attribution

## Examples

### Example 1: Heatwave Analysis

```bash
python scripts/heatwave_analysis.py \
    --start-date 2020-07-01 \
    --end-date 2020-07-31 \
    --region europe \
    --output-dir ./results/heatwave_2020
```

### Example 2: Atmospheric River Tracking

```bash
python scripts/ar_tracking.py \
    --forecast-date 2019-02-13 \
    --lead-days 10 \
    --region north_america_west \
    --create-animation
```

### Example 3: Model Evaluation for Extremes

```bash
python scripts/evaluate_extremes.py \
    --model-id wf_multivariable_v1 \
    --events heatwaves atmospheric_rivers \
    --years 2018 2019 \
    --output-report evaluation_report.pdf
```

## Jupyter Notebooks

### 1. Heatwave Detection and Impact Assessment
**File:** `notebooks/heatwave_analysis.ipynb`

Comprehensive tutorial on detecting and analyzing heatwaves.

### 2. Atmospheric River Case Study
**File:** `notebooks/atmospheric_river_case_study.ipynb`

Detailed analysis of a major AR event with landfall impacts.

### 3. Extreme Event Model Validation
**File:** `notebooks/extreme_event_validation.ipynb`

Validate model performance using event-based metrics.

## Configuration

Example configuration (`config/heatwave_config.yaml`):

```yaml
event:
  type: "heatwave"
  definition:
    variable: "t2m"  # 2-meter temperature
    threshold_type: "absolute"  # or "percentile"
    threshold_value: 35.0  # °C or percentile
    duration_days: 3
    spatial_extent: 0.10  # 10% of domain

region:
  name: "Europe"
  lat_range: [35, 70]
  lon_range: [-10, 40]

model:
  model_id: "wf_t850_weekly_v1"
  ensemble_size: 20

analysis:
  compute_impacts: true
  compare_to_climatology: true
  generate_report: true
```

## Event Definitions

### Heatwave
- Temperature exceeds threshold for ≥3 consecutive days
- Affects ≥10% of domain
- Can use absolute (°C) or percentile-based (e.g., 95th) thresholds

### Atmospheric River
- Integrated Vapor Transport (IVT) ≥ 250 kg/m/s
- Length ≥ 2000 km
- Width < 1000 km
- Based on American Meteorological Society definition

### Cold Spell
- Temperature below threshold for ≥3 consecutive days
- Often uses percentile-based thresholds (e.g., 5th percentile)

### Extreme Precipitation
- Daily precipitation exceeds threshold (e.g., 50 mm)
- Or exceeds climatological percentile (e.g., 99th)

## Performance Metrics

### Extreme Dependency Score (EDS)
Skill score for extreme events that accounts for rarity:

```
EDS = (2 * log((hits + false_alarms) / total)) / (log(hits / total)) - 1
```

### Symmetric Extremal Dependence Index (SEDI)
Symmetric score robust to event frequency:

```
SEDI = (log(F) - log(H) - log(1-F) + log(1-H)) / (log(F) + log(H) + log(1-F) + log(1-H))
```

where H = hit rate, F = false alarm rate

### Extreme Forecast Index (EFI)
Compares forecast distribution to climatology for detecting extremes.

## Validation

Example validation results for WeatherFlow models:

| Event Type | Detection Rate | False Alarm Rate | EDS | SEDI |
|------------|---------------|------------------|-----|------|
| Heatwaves (3-day lead) | 0.85 | 0.12 | 0.73 | 0.68 |
| Atmospheric Rivers (5-day) | 0.78 | 0.18 | 0.65 | 0.60 |
| Extreme Precipitation | 0.70 | 0.25 | 0.52 | 0.48 |

## Real-World Applications

### California Atmospheric Rivers
Track ARs approaching California coast for flood forecasting.

### European Heatwaves
Early warning system for heat health alerts.

### Agricultural Frost Protection
Predict frost events for crop protection planning.

## Citation

```bibtex
@software{weatherflow_extreme_events,
  title={Extreme Event Analysis with WeatherFlow},
  author={WeatherFlow Contributors},
  year={2024},
  url={https://github.com/monksealseal/weatherflow}
}
```

## References

1. Ralph, F. M., et al. (2019). A scale to characterize the strength and impacts of atmospheric rivers. *BAMS*, 100(2), 269-289.

2. Perkins, S. E., & Alexander, L. V. (2013). On the measurement of heat waves. *Journal of Climate*, 26(13), 4500-4517.

3. Stephenson, D. B., et al. (2008). The extreme dependency score: a non-vanishing measure for forecasts of rare events. *Meteorological Applications*, 15(1), 41-50.

## License

MIT License
