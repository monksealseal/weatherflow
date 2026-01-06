# WeatherFlow Streamlit App

An interactive web application that runs ALL the Python code from the WeatherFlow repository.

## Features

- **Wind Power Forecasting**: Use WindPowerConverter to convert weather forecasts to power output
- **Solar Power Analysis**: Use SolarPowerConverter with pvlib for solar energy calculations
- **Extreme Event Detection**: Detect heatwaves, atmospheric rivers, and extreme precipitation
- **Flow Matching Models**: Interactive WeatherFlowMatch and WeatherFlowODE demonstrations
- **GCM Simulation**: Run General Circulation Model simulations
- **Graduate Education**: Interactive atmospheric dynamics learning tools
- **Visualization**: Weather data visualization with maps, flow fields, and animations
- **Physics Losses**: Explore physics-informed machine learning loss functions
- **Experiments**: Ablation studies, WeatherBench evaluation, and model zoo

## Local Development

1. Install dependencies:
```bash
cd streamlit_app
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run Home.py
```

3. Open http://localhost:8501 in your browser

## Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Set the main file path to `streamlit_app/Home.py`
5. Deploy!

## Project Structure

```
streamlit_app/
├── Home.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .streamlit/
│   ├── config.toml           # Streamlit configuration
│   └── secrets.toml.example  # Example secrets file
└── pages/
    ├── 1_Wind_Power.py       # Wind power forecasting
    ├── 2_Solar_Power.py      # Solar power analysis
    ├── 3_Extreme_Events.py   # Extreme event detection
    ├── 4_Flow_Matching.py    # ML flow matching models
    ├── 5_GCM_Simulation.py   # General Circulation Model
    ├── 6_Education.py        # Graduate atmospheric dynamics
    ├── 7_Visualization.py    # Weather visualization tools
    ├── 8_Physics_Losses.py   # Physics-informed losses
    └── 9_Experiments.py      # Experiments and model zoo
```

## Python Modules Used

This app integrates code from:

- `weatherflow/` - Core flow matching models and physics
- `applications/renewable_energy/` - Wind and solar power converters
- `applications/extreme_event_analysis/` - Event detectors
- `gcm/` - General Circulation Model
- `foundation_model/` - Foundation model components
- `examples/` - Example implementations
- `experiments/` - Ablation studies and evaluation
- `model_zoo/` - Pre-trained models and training
