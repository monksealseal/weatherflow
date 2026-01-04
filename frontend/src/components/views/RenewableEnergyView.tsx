import './RenewableEnergyView.css';

interface TurbineSpec {
  name: string;
  ratedPower: string;
  cutInSpeed: string;
  ratedSpeed: string;
  cutOutSpeed: string;
  hubHeight: string;
  rotorDiameter: string;
}

const TURBINE_LIBRARY: Record<string, TurbineSpec> = {
  'IEA-3.4MW': {
    name: 'IEA 3.4 MW',
    ratedPower: '3.4 MW',
    cutInSpeed: '3.0 m/s',
    ratedSpeed: '13.0 m/s',
    cutOutSpeed: '25.0 m/s',
    hubHeight: '110 m',
    rotorDiameter: '130 m'
  },
  'NREL-5MW': {
    name: 'NREL 5 MW Reference',
    ratedPower: '5.0 MW',
    cutInSpeed: '3.0 m/s',
    ratedSpeed: '11.4 m/s',
    cutOutSpeed: '25.0 m/s',
    hubHeight: '90 m',
    rotorDiameter: '126 m'
  },
  'Vestas-V90': {
    name: 'Vestas V90 2.0 MW',
    ratedPower: '2.0 MW',
    cutInSpeed: '4.0 m/s',
    ratedSpeed: '15.0 m/s',
    cutOutSpeed: '25.0 m/s',
    hubHeight: '80 m',
    rotorDiameter: '90 m'
  }
};

export default function RenewableEnergyView() {
  return (
    <div className="view-container renewable-energy-view">
      <div className="view-header">
        <h1>⚡ Renewable Energy Forecasting</h1>
        <p className="view-subtitle">
          Wind and solar power prediction using weather forecasts
        </p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">🌬️</div>
        <div className="banner-content">
          <h3>Weather to Power Conversion</h3>
          <p>
            Convert weather forecasts into renewable energy production estimates using
            turbine power curves, solar irradiance models, and atmospheric corrections.
          </p>
        </div>
      </div>

      <div className="energy-types">
        <div className="energy-card wind">
          <h2>💨 Wind Power</h2>
          <p>
            Convert wind speed forecasts to power output using turbine specifications
            and power curves. Includes hub height extrapolation and wind farm wake effects.
          </p>
          
          <h3>Key Features</h3>
          <ul>
            <li>Standard turbine library (IEA, NREL, commercial models)</li>
            <li>Hub height wind speed extrapolation</li>
            <li>Power curve conversion with uncertainty</li>
            <li>Wake effect modeling for wind farms</li>
            <li>Capacity factor estimation</li>
          </ul>

          <h3>Available Turbine Models</h3>
          <div className="turbine-specs">
            {Object.entries(TURBINE_LIBRARY).map(([key, spec]) => (
              <details key={key} className="turbine-details">
                <summary>{spec.name}</summary>
                <div className="spec-grid">
                  <div className="spec-item">
                    <strong>Rated Power:</strong> {spec.ratedPower}
                  </div>
                  <div className="spec-item">
                    <strong>Hub Height:</strong> {spec.hubHeight}
                  </div>
                  <div className="spec-item">
                    <strong>Cut-in Speed:</strong> {spec.cutInSpeed}
                  </div>
                  <div className="spec-item">
                    <strong>Rated Speed:</strong> {spec.ratedSpeed}
                  </div>
                  <div className="spec-item">
                    <strong>Cut-out Speed:</strong> {spec.cutOutSpeed}
                  </div>
                  <div className="spec-item">
                    <strong>Rotor Diameter:</strong> {spec.rotorDiameter}
                  </div>
                </div>
              </details>
            ))}
          </div>
        </div>

        <div className="energy-card solar">
          <h2>☀️ Solar Power</h2>
          <p>
            Convert solar irradiance and weather conditions to photovoltaic power output.
            Includes temperature effects, cloud attenuation, and panel orientation.
          </p>
          
          <h3>Key Features</h3>
          <ul>
            <li>Clear-sky irradiance models</li>
            <li>Cloud cover attenuation</li>
            <li>Temperature-dependent efficiency</li>
            <li>Panel tilt and orientation optimization</li>
            <li>Diffuse and direct component separation</li>
          </ul>

          <h3>Solar Models</h3>
          <div className="solar-models">
            <div className="model-card">
              <h4>Clear-Sky Irradiance</h4>
              <p>Calculate potential solar irradiance based on sun position and atmospheric conditions</p>
            </div>
            <div className="model-card">
              <h4>Cloud Attenuation</h4>
              <p>Reduce irradiance based on cloud cover from weather forecasts</p>
            </div>
            <div className="model-card">
              <h4>Temperature Correction</h4>
              <p>Adjust panel efficiency based on ambient temperature</p>
            </div>
          </div>
        </div>
      </div>

      <div className="code-section">
        <h2>📖 Usage Examples</h2>
        
        <div className="code-example">
          <h3>Wind Power Conversion</h3>
          <pre><code>{`from weatherflow.applications.renewable_energy import WindPowerConverter, TURBINE_LIBRARY
import torch

# Initialize converter with turbine specs
converter = WindPowerConverter(
    turbine=TURBINE_LIBRARY['NREL-5MW'],
    num_turbines=50,  # Wind farm size
    farm_efficiency=0.85  # Account for wake effects
)

# Wind speed forecast at 10m height [batch, time, lat, lon]
wind_speed_10m = torch.randn(1, 24, 32, 64) * 5 + 7  # ~7 m/s mean

# Convert to power output
power_output = converter.wind_to_power(
    wind_speed=wind_speed_10m,
    wind_direction=None,  # Optional for wake modeling
    temperature=None,  # Optional for air density correction
    surface_pressure=None
)

# Results in MW
print(f"Total power: {power_output.sum().item():.2f} MW")
print(f"Capacity factor: {converter.capacity_factor(power_output):.2%}")`}</code></pre>
        </div>

        <div className="code-example">
          <h3>Solar Power Prediction</h3>
          <pre><code>{`from weatherflow.applications.renewable_energy import SolarPowerConverter
import torch
from datetime import datetime

# Initialize solar converter
converter = SolarPowerConverter(
    panel_capacity=100.0,  # 100 MW farm
    panel_efficiency=0.18,
    temperature_coefficient=-0.004,  # %/°C
    tilt=30,  # Panel tilt angle
    azimuth=180  # South-facing
)

# Weather forecasts
cloud_cover = torch.rand(1, 24, 32, 64)  # 0-1 fraction
temperature = torch.randn(1, 24, 32, 64) * 5 + 20  # ~20°C

# Calculate power output
times = [datetime(2024, 6, 15, h) for h in range(24)]
power_output = converter.solar_to_power(
    cloud_cover=cloud_cover,
    temperature=temperature,
    times=times,
    latitude=40.0
)

print(f"Daily energy: {power_output.sum().item():.2f} MWh")`}</code></pre>
        </div>

        <div className="code-example">
          <h3>Combined Wind-Solar Farm</h3>
          <pre><code>{`from weatherflow.applications.renewable_energy import (
    WindPowerConverter,
    SolarPowerConverter,
    TURBINE_LIBRARY
)
from weatherflow.models import WeatherFlowODE
from weatherflow.data import ERA5Dataset

# Load weather forecast model
model = ...  # Your trained model
ode_solver = WeatherFlowODE(model)

# Load initial conditions
dataset = ERA5Dataset(
    variables=['u', 'v', 't'],
    pressure_levels=[10, 100],  # Surface and 100m
    time_slice=('2024-01-01', '2024-01-01')
)
x0 = dataset[0]['input']

# Generate 48-hour forecast
times = torch.linspace(0, 1, 48)
forecast = ode_solver(x0.unsqueeze(0), times)

# Extract wind and temperature
wind_u = forecast[:, :, 0]  # U-component
wind_v = forecast[:, :, 1]  # V-component
wind_speed = torch.sqrt(wind_u**2 + wind_v**2)
temperature = forecast[:, :, 2]

# Convert to power
wind_converter = WindPowerConverter(TURBINE_LIBRARY['IEA-3.4MW'])
solar_converter = SolarPowerConverter(panel_capacity=50.0)

wind_power = wind_converter.wind_to_power(wind_speed)
solar_power = solar_converter.solar_to_power(
    cloud_cover=None,  # Estimate from other variables
    temperature=temperature,
    times=times
)

# Combined output
total_power = wind_power + solar_power
print(f"Average power: {total_power.mean().item():.2f} MW")`}</code></pre>
        </div>
      </div>

      <div className="applications-section">
        <h2>🎯 Real-World Applications</h2>
        <div className="applications-grid">
          <div className="application-card">
            <h3>⚡ Grid Integration</h3>
            <p>Forecast renewable energy production for grid operators to balance supply and demand</p>
          </div>
          <div className="application-card">
            <h3>💰 Energy Trading</h3>
            <p>Predict power output for electricity market bidding and trading strategies</p>
          </div>
          <div className="application-card">
            <h3>🏗️ Site Selection</h3>
            <p>Evaluate potential renewable energy sites using historical weather data</p>
          </div>
          <div className="application-card">
            <h3>🔋 Storage Optimization</h3>
            <p>Optimize battery storage charging/discharging based on production forecasts</p>
          </div>
        </div>
      </div>

      <div className="reference-section">
        <h2>📚 References</h2>
        <ul>
          <li>IEA Wind Task 37: <code>applications/renewable_energy/wind_power.py</code></li>
          <li>Solar resource assessment: <code>applications/renewable_energy/solar_power.py</code></li>
          <li>NREL 5-MW Reference Turbine specifications</li>
          <li>Sandia PV performance model implementation</li>
        </ul>
      </div>
    </div>
  );
}
