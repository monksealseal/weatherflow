"""
Solar Power Calculator - Interactive PV system power estimation

Uses the actual SolarPowerConverter class from applications/renewable_energy/solar_power.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from applications.renewable_energy.solar_power import (
    SolarPowerConverter,
    PV_LIBRARY,
    PVSystemSpec
)

st.set_page_config(page_title="Solar Power Calculator", page_icon="☀️", layout="wide")

st.title("☀️ Solar Power Calculator")
st.markdown("""
Convert solar irradiance and temperature forecasts to PV power output.
This runs the actual `SolarPowerConverter` class from the repository.
""")

# Sidebar configuration
st.sidebar.header("PV System Configuration")

# Panel type selection
panel_type = st.sidebar.selectbox(
    "Panel Type",
    options=list(PV_LIBRARY.keys()),
    format_func=lambda x: f"{PV_LIBRARY[x].name}"
)

panel = PV_LIBRARY[panel_type]

# Display panel specs
with st.sidebar.expander("Panel Specifications", expanded=True):
    st.markdown(f"""
    - **Module Type**: {panel.module_type}
    - **Efficiency**: {panel.module_efficiency:.1%}
    - **Temp Coefficient**: {panel.temperature_coefficient}%/°C
    - **Default Tilt**: {panel.tilt_angle}°
    - **Default Azimuth**: {panel.azimuth}° (180=South)
    - **Inverter Efficiency**: {panel.inverter_efficiency:.1%}
    - **System Losses**: {panel.system_losses:.1%}
    """)

# System parameters
capacity = st.sidebar.slider("System Capacity (MW DC)", 1.0, 500.0, 100.0)
tilt_angle = st.sidebar.slider("Tilt Angle (°)", 0.0, 90.0, 30.0)
azimuth = st.sidebar.slider("Azimuth (°, 180=South)", 0.0, 360.0, 180.0)
tracking = st.sidebar.checkbox("Single-Axis Tracking")

# Location
st.sidebar.subheader("System Location")
farm_lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 35.0)
farm_lon = st.sidebar.number_input("Longitude", -180.0, 180.0, -110.0)

# Create converter
converter = SolarPowerConverter(
    panel_type=panel_type,
    capacity=capacity,
    tilt_angle=tilt_angle,
    azimuth=azimuth,
    farm_location={'lat': farm_lat, 'lon': farm_lon},
    tracking=tracking
)

# Main content - tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Daily Profile",
    "🔢 Single Calculation",
    "📊 Annual Analysis",
    "🗺️ Weather Forecast Conversion"
])

# Tab 1: Daily Profile
with tab1:
    st.header("Daily Power Profile")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Generate daily profile
        hours = np.arange(0, 24, 0.5)

        # GHI follows sinusoidal pattern
        sunrise = 6
        sunset = 18
        peak_ghi = st.slider("Peak GHI (W/m²)", 500, 1200, 1000)

        ghi = np.zeros_like(hours)
        daylight_mask = (hours >= sunrise) & (hours <= sunset)
        ghi[daylight_mask] = peak_ghi * np.sin(np.pi * (hours[daylight_mask] - sunrise) / (sunset - sunrise))
        ghi = np.maximum(ghi, 0)

        # Temperature pattern
        min_temp = st.slider("Min Temperature (°C)", -10, 20, 15)
        max_temp = st.slider("Max Temperature (°C)", 20, 45, 30)

        # Temperature peaks a few hours after solar noon
        temp_c = min_temp + (max_temp - min_temp) * (0.5 + 0.5 * np.sin(np.pi * (hours - 6) / 12))
        temp_k = temp_c + 273.15

        # Calculate power
        power = converter.irradiance_to_power(ghi, temp_k)

        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Solar Irradiance (GHI)', 'Temperature', 'Power Output', 'Efficiency'),
            vertical_spacing=0.12
        )

        # GHI
        fig.add_trace(
            go.Scatter(x=hours, y=ghi, name='GHI',
                      line=dict(color='#ffa726', width=3),
                      fill='tozeroy', fillcolor='rgba(255, 167, 38, 0.3)'),
            row=1, col=1
        )

        # Temperature
        fig.add_trace(
            go.Scatter(x=hours, y=temp_c, name='Temperature',
                      line=dict(color='#ef5350', width=2)),
            row=1, col=2
        )

        # Power
        fig.add_trace(
            go.Scatter(x=hours, y=power, name='AC Power',
                      line=dict(color='#42a5f5', width=3),
                      fill='tozeroy', fillcolor='rgba(66, 165, 245, 0.3)'),
            row=2, col=1
        )

        # Efficiency (only during daylight)
        efficiency = np.zeros_like(hours)
        nonzero_ghi = ghi > 10
        efficiency[nonzero_ghi] = power[nonzero_ghi] / (ghi[nonzero_ghi] * capacity / 1000) * 100
        efficiency = np.clip(efficiency, 0, 100)

        fig.add_trace(
            go.Scatter(x=hours, y=efficiency, name='System Efficiency',
                      line=dict(color='#66bb6a', width=2)),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Hour", row=2, col=1)
        fig.update_xaxes(title_text="Hour", row=2, col=2)
        fig.update_yaxes(title_text="W/m²", row=1, col=1)
        fig.update_yaxes(title_text="°C", row=1, col=2)
        fig.update_yaxes(title_text="MW", row=2, col=1)
        fig.update_yaxes(title_text="%", row=2, col=2)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Daily Statistics")

        daily_energy = converter.daily_energy(power[::2])  # Hourly values
        st.metric("Daily Energy Production", f"{daily_energy:,.1f} MWh")

        capacity_factor = converter.capacity_factor(power)
        st.metric("Daily Capacity Factor", f"{capacity_factor:.1%}")

        peak_power = np.max(power)
        st.metric("Peak Power", f"{peak_power:.1f} MW")

        peak_hour = hours[np.argmax(power)]
        st.metric("Peak Hour", f"{int(peak_hour)}:{int((peak_hour % 1) * 60):02d}")

        st.markdown("---")

        # Performance ratio
        theoretical_max = capacity * panel.module_efficiency
        pr = daily_energy / (peak_ghi * 12 / 1000 * capacity * panel.module_efficiency) if peak_ghi > 0 else 0
        st.metric("Performance Ratio", f"{min(pr, 1.0):.1%}")

        st.markdown("---")
        st.markdown("### Cell Temperature")

        # Calculate cell temp at peak
        peak_idx = np.argmax(ghi)
        cell_temp = converter._cell_temperature(
            np.array([temp_c[peak_idx]]),
            np.array([ghi[peak_idx]])
        )[0]
        st.metric("Peak Cell Temperature", f"{cell_temp:.1f}°C")

        temp_loss = abs(panel.temperature_coefficient) * (cell_temp - 25)
        st.metric("Temperature Derate", f"{temp_loss:.1f}%")

# Tab 2: Single Calculation
with tab2:
    st.header("Single Point Calculation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")

        ghi_input = st.slider("Global Horizontal Irradiance (W/m²)", 0, 1200, 800)
        temp_input = st.slider("Ambient Temperature (°C)", -20, 50, 25)
        wind_input = st.slider("Wind Speed (m/s)", 0.0, 20.0, 2.0)

        use_zenith = st.checkbox("Specify Solar Zenith Angle")
        if use_zenith:
            zenith_input = st.slider("Solar Zenith Angle (°)", 0, 90, 30)
            zenith_arr = np.array([zenith_input])
        else:
            zenith_arr = None

    with col2:
        st.subheader("Results")

        # Calculate
        ghi_arr = np.array([ghi_input])
        temp_arr = np.array([temp_input + 273.15])
        wind_arr = np.array([wind_input])

        power_output = converter.irradiance_to_power(
            ghi_arr, temp_arr,
            wind_speed=wind_arr,
            solar_zenith=zenith_arr
        )[0]

        # Display results
        st.metric("AC Power Output", f"{power_output:.2f} MW")

        instant_cf = power_output / capacity
        st.metric("Instantaneous Capacity Factor", f"{instant_cf:.1%}")

        # Cell temperature
        cell_temp = converter._cell_temperature(
            np.array([temp_input]),
            ghi_arr,
            wind_arr
        )[0]
        st.metric("Cell Temperature", f"{cell_temp:.1f}°C")

        # Breakdown
        st.markdown("### Power Breakdown")

        dc_power = (capacity * panel.module_efficiency * (ghi_input / 1000) *
                    (1 + panel.temperature_coefficient / 100 * (cell_temp - 25)))

        breakdown_data = {
            'Stage': ['DC Generation', 'After System Losses', 'AC Output'],
            'Power (MW)': [
                dc_power,
                dc_power * (1 - panel.system_losses),
                power_output
            ]
        }

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=['DC Power', 'System Losses', 'AC Output'],
            y=[dc_power, -dc_power * panel.system_losses, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ef5350"}},
            increasing={"marker": {"color": "#66bb6a"}},
            totals={"marker": {"color": "#42a5f5"}}
        ))

        fig.update_layout(title="Power Flow", height=300)
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Annual Analysis
with tab3:
    st.header("Annual Energy Analysis")

    st.markdown("Simulate a full year of solar production using typical meteorological data patterns.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Annual Parameters")

        annual_ghi = st.slider("Annual Average GHI (kWh/m²/day)", 3.0, 7.0, 5.5)
        clearness_index = st.slider("Clearness Index", 0.4, 0.8, 0.6)

        seasonal_variation = st.slider("Seasonal Variation", 0.1, 0.5, 0.3)
        weather_noise = st.slider("Weather Variability", 0.0, 0.3, 0.15)

    # Generate annual data
    np.random.seed(42)
    days = np.arange(365)

    # Seasonal pattern
    seasonal = 1 + seasonal_variation * np.cos(2 * np.pi * (days - 172) / 365)

    # Daily GHI
    daily_ghi = annual_ghi * seasonal * (1 + weather_noise * np.random.randn(365))
    daily_ghi = np.clip(daily_ghi, 0, 10)

    # Temperature pattern
    temp_mean = 15 + 15 * np.cos(2 * np.pi * (days - 200) / 365)
    temp_daily = temp_mean + np.random.randn(365) * 5

    # Estimate daily energy production
    # Use peak sun hours approach
    daily_energy = np.zeros(365)

    for i, (ghi, temp) in enumerate(zip(daily_ghi, temp_daily)):
        # Approximate using 6 hours of equivalent peak production
        peak_hours = ghi / 1.0  # kWh/m²/day ≈ peak sun hours
        avg_cell_temp = temp + 20  # Rough estimate
        temp_factor = 1 + panel.temperature_coefficient / 100 * (avg_cell_temp - 25)
        daily_energy[i] = (capacity * panel.module_efficiency * temp_factor *
                          (1 - panel.system_losses) * panel.inverter_efficiency * peak_hours)

    with col2:
        # Monthly aggregation
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly_energy = []
        day_idx = 0
        for days_in_month in month_days:
            monthly_energy.append(np.sum(daily_energy[day_idx:day_idx + days_in_month]))
            day_idx += days_in_month

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Energy Production', 'Monthly Energy',
                          'Cumulative Energy', 'Capacity Factor by Month'),
            specs=[[{}, {}], [{}, {}]]
        )

        # Daily energy
        fig.add_trace(
            go.Scatter(x=list(range(365)), y=daily_energy, name='Daily',
                      line=dict(color='#ffa726', width=1)),
            row=1, col=1
        )

        # Monthly bar chart
        fig.add_trace(
            go.Bar(x=month_names, y=monthly_energy, name='Monthly',
                  marker_color='#42a5f5'),
            row=1, col=2
        )

        # Cumulative
        fig.add_trace(
            go.Scatter(x=list(range(365)), y=np.cumsum(daily_energy), name='Cumulative',
                      line=dict(color='#66bb6a', width=2),
                      fill='tozeroy', fillcolor='rgba(102, 187, 106, 0.2)'),
            row=2, col=1
        )

        # Monthly capacity factor
        monthly_cf = [e / (capacity * 24 * d) for e, d in zip(monthly_energy, month_days)]
        fig.add_trace(
            go.Bar(x=month_names, y=[cf * 100 for cf in monthly_cf], name='CF',
                  marker_color='#7c4dff'),
            row=2, col=2
        )

        fig.update_yaxes(title_text="MWh", row=1, col=1)
        fig.update_yaxes(title_text="MWh", row=1, col=2)
        fig.update_yaxes(title_text="MWh", row=2, col=1)
        fig.update_yaxes(title_text="%", row=2, col=2)
        fig.update_xaxes(title_text="Day of Year", row=1, col=1)
        fig.update_xaxes(title_text="Day of Year", row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Annual statistics
    st.subheader("Annual Statistics")

    stat_cols = st.columns(5)
    with stat_cols[0]:
        st.metric("Annual Energy", f"{np.sum(daily_energy):,.0f} MWh")
    with stat_cols[1]:
        annual_cf = np.sum(daily_energy) / (capacity * 8760)
        st.metric("Annual Capacity Factor", f"{annual_cf:.1%}")
    with stat_cols[2]:
        specific_yield = np.sum(daily_energy) / capacity
        st.metric("Specific Yield", f"{specific_yield:,.0f} kWh/kWp")
    with stat_cols[3]:
        best_month = month_names[np.argmax(monthly_energy)]
        st.metric("Best Month", best_month)
    with stat_cols[4]:
        worst_month = month_names[np.argmin(monthly_energy)]
        st.metric("Worst Month", worst_month)

# Tab 4: Weather Forecast Conversion
with tab4:
    st.header("Convert Weather Forecast to Power Forecast")

    st.markdown("""
    This demonstrates the `convert_forecast()` method which takes solar radiation
    and temperature data and converts them to power forecasts.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Forecast Input")

        forecast_hours = st.slider("Forecast Length (hours)", 6, 72, 48, key="solar_forecast")

        # Generate synthetic forecast
        np.random.seed(456)
        hours = np.arange(forecast_hours)

        # GHI with day/night cycle
        ghi = np.zeros(forecast_hours)
        for h in hours:
            hour_of_day = h % 24
            if 6 <= hour_of_day <= 18:
                ghi[int(h)] = 800 * np.sin(np.pi * (hour_of_day - 6) / 12)
        ghi = ghi * (0.8 + 0.4 * np.random.rand(forecast_hours))
        ghi = np.maximum(ghi, 0)

        # Temperature
        t2m = 273.15 + 20 + 10 * np.sin(np.pi * (hours % 24 - 6) / 12)
        t2m += np.random.randn(forecast_hours) * 2

        # Wind
        u10 = 2 + np.random.randn(forecast_hours) * 0.5
        v10 = 1 + np.random.randn(forecast_hours) * 0.3

        # Create forecast dict
        weather_forecast = {
            'ghi': ghi,
            't2m': t2m,
            'u10': u10,
            'v10': v10
        }

        # Display input
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Solar Irradiance (GHI)', 'Temperature'))

        fig.add_trace(
            go.Scatter(x=hours, y=ghi, name='GHI', fill='tozeroy',
                      line=dict(color='#ffa726')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=hours, y=t2m - 273.15, name='T (°C)',
                      line=dict(color='#ef5350')),
            row=2, col=1
        )

        fig.update_yaxes(title_text="W/m²", row=1, col=1)
        fig.update_yaxes(title_text="°C", row=2, col=1)
        fig.update_xaxes(title_text="Forecast Hour", row=2, col=1)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Power Forecast Results")

        # Convert forecast
        result = converter.convert_forecast(weather_forecast)

        # Display results
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hours, y=result['power_mw'],
            mode='lines',
            name='Power Forecast',
            line=dict(color='#42a5f5', width=3),
            fill='tozeroy',
            fillcolor='rgba(66, 165, 245, 0.3)'
        ))

        fig.update_layout(
            title='Power Forecast',
            xaxis_title='Forecast Hour',
            yaxis_title='Power (MW)',
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cell temperature
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hours, y=result['cell_temperature'],
            mode='lines',
            name='Cell Temp',
            line=dict(color='#ef5350', width=2)
        ))
        fig2.update_layout(
            title='Cell Temperature',
            xaxis_title='Forecast Hour',
            yaxis_title='Temperature (°C)',
            height=200
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Statistics
        st.markdown("### Forecast Statistics")

        result_cols = st.columns(3)
        with result_cols[0]:
            st.metric("Mean Power", f"{result['mean_power']:.2f} MW")
        with result_cols[1]:
            st.metric("Max Power", f"{result['max_power']:.2f} MW")
        with result_cols[2]:
            st.metric("Capacity Factor", f"{result['capacity_factor']:.1%}")

# Code reference
with st.expander("📝 Code Reference"):
    st.markdown("""
    This page uses the following code from the repository:

    ```python
    # From applications/renewable_energy/solar_power.py
    from applications.renewable_energy.solar_power import (
        SolarPowerConverter,
        PV_LIBRARY,
        PVSystemSpec
    )

    # Create converter
    converter = SolarPowerConverter(
        panel_type='mono-Si-standard',
        capacity=100.0,  # MW
        tilt_angle=30.0,
        azimuth=180.0,  # South-facing
        tracking=False
    )

    # Convert irradiance to power
    power = converter.irradiance_to_power(ghi, temperature)

    # Calculate daily energy
    energy = converter.daily_energy(hourly_power)

    # Convert weather forecast
    result = converter.convert_forecast(weather_forecast)
    ```
    """)
