"""
Wind Power Calculator - Interactive wind farm power estimation

Uses the actual WindPowerConverter class from applications/renewable_energy/wind_power.py
Supports ERA5 real wind data or demo mode with synthetic data.
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from applications.renewable_energy.wind_power import (
    WindPowerConverter,
    TURBINE_LIBRARY,
    TurbineSpec
)

# Import ERA5 utilities
try:
    from era5_utils import (
        ensure_era5_data_or_demo,
        get_era5_wind_data,
        get_era5_time_range,
        get_era5_data_banner,
        has_era5_data,
        get_active_era5_data,
        generate_synthetic_era5_like_data
    )
    ERA5_UTILS_AVAILABLE = True
except ImportError:
    ERA5_UTILS_AVAILABLE = False

st.set_page_config(page_title="Wind Power Calculator", page_icon="🌬️", layout="wide")

st.title("🌬️ Wind Power Calculator")
st.markdown("""
Convert wind speed forecasts to power output using real turbine power curves.
This runs the actual `WindPowerConverter` class from the repository.
""")

# Show data source banner
if ERA5_UTILS_AVAILABLE:
    banner = get_era5_data_banner()
    if "Demo Mode" in banner or "⚠️" in banner:
        st.info(f"📊 {banner}")
    else:
        st.success(f"📊 {banner}")

# Sidebar configuration
st.sidebar.header("Wind Farm Configuration")

# Turbine selection
turbine_type = st.sidebar.selectbox(
    "Turbine Type",
    options=list(TURBINE_LIBRARY.keys()),
    format_func=lambda x: f"{x} ({TURBINE_LIBRARY[x].rated_power} MW)"
)

turbine = TURBINE_LIBRARY[turbine_type]

# Display turbine specs
with st.sidebar.expander("Turbine Specifications", expanded=True):
    st.markdown(f"""
    - **Name**: {turbine.name}
    - **Rated Power**: {turbine.rated_power} MW
    - **Cut-in Speed**: {turbine.cut_in_speed} m/s
    - **Rated Speed**: {turbine.rated_speed} m/s
    - **Cut-out Speed**: {turbine.cut_out_speed} m/s
    - **Hub Height**: {turbine.hub_height} m
    - **Rotor Diameter**: {turbine.rotor_diameter} m
    """)

# Farm parameters
num_turbines = st.sidebar.slider("Number of Turbines", 1, 200, 50)
array_efficiency = st.sidebar.slider("Array Efficiency", 0.80, 1.0, 0.95)
availability = st.sidebar.slider("Turbine Availability", 0.90, 1.0, 0.97)

# Location
st.sidebar.subheader("Farm Location")
farm_lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 45.0)
farm_lon = st.sidebar.number_input("Longitude", -180.0, 180.0, -95.0)

# Create converter
converter = WindPowerConverter(
    turbine_type=turbine_type,
    num_turbines=num_turbines,
    farm_location={'lat': farm_lat, 'lon': farm_lon},
    array_efficiency=array_efficiency,
    availability=availability
)

# Main content - tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Power Curve",
    "🔢 Single Calculation",
    "📊 Time Series Analysis",
    "🗺️ Weather Forecast Conversion",
    "🌍 ERA5 Wind Data"
])

# Tab 1: Power Curve
with tab1:
    st.header("Turbine Power Curve")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Generate power curve
        wind_speeds = np.linspace(0, 30, 200)
        single_turbine_power = converter.wind_speed_to_power(wind_speeds)
        farm_power = converter.farm_power(wind_speeds)

        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Single Turbine Power Curve', 'Full Farm Power Output')
        )

        fig.add_trace(
            go.Scatter(
                x=wind_speeds,
                y=single_turbine_power,
                mode='lines',
                name='Single Turbine',
                line=dict(color='#1e88e5', width=3),
                fill='tozeroy',
                fillcolor='rgba(30, 136, 229, 0.2)'
            ),
            row=1, col=1
        )

        # Add cut-in, rated, cut-out markers
        fig.add_vline(x=turbine.cut_in_speed, line_dash="dash", line_color="green",
                      annotation_text="Cut-in", row=1, col=1)
        fig.add_vline(x=turbine.rated_speed, line_dash="dash", line_color="orange",
                      annotation_text="Rated", row=1, col=1)
        fig.add_vline(x=turbine.cut_out_speed, line_dash="dash", line_color="red",
                      annotation_text="Cut-out", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=wind_speeds,
                y=farm_power,
                mode='lines',
                name=f'Farm ({num_turbines} turbines)',
                line=dict(color='#7c4dff', width=3),
                fill='tozeroy',
                fillcolor='rgba(124, 77, 255, 0.2)'
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Wind Speed (m/s)")
        fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
        fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
        fig.update_layout(height=500, showlegend=True)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key Statistics")

        max_farm_capacity = turbine.rated_power * num_turbines
        st.metric("Maximum Farm Capacity", f"{max_farm_capacity:.1f} MW")
        st.metric("Effective Capacity (with losses)",
                  f"{max_farm_capacity * array_efficiency * availability:.1f} MW")

        # Calculate capacity factor for typical wind distribution
        weibull_winds = np.random.weibull(2.0, 10000) * 8  # Shape=2, Scale~8 m/s
        cf = converter.capacity_factor(weibull_winds)
        st.metric("Typical Capacity Factor", f"{cf:.1%}")

        st.markdown("---")
        st.markdown("### Wind Speed Regions")
        st.markdown(f"""
        - **Region I** (0 - {turbine.cut_in_speed} m/s): No power
        - **Region II** ({turbine.cut_in_speed} - {turbine.rated_speed} m/s): Cubic increase
        - **Region III** ({turbine.rated_speed} - {turbine.cut_out_speed} m/s): Rated power
        - **Region IV** (>{turbine.cut_out_speed} m/s): Shutdown
        """)

# Tab 2: Single Calculation
with tab2:
    st.header("Single Wind Speed Calculation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")

        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 35.0, 10.0, 0.5)

        use_density_correction = st.checkbox("Apply Air Density Correction")

        if use_density_correction:
            temperature = st.slider("Temperature (°C)", -30.0, 45.0, 15.0)
            pressure = st.slider("Pressure (hPa)", 900.0, 1050.0, 1013.25)
            temp_k = temperature + 273.15
            pressure_pa = pressure * 100
        else:
            temp_k = None
            pressure_pa = None

        adjust_height = st.checkbox("Adjust from Measurement Height")
        if adjust_height:
            measurement_height = st.slider("Measurement Height (m)", 2.0, 100.0, 10.0)
            terrain_roughness = st.selectbox(
                "Terrain Type",
                options=[
                    ("Open water", 0.0002),
                    ("Open terrain", 0.03),
                    ("Suburban", 0.3),
                    ("Urban", 1.0)
                ],
                format_func=lambda x: x[0]
            )[1]

    with col2:
        st.subheader("Results")

        # Calculate
        if adjust_height:
            adjusted_wind = converter.adjust_height(
                np.array([wind_speed]),
                measurement_height,
                terrain_roughness
            )[0]
            st.info(f"Wind speed at hub height ({turbine.hub_height}m): **{adjusted_wind:.2f} m/s**")
            calc_wind = adjusted_wind
        else:
            calc_wind = wind_speed

        if use_density_correction:
            power = converter.wind_speed_to_power(
                np.array([calc_wind]),
                temperature=np.array([temp_k]),
                pressure=np.array([pressure_pa])
            )[0]
        else:
            power = converter.wind_speed_to_power(np.array([calc_wind]))[0]

        farm_total = converter.farm_power(np.array([calc_wind]))[0]

        st.metric("Single Turbine Power", f"{power:.3f} MW")
        st.metric("Total Farm Power", f"{farm_total:.2f} MW")

        capacity_factor = farm_total / (turbine.rated_power * num_turbines)
        st.metric("Instantaneous Capacity Factor", f"{capacity_factor:.1%}")

        # Visual gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=farm_total,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Farm Power (MW)"},
            gauge={
                'axis': {'range': [0, turbine.rated_power * num_turbines]},
                'bar': {'color': "#1e88e5"},
                'steps': [
                    {'range': [0, turbine.rated_power * num_turbines * 0.3], 'color': "#e3f2fd"},
                    {'range': [turbine.rated_power * num_turbines * 0.3,
                              turbine.rated_power * num_turbines * 0.7], 'color': "#90caf9"},
                    {'range': [turbine.rated_power * num_turbines * 0.7,
                              turbine.rated_power * num_turbines], 'color': "#42a5f5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': turbine.rated_power * num_turbines * array_efficiency * availability
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Time Series Analysis
with tab3:
    st.header("Time Series Power Analysis")

    st.markdown("Generate synthetic wind data or upload your own to analyze power production over time.")

    data_source = st.radio("Data Source", ["Generate Synthetic Data", "Upload CSV"])

    if data_source == "Generate Synthetic Data":
        col1, col2 = st.columns(2)
        with col1:
            n_hours = st.slider("Duration (hours)", 24, 8760, 168)
            mean_wind = st.slider("Mean Wind Speed (m/s)", 3.0, 15.0, 8.0)
        with col2:
            wind_variability = st.slider("Wind Variability", 0.1, 0.5, 0.3)
            include_diurnal = st.checkbox("Include Diurnal Pattern", value=True)

        # Generate synthetic wind data
        np.random.seed(42)
        hours = np.arange(n_hours)

        # Base wind from Weibull
        base_wind = np.random.weibull(2.0, n_hours) * mean_wind / 0.886

        # Add autocorrelation
        smoothed_wind = np.convolve(base_wind, np.ones(6)/6, mode='same')

        # Add diurnal pattern
        if include_diurnal:
            diurnal = 1 + 0.15 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
            wind_series = smoothed_wind * diurnal
        else:
            wind_series = smoothed_wind

        wind_series = np.clip(wind_series, 0, 35)

    else:
        uploaded_file = st.file_uploader("Upload wind speed CSV (column: 'wind_speed')")
        if uploaded_file is not None:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            wind_series = df['wind_speed'].values
            n_hours = len(wind_series)
            hours = np.arange(n_hours)
        else:
            st.warning("Please upload a CSV file with a 'wind_speed' column")
            st.stop()

    # Calculate power
    power_series = converter.farm_power(wind_series)

    # Create visualization
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Wind Speed', 'Power Output', 'Cumulative Energy'),
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.33]
    )

    fig.add_trace(
        go.Scatter(x=hours, y=wind_series, name='Wind Speed',
                   line=dict(color='#26a69a', width=1)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=hours, y=power_series, name='Power',
                   line=dict(color='#1e88e5', width=1),
                   fill='tozeroy', fillcolor='rgba(30, 136, 229, 0.3)'),
        row=2, col=1
    )

    cumulative_energy = np.cumsum(power_series)  # MWh (hourly data)
    fig.add_trace(
        go.Scatter(x=hours, y=cumulative_energy, name='Cumulative Energy',
                   line=dict(color='#7c4dff', width=2)),
        row=3, col=1
    )

    fig.update_yaxes(title_text="m/s", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="MWh", row=3, col=1)
    fig.update_xaxes(title_text="Hour", row=3, col=1)

    fig.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.subheader("Production Statistics")

    stat_cols = st.columns(5)
    with stat_cols[0]:
        st.metric("Total Energy", f"{cumulative_energy[-1]:,.0f} MWh")
    with stat_cols[1]:
        st.metric("Capacity Factor", f"{converter.capacity_factor(wind_series):.1%}")
    with stat_cols[2]:
        st.metric("Mean Power", f"{np.mean(power_series):.1f} MW")
    with stat_cols[3]:
        st.metric("Peak Power", f"{np.max(power_series):.1f} MW")
    with stat_cols[4]:
        hours_at_rated = np.sum(power_series > turbine.rated_power * num_turbines * 0.9)
        st.metric("Hours at >90% Rated", f"{hours_at_rated}")

# Tab 4: Weather Forecast Conversion
with tab4:
    st.header("Convert Weather Forecast to Power Forecast")

    st.markdown("""
    This demonstrates the `convert_forecast()` method which takes U/V wind components
    (like from ERA5 or other NWP models) and converts them to power forecasts.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Forecast Input")

        forecast_hours = st.slider("Forecast Length (hours)", 6, 72, 24)

        # Generate synthetic forecast
        np.random.seed(123)
        hours = np.arange(forecast_hours)

        # U and V components with some structure
        base_u = 5 + 3 * np.sin(2 * np.pi * hours / 24)
        base_v = 2 + 2 * np.cos(2 * np.pi * hours / 24)

        u_wind = base_u + np.random.normal(0, 1, forecast_hours)
        v_wind = base_v + np.random.normal(0, 0.5, forecast_hours)

        # Optional: temperature and pressure
        temperature = 288 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 1, forecast_hours)
        pressure = 101325 + np.random.normal(0, 500, forecast_hours)

        # Create forecast dict
        weather_forecast = {
            'u': u_wind,
            'v': v_wind,
            't': temperature,
            'sp': pressure
        }

        # Display input
        fig = make_subplots(rows=2, cols=2, subplot_titles=('U Wind', 'V Wind', 'Temperature', 'Pressure'))
        fig.add_trace(go.Scatter(x=hours, y=u_wind, name='U'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hours, y=v_wind, name='V'), row=1, col=2)
        fig.add_trace(go.Scatter(x=hours, y=temperature - 273.15, name='T (°C)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=hours, y=pressure / 100, name='P (hPa)'), row=2, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Power Forecast Results")

        # Convert forecast
        result = converter.convert_forecast(weather_forecast, measurement_height=10.0)

        # Display results
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=result['power_mw'],
            mode='lines+markers',
            name='Power Forecast',
            line=dict(color='#1e88e5', width=3),
            fill='tozeroy',
            fillcolor='rgba(30, 136, 229, 0.3)'
        ))

        fig.update_layout(
            title='Power Forecast',
            xaxis_title='Forecast Hour',
            yaxis_title='Power (MW)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        st.markdown("### Forecast Statistics")

        result_cols = st.columns(3)
        with result_cols[0]:
            st.metric("Mean Power", f"{result['mean_power']:.2f} MW")
        with result_cols[1]:
            st.metric("Max Power", f"{result['max_power']:.2f} MW")
        with result_cols[2]:
            st.metric("Std Dev", f"{result['std_power']:.2f} MW")

        # Wind rose
        fig_rose = go.Figure()
        fig_rose.add_trace(go.Scatterpolar(
            r=result['wind_speed_hub'],
            theta=result['wind_direction'],
            mode='markers',
            marker=dict(
                size=8,
                color=result['power_mw'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Power (MW)')
            )
        ))
        fig_rose.update_layout(
            title='Wind Rose (colored by power)',
            polar=dict(radialaxis=dict(title='Wind Speed (m/s)')),
            height=350
        )
        st.plotly_chart(fig_rose, use_container_width=True)

# Code reference
with st.expander("📝 Code Reference"):
    st.markdown("""
    This page uses the following code from the repository:

    ```python
    # From applications/renewable_energy/wind_power.py
    from applications.renewable_energy.wind_power import (
        WindPowerConverter,
        TURBINE_LIBRARY,
        TurbineSpec
    )

    # Create converter
    converter = WindPowerConverter(
        turbine_type='IEA-3.4MW',
        num_turbines=50,
        array_efficiency=0.95
    )

    # Convert wind speed to power
    power = converter.wind_speed_to_power(wind_speed)

    # Full farm power
    farm_power = converter.farm_power(wind_speed)

    # Convert weather forecast
    result = converter.convert_forecast(weather_forecast)
    ```
    """)

# Tab 5: ERA5 Wind Data
with tab5:
    st.header("🌍 ERA5 Real Wind Data Analysis")
    
    st.markdown("""
    Analyze wind power production using **real ERA5 reanalysis data**.
    This demonstrates how to use actual atmospheric observations for wind farm planning.
    """)
    
    if ERA5_UTILS_AVAILABLE and has_era5_data():
        data, metadata = get_active_era5_data()
        
        st.success(f"✅ Using Real ERA5 Data: **{metadata.get('name', 'Unknown')}**")
        st.markdown(f"- **Period:** {metadata.get('start_date', '?')} to {metadata.get('end_date', '?')}")
        st.markdown(f"- **Source:** {metadata.get('source', 'ERA5')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Time Selection")
            
            # Get time range
            start_time, end_time = get_era5_time_range()
            
            if start_time is not None:
                time_values = data.time.values
                time_options = [str(t)[:19] for t in time_values]
                
                selected_time_idx = st.selectbox(
                    "Select Time",
                    options=range(len(time_options)),
                    format_func=lambda x: time_options[x],
                    key="era5_wind_time"
                )
                
                # Get pressure levels if available
                levels = []
                if "level" in data.coords:
                    levels = sorted([int(l) for l in data.level.values])
                    selected_level = st.selectbox(
                        "Pressure Level (hPa)",
                        options=levels,
                        index=min(1, len(levels) - 1),
                        key="era5_wind_level"
                    )
                else:
                    selected_level = None
            else:
                st.warning("Could not determine time range")
                selected_time_idx = 0
                selected_level = None
        
        with col2:
            st.subheader("Wind Analysis Options")
            
            analysis_type = st.radio(
                "Analysis Type",
                ["Spatial Wind Map", "Power Production Map", "Time Series at Point"]
            )
        
        st.markdown("---")
        
        # Get wind data
        u_data, v_data, lats, lons = get_era5_wind_data(selected_time_idx, selected_level)
        
        if u_data is not None and v_data is not None:
            # Calculate wind speed
            wind_speed = np.sqrt(u_data**2 + v_data**2)
            
            if analysis_type == "Spatial Wind Map":
                st.subheader("ERA5 Wind Speed Map")
                
                fig = go.Figure(data=go.Heatmap(
                    z=wind_speed,
                    x=lons,
                    y=lats,
                    colorscale="YlOrRd",
                    colorbar=dict(title="Wind Speed (m/s)")
                ))
                
                title = f"ERA5 Wind Speed"
                if selected_level:
                    title += f" at {selected_level} hPa"
                title += f" - {time_options[selected_time_idx]}"
                
                fig.update_layout(
                    title=title,
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Wind", f"{np.mean(wind_speed):.1f} m/s")
                col2.metric("Max Wind", f"{np.max(wind_speed):.1f} m/s")
                col3.metric("Min Wind", f"{np.min(wind_speed):.1f} m/s")
                col4.metric("Std Dev", f"{np.std(wind_speed):.1f} m/s")
                
            elif analysis_type == "Power Production Map":
                st.subheader("Estimated Power Production Map")
                
                # Calculate power at each grid point
                power_map = converter.wind_speed_to_power(wind_speed.flatten()).reshape(wind_speed.shape)
                
                fig = go.Figure(data=go.Heatmap(
                    z=power_map,
                    x=lons,
                    y=lats,
                    colorscale="Viridis",
                    colorbar=dict(title="Power (MW)")
                ))
                
                fig.update_layout(
                    title=f"Estimated Wind Power Production ({turbine_type})",
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Power", f"{np.mean(power_map):.2f} MW")
                col2.metric("Max Power", f"{np.max(power_map):.2f} MW")
                
                # Capacity factor based on spatial average
                cf = np.mean(power_map) / turbine.rated_power * 100
                col3.metric("Avg Capacity Factor", f"{cf:.1f}%")
                
                # Areas above rated
                above_rated = np.sum(power_map >= turbine.rated_power * 0.9) / power_map.size * 100
                col4.metric("% Above 90% Rated", f"{above_rated:.1f}%")
                
            else:  # Time Series at Point
                st.subheader("Wind Time Series Analysis")
                
                # Select location
                lat_options = lats.tolist()
                lon_options = lons.tolist()
                
                point_lat = st.select_slider("Latitude", options=lat_options, value=lat_options[len(lat_options)//2])
                point_lon = st.select_slider("Longitude", options=lon_options, value=lon_options[len(lon_options)//2])
                
                # Get lat/lon indices
                lat_idx = np.argmin(np.abs(lats - point_lat))
                lon_idx = np.argmin(np.abs(lons - point_lon))
                
                # Extract time series
                u_var = None
                v_var = None
                for name in ["u_component_of_wind", "u", "U"]:
                    if name in data.data_vars:
                        u_var = name
                        break
                for name in ["v_component_of_wind", "v", "V"]:
                    if name in data.data_vars:
                        v_var = name
                        break
                
                if u_var and v_var:
                    if "latitude" in data.coords:
                        u_ts = data[u_var].isel(latitude=lat_idx, longitude=lon_idx)
                        v_ts = data[v_var].isel(latitude=lat_idx, longitude=lon_idx)
                    else:
                        u_ts = data[u_var].isel(lat=lat_idx, lon=lon_idx)
                        v_ts = data[v_var].isel(lat=lat_idx, lon=lon_idx)
                    
                    if selected_level and "level" in u_ts.dims:
                        u_ts = u_ts.sel(level=selected_level)
                        v_ts = v_ts.sel(level=selected_level)
                    
                    wind_ts = np.sqrt(u_ts.values**2 + v_ts.values**2)
                    power_ts = converter.wind_speed_to_power(wind_ts)
                    
                    times = [str(t)[:19] for t in data.time.values]
                    
                    fig = make_subplots(rows=2, cols=1, subplot_titles=("Wind Speed", "Power Output"))
                    
                    fig.add_trace(
                        go.Scatter(x=times, y=wind_ts, name="Wind Speed", line=dict(color="#26a69a")),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=times, y=power_ts, name="Power", fill="tozeroy", line=dict(color="#1e88e5")),
                        row=2, col=1
                    )
                    
                    fig.update_yaxes(title_text="m/s", row=1, col=1)
                    fig.update_yaxes(title_text="MW", row=2, col=1)
                    fig.update_layout(height=500, showlegend=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Wind", f"{np.mean(wind_ts):.1f} m/s")
                    col2.metric("Mean Power", f"{np.mean(power_ts):.2f} MW")
                    cf = np.mean(power_ts) / turbine.rated_power * 100
                    col3.metric("Capacity Factor", f"{cf:.1f}%")
        else:
            st.warning("Could not extract wind data. Make sure your dataset contains u and v wind components.")
    
    else:
        st.warning("""
        ⚠️ **ERA5 Data Not Available**
        
        To use real ERA5 wind data:
        1. Go to the **Data Manager** page
        2. Download a sample dataset (e.g., "General Sample 2023")
        3. Click "Use This Dataset"
        4. Return here to analyze real wind data
        
        The other tabs on this page demonstrate functionality using synthetic data.
        """)
