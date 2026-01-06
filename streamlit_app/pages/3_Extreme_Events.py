"""
Extreme Event Detection - Heatwaves, Atmospheric Rivers, Extreme Precipitation

Uses the actual detector classes from applications/extreme_event_analysis/detectors.py
Supports ERA5 real data or demo mode with synthetic data.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))

from applications.extreme_event_analysis.detectors import (
    HeatwaveDetector,
    AtmosphericRiverDetector,
    ExtremePrecipitationDetector,
    ExtremeEvent
)

# Import ERA5 utilities
try:
    from era5_utils import (
        get_era5_data_banner,
        has_era5_data,
        get_active_era5_data,
        get_era5_temperature,
        get_era5_time_range,
    )
    ERA5_UTILS_AVAILABLE = True
except ImportError:
    ERA5_UTILS_AVAILABLE = False

st.set_page_config(page_title="Extreme Event Detection", page_icon="🌡️", layout="wide")

st.title("🌡️ Extreme Event Detection")
st.markdown("""
Detect and characterize extreme weather events using physics-based algorithms.
This runs the actual detector classes from the repository.
""")

# Show data source banner
if ERA5_UTILS_AVAILABLE:
    banner = get_era5_data_banner()
    if "Demo Mode" in banner or "⚠️" in banner:
        st.info(f"📊 {banner}")
    else:
        st.success(f"📊 {banner}")

# Tabs for different event types
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 Heatwave Detection",
    "🌊 Atmospheric Rivers",
    "🌧️ Extreme Precipitation",
    "🌍 ERA5 Event Analysis"
])

# Tab 1: Heatwave Detection
with tab1:
    st.header("Heatwave Detection")

    st.markdown("""
    A heatwave is defined as a period when temperature exceeds a threshold
    for a minimum duration over a minimum spatial extent.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Detection Parameters")

        threshold_type = st.radio("Threshold Type", ["Absolute Temperature", "Percentile"])

        if threshold_type == "Absolute Temperature":
            temp_threshold = st.slider("Temperature Threshold (°C)", 25.0, 45.0, 35.0)
            percentile_threshold = None
        else:
            percentile_threshold = st.slider("Percentile Threshold", 80.0, 99.0, 95.0)
            temp_threshold = None

        duration_days = st.slider("Minimum Duration (days)", 1, 7, 3)
        spatial_extent = st.slider("Minimum Spatial Extent", 0.05, 0.5, 0.1,
                                   help="Fraction of domain that must exceed threshold")

        st.markdown("---")
        st.subheader("Generate Synthetic Data")

        n_days = st.slider("Simulation Days", 10, 60, 30)
        grid_size = st.slider("Grid Size", 16, 64, 32)

        include_heatwave = st.checkbox("Include Heatwave Event", value=True)
        if include_heatwave:
            heatwave_start = st.slider("Heatwave Start Day", 5, n_days - 10, 10)
            heatwave_duration = st.slider("Heatwave Duration", 2, 10, 5)
            heatwave_intensity = st.slider("Heatwave Intensity (°C above normal)", 5.0, 20.0, 10.0)

    with col2:
        # Generate synthetic temperature data
        np.random.seed(42)

        n_times = n_days * 4  # 6-hourly data
        lats = np.linspace(30, 50, grid_size)
        lons = np.linspace(-120, -80, grid_size)

        # Base temperature field with spatial structure
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        base_temp = 25 + 5 * np.cos(np.radians(lat_grid - 40) * 3)  # Cooler at higher latitudes

        # Create time series with diurnal cycle
        temperature = np.zeros((n_times, grid_size, grid_size))
        times = [datetime(2024, 7, 1) + timedelta(hours=6*i) for i in range(n_times)]

        for t in range(n_times):
            hour = (t * 6) % 24
            diurnal = 5 * np.sin(np.pi * (hour - 6) / 12)  # Peak at 14:00
            noise = np.random.randn(grid_size, grid_size) * 2
            temperature[t] = base_temp + diurnal + noise

        # Add heatwave if requested
        if include_heatwave:
            hw_start_idx = heatwave_start * 4
            hw_end_idx = (heatwave_start + heatwave_duration) * 4

            # Create spatial pattern for heatwave
            hw_center_lat = 40
            hw_center_lon = -100
            hw_radius = 10

            dist_from_center = np.sqrt((lat_grid - hw_center_lat)**2 +
                                       (lon_grid - hw_center_lon)**2)
            hw_spatial = np.exp(-(dist_from_center / hw_radius)**2)

            for t in range(hw_start_idx, min(hw_end_idx, n_times)):
                temperature[t] += heatwave_intensity * hw_spatial

        # Convert to Kelvin
        temperature_k = temperature + 273.15

        # Create detector
        detector = HeatwaveDetector(
            temperature_threshold=temp_threshold,
            percentile_threshold=percentile_threshold,
            duration_days=duration_days,
            spatial_extent=spatial_extent
        )

        # Detect events
        events = detector.detect(
            temperature_k,
            times=np.array(times),
            lats=lats,
            lons=lons
        )

        # Visualization
        st.subheader(f"Detection Results: {len(events)} Heatwave(s) Found")

        # Time series of domain-average temperature
        domain_avg = np.mean(temperature, axis=(1, 2))

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Domain-Average Temperature', 'Spatial Distribution',
                          'Threshold Exceedance', 'Event Timeline'),
            specs=[[{}, {}], [{}, {}]]
        )

        # Domain average
        fig.add_trace(
            go.Scatter(x=list(range(n_times)), y=domain_avg,
                      name='Avg Temp', line=dict(color='#ef5350')),
            row=1, col=1
        )

        if temp_threshold:
            fig.add_hline(y=temp_threshold, line_dash="dash", line_color="red",
                         annotation_text="Threshold", row=1, col=1)

        # Spatial distribution (snapshot at peak)
        if include_heatwave:
            peak_idx = heatwave_start * 4 + heatwave_duration * 2
        else:
            peak_idx = n_times // 2

        fig.add_trace(
            go.Heatmap(z=temperature[peak_idx], x=lons, y=lats,
                      colorscale='RdYlBu_r', name='Temperature'),
            row=1, col=2
        )

        # Threshold exceedance over time
        threshold_val = temp_threshold if temp_threshold else np.percentile(temperature, percentile_threshold)
        exceedance = (temperature > threshold_val).mean(axis=(1, 2))

        fig.add_trace(
            go.Scatter(x=list(range(n_times)), y=exceedance * 100,
                      name='% Exceeding', fill='tozeroy',
                      line=dict(color='#ff7043')),
            row=2, col=1
        )
        fig.add_hline(y=spatial_extent * 100, line_dash="dash", line_color="green",
                     annotation_text="Spatial Threshold", row=2, col=1)

        # Event timeline
        for i, event in enumerate(events):
            start_day = (event.start_time - datetime(2024, 7, 1)).days
            end_day = (event.end_time - datetime(2024, 7, 1)).days

            fig.add_trace(
                go.Scatter(
                    x=[start_day, end_day],
                    y=[i, i],
                    mode='lines+markers',
                    name=f'Event {i+1}',
                    line=dict(width=10, color='#d32f2f'),
                    marker=dict(size=12)
                ),
                row=2, col=2
            )

        fig.update_xaxes(title_text="Time Step (6-hourly)", row=1, col=1)
        fig.update_xaxes(title_text="Longitude", row=1, col=2)
        fig.update_xaxes(title_text="Time Step", row=2, col=1)
        fig.update_xaxes(title_text="Day", row=2, col=2)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Latitude", row=1, col=2)
        fig.update_yaxes(title_text="% Domain", row=2, col=1)
        fig.update_yaxes(title_text="Event ID", row=2, col=2)

        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Event details
        if events:
            st.subheader("Event Details")
            for i, event in enumerate(events):
                with st.expander(f"Heatwave Event {i+1}", expanded=True):
                    event_cols = st.columns(4)
                    with event_cols[0]:
                        st.metric("Duration", f"{event.duration_hours:.0f} hours")
                    with event_cols[1]:
                        st.metric("Peak Temperature", f"{event.peak_value:.1f}°C")
                    with event_cols[2]:
                        st.metric("Mean Temperature", f"{event.mean_value:.1f}°C")
                    with event_cols[3]:
                        st.metric("Affected Area", f"{event.affected_area_km2:,.0f} km²")

                    st.markdown(f"""
                    - **Start**: {event.start_time}
                    - **End**: {event.end_time}
                    - **Center**: ({event.center_lat:.1f}°N, {event.center_lon:.1f}°W)
                    """)

# Tab 2: Atmospheric Rivers
with tab2:
    st.header("Atmospheric River Detection")

    st.markdown("""
    Atmospheric Rivers (ARs) are identified by high values of Integrated Vapor Transport (IVT)
    organized in long, narrow corridors.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Detection Parameters")

        ivt_threshold = st.slider("IVT Threshold (kg/m/s)", 150.0, 500.0, 250.0)
        length_threshold = st.slider("Minimum Length (km)", 1000.0, 3000.0, 2000.0)
        width_threshold = st.slider("Maximum Width (km)", 500.0, 1500.0, 1000.0)
        min_duration = st.slider("Minimum Duration (hours)", 3, 24, 6)

        st.markdown("---")
        st.subheader("Generate Synthetic Data")

        ar_n_times = st.slider("Time Steps", 10, 50, 20, key="ar_times")
        ar_grid_size = st.slider("Grid Size", 32, 96, 64, key="ar_grid")

        include_ar = st.checkbox("Include AR Event", value=True)

    with col2:
        # Generate synthetic IVT data
        np.random.seed(123)

        ar_lats = np.linspace(20, 60, ar_grid_size)
        ar_lons = np.linspace(-160, -100, ar_grid_size)

        lat_grid, lon_grid = np.meshgrid(ar_lats, ar_lons, indexing='ij')

        # Base IVT field (background moisture flux)
        ivt_base = 100 + 50 * np.random.rand(ar_n_times, ar_grid_size, ar_grid_size)

        if include_ar:
            # Create AR structure (elongated moisture plume)
            ar_center_lat = 40
            ar_center_lon = -130
            ar_orientation = 45  # degrees from east

            # Elongated ellipse
            rot_angle = np.radians(ar_orientation)

            for t in range(ar_n_times // 3, 2 * ar_n_times // 3):
                # AR moves eastward
                offset = (t - ar_n_times // 3) * 2

                rotated_lat = (lat_grid - ar_center_lat) * np.cos(rot_angle) - \
                             (lon_grid - ar_center_lon - offset) * np.sin(rot_angle)
                rotated_lon = (lat_grid - ar_center_lat) * np.sin(rot_angle) + \
                             (lon_grid - ar_center_lon - offset) * np.cos(rot_angle)

                ar_shape = np.exp(-(rotated_lat**2 / 100 + rotated_lon**2 / 400))
                ivt_base[t] += 400 * ar_shape

        # Create detector
        ar_detector = AtmosphericRiverDetector(
            ivt_threshold=ivt_threshold,
            length_threshold=length_threshold,
            width_threshold=width_threshold,
            min_duration_hours=min_duration
        )

        # Detect
        ar_times = [datetime(2024, 1, 1) + timedelta(hours=6*i) for i in range(ar_n_times)]

        ar_events = ar_detector.detect(
            ivt=ivt_base,
            times=np.array(ar_times),
            lats=ar_lats,
            lons=ar_lons
        )

        st.subheader(f"Detection Results: {len(ar_events)} AR(s) Found")

        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('IVT Field (Peak)', 'Maximum IVT Over Time',
                          'AR Mask', 'Event Characteristics'),
            specs=[[{}, {}], [{}, {}]]
        )

        # Find peak time
        peak_t = np.argmax(ivt_base.max(axis=(1, 2)))

        # IVT heatmap
        fig.add_trace(
            go.Heatmap(z=ivt_base[peak_t], x=ar_lons, y=ar_lats,
                      colorscale='Blues', name='IVT'),
            row=1, col=1
        )

        # Add AR threshold contour
        fig.add_trace(
            go.Contour(z=ivt_base[peak_t], x=ar_lons, y=ar_lats,
                      contours=dict(start=ivt_threshold, end=ivt_threshold, size=1),
                      line=dict(color='red', width=2),
                      showscale=False, name='Threshold'),
            row=1, col=1
        )

        # Max IVT time series
        max_ivt = ivt_base.max(axis=(1, 2))
        fig.add_trace(
            go.Scatter(x=list(range(ar_n_times)), y=max_ivt,
                      name='Max IVT', line=dict(color='#1e88e5', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=ivt_threshold, line_dash="dash", line_color="red", row=1, col=2)

        # AR mask
        ar_mask = ivt_base[peak_t] > ivt_threshold
        fig.add_trace(
            go.Heatmap(z=ar_mask.astype(float), x=ar_lons, y=ar_lats,
                      colorscale='Reds', name='AR Mask'),
            row=2, col=1
        )

        # Event characteristics
        if ar_events:
            lengths = [e.metadata.get('length_km', 0) for e in ar_events]
            widths = [e.metadata.get('width_km', 0) for e in ar_events]
            peak_ivts = [e.peak_value for e in ar_events]

            fig.add_trace(
                go.Scatter(x=lengths, y=widths, mode='markers',
                          marker=dict(size=15, color=peak_ivts,
                                    colorscale='Viridis', showscale=True,
                                    colorbar=dict(title='Peak IVT')),
                          name='AR Events'),
                row=2, col=2
            )
            fig.add_hline(y=width_threshold, line_dash="dash", line_color="green", row=2, col=2)
            fig.add_vline(x=length_threshold, line_dash="dash", line_color="green", row=2, col=2)

        fig.update_xaxes(title_text="Longitude", row=1, col=1)
        fig.update_xaxes(title_text="Time Step", row=1, col=2)
        fig.update_xaxes(title_text="Longitude", row=2, col=1)
        fig.update_xaxes(title_text="Length (km)", row=2, col=2)
        fig.update_yaxes(title_text="Latitude", row=1, col=1)
        fig.update_yaxes(title_text="IVT (kg/m/s)", row=1, col=2)
        fig.update_yaxes(title_text="Latitude", row=2, col=1)
        fig.update_yaxes(title_text="Width (km)", row=2, col=2)

        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Event details
        if ar_events:
            st.subheader("AR Event Details")
            for i, event in enumerate(ar_events):
                with st.expander(f"Atmospheric River {i+1}", expanded=True):
                    event_cols = st.columns(4)
                    with event_cols[0]:
                        st.metric("Peak IVT", f"{event.peak_value:.0f} kg/m/s")
                    with event_cols[1]:
                        st.metric("Length", f"{event.metadata.get('length_km', 0):,.0f} km")
                    with event_cols[2]:
                        st.metric("Width", f"{event.metadata.get('width_km', 0):,.0f} km")
                    with event_cols[3]:
                        st.metric("Area", f"{event.affected_area_km2:,.0f} km²")

# Tab 3: Extreme Precipitation
with tab3:
    st.header("Extreme Precipitation Detection")

    st.markdown("""
    Detect extreme precipitation events based on intensity thresholds
    and spatial extent criteria.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Detection Parameters")

        precip_threshold_type = st.radio("Threshold Type", ["Absolute (mm/day)", "Percentile"],
                                         key="precip_type")

        if precip_threshold_type == "Absolute (mm/day)":
            precip_threshold = st.slider("Precipitation Threshold (mm/day)", 20.0, 200.0, 50.0)
            precip_percentile = None
        else:
            precip_percentile = st.slider("Percentile Threshold", 90.0, 99.9, 99.0)
            precip_threshold = None

        min_area = st.slider("Minimum Affected Area (km²)", 1000, 100000, 10000)

        st.markdown("---")
        st.subheader("Generate Synthetic Data")

        precip_days = st.slider("Simulation Days", 10, 90, 30, key="precip_days")
        precip_grid = st.slider("Grid Size", 32, 96, 48, key="precip_grid")

        include_extreme = st.checkbox("Include Extreme Event", value=True, key="precip_event")

    with col2:
        # Generate synthetic precipitation data
        np.random.seed(789)

        p_lats = np.linspace(25, 50, precip_grid)
        p_lons = np.linspace(-130, -70, precip_grid)
        lat_grid, lon_grid = np.meshgrid(p_lats, p_lons, indexing='ij')

        # Base precipitation (gamma distribution - typical for daily precip)
        precip_data = np.random.gamma(shape=0.5, scale=5, size=(precip_days, precip_grid, precip_grid))

        # Add spatial coherence
        from scipy.ndimage import gaussian_filter
        for t in range(precip_days):
            precip_data[t] = gaussian_filter(precip_data[t], sigma=2)

        if include_extreme:
            # Add extreme event (e.g., landfalling hurricane or atmospheric river)
            extreme_day = precip_days // 2
            extreme_center_lat = 35
            extreme_center_lon = -90

            dist = np.sqrt((lat_grid - extreme_center_lat)**2 +
                          (lon_grid - extreme_center_lon)**2)
            extreme_pattern = 150 * np.exp(-(dist / 5)**2)

            precip_data[extreme_day] += extreme_pattern
            precip_data[extreme_day + 1] += extreme_pattern * 0.5

        # Create detector
        precip_detector = ExtremePrecipitationDetector(
            threshold_mm=precip_threshold,
            percentile_threshold=precip_percentile,
            min_area_km2=min_area
        )

        # Detect
        p_times = [datetime(2024, 9, 1) + timedelta(days=i) for i in range(precip_days)]

        precip_events = precip_detector.detect(
            precip_data,
            times=np.array(p_times),
            lats=p_lats,
            lons=p_lons
        )

        st.subheader(f"Detection Results: {len(precip_events)} Extreme Event(s) Found")

        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Maximum Precipitation', 'Precipitation Field (Peak Day)',
                          'Exceedance Area Over Time', 'Event Intensity Distribution'),
            specs=[[{}, {}], [{}, {}]]
        )

        # Max precip time series
        max_precip = precip_data.max(axis=(1, 2))
        fig.add_trace(
            go.Scatter(x=list(range(precip_days)), y=max_precip,
                      name='Max Precip', line=dict(color='#1565c0', width=2)),
            row=1, col=1
        )

        threshold_val = precip_threshold if precip_threshold else np.percentile(precip_data, precip_percentile)
        fig.add_hline(y=threshold_val, line_dash="dash", line_color="red", row=1, col=1)

        # Peak day spatial
        peak_day = np.argmax(max_precip)
        fig.add_trace(
            go.Heatmap(z=precip_data[peak_day], x=p_lons, y=p_lats,
                      colorscale='Blues', name='Precipitation'),
            row=1, col=2
        )

        # Exceedance area
        exceedance_area = (precip_data > threshold_val).sum(axis=(1, 2))
        fig.add_trace(
            go.Bar(x=list(range(precip_days)), y=exceedance_area,
                  name='Exceeding Cells', marker_color='#42a5f5'),
            row=2, col=1
        )

        # Event intensity histogram
        if precip_events:
            intensities = [e.peak_value for e in precip_events]
            fig.add_trace(
                go.Histogram(x=intensities, nbinsx=20, name='Event Intensities',
                           marker_color='#1e88e5'),
                row=2, col=2
            )

        fig.update_xaxes(title_text="Day", row=1, col=1)
        fig.update_xaxes(title_text="Longitude", row=1, col=2)
        fig.update_xaxes(title_text="Day", row=2, col=1)
        fig.update_xaxes(title_text="Peak Precipitation (mm)", row=2, col=2)
        fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
        fig.update_yaxes(title_text="Latitude", row=1, col=2)
        fig.update_yaxes(title_text="Grid Cells", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Event summary
        if precip_events:
            st.subheader("Extreme Precipitation Events")

            for i, event in enumerate(precip_events):
                with st.expander(f"Event {i+1}: {event.start_time.strftime('%Y-%m-%d')}", expanded=i==0):
                    event_cols = st.columns(4)
                    with event_cols[0]:
                        st.metric("Peak Precipitation", f"{event.peak_value:.1f} mm")
                    with event_cols[1]:
                        st.metric("Mean Precipitation", f"{event.mean_value:.1f} mm")
                    with event_cols[2]:
                        st.metric("Affected Area", f"{event.affected_area_km2:,.0f} km²")
                    with event_cols[3]:
                        st.metric("Duration", f"{event.duration_hours:.0f} hours")

# Code reference
with st.expander("📝 Code Reference"):
    st.markdown("""
    This page uses the following code from the repository:

    ```python
    # From applications/extreme_event_analysis/detectors.py
    from applications.extreme_event_analysis.detectors import (
        HeatwaveDetector,
        AtmosphericRiverDetector,
        ExtremePrecipitationDetector,
        ExtremeEvent
    )

    # Heatwave detection
    hw_detector = HeatwaveDetector(
        temperature_threshold=35.0,  # °C
        duration_days=3,
        spatial_extent=0.1  # 10% of domain
    )
    events = hw_detector.detect(temperature, times, lats, lons)

    # Atmospheric River detection
    ar_detector = AtmosphericRiverDetector(
        ivt_threshold=250.0,  # kg/m/s
        length_threshold=2000.0,  # km
        width_threshold=1000.0  # km
    )
    ar_events = ar_detector.detect(ivt, times, lats, lons)

    # Extreme Precipitation detection
    precip_detector = ExtremePrecipitationDetector(
        threshold_mm=50.0,
        min_area_km2=10000
    )
    precip_events = precip_detector.detect(precipitation, times, lats, lons)
    ```
    """)

# Tab 4: ERA5 Event Analysis
with tab4:
    st.header("🌍 Detect Events in ERA5 Data")
    
    st.markdown("""
    Run extreme event detection algorithms on **real ERA5 reanalysis data**.
    This allows you to identify actual historical weather events.
    """)
    
    if ERA5_UTILS_AVAILABLE and has_era5_data():
        data, metadata = get_active_era5_data()
        
        st.success(f"✅ Using Real ERA5 Data: **{metadata.get('name', 'Unknown')}**")
        st.markdown(f"- **Period:** {metadata.get('start_date', '?')} to {metadata.get('end_date', '?')}")
        
        # Check available variables
        available_vars = list(data.data_vars)
        has_temp = any(v in available_vars for v in ["temperature", "t", "T"])
        has_wind = all(any(v in available_vars for v in names) 
                      for names in [["u_component_of_wind", "u"], ["v_component_of_wind", "v"]])
        
        st.markdown("**Available for analysis:**")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"{'✅' if has_temp else '❌'} Temperature (Heatwave)")
        col2.markdown(f"{'✅' if has_wind else '❌'} Wind (Atmospheric Rivers)")
        col3.markdown(f"❌ Precipitation (not in sample)")
        
        if has_temp:
            st.markdown("---")
            st.subheader("🔥 Heatwave Detection on ERA5 Data")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Detection parameters
                era5_temp_threshold = st.slider(
                    "Temperature Threshold (K)",
                    280.0, 320.0, 300.0,
                    help="Temperature above which conditions are considered extreme"
                )
                
                era5_duration = st.slider(
                    "Minimum Duration (days)",
                    1, 7, 2
                )
                
                era5_spatial = st.slider(
                    "Minimum Spatial Extent",
                    0.05, 0.5, 0.1
                )
                
                # Select pressure level if available
                if "level" in data.coords:
                    levels = sorted([int(l) for l in data.level.values])
                    era5_level = st.selectbox(
                        "Pressure Level (hPa)",
                        options=levels,
                        index=0  # Surface level
                    )
                else:
                    era5_level = None
                
                run_detection = st.button("🔍 Run Heatwave Detection", type="primary")
            
            with col2:
                if run_detection:
                    with st.spinner("Analyzing ERA5 temperature data..."):
                        try:
                            # Get temperature variable name
                            temp_var = None
                            for name in ["temperature", "t", "T"]:
                                if name in available_vars:
                                    temp_var = name
                                    break
                            
                            # Get coordinates
                            if "latitude" in data.coords:
                                lats = data.latitude.values
                                lons = data.longitude.values
                            else:
                                lats = data.lat.values
                                lons = data.lon.values
                            
                            # Extract temperature data
                            temp_data = data[temp_var]
                            if era5_level is not None and "level" in temp_data.dims:
                                temp_data = temp_data.sel(level=era5_level)
                            
                            # Get as numpy array
                            temp_values = temp_data.values
                            times = np.array([datetime.fromisoformat(str(t)[:19]) for t in data.time.values])
                            
                            # Create detector
                            detector = HeatwaveDetector(
                                temperature_threshold=era5_temp_threshold,
                                duration_days=era5_duration,
                                spatial_extent=era5_spatial
                            )
                            
                            # Detect events
                            events = detector.detect(temp_values, times, lats, lons)
                            
                            st.success(f"✅ Found **{len(events)} heatwave event(s)**")
                            
                            if events:
                                for i, event in enumerate(events):
                                    with st.expander(f"Event {i+1}: {event.start_time.strftime('%Y-%m-%d')}", expanded=True):
                                        ev_cols = st.columns(4)
                                        ev_cols[0].metric("Duration", f"{event.duration_hours:.0f} hours")
                                        ev_cols[1].metric("Peak Temperature", f"{event.peak_value:.1f} K")
                                        ev_cols[2].metric("Mean Temperature", f"{event.mean_value:.1f} K")
                                        ev_cols[3].metric("Affected Area", f"{event.affected_area_km2:,.0f} km²")
                            else:
                                st.info("No heatwave events detected with current thresholds. Try adjusting parameters.")
                            
                            # Show temperature time series
                            st.subheader("Temperature Time Series")
                            domain_mean = np.mean(temp_values, axis=(1, 2))
                            domain_max = np.max(temp_values, axis=(1, 2))
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=times, y=domain_mean,
                                name="Domain Mean", line=dict(color="#1e88e5")
                            ))
                            fig.add_trace(go.Scatter(
                                x=times, y=domain_max,
                                name="Domain Max", line=dict(color="#ef5350")
                            ))
                            fig.add_hline(y=era5_temp_threshold, line_dash="dash", 
                                         line_color="red", annotation_text="Threshold")
                            
                            fig.update_layout(
                                title="ERA5 Temperature (Real Data)",
                                xaxis_title="Time",
                                yaxis_title="Temperature (K)",
                                height=350
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error during detection: {e}")
                else:
                    st.info("Configure detection parameters and click 'Run Heatwave Detection' to analyze ERA5 data.")
        
        else:
            st.warning("Temperature data not available in current dataset. Please load a dataset with temperature.")
    
    else:
        st.warning("""
        ⚠️ **ERA5 Data Not Available**
        
        To run event detection on real data:
        1. Go to the **Data Manager** page
        2. Download a sample dataset (e.g., "European Heat Wave 2003" for heatwave analysis)
        3. Click "Use This Dataset"
        4. Return here to detect events in real data
        
        The other tabs demonstrate event detection using synthetic data.
        """)
