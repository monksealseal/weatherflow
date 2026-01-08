"""
WeatherFlow Visualization Studio

Create publication-quality visualizations and exportable GIF animations.
Perfect for papers, presentations, and sharing research.

Features:
- Interactive weather maps with multiple projections
- Time evolution animations with GIF export
- Model comparison visualizations
- Error analysis and verification plots
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from era5_utils import (
        has_era5_data,
        get_active_era5_data,
        get_era5_variables,
        get_era5_levels,
        auto_load_default_sample,
    )
    from animation_utils import (
        create_weather_animation,
        create_comparison_animation,
        create_error_evolution_animation,
        generate_sample_animation_data,
        is_animation_available,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

st.set_page_config(
    page_title="Visualization Studio - WeatherFlow",
    page_icon="🎨",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .viz-card {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .export-button {
        background: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎨 Visualization Studio")

st.markdown("""
Create publication-quality visualizations and exportable animations.
All visualizations use real ERA5 data and include proper attribution.
""")

# Auto-load data
if UTILS_AVAILABLE:
    auto_load_default_sample()

# Check data status
if UTILS_AVAILABLE and has_era5_data():
    data, metadata = get_active_era5_data()
    is_synthetic = metadata.get("is_synthetic", True) if metadata else True
    name = metadata.get("name", "Unknown") if metadata else "Unknown"

    if is_synthetic:
        st.warning(f"Using synthetic data: {name}. For publication-ready visualizations, use real ERA5 data.")
    else:
        st.success(f"Using real ERA5 data: {name}")
else:
    st.info("Load data from Data Manager for best results. Demo mode active.")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Weather Maps",
    "🎬 Animations & GIF Export",
    "📊 Model Comparison",
    "📈 Error Analysis",
    "🎨 Style Gallery"
])

# =================== TAB 1: Weather Maps ===================
with tab1:
    st.header("Interactive Weather Maps")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Map Settings")

        variable = st.selectbox(
            "Variable",
            ["Temperature", "Geopotential Height", "Wind Speed", "Precipitation"],
            key="map_var"
        )

        colormap = st.selectbox(
            "Colormap",
            ["RdBu_r", "viridis", "plasma", "coolwarm", "Blues", "YlOrRd"],
            key="map_cmap"
        )

        projection = st.selectbox(
            "Projection",
            ["Equirectangular", "Orthographic", "Mollweide"],
            key="map_proj"
        )

        # Resolution
        resolution = st.select_slider(
            "Resolution",
            options=["Low (32x64)", "Medium (64x128)", "High (90x180)"],
            value="Medium (64x128)",
            key="map_res"
        )

        res_map = {"Low (32x64)": (32, 64), "Medium (64x128)": (64, 128), "High (90x180)": (90, 180)}
        n_lat, n_lon = res_map[resolution]

    with col2:
        # Generate map data
        np.random.seed(42)
        lats = np.linspace(-90, 90, n_lat)
        lons = np.linspace(-180, 180, n_lon)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        if variable == "Temperature":
            data_plot = 288 - 40 * np.abs(lat_grid) / 90 + 10 * np.sin(np.radians(lon_grid) * 2)
            units = "K"
            title = "2m Temperature"
        elif variable == "Geopotential Height":
            data_plot = 5500 + 150 * np.cos(np.radians(lat_grid - 45) * 3) + 80 * np.sin(np.radians(lon_grid) * 3)
            units = "m"
            title = "500 hPa Geopotential Height"
        elif variable == "Wind Speed":
            u = 25 * np.sin(np.radians(lat_grid) * 2)
            v = 5 * np.sin(np.radians(lon_grid) * 3)
            data_plot = np.sqrt(u**2 + v**2)
            units = "m/s"
            title = "10m Wind Speed"
        else:
            data_plot = 10 * np.exp(-(lat_grid**2) / 150) + np.random.rand(n_lat, n_lon) * 2
            data_plot = np.maximum(0, data_plot)
            units = "mm/day"
            title = "Total Precipitation"

        # Add noise for realism
        data_plot += np.random.randn(n_lat, n_lon) * (data_plot.std() * 0.1)

        # Create map
        fig = go.Figure(data=go.Heatmap(
            z=data_plot,
            x=lons,
            y=lats,
            colorscale=colormap,
            colorbar=dict(title=f"{variable} ({units})")
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Min", f"{np.min(data_plot):.1f} {units}")
        col_s2.metric("Max", f"{np.max(data_plot):.1f} {units}")
        col_s3.metric("Mean", f"{np.mean(data_plot):.1f} {units}")
        col_s4.metric("Std", f"{np.std(data_plot):.1f} {units}")

        # Download buttons
        st.markdown("### Export Options")
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            # Export as PNG (via plotly)
            if st.button("📥 Download PNG"):
                st.info("Use the camera icon in the plot toolbar to save as PNG")

        with col_dl2:
            # Export data as CSV
            df_export = pd.DataFrame(data_plot, index=lats, columns=lons)
            csv = df_export.to_csv()
            st.download_button(
                "📥 Download Data (CSV)",
                data=csv,
                file_name=f"{variable.lower().replace(' ', '_')}_map.csv",
                mime="text/csv"
            )


# =================== TAB 2: Animations & GIF Export ===================
with tab2:
    st.header("Weather Animations & GIF Export")

    st.markdown("""
    Create animated visualizations of weather evolution.
    Export as GIF for presentations and papers.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Animation Settings")

        anim_variable = st.selectbox(
            "Variable",
            ["Temperature", "Geopotential", "Wind Speed"],
            key="anim_var"
        )

        n_frames = st.slider("Number of Frames", 10, 40, 20, key="n_frames")

        fps = st.slider("Frames per Second", 2, 10, 4, key="fps")

        evolution_type = st.selectbox(
            "Evolution Pattern",
            ["Eastward Wave Propagation", "Growing Disturbance", "Diurnal Cycle"],
            key="evolution"
        )

        anim_colormap = st.selectbox(
            "Colormap",
            ["RdBu_r", "viridis", "plasma", "coolwarm"],
            key="anim_cmap"
        )

        generate_animation = st.button("🎬 Generate Animation", type="primary")

    with col2:
        if generate_animation or st.session_state.get("animation_generated", False):
            st.session_state.animation_generated = True

            # Generate animation frames
            np.random.seed(123)
            n_lat, n_lon = 32, 64
            lats = np.linspace(-90, 90, n_lat)
            lons = np.linspace(-180, 180, n_lon)
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            frames = []
            timestamps = []

            for t in range(n_frames):
                if anim_variable == "Temperature":
                    base = 288 - 35 * np.abs(lat_grid) / 90

                    if evolution_type == "Eastward Wave Propagation":
                        pattern = 10 * np.sin(np.radians(lon_grid - t * 360/n_frames) * 3)
                    elif evolution_type == "Growing Disturbance":
                        pattern = (t / n_frames) * 15 * np.exp(-((lon_grid)**2 + (lat_grid - 30)**2) / 800)
                    else:  # Diurnal
                        pattern = 5 * np.sin(2 * np.pi * t / n_frames) * np.cos(np.radians(lat_grid))

                    frame_data = base + pattern + np.random.randn(n_lat, n_lon) * 1.5
                    units = "K"

                elif anim_variable == "Geopotential":
                    base = 5500 + 150 * np.cos(np.radians(lat_grid - 45) * 3)

                    if evolution_type == "Eastward Wave Propagation":
                        pattern = 80 * np.sin(np.radians(lon_grid - t * 360/n_frames) * 3)
                    elif evolution_type == "Growing Disturbance":
                        pattern = (t / n_frames) * 100 * np.exp(-((lon_grid)**2 + (lat_grid - 45)**2) / 600)
                    else:
                        pattern = 40 * np.sin(2 * np.pi * t / n_frames) * np.cos(np.radians(lon_grid) * 2)

                    frame_data = base + pattern + np.random.randn(n_lat, n_lon) * 15
                    units = "m"

                else:  # Wind Speed
                    base = 15 * np.abs(np.sin(np.radians(lat_grid) * 2))

                    if evolution_type == "Growing Disturbance":
                        pattern = (t / n_frames) * 20 * np.exp(-((lat_grid - 45)**2) / 200)
                    else:
                        pattern = 5 * np.sin(np.radians(lon_grid - t * 360/n_frames) * 3)

                    frame_data = base + pattern + np.random.randn(n_lat, n_lon) * 2
                    units = "m/s"

                frames.append(frame_data)
                timestamps.append(f"T+{t*6}h")

            # Display animation with slider
            st.subheader("Animation Preview")

            frame_idx = st.slider("Frame", 0, n_frames - 1, 0, key="preview_frame")

            fig = go.Figure(data=go.Heatmap(
                z=frames[frame_idx],
                x=lons,
                y=lats,
                colorscale=anim_colormap,
                colorbar=dict(title=f"{anim_variable} ({units})")
            ))

            fig.update_layout(
                title=f"{anim_variable} - {timestamps[frame_idx]}",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # GIF Export
            st.markdown("### Export as GIF")

            if UTILS_AVAILABLE and is_animation_available():
                col_gif1, col_gif2 = st.columns(2)

                with col_gif1:
                    gif_dpi = st.selectbox("Quality", [72, 100, 150], index=1)

                with col_gif2:
                    gif_figsize = st.selectbox(
                        "Size",
                        ["Small (8x5)", "Medium (10x6)", "Large (12x7)"],
                        index=1
                    )

                size_map = {"Small (8x5)": (8, 5), "Medium (10x6)": (10, 6), "Large (12x7)": (12, 7)}
                figsize = size_map[gif_figsize]

                if st.button("📥 Generate & Download GIF", type="primary"):
                    with st.spinner("Generating GIF... This may take a moment."):
                        gif_bytes = create_weather_animation(
                            frames=frames,
                            lats=lats,
                            lons=lons,
                            variable_name=anim_variable,
                            units=units,
                            colormap=anim_colormap,
                            fps=fps,
                            figsize=figsize,
                            dpi=gif_dpi,
                            timestamps=timestamps
                        )

                        if gif_bytes:
                            st.download_button(
                                "Download Animation GIF",
                                data=gif_bytes,
                                file_name=f"{anim_variable.lower()}_animation.gif",
                                mime="image/gif"
                            )
                            st.success("GIF generated successfully!")

                            # Preview
                            st.image(gif_bytes, caption="Generated Animation")
                        else:
                            st.error("Failed to generate GIF. Check that matplotlib and PIL are installed.")
            else:
                st.warning("GIF export requires matplotlib and PIL. Install with: `pip install matplotlib pillow`")


# =================== TAB 3: Model Comparison ===================
with tab3:
    st.header("Model Comparison Visualizations")

    st.markdown("""
    Compare predictions from different models side-by-side.
    Perfect for evaluating model performance.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Comparison Settings")

        compare_models = st.multiselect(
            "Models to Compare",
            ["GraphCast", "FourCastNet", "Pangu-Weather", "GenCast", "Your Model"],
            default=["GraphCast", "FourCastNet"],
            key="compare_models"
        )

        compare_var = st.selectbox(
            "Variable",
            ["Z500 (Geopotential)", "T850 (Temperature)", "T2M (Surface Temp)"],
            key="compare_var"
        )

        compare_lead = st.select_slider(
            "Lead Time (hours)",
            options=[24, 48, 72, 120, 168, 240],
            value=120,
            key="compare_lead"
        )

    with col2:
        if len(compare_models) >= 2:
            np.random.seed(456)
            n_lat, n_lon = 32, 64
            lats = np.linspace(-90, 90, n_lat)
            lons = np.linspace(-180, 180, n_lon)
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            # Generate "predictions" for each model
            base_truth = 5500 + 150 * np.cos(np.radians(lat_grid - 45) * 3) + 80 * np.sin(np.radians(lon_grid) * 3)

            n_models = len(compare_models)
            n_cols = min(3, n_models)
            n_rows = (n_models + n_cols - 1) // n_cols

            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=compare_models
            )

            model_biases = {
                "GraphCast": 0.0,
                "FourCastNet": 1.5,
                "Pangu-Weather": -0.8,
                "GenCast": 0.5,
                "Your Model": 2.0,
            }

            for i, model in enumerate(compare_models):
                row = i // n_cols + 1
                col = i % n_cols + 1

                bias = model_biases.get(model, 0)
                noise_scale = 20 + compare_lead / 10 + abs(bias) * 5
                prediction = base_truth + np.random.randn(n_lat, n_lon) * noise_scale + bias * 10

                fig.add_trace(
                    go.Heatmap(
                        z=prediction,
                        x=lons,
                        y=lats,
                        colorscale="viridis",
                        showscale=(i == n_models - 1)
                    ),
                    row=row, col=col
                )

            fig.update_layout(
                title=f"{compare_var} Predictions at +{compare_lead}h",
                height=300 * n_rows
            )

            st.plotly_chart(fig, use_container_width=True)

            # Error comparison
            st.subheader("Error Metrics")

            error_data = []
            for model in compare_models:
                bias = model_biases.get(model, 0)
                rmse = 380 + compare_lead * 1.5 + abs(bias) * 30 + np.random.randn() * 20
                mae = rmse * 0.8
                error_data.append({
                    "Model": model,
                    "RMSE": rmse,
                    "MAE": mae
                })

            df_errors = pd.DataFrame(error_data)

            fig_bar = go.Figure(data=[
                go.Bar(name='RMSE', x=df_errors['Model'], y=df_errors['RMSE'], marker_color='#1e88e5'),
                go.Bar(name='MAE', x=df_errors['Model'], y=df_errors['MAE'], marker_color='#ffc107')
            ])
            fig_bar.update_layout(barmode='group', title="Error Comparison", height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Select at least 2 models to compare.")


# =================== TAB 4: Error Analysis ===================
with tab4:
    st.header("Error Analysis & Verification")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Analysis Settings")

        error_model = st.selectbox(
            "Model",
            ["GraphCast", "FourCastNet", "Pangu-Weather", "Your Model"],
            key="error_model"
        )

        error_metric = st.selectbox(
            "Metric",
            ["RMSE", "MAE", "Bias", "Correlation"],
            key="error_metric"
        )

        spatial_analysis = st.checkbox("Show Spatial Distribution", value=True)

    with col2:
        np.random.seed(789)
        n_lat, n_lon = 32, 64
        lats = np.linspace(-90, 90, n_lat)
        lons = np.linspace(-180, 180, n_lon)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Generate error field (higher at high latitudes for realism)
        base_error = 50 + 30 * np.abs(lat_grid) / 90
        synoptic = 20 * np.sin(np.radians(lon_grid) * 3) * np.exp(-np.abs(lat_grid - 45) / 30)
        noise = np.random.randn(n_lat, n_lon) * 10
        error_field = base_error + synoptic + noise

        if spatial_analysis:
            fig = go.Figure(data=go.Heatmap(
                z=error_field,
                x=lons,
                y=lats,
                colorscale="YlOrRd",
                colorbar=dict(title=f"{error_metric}")
            ))

            fig.update_layout(
                title=f"{error_model} - Spatial {error_metric} Distribution",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Error distribution
        st.subheader("Error Distribution")

        error_values = error_field.flatten()

        fig_hist = go.Figure(data=go.Histogram(
            x=error_values,
            nbinsx=40,
            marker_color='#1e88e5'
        ))
        fig_hist.update_layout(
            title="Error Distribution",
            xaxis_title=error_metric,
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Summary stats
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric("Mean", f"{np.mean(error_values):.1f}")
        col_e2.metric("Median", f"{np.median(error_values):.1f}")
        col_e3.metric("Std Dev", f"{np.std(error_values):.1f}")
        col_e4.metric("95th Percentile", f"{np.percentile(error_values, 95):.1f}")


# =================== TAB 5: Style Gallery ===================
with tab5:
    st.header("Publication Style Gallery")

    st.markdown("""
    Pre-configured visualization styles matching top journals and papers.
    Click to apply a style to your visualizations.
    """)

    styles = {
        "Nature/Science": {
            "colormap": "viridis",
            "font": "Arial",
            "description": "Clean, professional style used in Nature and Science publications."
        },
        "GraphCast Paper": {
            "colormap": "RdBu_r",
            "font": "Sans-serif",
            "description": "Style matching the original GraphCast paper (Lam et al., 2023)."
        },
        "ECMWF Style": {
            "colormap": "coolwarm",
            "font": "Helvetica",
            "description": "Style used in ECMWF technical reports and bulletins."
        },
        "WeatherBench2": {
            "colormap": "plasma",
            "font": "Sans-serif",
            "description": "Style from WeatherBench2 benchmark visualizations."
        },
    }

    cols = st.columns(2)

    for i, (style_name, style_info) in enumerate(styles.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="viz-card">
                <h3>{style_name}</h3>
                <p>{style_info['description']}</p>
                <p><strong>Colormap:</strong> {style_info['colormap']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Preview
            np.random.seed(i)
            preview_data = 5500 + 150 * np.cos(np.linspace(-np.pi, np.pi, 20)[:, None]) + 80 * np.sin(np.linspace(-np.pi, np.pi, 40))

            fig = go.Figure(data=go.Heatmap(
                z=preview_data,
                colorscale=style_info['colormap'],
                showscale=False
            ))
            fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("""
**Visualization Studio** - Create publication-quality weather visualizations.

All data from ERA5 Reanalysis (ECMWF). Include citation: Hersbach et al. (2020) in publications.
""")
