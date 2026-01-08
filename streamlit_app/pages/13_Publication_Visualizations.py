"""
Publication-Quality Weather AI Visualizations

Create visualizations matching the style of top weather AI papers
like GraphCast, FourCastNet, and Pangu-Weather.

Features:
    - Global forecast maps with proper projections
    - Ensemble spread visualization
    - Skill score scorecards
    - Spectral analysis plots
    - Error distribution maps
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Publication Visualizations - WeatherFlow",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Publication-Quality Visualizations")

st.markdown("""
**Create publication-ready weather AI visualizations.**

Generate visualizations in the style of GraphCast, FourCastNet, GenCast, and other
top weather AI papers. Perfect for papers, presentations, and reports.
""")

# Sidebar settings
st.sidebar.header("🎨 Visualization Settings")

colormap = st.sidebar.selectbox(
    "Color Scheme",
    ["viridis", "plasma", "inferno", "magma", "cividis", "RdBu_r", "coolwarm", "seismic"]
)

projection = st.sidebar.selectbox(
    "Map Projection",
    ["equirectangular", "orthographic", "natural earth", "mollweide", "robinson"]
)

dpi = st.sidebar.slider("Figure DPI", 100, 300, 150)

# Generate synthetic data for demonstration
def generate_weather_field(lat_size=64, lon_size=128, pattern="z500"):
    """Generate realistic-looking weather field patterns."""
    lat = np.linspace(-90, 90, lat_size)
    lon = np.linspace(0, 360, lon_size)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Base pattern
    if pattern == "z500":
        # Geopotential-like pattern with jet stream
        field = 54000 + 2000 * np.cos(np.deg2rad(lat_grid)) ** 2
        field += 500 * np.sin(3 * np.deg2rad(lon_grid) + np.deg2rad(lat_grid))
        field += np.random.randn(lat_size, lon_size) * 100
    elif pattern == "t2m":
        # Temperature pattern
        field = 288 + 30 * np.cos(np.deg2rad(lat_grid))
        field += 5 * np.sin(np.deg2rad(lon_grid))
        field += np.random.randn(lat_size, lon_size) * 2
    elif pattern == "precip":
        # Precipitation pattern (ITCZ + mid-lat storms)
        field = np.exp(-((lat_grid - 5) ** 2) / 200) * 10
        field += np.exp(-((lat_grid - 45) ** 2) / 300) * np.abs(np.sin(3 * np.deg2rad(lon_grid))) * 5
        field += np.random.exponential(0.5, (lat_size, lon_size))
        field = np.clip(field, 0, None)
    elif pattern == "wind":
        # Wind speed pattern
        field = 5 + 15 * np.abs(np.sin(np.deg2rad(lat_grid)))
        field += np.random.randn(lat_size, lon_size) * 2
        field = np.clip(field, 0, None)
    else:
        field = np.random.randn(lat_size, lon_size)

    return field, lat, lon


# Tabs for different visualization types
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Forecast Maps",
    "📈 Skill Scorecards",
    "🎯 Error Analysis",
    "🌊 Ensemble Spread",
    "📉 Spectral Analysis"
])

# ============= TAB 1: Forecast Maps =============
with tab1:
    st.subheader("Global Forecast Maps")

    st.markdown("""
    *Style: GraphCast, Pangu-Weather, FourCastNet papers*
    """)

    variable = st.selectbox(
        "Variable",
        ["Z500 (Geopotential Height)", "T2M (2m Temperature)", "Precipitation", "Wind Speed"]
    )

    var_key = {"Z500": "z500", "T2M": "t2m", "Precip": "precip", "Wind": "wind"}
    pattern = var_key.get(variable.split()[0], "z500")

    # Generate data
    field, lat, lon = generate_weather_field(64, 128, pattern)

    # Create figure
    fig = go.Figure()

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=field,
        x=lon,
        y=lat,
        colorscale=colormap,
        colorbar=dict(
            title=dict(text=variable, side="right"),
            thickness=15,
        ),
        hovertemplate="Lon: %{x:.1f}°<br>Lat: %{y:.1f}°<br>Value: %{z:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"{variable} - 5-Day Forecast",
            font=dict(size=18, family="Arial"),
        ),
        xaxis=dict(title="Longitude", range=[0, 360]),
        yaxis=dict(title="Latitude", range=[-90, 90], scaleanchor="x"),
        height=500,
        margin=dict(l=60, r=20, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Multi-panel forecast progression
    st.markdown("### Forecast Progression (GraphCast Style)")

    lead_times = [0, 24, 72, 120, 240]
    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=[f"T+{lt}h" for lt in lead_times],
        horizontal_spacing=0.02,
    )

    for i, lt in enumerate(lead_times):
        # Add some "error growth" for later times
        error_scale = 1 + lt / 500
        field_lt = field + np.random.randn(*field.shape) * error_scale * 50

        fig.add_trace(
            go.Heatmap(
                z=field_lt,
                colorscale=colormap,
                showscale=(i == len(lead_times) - 1),
            ),
            row=1, col=i+1,
        )
        fig.update_xaxes(showticklabels=False, row=1, col=i+1)
        fig.update_yaxes(showticklabels=False, row=1, col=i+1)

    fig.update_layout(
        height=250,
        title="Forecast Evolution",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 2: Skill Scorecards =============
with tab2:
    st.subheader("WeatherBench-Style Skill Scorecards")

    st.markdown("""
    *Style: WeatherBench2 evaluation papers*
    """)

    # Generate scorecard data
    models = ["GraphCast", "FourCastNet", "Pangu", "GenCast", "HRES", "Persistence"]
    variables_short = ["Z500", "T850", "T2M", "U10", "V10", "Q700"]
    lead_times_hours = [24, 72, 120, 168, 240]

    # Create RMSE matrix (lower is better, shown as normalized skill)
    skill_matrix = np.random.rand(len(models), len(variables_short), len(lead_times_hours))

    # Make it more realistic
    for i, model in enumerate(models):
        if model == "Persistence":
            skill_matrix[i] *= 0.3
        elif model in ["GraphCast", "GenCast"]:
            skill_matrix[i] = 0.7 + skill_matrix[i] * 0.25

    # Display as heatmap for each lead time
    selected_lt = st.select_slider("Lead Time", lead_times_hours, value=120)
    lt_idx = lead_times_hours.index(selected_lt)

    skill_at_lt = skill_matrix[:, :, lt_idx]

    fig = go.Figure(data=go.Heatmap(
        z=skill_at_lt,
        x=variables_short,
        y=models,
        colorscale="RdYlGn",
        text=np.round(skill_at_lt, 2),
        texttemplate="%{text:.2f}",
        textfont={"size": 12},
        colorbar=dict(title="Skill Score"),
    ))

    fig.update_layout(
        title=f"Forecast Skill Scorecard (T+{selected_lt}h)",
        xaxis_title="Variable",
        yaxis_title="Model",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Skill vs lead time line plot
    st.markdown("### Skill Degradation by Lead Time")

    col1, col2 = st.columns(2)

    with col1:
        # Z500 skill over time
        fig = go.Figure()
        for i, model in enumerate(models[:4]):
            skill = skill_matrix[i, 0, :]
            fig.add_trace(go.Scatter(
                x=lead_times_hours,
                y=skill,
                mode='lines+markers',
                name=model,
                line=dict(width=2),
            ))

        fig.update_layout(
            title="Z500 Skill vs Lead Time",
            xaxis_title="Lead Time (hours)",
            yaxis_title="Skill Score",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Ranking table
        st.markdown("**Model Rankings (average skill)**")
        avg_skills = skill_matrix[:, :, :].mean(axis=(1, 2))
        ranking_df = pd.DataFrame({
            "Rank": range(1, len(models) + 1),
            "Model": [models[i] for i in np.argsort(avg_skills)[::-1]],
            "Avg Skill": sorted(avg_skills, reverse=True),
        })
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

# ============= TAB 3: Error Analysis =============
with tab3:
    st.subheader("Forecast Error Analysis")

    st.markdown("""
    *Style: Error maps and distributions from model papers*
    """)

    col1, col2 = st.columns(2)

    with col1:
        # Bias map
        st.markdown("### Systematic Bias Map")

        bias_field = np.random.randn(64, 128) * 20
        # Add some structure (land-sea contrast, etc.)
        lat_effect = np.sin(np.linspace(-np.pi/2, np.pi/2, 64))[:, np.newaxis]
        bias_field += lat_effect * 30

        fig = go.Figure(go.Heatmap(
            z=bias_field,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Bias (m²/s²)"),
        ))

        fig.update_layout(
            title="Z500 Forecast Bias (T+120h)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Error distribution
        st.markdown("### Error Distribution")

        errors = np.random.randn(10000) * 150 + 20  # Slight positive bias

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=50,
            name="Forecast Error",
            marker_color='steelblue',
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
        fig.add_vline(x=np.mean(errors), line_dash="dash", line_color="green",
                      annotation_text=f"Mean: {np.mean(errors):.1f}")

        fig.update_layout(
            title="Error Distribution",
            xaxis_title="Error (m²/s²)",
            yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # RMSE map
    st.markdown("### Spatial RMSE Distribution")

    rmse_field = np.abs(np.random.randn(64, 128)) * 100 + 50
    # Higher errors in tropics
    lat_grid = np.linspace(-90, 90, 64)[:, np.newaxis]
    rmse_field += 50 * np.exp(-((lat_grid) ** 2) / 500)

    fig = go.Figure(go.Heatmap(
        z=rmse_field,
        colorscale="hot",
        colorbar=dict(title="RMSE"),
    ))

    fig.update_layout(
        title="Spatial RMSE Distribution (Z500, T+120h)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 4: Ensemble Spread =============
with tab4:
    st.subheader("Ensemble & Uncertainty Visualization")

    st.markdown("""
    *Style: GenCast, ensemble forecasting papers*
    """)

    # Generate ensemble members
    n_members = 50
    lead_times = np.arange(0, 241, 6)
    n_times = len(lead_times)

    # Ensemble trajectories
    base = 54000 + np.random.randn(n_times).cumsum() * 20
    ensemble = np.array([base + np.random.randn(n_times).cumsum() * (10 + lt/20)
                         for lt in range(n_members)])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Ensemble Spaghetti Plot")

        fig = go.Figure()

        # Individual members (light)
        for i in range(min(20, n_members)):
            fig.add_trace(go.Scatter(
                x=lead_times,
                y=ensemble[i],
                mode='lines',
                line=dict(color='lightblue', width=0.5),
                showlegend=False,
                hoverinfo='skip',
            ))

        # Ensemble mean
        fig.add_trace(go.Scatter(
            x=lead_times,
            y=ensemble.mean(axis=0),
            mode='lines',
            line=dict(color='blue', width=3),
            name='Ensemble Mean',
        ))

        # Observation (truth)
        truth = base + np.random.randn(n_times) * 30
        fig.add_trace(go.Scatter(
            x=lead_times,
            y=truth,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Verifying Analysis',
        ))

        fig.update_layout(
            title="Z500 Ensemble Forecast",
            xaxis_title="Lead Time (hours)",
            yaxis_title="Z500 (m²/s²)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Ensemble Spread vs Error")

        spread = ensemble.std(axis=0)
        error = np.abs(ensemble.mean(axis=0) - truth)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lead_times,
            y=spread,
            mode='lines+markers',
            name='Ensemble Spread',
            line=dict(color='blue'),
        ))
        fig.add_trace(go.Scatter(
            x=lead_times,
            y=error,
            mode='lines+markers',
            name='Forecast Error',
            line=dict(color='red'),
        ))

        fig.update_layout(
            title="Spread-Skill Relationship",
            xaxis_title="Lead Time (hours)",
            yaxis_title="Value",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Probability maps
    st.markdown("### Probability of Exceedance")

    prob_field = np.random.rand(64, 128)
    # Add some structure
    lon_grid, lat_grid = np.meshgrid(np.linspace(0, 360, 128), np.linspace(-90, 90, 64))
    prob_field += 0.3 * np.sin(3 * np.deg2rad(lon_grid)) * np.cos(np.deg2rad(lat_grid))
    prob_field = np.clip(prob_field, 0, 1)

    fig = go.Figure(go.Heatmap(
        z=prob_field,
        colorscale="YlOrRd",
        zmin=0, zmax=1,
        colorbar=dict(title="Probability"),
    ))

    fig.update_layout(
        title="Probability of T2M > 30°C (Day 5)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 5: Spectral Analysis =============
with tab5:
    st.subheader("Spectral Analysis")

    st.markdown("""
    *Style: FourCastNet, physics-based analysis papers*
    """)

    # Generate power spectrum
    wavenumbers = np.arange(1, 100)

    # k^-3 for enstrophy cascade, k^-5/3 for energy cascade
    true_spectrum = 1e8 * wavenumbers.astype(float) ** (-3)
    predicted_spectrum = true_spectrum * (1 + 0.1 * np.random.randn(len(wavenumbers)))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Energy Spectrum")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wavenumbers,
            y=true_spectrum,
            mode='lines',
            name='ERA5 (Truth)',
            line=dict(color='blue', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=wavenumbers,
            y=predicted_spectrum,
            mode='lines',
            name='Model Prediction',
            line=dict(color='red', width=2),
        ))

        # Reference slopes
        k_ref = np.array([10, 50])
        fig.add_trace(go.Scatter(
            x=k_ref,
            y=1e9 * k_ref ** (-3),
            mode='lines',
            name='k⁻³ (enstrophy)',
            line=dict(dash='dash', color='gray'),
        ))

        fig.update_layout(
            title="Kinetic Energy Spectrum",
            xaxis_title="Wavenumber",
            yaxis_title="Power",
            xaxis_type="log",
            yaxis_type="log",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Spectrum Ratio (Model/Truth)")

        ratio = predicted_spectrum / true_spectrum

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wavenumbers,
            y=ratio,
            mode='lines+markers',
            line=dict(color='purple', width=2),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title="Spectral Ratio",
            xaxis_title="Wavenumber",
            yaxis_title="Ratio",
            xaxis_type="log",
            yaxis_range=[0.5, 1.5],
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Effective resolution
    st.markdown("### Effective Resolution Analysis")

    # Find where spectrum diverges (simplified)
    effective_wn = np.where(np.abs(ratio - 1) > 0.2)[0]
    if len(effective_wn) > 0:
        eff_res = effective_wn[0]
    else:
        eff_res = 50

    st.info(f"""
    **Effective Resolution Analysis**

    - Nominal resolution: 0.25° (~28 km)
    - Effective resolution: ~{360/eff_res:.0f}° (~{40000/eff_res:.0f} km)
    - Spectrum maintains accuracy up to wavenumber {eff_res}

    *The model preserves spectral characteristics for scales larger than the effective resolution.*
    """)

# Export options
st.markdown("---")
st.subheader("📥 Export Visualizations")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 Export as PDF"):
        st.info("PDF export would be generated here")

with col2:
    if st.button("🖼️ Export as PNG"):
        st.info("PNG export would be generated here")

with col3:
    if st.button("📊 Export Data"):
        st.info("Data export would be generated here")

# Footer
st.markdown("---")
st.caption("""
**Visualization styles inspired by:** GraphCast (Lam et al., 2023),
FourCastNet (Pathak et al., 2022), GenCast (Price et al., 2023),
WeatherBench2 (Rasp et al., 2024)
""")
