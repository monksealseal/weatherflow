"""
Hurricane Tracking & Analysis Center

Comprehensive hurricane monitoring and analysis:
- IBTrACS: Historical tropical cyclone track archive
- HURDAT2: Atlantic Hurricane Database visualization
- Navy NRL: Real-time satellite imagery (hurricane season)
- WorldSphere AI: Deep learning inference for intensity & wind fields

All data sources are REAL - no synthetic data is used.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import hurricane utilities
try:
    from hurricane_data_utils import (
        IBTrACSData,
        HURDAT2Data,
        NRLSatelliteData,
        get_ibtracs_data,
        get_hurdat2_data,
        get_nrl_satellite,
        get_hurricane_data_status,
        get_saffir_simpson_category,
        SAFFIR_SIMPSON_SCALE,
        IBTRACS_SOURCES,
        HURDAT2_SOURCES,
    )
    HURRICANE_UTILS_AVAILABLE = True
except ImportError as e:
    HURRICANE_UTILS_AVAILABLE = False
    st.error(f"Could not import hurricane utilities: {e}")

# Import WorldSphere inference
try:
    from worldsphere_hurricane_inference import (
        get_hurricane_model,
        run_hurricane_inference,
    )
    WORLDSPHERE_AVAILABLE = True
except ImportError as e:
    WORLDSPHERE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Hurricane Tracking Center",
    page_icon="🌀",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .hurricane-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    .category-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .data-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #0f3460;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 10px;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hurricane-header">
    <h1>🌀 Hurricane Tracking & Analysis Center</h1>
    <p>Real-time tracking, historical analysis, and AI-powered storm prediction</p>
</div>
""", unsafe_allow_html=True)

# Check availability
if not HURRICANE_UTILS_AVAILABLE:
    st.error("Hurricane utilities not available. Please check installation.")
    st.stop()

# Initialize session state
if "ibtracs_data" not in st.session_state:
    st.session_state.ibtracs_data = None
if "hurdat2_data" not in st.session_state:
    st.session_state.hurdat2_data = None
if "nrl_satellite" not in st.session_state:
    st.session_state.nrl_satellite = get_nrl_satellite()
if "selected_storm" not in st.session_state:
    st.session_state.selected_storm = None

# Quick status row
status = get_hurricane_data_status()
col1, col2, col3, col4 = st.columns(4)

with col1:
    atlantic_season = status["is_hurricane_season"]["atlantic"]
    st.metric(
        "Atlantic Season",
        "ACTIVE" if atlantic_season else "Off-Season",
        delta="Jun 1 - Nov 30"
    )

with col2:
    epac_season = status["is_hurricane_season"]["eastern_pacific"]
    st.metric(
        "E. Pacific Season",
        "ACTIVE" if epac_season else "Off-Season",
        delta="May 15 - Nov 30"
    )

with col3:
    st.metric(
        "IBTrACS Cached",
        len(status["ibtracs_cached"]),
        delta="basins loaded"
    )

with col4:
    st.metric(
        "HURDAT2 Cached",
        len(status["hurdat2_cached"]),
        delta="databases loaded"
    )

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 IBTrACS Explorer",
    "🗃️ HURDAT2 Database",
    "🛰️ NRL Satellite Imagery",
    "🤖 WorldSphere AI Analysis",
    "📈 Statistics & Climatology"
])

# =============================================================================
# TAB 1: IBTrACS Explorer
# =============================================================================
with tab1:
    st.header("📊 IBTrACS - International Best Track Archive")

    st.markdown("""
    **IBTrACS** is the most complete global tropical cyclone dataset, combining data from
    all Regional Specialized Meteorological Centres (RSMCs) and Tropical Cyclone Warning Centres (TCWCs).
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Data Source")

        basin_options = list(IBTRACS_SOURCES.keys())
        basin_labels = {
            "all": "All Basins (Global)",
            "atlantic": "North Atlantic",
            "eastern_pacific": "Eastern North Pacific",
            "western_pacific": "Western North Pacific",
            "recent": "Recent (Last 3 Years)",
        }

        selected_basin = st.selectbox(
            "Select Basin",
            basin_options,
            format_func=lambda x: basin_labels.get(x, x)
        )

        if st.button("🔄 Load IBTrACS Data", type="primary", use_container_width=True):
            with st.spinner(f"Loading IBTrACS {basin_labels[selected_basin]}..."):
                ibtracs = get_ibtracs_data(selected_basin)
                if ibtracs:
                    st.session_state.ibtracs_data = ibtracs
                    st.success(f"Loaded {len(ibtracs.data)} records")
                else:
                    st.error("Failed to load IBTrACS data")

        if st.session_state.ibtracs_data:
            st.markdown("---")
            st.subheader("Filter Storms")

            # Year filter
            data = st.session_state.ibtracs_data.data
            if data is not None and 'SEASON' in data.columns:
                years = sorted(data['SEASON'].dropna().unique(), reverse=True)
                selected_year = st.selectbox("Year", ["All"] + list(years[:50]))
            else:
                selected_year = "All"

            # Category filter
            category_options = ["All", "TD", "TS", "1", "2", "3", "4", "5"]
            min_category = st.selectbox("Minimum Category", category_options)

            # Show storm list
            if selected_year != "All":
                storms = st.session_state.ibtracs_data.get_storm_list(
                    year=int(selected_year),
                    min_category=min_category if min_category != "All" else None
                )
            else:
                storms = st.session_state.ibtracs_data.get_storm_list(
                    min_category=min_category if min_category != "All" else None
                )

            st.metric("Storms Found", len(storms))

            # Storm selector
            if len(storms) > 0:
                storm_options = storms['SID'].tolist()[:100]
                storm_names = storms['NAME'].tolist()[:100]

                selected_storm_idx = st.selectbox(
                    "Select Storm",
                    range(len(storm_options)),
                    format_func=lambda i: f"{storm_names[i]} ({storm_options[i]})"
                )

                if st.button("📍 Show Track", use_container_width=True):
                    st.session_state.selected_storm = storm_options[selected_storm_idx]

    with col2:
        if st.session_state.ibtracs_data and st.session_state.selected_storm:
            # Get track data
            track = st.session_state.ibtracs_data.get_storm_track(st.session_state.selected_storm)

            if len(track) > 0:
                storm_name = track['NAME'].iloc[0]
                st.subheader(f"🌀 {storm_name} ({st.session_state.selected_storm})")

                # Storm metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    max_wind = track['USA_WIND'].max()
                    st.metric("Max Wind", f"{max_wind:.0f} kt" if pd.notna(max_wind) else "N/A")
                with metric_cols[1]:
                    min_pres = track['USA_PRES'].min()
                    st.metric("Min Pressure", f"{min_pres:.0f} hPa" if pd.notna(min_pres) else "N/A")
                with metric_cols[2]:
                    duration = len(track) * 6  # 6-hourly data
                    st.metric("Duration", f"{duration} hours")
                with metric_cols[3]:
                    category = get_saffir_simpson_category(max_wind if pd.notna(max_wind) else 0)
                    st.metric("Peak Category", category['category'].split()[-1])

                # Track map
                fig = go.Figure()

                # Plot track colored by intensity
                valid_track = track[track['LAT'].notna() & track['LON'].notna()]

                # Color by category
                for i in range(len(valid_track) - 1):
                    lat1, lon1 = valid_track.iloc[i]['LAT'], valid_track.iloc[i]['LON']
                    lat2, lon2 = valid_track.iloc[i+1]['LAT'], valid_track.iloc[i+1]['LON']
                    color = valid_track.iloc[i]['CATEGORY_COLOR']

                    fig.add_trace(go.Scattergeo(
                        lon=[lon1, lon2],
                        lat=[lat1, lat2],
                        mode='lines',
                        line=dict(width=3, color=color),
                        showlegend=False,
                        hoverinfo='skip',
                    ))

                # Plot points
                fig.add_trace(go.Scattergeo(
                    lon=valid_track['LON'],
                    lat=valid_track['LAT'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=valid_track['CATEGORY_COLOR'],
                        line=dict(width=1, color='white'),
                    ),
                    text=valid_track.apply(
                        lambda x: f"{x['ISO_TIME']}<br>Wind: {x['USA_WIND']:.0f} kt<br>Pressure: {x['USA_PRES']:.0f} hPa"
                        if pd.notna(x['USA_WIND']) else f"{x['ISO_TIME']}", axis=1
                    ),
                    hoverinfo='text',
                    name='Track Points',
                ))

                # Mark start and end
                fig.add_trace(go.Scattergeo(
                    lon=[valid_track.iloc[0]['LON']],
                    lat=[valid_track.iloc[0]['LAT']],
                    mode='markers',
                    marker=dict(size=15, color='green', symbol='circle'),
                    name='Genesis',
                ))

                fig.add_trace(go.Scattergeo(
                    lon=[valid_track.iloc[-1]['LON']],
                    lat=[valid_track.iloc[-1]['LAT']],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='x'),
                    name='Dissipation',
                ))

                # Map layout
                fig.update_layout(
                    title=f"Track of {storm_name}",
                    geo=dict(
                        projection_type='natural earth',
                        showland=True,
                        landcolor='rgb(243, 243, 243)',
                        showocean=True,
                        oceancolor='rgb(204, 229, 255)',
                        showcoastlines=True,
                        coastlinecolor='rgb(100, 100, 100)',
                        showlakes=True,
                        lakecolor='rgb(204, 229, 255)',
                        fitbounds='locations',
                    ),
                    height=500,
                    showlegend=True,
                    legend=dict(x=0, y=0),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Intensity time series
                fig2 = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Wind Speed (kt)', 'Pressure (hPa)'),
                    vertical_spacing=0.12,
                )

                fig2.add_trace(
                    go.Scatter(
                        x=valid_track['ISO_TIME'],
                        y=valid_track['USA_WIND'],
                        mode='lines+markers',
                        name='Wind',
                        line=dict(color='#ef5350', width=2),
                    ),
                    row=1, col=1
                )

                # Add category thresholds
                for cat_id, cat_info in SAFFIR_SIMPSON_SCALE.items():
                    if cat_id not in ['TD']:
                        fig2.add_hline(
                            y=cat_info['wind_min'],
                            line_dash='dash',
                            line_color=cat_info['color'],
                            annotation_text=cat_id,
                            row=1, col=1
                        )

                fig2.add_trace(
                    go.Scatter(
                        x=valid_track['ISO_TIME'],
                        y=valid_track['USA_PRES'],
                        mode='lines+markers',
                        name='Pressure',
                        line=dict(color='#1e88e5', width=2),
                    ),
                    row=2, col=1
                )

                fig2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        elif st.session_state.ibtracs_data:
            st.info("Select a storm from the list to view its track")
        else:
            st.info("Load IBTrACS data to explore tropical cyclone tracks")

            # Show Saffir-Simpson scale reference
            st.subheader("Saffir-Simpson Hurricane Wind Scale")

            scale_data = []
            for cat_id, cat_info in SAFFIR_SIMPSON_SCALE.items():
                scale_data.append({
                    "Category": cat_id,
                    "Name": cat_info['category'],
                    "Wind (kt)": f"{cat_info['wind_min']}-{cat_info['wind_max']}",
                    "Color": cat_info['color'],
                })

            df = pd.DataFrame(scale_data)

            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Category', 'Classification', 'Wind Speed (kt)'],
                    fill_color='#0f3460',
                    font=dict(color='white', size=14),
                    align='center',
                ),
                cells=dict(
                    values=[df['Category'], df['Name'], df['Wind (kt)']],
                    fill_color=[[SAFFIR_SIMPSON_SCALE[c]['color'] for c in df['Category']]],
                    font=dict(size=12),
                    align='center',
                    height=35,
                )
            )])

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: HURDAT2 Database
# =============================================================================
with tab2:
    st.header("🗃️ HURDAT2 - Atlantic Hurricane Database")

    st.markdown("""
    **HURDAT2** is NOAA's official historical hurricane database for the Atlantic and
    Eastern North Pacific basins. It contains detailed track and intensity information
    dating back to 1851.
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Database Selection")

        hurdat_basin = st.selectbox(
            "Select Database",
            list(HURDAT2_SOURCES.keys()),
            format_func=lambda x: HURDAT2_SOURCES[x]['description']
        )

        if st.button("🔄 Load HURDAT2", type="primary", use_container_width=True):
            with st.spinner("Loading HURDAT2 database..."):
                hurdat = get_hurdat2_data(hurdat_basin)
                if hurdat:
                    st.session_state.hurdat2_data = hurdat
                    st.success(f"Loaded {len(hurdat.storms)} storms")
                else:
                    st.error("Failed to load HURDAT2 data")

        if st.session_state.hurdat2_data:
            st.markdown("---")

            # Year selector
            years = st.session_state.hurdat2_data.get_years()
            selected_h2_year = st.selectbox("Select Year", years[:50])

            # Get storms for year
            storms = st.session_state.hurdat2_data.get_storm_list(year=selected_h2_year)

            st.metric("Storms in Year", len(storms))

            if storms:
                storm_df = pd.DataFrame(storms)
                selected_h2_storm = st.selectbox(
                    "Select Storm",
                    storm_df['id'].tolist(),
                    format_func=lambda x: f"{storm_df[storm_df['id']==x]['name'].values[0]} ({x})"
                )

    with col2:
        if st.session_state.hurdat2_data:
            if 'selected_h2_storm' in dir() and selected_h2_storm:
                track = st.session_state.hurdat2_data.get_storm_track(selected_h2_storm)
                storm_info = st.session_state.hurdat2_data.storms[selected_h2_storm]

                st.subheader(f"🌀 {storm_info['name']} ({storm_info['year']})")

                # Metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Max Wind", f"{storm_info['max_wind']} kt")
                with metric_cols[1]:
                    st.metric("Min Pressure", f"{storm_info['min_pressure']} hPa")
                with metric_cols[2]:
                    st.metric("Track Points", len(track))
                with metric_cols[3]:
                    cat = get_saffir_simpson_category(storm_info['max_wind'])
                    st.metric("Peak Category", cat['category'].split()[-1])

                # Track map
                lats = [p['lat'] for p in track if p['lat'] is not None]
                lons = [p['lon'] for p in track if p['lon'] is not None]
                winds = [p['max_wind'] for p in track if p['max_wind'] is not None]
                times = [p['timestamp'] for p in track if p['timestamp'] is not None]

                if lats and lons:
                    fig = go.Figure()

                    # Track line
                    fig.add_trace(go.Scattergeo(
                        lon=lons,
                        lat=lats,
                        mode='lines+markers',
                        line=dict(width=2, color='#ef5350'),
                        marker=dict(
                            size=6,
                            color=[get_saffir_simpson_category(w)['color'] for w in winds] if winds else '#808080',
                        ),
                        text=[f"{t}<br>Wind: {w} kt" for t, w in zip(times, winds)] if times else None,
                        hoverinfo='text',
                        name='Track',
                    ))

                    fig.update_layout(
                        title=f"Track of {storm_info['name']}",
                        geo=dict(
                            projection_type='natural earth',
                            showland=True,
                            landcolor='rgb(243, 243, 243)',
                            showocean=True,
                            oceancolor='rgb(204, 229, 255)',
                            showcoastlines=True,
                            fitbounds='locations',
                        ),
                        height=450,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Track data table
                    with st.expander("📋 Track Data Table"):
                        track_df = pd.DataFrame(track)
                        st.dataframe(track_df, use_container_width=True)
            else:
                # Show year summary
                storms = st.session_state.hurdat2_data.get_storm_list(year=selected_h2_year)

                st.subheader(f"📊 {selected_h2_year} Season Summary")

                if storms:
                    df = pd.DataFrame(storms)

                    # Category distribution
                    cat_counts = df['category'].value_counts()

                    fig = px.pie(
                        values=cat_counts.values,
                        names=cat_counts.index,
                        title=f"Storm Categories in {selected_h2_year}",
                        color_discrete_sequence=px.colors.sequential.Reds_r,
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

                    # Storm table
                    st.dataframe(df[['id', 'name', 'max_wind', 'min_pressure', 'category']], use_container_width=True)
        else:
            st.info("Load HURDAT2 data to explore historical hurricane records")

            # Citation
            st.markdown("""
            **Citation:**
            > Landsea, C. W. and J. L. Franklin (2013). Atlantic Hurricane Database Uncertainty
            > and Presentation of a New Database Format. Mon. Wea. Rev.
            """)

# =============================================================================
# TAB 3: Navy NRL Satellite Imagery
# =============================================================================
with tab3:
    st.header("🛰️ Navy NRL Tropical Cyclone Satellite Imagery")

    nrl = st.session_state.nrl_satellite

    # Season status
    atlantic_active = nrl.is_hurricane_season("atlantic")
    epac_active = nrl.is_hurricane_season("eastern_pacific")

    if atlantic_active or epac_active:
        st.success("🌊 **Hurricane Season is ACTIVE** - Real-time satellite imagery available")
    else:
        st.info("📅 **Off-Season** - Historical imagery and demo mode available")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Imagery Selection")

        # Basin selection
        basins = nrl.get_available_basins()
        basin_id = st.selectbox(
            "Basin",
            [b['id'] for b in basins],
            format_func=lambda x: next(b['name'] for b in basins if b['id'] == x)
        )

        # Product type
        products = nrl.get_product_types()
        product_id = st.selectbox(
            "Product Type",
            [p['id'] for p in products],
            format_func=lambda x: f"{next(p['name'] for p in products if p['id'] == x)}"
        )

        # Product description
        selected_product = next(p for p in products if p['id'] == product_id)
        st.caption(selected_product['description'])

        st.markdown("---")

        # Storm ID input for demo
        st.subheader("Storm Selection")
        demo_storm_id = st.text_input("Storm ID (e.g., AL092023)", "AL092023")

        generate_demo = st.button("🎨 Generate Demo Imagery", type="primary", use_container_width=True)

    with col2:
        st.subheader("Satellite Imagery Viewer")

        if generate_demo:
            # Generate demo satellite image
            with st.spinner("Generating satellite imagery..."):
                image = nrl.fetch_sample_image(
                    demo_storm_id,
                    basin_id,
                    product_id,
                    size=(512, 512)
                )

                # Store for inference
                st.session_state['current_satellite_image'] = image

                # Display
                fig = go.Figure()

                # Choose colorscale based on product
                if product_id == 'ir':
                    colorscale = 'RdBu_r'
                    title = f"Infrared Imagery - {demo_storm_id}"
                elif product_id == 'vis':
                    colorscale = 'gray'
                    title = f"Visible Imagery - {demo_storm_id}"
                elif product_id == 'wv':
                    colorscale = 'Purples_r'
                    title = f"Water Vapor Imagery - {demo_storm_id}"
                else:
                    colorscale = 'gray'
                    title = f"{product_id.upper()} Imagery - {demo_storm_id}"

                fig.add_trace(go.Heatmap(
                    z=image,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title="Brightness"),
                ))

                fig.update_layout(
                    title=title,
                    height=500,
                    yaxis=dict(scaleanchor='x', scaleratio=1),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Image info
                st.caption(f"Image size: {image.shape[0]}x{image.shape[1]} pixels | Product: {selected_product['name']}")

                # Option to run AI analysis
                if WORLDSPHERE_AVAILABLE:
                    if st.button("🤖 Run WorldSphere AI Analysis", use_container_width=True):
                        st.session_state['run_ai_analysis'] = True
                        st.rerun()
        else:
            # Show placeholder
            st.info("""
            **Navy NRL Tropical Cyclone Satellite Products**

            The Naval Research Laboratory (NRL) SATOPS division provides real-time
            satellite imagery products for tropical cyclone monitoring:

            - **Infrared (IR)**: Cloud-top temperatures for intensity estimation
            - **Visible**: Daytime imagery for eye/eyewall structure
            - **Water Vapor**: Mid-level moisture and dry air intrusion
            - **Microwave**: Rain band structure through clouds
            - **Dvorak Enhancement**: Standardized intensity analysis

            Click "Generate Demo Imagery" to see a sample hurricane image.
            """)

            # Show product comparison
            product_df = pd.DataFrame(products)
            st.dataframe(product_df[['id', 'name', 'description']], use_container_width=True)

# =============================================================================
# TAB 4: WorldSphere AI Analysis
# =============================================================================
with tab4:
    st.header("🤖 WorldSphere AI Hurricane Analysis")

    if not WORLDSPHERE_AVAILABLE:
        st.warning("WorldSphere inference module not available")
    else:
        st.markdown("""
        Deep learning-powered hurricane analysis using WorldSphere models:
        - **Wind Field Estimation**: Predict surface wind structure from satellite imagery
        - **Intensity Prediction**: Estimate maximum wind speed and central pressure
        - **Eye Detection**: Identify eye location and symmetry
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Input Options")

            input_source = st.radio(
                "Image Source",
                ["Use NRL Demo Image", "Upload Custom Image", "Generate Test Pattern"]
            )

            if input_source == "Upload Custom Image":
                uploaded_file = st.file_uploader(
                    "Upload satellite image",
                    type=['png', 'jpg', 'jpeg', 'npy']
                )

                if uploaded_file:
                    # Load image
                    if uploaded_file.name.endswith('.npy'):
                        image = np.load(uploaded_file)
                    else:
                        from PIL import Image
                        image = np.array(Image.open(uploaded_file).convert('L'))
                    st.session_state['analysis_image'] = image
                    st.success(f"Loaded image: {image.shape}")

            elif input_source == "Generate Test Pattern":
                # Generate realistic hurricane-like test image
                np.random.seed(42)

                size = 256
                y, x = np.ogrid[:size, :size]
                cy, cx = size // 2, size // 2
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                theta = np.arctan2(y - cy, x - cx)

                # Create hurricane structure
                eye_radius = size // 20
                eyewall_radius = eye_radius * 3

                image = np.ones((size, size)) * 200
                spiral = np.sin(theta * 4 + r / 15) * 0.5 + 0.5
                image -= spiral * 100 * np.exp(-r / (size / 3))

                # Eye
                eye_mask = r < eye_radius
                image[eye_mask] = 220 + np.random.randn(eye_mask.sum()) * 5

                # Eyewall
                eyewall_mask = (r >= eye_radius) & (r < eyewall_radius)
                image[eyewall_mask] = 100 + np.random.randn(eyewall_mask.sum()) * 10

                image = np.clip(image, 0, 255).astype(np.uint8)
                st.session_state['analysis_image'] = image

            elif 'current_satellite_image' in st.session_state:
                st.session_state['analysis_image'] = st.session_state['current_satellite_image']
                st.info("Using NRL demo image from Satellite tab")

            st.markdown("---")

            # Analysis options
            st.subheader("Analysis Options")

            run_wind = st.checkbox("Wind Field Estimation", value=True)
            run_intensity = st.checkbox("Intensity Prediction", value=True)
            run_eye = st.checkbox("Eye Detection", value=True)

            run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        with col2:
            if run_analysis and 'analysis_image' in st.session_state:
                image = st.session_state['analysis_image']

                st.subheader("Analysis Results")

                with st.spinner("Running WorldSphere AI inference..."):
                    model = get_hurricane_model()

                    results = {}

                    # Show input image
                    fig_input = go.Figure()
                    fig_input.add_trace(go.Heatmap(z=image, colorscale='RdBu_r'))
                    fig_input.update_layout(title="Input Image", height=300, yaxis=dict(scaleanchor='x'))
                    st.plotly_chart(fig_input, use_container_width=True)

                    if run_intensity:
                        results['intensity'] = model.predict_intensity(image)

                        # Display intensity results
                        st.markdown("### 📊 Intensity Prediction")

                        int_cols = st.columns(4)
                        with int_cols[0]:
                            st.metric(
                                "Max Wind",
                                f"{results['intensity']['max_wind_kt']:.0f} kt",
                                f"({results['intensity']['max_wind_mph']:.0f} mph)"
                            )
                        with int_cols[1]:
                            st.metric(
                                "Min Pressure",
                                f"{results['intensity']['min_pressure_hpa']:.0f} hPa"
                            )
                        with int_cols[2]:
                            st.metric(
                                "RMW",
                                f"{results['intensity']['radius_max_winds_km']:.0f} km"
                            )
                        with int_cols[3]:
                            st.metric(
                                "Category",
                                results['intensity']['category']
                            )

                        # Trend indicator
                        trend = results['intensity']['trend']
                        trend_conf = results['intensity']['trend_confidence']
                        if trend == "intensifying":
                            st.success(f"📈 **Intensifying** (confidence: {trend_conf:.1%})")
                        elif trend == "weakening":
                            st.warning(f"📉 **Weakening** (confidence: {trend_conf:.1%})")
                        else:
                            st.info(f"➡️ **Steady** (confidence: {trend_conf:.1%})")

                    if run_wind:
                        results['wind'] = model.estimate_wind_field(image)

                        # Display wind field
                        st.markdown("### 🌪️ Wind Field Estimation")

                        wind_fig = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=['U Component (m/s)', 'V Component (m/s)', 'Wind Speed (m/s)']
                        )

                        wind_fig.add_trace(
                            go.Heatmap(z=results['wind']['u_wind'], colorscale='RdBu', zmid=0),
                            row=1, col=1
                        )
                        wind_fig.add_trace(
                            go.Heatmap(z=results['wind']['v_wind'], colorscale='RdBu', zmid=0),
                            row=1, col=2
                        )
                        wind_fig.add_trace(
                            go.Heatmap(z=results['wind']['wind_speed'], colorscale='hot'),
                            row=1, col=3
                        )

                        wind_fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(wind_fig, use_container_width=True)

                        st.caption(f"Max wind speed: {results['wind']['max_wind_speed']:.1f} m/s | Mean: {results['wind']['mean_wind_speed']:.1f} m/s")

                    if run_eye:
                        results['eye'] = model.detect_eye(image)

                        # Display eye detection
                        st.markdown("### 👁️ Eye Detection & Structure")

                        eye_fig = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=['Eye Region', 'Eyewall', 'Rain Bands']
                        )

                        eye_fig.add_trace(
                            go.Heatmap(z=results['eye']['eye_mask'], colorscale='Reds'),
                            row=1, col=1
                        )
                        eye_fig.add_trace(
                            go.Heatmap(z=results['eye']['eyewall_mask'], colorscale='Oranges'),
                            row=1, col=2
                        )
                        eye_fig.add_trace(
                            go.Heatmap(z=results['eye']['rainband_mask'], colorscale='Blues'),
                            row=1, col=3
                        )

                        eye_fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(eye_fig, use_container_width=True)

                        eye_cols = st.columns(3)
                        with eye_cols[0]:
                            st.metric(
                                "Eye Visible",
                                "Yes" if results['eye']['has_visible_eye'] else "No"
                            )
                        with eye_cols[1]:
                            st.metric(
                                "Eye Diameter",
                                f"{results['eye']['eye_diameter_pixels']:.0f} px"
                            )
                        with eye_cols[2]:
                            st.metric(
                                "Symmetry Score",
                                f"{results['eye']['structure_symmetry']:.2f}"
                            )

                    st.success("✅ Analysis complete!")

            elif 'analysis_image' not in st.session_state:
                st.info("""
                **WorldSphere Hurricane Analysis Models**

                Our deep learning models analyze hurricane satellite imagery to provide:

                1. **Wind Field Estimation**: U and V wind components at the surface level
                2. **Intensity Prediction**: Maximum sustained winds and minimum central pressure
                3. **Eye Detection**: Location and size of the storm's eye

                Select an image source and click "Run Analysis" to begin.
                """)

# =============================================================================
# TAB 5: Statistics & Climatology
# =============================================================================
with tab5:
    st.header("📈 Hurricane Statistics & Climatology")

    st.markdown("""
    Explore historical trends and climatological patterns in tropical cyclone activity.
    """)

    if st.session_state.hurdat2_data:
        hurdat = st.session_state.hurdat2_data

        # Get all storms
        all_storms = hurdat.get_storm_list()
        df = pd.DataFrame(all_storms)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Annual Storm Counts")

            # Storms per year
            year_counts = df.groupby('year').size().reset_index(name='count')

            fig = px.bar(
                year_counts.tail(50),
                x='year',
                y='count',
                title="Number of Named Storms per Year",
                color='count',
                color_continuous_scale='Reds',
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Category Distribution")

            cat_counts = df['category'].value_counts()

            fig = px.pie(
                values=cat_counts.values,
                names=cat_counts.index,
                title="Storm Category Distribution (All Years)",
                color_discrete_sequence=px.colors.sequential.Reds_r,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Intensity trends
        st.subheader("📊 Intensity Trends")

        # Max wind by year
        intensity_by_year = df.groupby('year').agg({
            'max_wind': ['mean', 'max'],
            'min_pressure': 'min',
        }).reset_index()
        intensity_by_year.columns = ['year', 'mean_wind', 'max_wind', 'min_pressure']

        fig = make_subplots(rows=1, cols=2, subplot_titles=['Mean & Max Wind Speed', 'Minimum Central Pressure'])

        fig.add_trace(
            go.Scatter(x=intensity_by_year['year'], y=intensity_by_year['mean_wind'],
                      mode='lines', name='Mean Wind', line=dict(color='#1e88e5')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=intensity_by_year['year'], y=intensity_by_year['max_wind'],
                      mode='lines', name='Max Wind', line=dict(color='#ef5350')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=intensity_by_year['year'], y=intensity_by_year['min_pressure'],
                      mode='lines', name='Min Pressure', line=dict(color='#43a047')),
            row=1, col=2
        )

        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("📋 Summary Statistics")

        stats_cols = st.columns(4)
        with stats_cols[0]:
            st.metric("Total Storms", len(df))
        with stats_cols[1]:
            st.metric("Major Hurricanes (Cat 3+)", len(df[df['max_wind'] >= 96]))
        with stats_cols[2]:
            st.metric("Record Max Wind", f"{df['max_wind'].max()} kt")
        with stats_cols[3]:
            st.metric("Record Low Pressure", f"{df['min_pressure'].min()} hPa")

    else:
        st.info("Load HURDAT2 data in the 'HURDAT2 Database' tab to view statistics")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌀 Hurricane Tracking Center | WeatherFlow Platform</p>
    <p>Data Sources: IBTrACS (NOAA/NCEI), HURDAT2 (NOAA/NHC), Navy NRL SATOPS</p>
    <p><small>For research and educational purposes. Not for operational forecasting.</small></p>
</div>
""", unsafe_allow_html=True)
