"""
Tropic World Visualization

Interactive visualizations for Tropic World GCM simulations, featuring:
- 3D interactive globe visualizations
- Vertical cross sections sorted by area fraction (warmest to coldest SST)
- Animated time evolution
- Air Temperature, Relative Humidity, Streamfunction, and Diabatic Heating

Based on Section 2.4 "Tropic World: Convection on a Planetary Scale"
from "Heuristic Models of the General Circulation"
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Tropic World",
    page_icon="🌴",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .cross-section-title {
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌴 Tropic World Visualization")
st.markdown("""
Explore the fascinating dynamics of **Tropic World** - an idealized non-rotating planet with uniform solar radiation.
This page provides beautiful, interactive visualizations of vertical atmospheric structure organized by SST.
""")

# Import GCM components
try:
    from gcm.core.model import GCM
    from gcm.grid.spherical import SphericalGrid
    from gcm.grid.vertical import VerticalGrid
    from gcm.core.state import ModelState
    GCM_AVAILABLE = True
except ImportError as e:
    GCM_AVAILABLE = False
    gcm_error = str(e)


# =============================================================================
# Custom Color Scales
# =============================================================================

def create_blue_to_red_colorscale():
    """Blue to Red colorscale for Air Temperature and Diabatic Heating"""
    return [
        [0.0, 'rgb(0, 0, 139)'],      # Dark blue
        [0.1, 'rgb(0, 0, 255)'],      # Blue
        [0.2, 'rgb(65, 105, 225)'],   # Royal blue
        [0.3, 'rgb(100, 149, 237)'],  # Cornflower blue
        [0.4, 'rgb(173, 216, 230)'],  # Light blue
        [0.5, 'rgb(255, 255, 255)'],  # White
        [0.6, 'rgb(255, 200, 150)'],  # Light orange/peach
        [0.7, 'rgb(255, 150, 100)'],  # Light red
        [0.8, 'rgb(255, 100, 50)'],   # Orange-red
        [0.9, 'rgb(255, 50, 0)'],     # Red-orange
        [1.0, 'rgb(139, 0, 0)'],      # Dark red
    ]


def create_white_to_orange_colorscale():
    """White to Orange colorscale for Relative Humidity and Streamfunction"""
    return [
        [0.0, 'rgb(255, 255, 255)'],   # White
        [0.15, 'rgb(255, 250, 240)'],  # Floral white
        [0.3, 'rgb(255, 239, 213)'],   # Papaya whip
        [0.45, 'rgb(255, 218, 185)'],  # Peach puff
        [0.6, 'rgb(255, 200, 150)'],   # Light orange
        [0.75, 'rgb(255, 165, 79)'],   # Orange
        [0.85, 'rgb(255, 140, 0)'],    # Dark orange
        [0.95, 'rgb(255, 100, 0)'],    # Deep orange
        [1.0, 'rgb(200, 80, 0)'],      # Burnt orange
    ]


def create_diverging_streamfunction_colorscale():
    """Diverging colorscale for streamfunction (blue-white-orange)"""
    return [
        [0.0, 'rgb(0, 100, 150)'],     # Teal blue
        [0.25, 'rgb(100, 180, 220)'],  # Light blue
        [0.5, 'rgb(255, 255, 255)'],   # White
        [0.75, 'rgb(255, 180, 100)'],  # Light orange
        [1.0, 'rgb(200, 80, 0)'],      # Burnt orange
    ]


# =============================================================================
# Tropic World Simulation Functions
# =============================================================================

def run_tropic_world_simulation(nlon, nlat, nlev, dt, duration_days, output_interval_hours,
                                 base_sst, sst_perturbation, progress_callback=None):
    """
    Run a Tropic World simulation and return all state history

    This uses the actual GCM Tropic World implementation which features:
    - Non-rotating planet (no Coriolis force)
    - Uniform solar radiation over entire surface
    - Slab ocean with no horizontal heat transport
    - Bulk aerodynamic surface fluxes for SST-wind feedback
    """
    if not GCM_AVAILABLE:
        return generate_synthetic_tropic_world_data(
            nlon, nlat, nlev, duration_days, output_interval_hours, base_sst, sst_perturbation,
            progress_callback=progress_callback
        )

    # Create GCM in Tropic World mode
    # This configuration matches the implementation in gcm/core/model.py:
    # - rotation_rate=0 for non-rotating planet
    # - uniform_solar=True for uniform radiation
    # - Slab ocean with no horizontal heat transport
    model = GCM(
        nlon=nlon,
        nlat=nlat,
        nlev=nlev,
        dt=dt,
        integration_method='rk3',
        co2_ppmv=400.0,
        tropic_world=True,           # Enable Tropic World mode
        tropic_world_sst=base_sst,   # Base SST (K)
        sst_perturbation=sst_perturbation,  # Perturbation amplitude (K)
        mixed_layer_depth=50.0       # Slab ocean depth (m)
    )

    # Initialize with tropical profile
    model.initialize(profile='tropical')

    # Get grid information
    lons = np.rad2deg(model.grid.lon)
    lats = np.rad2deg(model.grid.lat)

    # Compute pressure levels from sigma coordinates
    sigma_levels = model.vgrid.sigma
    pressures = sigma_levels * 1013.25  # Reference pressure in hPa

    # Run simulation with state capture
    total_seconds = duration_days * 86400.0
    output_interval = output_interval_hours * 3600.0
    n_steps = int(total_seconds / model.dt)
    output_frequency = max(1, int(output_interval / model.dt))

    state_history = []

    for step in range(n_steps):
        # Integrate one time step using the model's tendency computation
        model.integrator.step(model.state, model.dt, model._compute_tendencies)

        if step % output_frequency == 0:
            # Capture state snapshot with all variables needed for visualization
            state_snapshot = capture_state_snapshot(model)
            state_history.append(state_snapshot)

            if progress_callback:
                # Pass state snapshot, grid info, and progress to callback for real-time visualization
                progress_callback(
                    (step + 1) / n_steps,
                    model.state.time / 86400.0,
                    state_snapshot,
                    lats,
                    pressures
                )

    return {
        'state_history': state_history,
        'diagnostics': model.diagnostics,
        'model': model,
        'lons': lons,
        'lats': lats,
        'pressures': pressures,
        'sigma_levels': sigma_levels
    }


def capture_state_snapshot(model):
    """
    Capture a snapshot of the model state for visualization

    Captures all variables needed for Tropic World cross-section visualization:
    - Temperature (T)
    - Relative Humidity (RH)
    - Winds (u, v, w) for streamfunction calculation
    - Diabatic Heating from physics tendencies
    - SST for sorting by area fraction
    """
    state = model.state

    # Copy primary state variables
    T = state.T.copy()
    q = state.q.copy()
    u = state.u.copy()
    v = state.v.copy()
    w = state.w.copy()
    p = state.p.copy()
    ps = state.ps.copy()

    # Get SST from ocean model
    sst = model.ocean.sst.copy()

    # Compute relative humidity
    # Using Clausius-Clapeyron for saturation vapor pressure
    T0, e0 = 273.15, 611.2
    Lv, Rv = 2.5e6, 461.5
    es = e0 * np.exp((Lv / Rv) * (1/T0 - 1/T))
    qsat = 0.622 * es / p
    qsat = np.maximum(qsat, 1e-10)  # Prevent division by zero
    rh = np.clip(100 * q / qsat, 0, 100)

    # Compute diabatic heating (sum of physics tendencies)
    # This includes radiation, convection, and cloud microphysics
    # These are the key heating/cooling terms that drive Tropic World circulation
    dT_radiation = state.physics_tendencies['radiation']['T'].copy()
    dT_convection = state.physics_tendencies['convection']['T'].copy()
    dT_cloud = state.physics_tendencies['cloud_micro']['T'].copy()
    dT_boundary_layer = state.physics_tendencies['boundary_layer']['T'].copy()

    # Total diabatic heating in K/day
    diabatic_heating = (dT_radiation + dT_convection + dT_cloud + dT_boundary_layer) * 86400

    # Compute vertical velocity in pressure coordinates (omega = dp/dt)
    # Approximate from sigma velocity
    omega = w.copy()  # Already in Pa/s if available

    return {
        'time': state.time / 86400.0,  # days
        'T': T,
        'q': q,
        'u': u,
        'v': v,
        'w': w,
        'omega': omega,
        'p': p,
        'ps': ps,
        'sst': sst,
        'rh': rh,
        'diabatic_heating': diabatic_heating,
        'tsurf': state.tsurf.copy(),
        # Store individual heating components for detailed analysis
        'dT_radiation': dT_radiation * 86400,
        'dT_convection': dT_convection * 86400,
        'dT_cloud': dT_cloud * 86400,
        'dT_boundary_layer': dT_boundary_layer * 86400
    }


def generate_synthetic_tropic_world_data(nlon, nlat, nlev, duration_days, output_interval_hours,
                                          base_sst, sst_perturbation, progress_callback=None):
    """
    Generate synthetic Tropic World data for demonstration when GCM is not available
    """
    np.random.seed(42)

    # Grid setup
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lats = np.linspace(-90, 90, nlat)
    sigma_levels = np.linspace(0.02, 1.0, nlev)
    pressures = sigma_levels * 1013.25  # hPa

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    n_timesteps = int(duration_days * 24 / output_interval_hours)
    state_history = []

    # Initial SST with warm pool
    sst_base = base_sst * np.ones((nlat, nlon))
    # Create a warm pool region
    warm_center_lon, warm_center_lat = 180, 0
    dist_from_warm = np.sqrt((lon_grid - warm_center_lon)**2 + (lat_grid - warm_center_lat)**2)
    sst_base += 3.0 * np.exp(-dist_from_warm**2 / 3000)
    sst_base += sst_perturbation * np.random.randn(nlat, nlon)

    for t in range(n_timesteps):
        time_days = t * output_interval_hours / 24

        # Evolve SST (oscillation)
        phase = 2 * np.pi * time_days / 365  # Annual-ish cycle
        sst = sst_base + 2 * np.sin(phase) * np.cos(np.radians(lat_grid))
        sst += 0.3 * np.random.randn(nlat, nlon)

        # Create 3D fields
        T = np.zeros((nlev, nlat, nlon))
        rh = np.zeros((nlev, nlat, nlon))
        u = np.zeros((nlev, nlat, nlon))
        v = np.zeros((nlev, nlat, nlon))
        w = np.zeros((nlev, nlat, nlon))
        diabatic_heating = np.zeros((nlev, nlat, nlon))
        p = np.zeros((nlev, nlat, nlon))

        for k in range(nlev):
            # Temperature (lapse rate)
            height_km = 15 * (1 - sigma_levels[k])
            T[k] = sst - 6.5 * height_km
            T[k] = np.maximum(T[k], 200)  # Tropopause limit

            # Higher RH over warm regions, decreasing with altitude
            sst_anom = sst - np.mean(sst)
            rh[k] = 60 + 30 * (sst_anom / np.max(np.abs(sst_anom))) * np.exp(-height_km / 5)
            rh[k] = np.clip(rh[k], 5, 100)

            # Convergent flow toward warm regions at surface, divergent aloft
            grad_sst_lon = np.gradient(sst, axis=1)
            grad_sst_lat = np.gradient(sst, axis=0)

            vertical_structure = np.sin(np.pi * (1 - sigma_levels[k]))  # Max in mid-levels
            surface_factor = sigma_levels[k]**2  # Strong near surface

            u[k] = -5 * grad_sst_lon * surface_factor
            v[k] = -5 * grad_sst_lat * surface_factor

            # Vertical velocity (rising over warm, sinking over cold)
            w[k] = -0.1 * sst_anom * vertical_structure

            # Diabatic heating (convective heating over warm regions)
            diabatic_heating[k] = 5 * (sst_anom / np.max(np.abs(sst_anom))) * np.exp(-(height_km - 8)**2 / 20)

            # Pressure
            p[k] = pressures[k] * np.ones((nlat, nlon))

        state_snapshot = {
            'time': time_days,
            'T': T,
            'rh': rh,
            'u': u,
            'v': v,
            'w': w,
            'p': p,
            'ps': 1013.25 * np.ones((nlat, nlon)),
            'sst': sst,
            'diabatic_heating': diabatic_heating,
            'tsurf': sst
        }
        state_history.append(state_snapshot)

        # Call progress callback with state for real-time visualization
        if progress_callback:
            progress_callback(
                (t + 1) / n_timesteps,
                time_days,
                state_snapshot,
                lats,
                pressures
            )

    # Generate diagnostics
    diagnostics = {
        'time': [s['time'] for s in state_history],
        'sst_contrast': [np.max(s['sst']) - np.min(s['sst']) for s in state_history],
        'global_mean_sst': [np.mean(s['sst']) for s in state_history],
        'sst_warm_fraction': [np.sum(s['sst'] > np.mean(s['sst'])) / s['sst'].size for s in state_history],
        'global_mean_T': [np.mean(s['T'][nlev//2]) for s in state_history],
        'kinetic_energy': [0.5 * np.mean(s['u']**2 + s['v']**2) for s in state_history]
    }

    return {
        'state_history': state_history,
        'diagnostics': diagnostics,
        'lons': lons,
        'lats': lats,
        'pressures': pressures,
        'sigma_levels': sigma_levels
    }


# =============================================================================
# Vertical Cross Section Functions
# =============================================================================

def compute_area_fraction_sorted_cross_section(field_3d, sst_2d, pressure_levels, lats):
    """
    Compute vertical cross section sorted by area fraction from warmest to coldest SST

    Parameters
    ----------
    field_3d : ndarray
        3D field with shape (nlev, nlat, nlon)
    sst_2d : ndarray
        2D SST field with shape (nlat, nlon)
    pressure_levels : ndarray
        Pressure levels (hPa)
    lats : ndarray
        Latitude values for area weighting

    Returns
    -------
    cross_section : ndarray
        2D cross section (nlev, n_bins)
    area_fractions : ndarray
        Area fraction values (0 = warmest, 1 = coldest)
    """
    nlev = field_3d.shape[0]
    nlat, nlon = sst_2d.shape
    n_bins = 50  # Number of area fraction bins

    # Flatten SST and compute area weights
    sst_flat = sst_2d.flatten()
    lat_weights = np.cos(np.radians(lats))
    area_weights = np.repeat(lat_weights[:, np.newaxis], nlon, axis=1).flatten()
    area_weights = area_weights / np.sum(area_weights)

    # Sort by SST (descending - warmest first)
    sort_idx = np.argsort(-sst_flat)
    sorted_weights = area_weights[sort_idx]
    cumulative_area = np.cumsum(sorted_weights)

    # Bin the data by cumulative area fraction
    area_fractions = np.linspace(0, 1, n_bins)
    cross_section = np.zeros((nlev, n_bins))

    for k in range(nlev):
        field_flat = field_3d[k].flatten()
        sorted_field = field_flat[sort_idx]

        for i, af in enumerate(area_fractions):
            # Find indices within this area fraction range
            if i == 0:
                mask = cumulative_area <= area_fractions[1]
            elif i == n_bins - 1:
                mask = cumulative_area > area_fractions[i-1]
            else:
                mask = (cumulative_area > area_fractions[i-1]) & (cumulative_area <= area_fractions[i+1] if i+1 < n_bins else True)

            if np.any(mask):
                weights_in_bin = sorted_weights[mask]
                cross_section[k, i] = np.average(sorted_field[mask], weights=weights_in_bin)
            else:
                cross_section[k, i] = np.nan

    return cross_section, area_fractions


def compute_streamfunction(v_3d, pressure_levels, lats):
    """
    Compute mass streamfunction from meridional velocity

    Streamfunction: psi = (2*pi*a*cos(lat)/g) * integral(v dp)

    Parameters
    ----------
    v_3d : ndarray
        3D meridional velocity field
    pressure_levels : ndarray
        Pressure levels (hPa)
    lats : ndarray
        Latitudes

    Returns
    -------
    psi : ndarray
        Mass streamfunction (kg/s)
    """
    a = 6.371e6  # Earth radius (m)
    g = 9.81     # Gravity (m/s^2)

    nlev = v_3d.shape[0]
    nlat, nlon = v_3d.shape[1], v_3d.shape[2]

    # Zonal mean of v
    v_zonal_mean = np.mean(v_3d, axis=2)

    # Create 3D streamfunction (same shape as v_3d for cross-section computation)
    psi_3d = np.zeros_like(v_3d)

    for k in range(nlev):
        for j in range(nlat):
            lat_rad = np.radians(lats[j])
            # Integrate from top
            if k == 0:
                dp = (pressure_levels[1] - pressure_levels[0]) * 100  # Convert to Pa
                psi_3d[k, j, :] = (2 * np.pi * a * np.cos(lat_rad) / g) * v_3d[k, j, :] * dp
            else:
                dp = (pressure_levels[k] - pressure_levels[k-1]) * 100
                psi_3d[k, j, :] = psi_3d[k-1, j, :] + (2 * np.pi * a * np.cos(lat_rad) / g) * v_3d[k, j, :] * dp

    return psi_3d / 1e9  # Scale to 10^9 kg/s


# =============================================================================
# Visualization Functions
# =============================================================================

def create_3d_globe(sst, lons, lats, title="SST", colorscale='RdBu_r'):
    """Create a beautiful 3D globe visualization"""

    # Convert to 3D spherical coordinates
    lon_rad = np.radians(lons)
    lat_rad = np.radians(lats)
    lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)

    # Spherical to Cartesian
    r = 1.0
    x = r * np.cos(lat_grid) * np.cos(lon_grid)
    y = r * np.cos(lat_grid) * np.sin(lon_grid)
    z = r * np.sin(lat_grid)

    # Create the surface
    fig = go.Figure(data=[
        go.Surface(
            x=x, y=y, z=z,
            surfacecolor=sst,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text='K', font=dict(size=14)),
                thickness=20,
                len=0.7
            ),
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5
            ),
            lightposition=dict(x=1, y=1, z=1)
        )
    ])

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            yaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            zaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500
    )

    return fig


def create_cross_section_plot(cross_section, area_fractions, pressure_levels,
                               title, colorscale, units, zmid=None, zmin=None, zmax=None):
    """Create a beautiful vertical cross section plot"""

    fig = go.Figure()

    # Handle z-scale
    if zmid is not None:
        fig.add_trace(go.Heatmap(
            z=cross_section,
            x=area_fractions * 100,  # Convert to percentage
            y=pressure_levels,
            colorscale=colorscale,
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(
                title=dict(text=units, font=dict(size=12)),
                thickness=15,
                len=0.9
            ),
            hoverongaps=False,
            hovertemplate='Area Fraction: %{x:.1f}%<br>Pressure: %{y:.0f} hPa<br>Value: %{z:.2f}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Heatmap(
            z=cross_section,
            x=area_fractions * 100,
            y=pressure_levels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(
                title=dict(text=units, font=dict(size=12)),
                thickness=15,
                len=0.9
            ),
            hoverongaps=False,
            hovertemplate='Area Fraction: %{x:.1f}%<br>Pressure: %{y:.0f} hPa<br>Value: %{z:.2f}<extra></extra>'
        ))

    # Add contour lines
    fig.add_trace(go.Contour(
        z=cross_section,
        x=area_fractions * 100,
        y=pressure_levels,
        showscale=False,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='black')
        ),
        line=dict(color='black', width=0.5),
        opacity=0.5
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, weight='bold')),
        xaxis=dict(
            title='Area Fraction (%) - Warmest to Coldest',
            ticksuffix='%',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title='Pressure (hPa)',
            autorange='reversed',  # Pressure decreases upward
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        height=450,
        margin=dict(l=60, r=20, t=60, b=60)
    )

    # Add annotations for warmest/coldest
    fig.add_annotation(x=0, y=1000, text="Warmest", showarrow=False,
                      font=dict(size=10, color='red'), xanchor='left', yanchor='top')
    fig.add_annotation(x=100, y=1000, text="Coldest", showarrow=False,
                      font=dict(size=10, color='blue'), xanchor='right', yanchor='top')

    return fig


def create_animated_cross_sections(state_history, lats, pressures, variable='T'):
    """Create animated cross section visualization"""

    n_frames = len(state_history)
    frames = []

    # Compute all cross sections
    all_cross_sections = []
    for state in state_history:
        if variable == 'T':
            field = state['T']
        elif variable == 'rh':
            field = state['rh']
        elif variable == 'diabatic_heating':
            field = state['diabatic_heating']
        elif variable == 'streamfunction':
            psi = compute_streamfunction(state['v'], pressures, lats)
            field = psi

        cs, area_frac = compute_area_fraction_sorted_cross_section(field, state['sst'], pressures, lats)
        all_cross_sections.append(cs)

    # Get global min/max for consistent colorscale
    all_data = np.array(all_cross_sections)
    vmin, vmax = np.nanpercentile(all_data, [2, 98])

    # Set colorscale based on variable
    if variable == 'T':
        colorscale = create_blue_to_red_colorscale()
        units = 'K'
        title = 'Air Temperature'
    elif variable == 'rh':
        colorscale = create_white_to_orange_colorscale()
        units = '%'
        title = 'Relative Humidity'
    elif variable == 'diabatic_heating':
        colorscale = create_blue_to_red_colorscale()
        units = 'K/day'
        title = 'Diabatic Heating'
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    elif variable == 'streamfunction':
        colorscale = create_diverging_streamfunction_colorscale()
        units = '10^9 kg/s'
        title = 'Mass Streamfunction'
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax

    # Create frames
    for i, cs in enumerate(all_cross_sections):
        frame = go.Frame(
            data=[
                go.Heatmap(
                    z=cs,
                    x=area_frac * 100,
                    y=pressures,
                    colorscale=colorscale,
                    zmin=vmin,
                    zmax=vmax,
                    colorbar=dict(title=units)
                )
            ],
            name=f'Day {state_history[i]["time"]:.1f}'
        )
        frames.append(frame)

    # Create base figure
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=all_cross_sections[0],
                x=area_frac * 100,
                y=pressures,
                colorscale=colorscale,
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title=units)
            )
        ],
        frames=frames
    )

    # Add animation controls
    fig.update_layout(
        title=dict(text=f'{title} - Animated', x=0.5, font=dict(size=16)),
        xaxis=dict(title='Area Fraction (%) - Warmest to Coldest'),
        yaxis=dict(title='Pressure (hPa)', autorange='reversed'),
        height=500,
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=500, redraw=True),
                                         fromcurrent=True, mode='immediate')]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate')])
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(method='animate',
                         args=[[f'Day {state_history[i]["time"]:.1f}'],
                               dict(mode='immediate', frame=dict(duration=500, redraw=True))],
                         label=f'{state_history[i]["time"]:.1f}')
                    for i in range(n_frames)
                ],
                x=0.1,
                len=0.8,
                y=0,
                currentvalue=dict(
                    prefix='Day: ',
                    visible=True,
                    xanchor='center'
                ),
                transition=dict(duration=300)
            )
        ]
    )

    return fig


def create_realtime_visualization(state, lats, pressures):
    """
    Create a quick visualization for real-time updates during simulation.
    Shows SST map and Temperature cross-section side by side.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'SST - Day {state["time"]:.1f}',
            f'Temperature Cross-Section - Day {state["time"]:.1f}'
        ),
        column_widths=[0.5, 0.5]
    )

    # SST map
    fig.add_trace(
        go.Heatmap(
            z=state['sst'],
            colorscale='RdBu_r',
            colorbar=dict(title='K', x=0.45, len=0.9),
            hovertemplate='SST: %{z:.1f} K<extra></extra>'
        ),
        row=1, col=1
    )

    # Temperature cross-section (sorted by SST area fraction)
    cs_T, area_frac = compute_area_fraction_sorted_cross_section(
        state['T'], state['sst'], pressures, lats
    )

    fig.add_trace(
        go.Heatmap(
            z=cs_T,
            x=area_frac * 100,
            y=pressures,
            colorscale=create_blue_to_red_colorscale(),
            colorbar=dict(title='K', x=1.0, len=0.9),
            hovertemplate='Area: %{x:.0f}%<br>P: %{y:.0f} hPa<br>T: %{z:.1f} K<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text='Longitude Index', row=1, col=1)
    fig.update_yaxes(title_text='Latitude Index', row=1, col=1)
    fig.update_xaxes(title_text='Area Fraction (%) - Warm→Cold', row=1, col=2)
    fig.update_yaxes(title_text='Pressure (hPa)', autorange='reversed', row=1, col=2)

    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        title=dict(text=f'🌡️ Real-Time Simulation: Day {state["time"]:.1f}', x=0.5, font=dict(size=16))
    )

    return fig


def create_sst_diagnostics_plot(diagnostics):
    """Create SST diagnostics time series plot"""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'SST Contrast (Max - Min)',
            'Global Mean SST',
            'Warm Area Fraction',
            'Kinetic Energy'
        )
    )

    times = diagnostics['time']

    # SST Contrast
    fig.add_trace(
        go.Scatter(x=times, y=diagnostics['sst_contrast'],
                  mode='lines', name='SST Contrast',
                  line=dict(color='#e74c3c', width=2)),
        row=1, col=1
    )

    # Global Mean SST
    fig.add_trace(
        go.Scatter(x=times, y=diagnostics['global_mean_sst'],
                  mode='lines', name='Mean SST',
                  line=dict(color='#3498db', width=2)),
        row=1, col=2
    )

    # Warm Fraction
    fig.add_trace(
        go.Scatter(x=times, y=diagnostics['sst_warm_fraction'],
                  mode='lines', name='Warm Fraction',
                  line=dict(color='#2ecc71', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=0.5, line_dash='dash', line_color='gray', row=2, col=1)

    # Kinetic Energy
    fig.add_trace(
        go.Scatter(x=times, y=diagnostics['kinetic_energy'],
                  mode='lines', name='KE',
                  line=dict(color='#9b59b6', width=2)),
        row=2, col=2
    )

    fig.update_xaxes(title_text='Time (days)', row=2, col=1)
    fig.update_xaxes(title_text='Time (days)', row=2, col=2)
    fig.update_yaxes(title_text='K', row=1, col=1)
    fig.update_yaxes(title_text='K', row=1, col=2)
    fig.update_yaxes(title_text='Fraction', row=2, col=1)
    fig.update_yaxes(title_text='m²/s²', row=2, col=2)

    fig.update_layout(
        height=500,
        showlegend=False,
        title=dict(text='Tropic World Diagnostics', x=0.5, font=dict(size=18))
    )

    return fig


# =============================================================================
# Sidebar Configuration
# =============================================================================

st.sidebar.header("Simulation Configuration")

st.sidebar.subheader("Grid Resolution")
nlon = st.sidebar.select_slider("Longitude Points", options=[32, 64, 96], value=64)
nlat = st.sidebar.select_slider("Latitude Points", options=[16, 32, 48], value=32)
nlev = st.sidebar.select_slider("Vertical Levels", options=[10, 15, 20], value=20)

st.sidebar.subheader("Simulation Settings")
duration_days = st.sidebar.slider("Duration (days)", 5, 100, 30)
output_interval = st.sidebar.slider("Output Interval (hours)", 6, 24, 12)
dt = st.sidebar.select_slider("Time Step (seconds)", options=[300, 600, 900], value=600)

st.sidebar.subheader("Tropic World Parameters")
base_sst = st.sidebar.slider("Base SST (K)", 295.0, 305.0, 300.0, 0.5)
sst_perturbation = st.sidebar.slider("SST Perturbation (K)", 0.1, 2.0, 0.5, 0.1)

# =============================================================================
# Main Content
# =============================================================================

# Information box
with st.expander("About Tropic World", expanded=False):
    st.markdown("""
    **Tropic World** is an idealized planet configuration that demonstrates fundamental
    atmospheric dynamics:

    - **Non-rotating**: No Coriolis force (unlike Earth)
    - **Uniform Solar Radiation**: Same solar heating everywhere
    - **Slab Ocean**: 50m deep mixed layer with no horizontal heat transport

    **Key Phenomena:**
    1. Spontaneous warm/cold SST regions despite uniform forcing
    2. SST contrast oscillation (~2-3 year period in model time)
    3. Stronger surface winds where SST contrasts are largest
    4. Convection concentrated over warm regions
    5. Higher humidity in warm regions (greenhouse feedback)

    **Visualization Approach:**
    The vertical cross sections sort data by **area fraction** from warmest to coldest SST,
    revealing the systematic relationship between surface temperature and atmospheric structure.
    """)

# Run simulation button
run_button = st.button("🚀 Run Tropic World Simulation", type="primary", width='stretch')

if run_button or 'tropic_world_results' in st.session_state:

    if run_button:
        # Run simulation with real-time visualization updates
        st.subheader("🔄 Simulation Progress")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create placeholder for real-time visualization
        viz_placeholder = st.empty()

        def update_progress(progress, day, state=None, lats=None, pressures=None):
            """Update progress and visualization during simulation"""
            progress_bar.progress(progress)
            status_text.text(f"Simulating day {day:.1f} of {duration_days}...")

            # Update real-time visualization if state is provided
            if state is not None and lats is not None and pressures is not None:
                try:
                    fig = create_realtime_visualization(state, lats, pressures)
                    viz_placeholder.plotly_chart(fig, width='stretch', key=f'realtime_{day:.2f}')
                except Exception:
                    # Skip visualization update if there's an error (e.g., during early steps)
                    pass

        results = run_tropic_world_simulation(
            nlon, nlat, nlev, dt, duration_days, output_interval,
            base_sst, sst_perturbation, update_progress
        )

        st.session_state['tropic_world_results'] = results
        progress_bar.empty()
        status_text.empty()
        viz_placeholder.empty()
        st.success("✅ Simulation complete! View results in the tabs below.")

    results = st.session_state['tropic_world_results']
    state_history = results['state_history']
    diagnostics = results['diagnostics']

    # Get grid info
    if 'lons' in results:
        lons = results['lons']
        lats = results['lats']
        pressures = results['pressures']
    else:
        # Reconstruct from state
        nlat, nlon = state_history[0]['sst'].shape
        nlev = state_history[0]['T'].shape[0]
        lons = np.linspace(0, 360, nlon, endpoint=False)
        lats = np.linspace(-90, 90, nlat)
        pressures = np.linspace(20, 1013, nlev)

    # Main visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 3D Globe",
        "📊 Cross Sections",
        "🎬 Animations",
        "📈 Diagnostics",
        "🔬 Detailed Analysis"
    ])

    # Tab 1: 3D Globe Visualization
    with tab1:
        st.header("3D Interactive Globe")

        col1, col2 = st.columns([1, 3])

        with col1:
            timestep = st.slider(
                "Select Timestep",
                0, len(state_history) - 1, 0,
                key='globe_timestep'
            )
            st.metric("Day", f"{state_history[timestep]['time']:.1f}")

            variable_3d = st.selectbox(
                "Variable to Display",
                ["SST", "Surface Wind Speed", "Mid-level Temperature", "Column Humidity"]
            )

        with col2:
            state = state_history[timestep]

            if variable_3d == "SST":
                data = state['sst']
                title = f"Sea Surface Temperature - Day {state['time']:.1f}"
                colorscale = 'RdBu_r'
            elif variable_3d == "Surface Wind Speed":
                data = np.sqrt(state['u'][-1]**2 + state['v'][-1]**2)
                title = f"Surface Wind Speed - Day {state['time']:.1f}"
                colorscale = 'Viridis'
            elif variable_3d == "Mid-level Temperature":
                data = state['T'][nlev//2]
                title = f"Mid-level Temperature - Day {state['time']:.1f}"
                colorscale = 'RdBu_r'
            else:  # Column Humidity
                data = np.mean(state['rh'], axis=0)
                title = f"Column-mean Relative Humidity - Day {state['time']:.1f}"
                colorscale = 'YlGnBu'

            fig_globe = create_3d_globe(data, lons, lats, title, colorscale)
            st.plotly_chart(fig_globe, width='stretch')

    # Tab 2: Vertical Cross Sections
    with tab2:
        st.header("Vertical Cross Sections")
        st.markdown("**Sorted by Area Fraction: Warmest SST (left) to Coldest SST (right)**")

        timestep_cs = st.slider(
            "Select Timestep for Cross Sections",
            0, len(state_history) - 1, len(state_history) - 1,
            key='cs_timestep'
        )

        state = state_history[timestep_cs]
        st.info(f"Showing Day {state['time']:.1f}")

        # Create 2x2 grid of cross sections
        col1, col2 = st.columns(2)

        with col1:
            # Air Temperature
            st.markdown("### Air Temperature")
            cs_T, area_frac = compute_area_fraction_sorted_cross_section(
                state['T'], state['sst'], pressures, lats
            )
            fig_T = create_cross_section_plot(
                cs_T, area_frac, pressures,
                "Air Temperature",
                create_blue_to_red_colorscale(),
                "K"
            )
            st.plotly_chart(fig_T, width='stretch')

            # Streamfunction
            st.markdown("### Mass Streamfunction")
            psi = compute_streamfunction(state['v'], pressures, lats)
            cs_psi, _ = compute_area_fraction_sorted_cross_section(
                psi, state['sst'], pressures, lats
            )
            vmax_psi = np.nanpercentile(np.abs(cs_psi), 95)
            fig_psi = create_cross_section_plot(
                cs_psi, area_frac, pressures,
                "Mass Streamfunction",
                create_diverging_streamfunction_colorscale(),
                "10⁹ kg/s",
                zmid=0,
                zmin=-vmax_psi,
                zmax=vmax_psi
            )
            st.plotly_chart(fig_psi, width='stretch')

        with col2:
            # Relative Humidity
            st.markdown("### Relative Humidity")
            cs_rh, _ = compute_area_fraction_sorted_cross_section(
                state['rh'], state['sst'], pressures, lats
            )
            fig_rh = create_cross_section_plot(
                cs_rh, area_frac, pressures,
                "Relative Humidity",
                create_white_to_orange_colorscale(),
                "%",
                zmin=0,
                zmax=100
            )
            st.plotly_chart(fig_rh, width='stretch')

            # Diabatic Heating
            st.markdown("### Diabatic Heating")
            cs_dh, _ = compute_area_fraction_sorted_cross_section(
                state['diabatic_heating'], state['sst'], pressures, lats
            )
            vmax_dh = np.nanpercentile(np.abs(cs_dh), 95)
            fig_dh = create_cross_section_plot(
                cs_dh, area_frac, pressures,
                "Diabatic Heating",
                create_blue_to_red_colorscale(),
                "K/day",
                zmid=0,
                zmin=-vmax_dh,
                zmax=vmax_dh
            )
            st.plotly_chart(fig_dh, width='stretch')

    # Tab 3: Animations
    with tab3:
        st.header("Animated Cross Sections")

        anim_var = st.selectbox(
            "Select Variable to Animate",
            ["Air Temperature", "Relative Humidity", "Diabatic Heating", "Streamfunction"]
        )

        var_map = {
            "Air Temperature": "T",
            "Relative Humidity": "rh",
            "Diabatic Heating": "diabatic_heating",
            "Streamfunction": "streamfunction"
        }

        with st.spinner("Generating animation..."):
            fig_anim = create_animated_cross_sections(
                state_history, lats, pressures, var_map[anim_var]
            )

        st.plotly_chart(fig_anim, width='stretch')

        st.markdown("""
        **How to Use:**
        - Click **Play** to animate through time
        - Use the **slider** to jump to specific days
        - **Hover** over the plot to see values
        """)

    # Tab 4: Diagnostics
    with tab4:
        st.header("Tropic World Diagnostics")

        # Check if diagnostics have data
        has_data = len(diagnostics.get('sst_contrast', [])) > 0

        if has_data:
            fig_diag = create_sst_diagnostics_plot(diagnostics)
            st.plotly_chart(fig_diag, width='stretch')
        else:
            st.info("Run a simulation to see diagnostics data.")

        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Max SST Contrast",
                f"{max(diagnostics['sst_contrast']):.2f} K" if has_data else "N/A"
            )
        with col2:
            st.metric(
                "Mean SST",
                f"{np.mean(diagnostics['global_mean_sst']):.2f} K" if has_data else "N/A"
            )
        with col3:
            st.metric(
                "Mean Warm Fraction",
                f"{np.mean(diagnostics['sst_warm_fraction']):.2%}" if has_data else "N/A"
            )
        with col4:
            st.metric(
                "Mean KE",
                f"{np.mean(diagnostics['kinetic_energy']):.2f} m²/s²" if has_data else "N/A"
            )

    # Tab 5: Detailed Analysis
    with tab5:
        st.header("Detailed Analysis")

        analysis_type = st.selectbox(
            "Select Analysis",
            ["SST Pattern Evolution", "Warm vs Cold Region Comparison", "Vertical Profiles", "Heating Components"]
        )

        if analysis_type == "SST Pattern Evolution":
            # Show SST patterns at different times
            n_panels = min(6, len(state_history))
            indices = np.linspace(0, len(state_history) - 1, n_panels, dtype=int)

            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f"Day {state_history[i]['time']:.1f}" for i in indices]
            )

            # Get global colorscale limits
            all_sst = np.array([state_history[i]['sst'] for i in indices])
            vmin, vmax = np.min(all_sst), np.max(all_sst)

            for idx, i in enumerate(indices):
                row = idx // 3 + 1
                col = idx % 3 + 1

                fig.add_trace(
                    go.Heatmap(
                        z=state_history[i]['sst'],
                        x=lons,
                        y=lats,
                        colorscale='RdBu_r',
                        zmin=vmin,
                        zmax=vmax,
                        showscale=(idx == len(indices) - 1),
                        colorbar=dict(title='K') if idx == len(indices) - 1 else None
                    ),
                    row=row, col=col
                )

            fig.update_layout(height=600, title='SST Pattern Evolution')
            st.plotly_chart(fig, width='stretch')

        elif analysis_type == "Warm vs Cold Region Comparison":
            state = state_history[-1]
            sst = state['sst']
            sst_mean = np.mean(sst)

            # Define warm and cold masks
            warm_mask = sst > sst_mean
            cold_mask = sst <= sst_mean

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Warm Regions (SST > mean)', 'Cold Regions (SST < mean)']
            )

            # Warm region mean profiles
            T_warm = np.mean(state['T'][:, warm_mask], axis=1)
            rh_warm = np.mean(state['rh'][:, warm_mask], axis=1)

            # Cold region mean profiles
            T_cold = np.mean(state['T'][:, cold_mask], axis=1)
            rh_cold = np.mean(state['rh'][:, cold_mask], axis=1)

            # Temperature comparison
            fig.add_trace(
                go.Scatter(x=T_warm, y=pressures, name='T (Warm)',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=T_cold, y=pressures, name='T (Cold)',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )

            # RH comparison
            fig.add_trace(
                go.Scatter(x=rh_warm, y=pressures, name='RH (Warm)',
                          line=dict(color='orange', width=2)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=rh_cold, y=pressures, name='RH (Cold)',
                          line=dict(color='lightblue', width=2)),
                row=1, col=2
            )

            fig.update_yaxes(autorange='reversed', title='Pressure (hPa)')
            fig.update_xaxes(title='Temperature (K)', row=1, col=1)
            fig.update_xaxes(title='Relative Humidity (%)', row=1, col=2)
            fig.update_layout(height=500)

            st.plotly_chart(fig, width='stretch')

            # Statistics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Warm Regions:**")
                st.write(f"- Mean SST: {np.mean(sst[warm_mask]):.2f} K")
                st.write(f"- Area Fraction: {np.sum(warm_mask)/sst.size:.1%}")
            with col2:
                st.markdown("**Cold Regions:**")
                st.write(f"- Mean SST: {np.mean(sst[cold_mask]):.2f} K")
                st.write(f"- Area Fraction: {np.sum(cold_mask)/sst.size:.1%}")

        elif analysis_type == "Vertical Profiles":
            state = state_history[-1]

            # Select location
            col1, col2 = st.columns(2)
            with col1:
                sel_lon = st.slider("Longitude", 0, 360, 180)
            with col2:
                sel_lat = st.slider("Latitude", -90, 90, 0)

            # Find nearest grid point
            lon_idx = np.argmin(np.abs(lons - sel_lon))
            lat_idx = np.argmin(np.abs(lats - sel_lat))

            st.write(f"Selected point: {lons[lon_idx]:.1f}°E, {lats[lat_idx]:.1f}°N")
            st.write(f"SST at this location: {state['sst'][lat_idx, lon_idx]:.2f} K")

            # Create vertical profile plots
            fig = make_subplots(
                rows=1, cols=4,
                subplot_titles=['Temperature', 'Rel. Humidity', 'U Wind', 'V Wind']
            )

            fig.add_trace(
                go.Scatter(x=state['T'][:, lat_idx, lon_idx], y=pressures,
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=state['rh'][:, lat_idx, lon_idx], y=pressures,
                          line=dict(color='orange', width=2)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=state['u'][:, lat_idx, lon_idx], y=pressures,
                          line=dict(color='blue', width=2)),
                row=1, col=3
            )
            fig.add_trace(
                go.Scatter(x=state['v'][:, lat_idx, lon_idx], y=pressures,
                          line=dict(color='green', width=2)),
                row=1, col=4
            )

            fig.update_yaxes(autorange='reversed')
            fig.update_yaxes(title='Pressure (hPa)', row=1, col=1)
            fig.update_xaxes(title='K', row=1, col=1)
            fig.update_xaxes(title='%', row=1, col=2)
            fig.update_xaxes(title='m/s', row=1, col=3)
            fig.update_xaxes(title='m/s', row=1, col=4)

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width='stretch')

        else:  # Heating Components
            st.subheader("Diabatic Heating Components")
            st.markdown("""
            Breakdown of diabatic heating into its physical components:
            - **Radiation**: Longwave and shortwave radiative heating/cooling
            - **Convection**: Convective heating from moist processes
            - **Cloud Microphysics**: Phase change heating (condensation/evaporation)
            - **Boundary Layer**: Surface heat exchange and mixing
            """)

            state = state_history[-1]

            # Check if heating components are available
            if 'dT_radiation' in state:
                # Create cross sections for each component
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Radiative Heating")
                    if np.any(state['dT_radiation'] != 0):
                        cs_rad, area_frac = compute_area_fraction_sorted_cross_section(
                            state['dT_radiation'], state['sst'], pressures, lats
                        )
                        vmax_rad = np.nanpercentile(np.abs(cs_rad), 95)
                        if vmax_rad > 0:
                            fig_rad = create_cross_section_plot(
                                cs_rad, area_frac, pressures,
                                "Radiative Heating",
                                create_blue_to_red_colorscale(),
                                "K/day",
                                zmid=0, zmin=-vmax_rad, zmax=vmax_rad
                            )
                            st.plotly_chart(fig_rad, width='stretch')
                        else:
                            st.info("Radiative heating is uniform/zero")
                    else:
                        st.info("Radiative heating data not available")

                    st.markdown("### Cloud Microphysics Heating")
                    if np.any(state['dT_cloud'] != 0):
                        cs_cloud, _ = compute_area_fraction_sorted_cross_section(
                            state['dT_cloud'], state['sst'], pressures, lats
                        )
                        vmax_cloud = np.nanpercentile(np.abs(cs_cloud), 95)
                        if vmax_cloud > 0:
                            fig_cloud = create_cross_section_plot(
                                cs_cloud, area_frac, pressures,
                                "Cloud Microphysics",
                                create_blue_to_red_colorscale(),
                                "K/day",
                                zmid=0, zmin=-vmax_cloud, zmax=vmax_cloud
                            )
                            st.plotly_chart(fig_cloud, width='stretch')
                        else:
                            st.info("Cloud microphysics heating is uniform/zero")
                    else:
                        st.info("Cloud heating data not available")

                with col2:
                    st.markdown("### Convective Heating")
                    if np.any(state['dT_convection'] != 0):
                        cs_conv, _ = compute_area_fraction_sorted_cross_section(
                            state['dT_convection'], state['sst'], pressures, lats
                        )
                        vmax_conv = np.nanpercentile(np.abs(cs_conv), 95)
                        if vmax_conv > 0:
                            fig_conv = create_cross_section_plot(
                                cs_conv, area_frac, pressures,
                                "Convective Heating",
                                create_blue_to_red_colorscale(),
                                "K/day",
                                zmid=0, zmin=-vmax_conv, zmax=vmax_conv
                            )
                            st.plotly_chart(fig_conv, width='stretch')
                        else:
                            st.info("Convective heating is uniform/zero")
                    else:
                        st.info("Convective heating data not available")

                    st.markdown("### Boundary Layer Heating")
                    if np.any(state['dT_boundary_layer'] != 0):
                        cs_bl, _ = compute_area_fraction_sorted_cross_section(
                            state['dT_boundary_layer'], state['sst'], pressures, lats
                        )
                        vmax_bl = np.nanpercentile(np.abs(cs_bl), 95)
                        if vmax_bl > 0:
                            fig_bl = create_cross_section_plot(
                                cs_bl, area_frac, pressures,
                                "Boundary Layer",
                                create_blue_to_red_colorscale(),
                                "K/day",
                                zmid=0, zmin=-vmax_bl, zmax=vmax_bl
                            )
                            st.plotly_chart(fig_bl, width='stretch')
                        else:
                            st.info("Boundary layer heating is uniform/zero")
                    else:
                        st.info("Boundary layer heating data not available")

                # Summary of heating budget
                st.subheader("Heating Budget Summary")

                # Compute domain-mean heating rates
                mean_rad = np.mean(state['dT_radiation'])
                mean_conv = np.mean(state['dT_convection'])
                mean_cloud = np.mean(state['dT_cloud'])
                mean_bl = np.mean(state['dT_boundary_layer'])
                total = mean_rad + mean_conv + mean_cloud + mean_bl

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Radiation", f"{mean_rad:.3f} K/day")
                with col2:
                    st.metric("Convection", f"{mean_conv:.3f} K/day")
                with col3:
                    st.metric("Cloud", f"{mean_cloud:.3f} K/day")
                with col4:
                    st.metric("Boundary Layer", f"{mean_bl:.3f} K/day")
                with col5:
                    st.metric("Total", f"{total:.3f} K/day")
            else:
                st.warning("Heating component data not available. Run simulation with full GCM to see heating breakdown.")

else:
    # Show placeholder content
    st.info("Click **Run Tropic World Simulation** to generate visualizations")

    # Show example of what will be displayed
    st.subheader("Preview: What You'll See")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **3D Globe Visualization**
        - Interactive rotating globe
        - SST, wind speed, temperature displays
        - Time evolution controls
        """)

        st.markdown("""
        **Vertical Cross Sections**
        - Air Temperature (Blue → Red)
        - Relative Humidity (White → Orange)
        - Streamfunction (White → Orange)
        - Diabatic Heating (Blue → Red)
        """)

    with col2:
        st.markdown("""
        **Key Features**
        - Pressure on Y-axis
        - Area fraction on X-axis (warmest → coldest)
        - Animated time evolution
        - Interactive contours
        """)

        st.markdown("""
        **Diagnostic Plots**
        - SST contrast evolution
        - Warm/cold area fractions
        - Kinetic energy
        - Comparison analysis
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Tropic World visualization based on Section 2.4 of "Heuristic Models of the General Circulation"<br>
    Demonstrates fundamental atmospheric instabilities and convective organization
</div>
""", unsafe_allow_html=True)
