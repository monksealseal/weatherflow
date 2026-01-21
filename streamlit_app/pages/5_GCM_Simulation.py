"""
General Circulation Model (GCM) Simulation

Interactive GCM simulation with comprehensive diagnostics including:
- Energy spectra (kinetic energy vs wavenumber)
- Mean zonal winds and meridional circulation
- Stationary and transient eddy statistics
- Hadley cell diagnostics

Supports both full-physics GCM and Held-Suarez benchmark forcing.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(page_title="GCM Simulation", page_icon="", layout="wide")

st.title("General Circulation Model (GCM)")
st.markdown("""
Run atmospheric simulations with comprehensive diagnostics for validating GCM physics.
This page uses the actual GCM code with proper spectral analysis, eddy decomposition,
and circulation diagnostics.
""")

# Import GCM components
try:
    from gcm.core.model import GCM
    from gcm.grid.spherical import SphericalGrid
    from gcm.grid.vertical import VerticalGrid
    from gcm.core.state import ModelState
    from gcm.physics.held_suarez import HeldSuarezGCM, HeldSuarezForcing
    from gcm.diagnostics import ComprehensiveGCMDiagnostics
    GCM_AVAILABLE = True
except ImportError as e:
    GCM_AVAILABLE = False
    gcm_error = str(e)

if not GCM_AVAILABLE:
    st.error(f"GCM module not available: {gcm_error}")
    st.info("Some imports may have failed. Check the error above.")

# Sidebar configuration
st.sidebar.header("GCM Configuration")

st.sidebar.subheader("Model Type")
model_type = st.sidebar.selectbox(
    "Simulation Type",
    ["Held-Suarez Benchmark", "Full Physics GCM", "Tropic World"],
    help="Held-Suarez is recommended for testing dynamics"
)

st.sidebar.subheader("Resolution")
nlon = st.sidebar.select_slider("Longitude Points", options=[32, 48, 64, 96], value=64)
nlat = st.sidebar.select_slider("Latitude Points", options=[16, 24, 32, 48], value=32)
nlev = st.sidebar.select_slider("Vertical Levels", options=[10, 15, 20, 26], value=20)

st.sidebar.subheader("Time Integration")
dt = st.sidebar.select_slider("Time Step (seconds)", options=[300, 450, 600, 900], value=600)
integration_method = st.sidebar.selectbox(
    "Integration Method",
    ["rk3", "euler", "ab2"],
    index=0,
    help="RK3 is most stable"
)

if model_type == "Full Physics GCM":
    st.sidebar.subheader("Physics Settings")
    co2_ppmv = st.sidebar.slider("CO2 (ppmv)", 280, 800, 400)
    initial_profile = st.sidebar.selectbox("Initial Profile", ["tropical", "midlatitude", "polar"])
elif model_type == "Tropic World":
    st.sidebar.subheader("Tropic World Settings")
    tropic_sst = st.sidebar.slider("Base SST (K)", 295, 305, 300)
    sst_perturb = st.sidebar.slider("SST Perturbation (K)", 0.1, 2.0, 0.5)
else:
    co2_ppmv = 400
    initial_profile = "tropical"

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Setup", "Run Simulation", "Circulation", "Energy Spectra", "Eddy Statistics"
])

# Initialize session state
if 'gcm_model' not in st.session_state:
    st.session_state.gcm_model = None
if 'gcm_diagnostics' not in st.session_state:
    st.session_state.gcm_diagnostics = None
if 'diag_history' not in st.session_state:
    st.session_state.diag_history = []

# Tab 1: Setup
with tab1:
    st.header("Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Grid Information")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Longitude points | {nlon} |
        | Latitude points | {nlat} |
        | Vertical levels | {nlev} |
        | Total grid points | {nlon * nlat * nlev:,} |
        | Zonal resolution | {360/nlon:.1f} deg |
        | Meridional resolution | {180/(nlat-1):.1f} deg |
        """)

        st.subheader("Simulation Type")
        if model_type == "Held-Suarez Benchmark":
            st.info("""
            **Held-Suarez (1994) Benchmark**

            A standard test for GCM dynamical cores:
            - Newtonian temperature relaxation
            - Rayleigh friction in boundary layer
            - No moisture or radiation
            - Produces realistic jets and eddies
            """)
        elif model_type == "Tropic World":
            st.info("""
            **Tropic World Simulation**

            Idealized non-rotating planet:
            - No Coriolis force
            - Uniform incoming solar radiation
            - Slab ocean with no heat transport
            - Spontaneous convective organization
            """)
        else:
            st.info("""
            **Full Physics GCM**

            Complete physics parameterizations:
            - Two-stream radiation
            - Mass-flux convection
            - Cloud microphysics
            - TKE boundary layer
            - Slab ocean
            """)

    with col2:
        st.subheader("Expected Climate Features")

        if model_type == "Held-Suarez Benchmark":
            expected = {
                "Subtropical jets": "~30 m/s at 30 deg, 200 hPa",
                "Hadley cell": "~100 x10^9 kg/s mass transport",
                "Eddy-driven jet": "~10-15 m/s at 45-50 deg",
                "Baroclinic eddies": "EKE max in mid-latitudes",
                "KE spectrum": "k^-3 slope in synoptic scales"
            }
        elif model_type == "Tropic World":
            expected = {
                "SST contrast": "Develops from initial perturbations",
                "Convective organization": "Warm pool with enhanced convection",
                "Surface winds": "Converge toward warm regions",
                "No jets": "No rotation means no geostrophic jets"
            }
        else:
            expected = {
                "Surface temperature": "300K equator, 260K poles",
                "Hadley cell": "Extends to ~30 deg lat",
                "Jet streams": "Subtropical and polar jets",
                "Precipitation": "ITCZ near equator"
            }

        for feature, description in expected.items():
            st.markdown(f"**{feature}**: {description}")

# Tab 2: Run Simulation
with tab2:
    st.header("Run GCM Simulation")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Simulation Settings")
        duration_days = st.slider("Duration (days)", 10, 200, 50 if model_type == "Held-Suarez Benchmark" else 30)
        diag_interval = st.slider("Diagnostic interval (days)", 1, 10, 5)

        st.markdown(f"""
        **Estimated runtime:**
        - Steps: {int(duration_days * 86400 / dt):,}
        - Output times: {int(duration_days / diag_interval)}
        """)

        if st.button("Initialize Model", type="secondary"):
            with st.spinner("Initializing GCM..."):
                try:
                    if model_type == "Held-Suarez Benchmark":
                        model = HeldSuarezGCM(
                            nlon=nlon, nlat=nlat, nlev=nlev,
                            dt=dt, integration_method=integration_method
                        )
                        model.initialize(perturbation=True)
                    elif model_type == "Tropic World":
                        model = GCM(
                            nlon=nlon, nlat=nlat, nlev=nlev, dt=dt,
                            integration_method=integration_method,
                            tropic_world=True,
                            tropic_world_sst=tropic_sst,
                            sst_perturbation=sst_perturb
                        )
                        model.initialize(profile='tropical')
                    else:
                        model = GCM(
                            nlon=nlon, nlat=nlat, nlev=nlev, dt=dt,
                            integration_method=integration_method,
                            co2_ppmv=co2_ppmv
                        )
                        model.initialize(profile=initial_profile)

                    st.session_state.gcm_model = model
                    st.session_state.gcm_diagnostics = ComprehensiveGCMDiagnostics(model.grid, model.vgrid)
                    st.session_state.diag_history = []
                    st.success("Model initialized!")
                except Exception as e:
                    st.error(f"Error initializing model: {e}")

        run_sim = st.button("Run Simulation", type="primary",
                           disabled=st.session_state.gcm_model is None)

    with col2:
        if run_sim and st.session_state.gcm_model is not None:
            model = st.session_state.gcm_model
            diag_module = st.session_state.gcm_diagnostics

            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_row = st.empty()
            plot_container = st.empty()

            n_outputs = int(duration_days / diag_interval)
            steps_per_output = int(diag_interval * 86400 / model.dt)

            start_time = time.time()

            try:
                for i in range(n_outputs):
                    # Run for one diagnostic interval
                    for _ in range(steps_per_output):
                        model.integrator.step(model.state, model.dt, model._compute_tendencies)
                        # Apply state limits to prevent unrealistic values
                        model.dynamics.apply_state_limits(model.state)

                    # Compute diagnostics
                    diag = diag_module.compute_all_diagnostics(model.state)
                    st.session_state.diag_history.append(diag)

                    # Update progress
                    progress = (i + 1) / n_outputs
                    progress_bar.progress(progress)
                    sim_day = model.state.time / 86400.0
                    status_text.text(f"Day {sim_day:.1f} / {duration_days} ({progress*100:.1f}%)")

                    # Update metrics
                    with metrics_row.container():
                        m_cols = st.columns(4)
                        with m_cols[0]:
                            st.metric("Simulation Day", f"{sim_day:.1f}")
                        with m_cols[1]:
                            T_mean = diag_module.zonal.zonal_mean(model.state.T[-1]).mean()
                            st.metric("Surface T", f"{T_mean:.1f} K")
                        with m_cols[2]:
                            ke = diag['energy']['kinetic_energy']
                            st.metric("KE", f"{ke:.1f} J/kg")
                        with m_cols[3]:
                            max_u = np.max(np.abs(model.state.u))
                            st.metric("Max Wind", f"{max_u:.1f} m/s")

                    # Update plot - zonal mean structure
                    zonal = diag['zonal']
                    lat = zonal['latitude']
                    p = zonal['pressure'][:, len(lat)//2]
                    u_bar = zonal['u_bar']

                    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Zonal Mean Zonal Wind [u]', 'Zonal Mean Temperature [T]'])

                    # Zonal wind
                    fig.add_trace(
                        go.Contour(x=lat, y=p, z=u_bar,
                                  colorscale='RdBu_r', zmid=0,
                                  contours=dict(showlines=True, size=5),
                                  colorbar=dict(title='m/s', x=0.45)),
                        row=1, col=1
                    )

                    # Temperature
                    fig.add_trace(
                        go.Contour(x=lat, y=p, z=zonal['T_bar'],
                                  colorscale='Spectral_r',
                                  contours=dict(showlines=True, size=10),
                                  colorbar=dict(title='K', x=1.0)),
                        row=1, col=2
                    )

                    fig.update_yaxes(autorange='reversed', title='Pressure (hPa)')
                    fig.update_xaxes(title='Latitude')
                    fig.update_layout(height=400)

                    plot_container.plotly_chart(fig, use_container_width=True)

                elapsed = time.time() - start_time
                st.success(f"Simulation complete! ({elapsed:.1f} seconds)")
            except Exception as e:
                st.error(f"Simulation error: {e}")

        elif st.session_state.gcm_model is None:
            st.info("Click 'Initialize Model' first, then 'Run Simulation'")

# Tab 3: Circulation Diagnostics
with tab3:
    st.header("Circulation Diagnostics")

    if len(st.session_state.diag_history) > 0:
        diag = st.session_state.diag_history[-1]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Zonal Mean Zonal Wind")
            zonal = diag['zonal']
            lat = zonal['latitude']
            p = zonal['pressure'][:, len(lat)//2]

            fig1 = go.Figure()
            fig1.add_trace(go.Contour(
                x=lat, y=p, z=zonal['u_bar'],
                colorscale='RdBu_r', zmid=0,
                contours=dict(showlines=True, size=5),
                colorbar=dict(title='m/s')
            ))
            fig1.update_yaxes(autorange='reversed', title='Pressure (hPa)')
            fig1.update_xaxes(title='Latitude (deg)')
            fig1.update_layout(height=400, title='[u] (m/s)')
            st.plotly_chart(fig1, use_container_width=True)

            # Jet diagnostics
            jet = diag['circulation']['jet_diagnostics']
            st.markdown(f"""
            **Jet Stream Analysis:**
            - NH Subtropical Jet: {jet['subtropical_jet_nh_speed']:.1f} m/s at {jet['subtropical_jet_nh_lat']:.1f} deg
            - SH Subtropical Jet: {jet['subtropical_jet_sh_speed']:.1f} m/s at {jet['subtropical_jet_sh_lat']:.1f} deg
            """)

        with col2:
            st.subheader("Meridional Streamfunction")
            circ = diag['circulation']
            psi = circ['streamfunction']

            if psi is not None:
                fig2 = go.Figure()
                psi_clean = np.nan_to_num(psi, nan=0.0)
                max_psi = max(abs(np.min(psi_clean)), abs(np.max(psi_clean)), 1.0)
                fig2.add_trace(go.Contour(
                    x=lat, y=p, z=psi_clean,
                    colorscale='RdBu_r', zmid=0,
                    zmin=-max_psi, zmax=max_psi,
                    contours=dict(showlines=True),
                    colorbar=dict(title='10^9 kg/s')
                ))
                fig2.update_yaxes(autorange='reversed', title='Pressure (hPa)')
                fig2.update_xaxes(title='Latitude (deg)')
                fig2.update_layout(height=400, title='Meridional Streamfunction')
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown(f"""
                **Hadley Cell Analysis:**
                - NH Strength: {circ['hadley_strength_nh']:.1f} x10^9 kg/s
                - SH Strength: {circ['hadley_strength_sh']:.1f} x10^9 kg/s
                """)

        # Time series
        if len(st.session_state.diag_history) > 1:
            st.subheader("Time Evolution")
            times = [d['time'] for d in st.session_state.diag_history]
            ke_vals = [d['energy']['kinetic_energy'] for d in st.session_state.diag_history]
            max_u = [np.max(np.abs(d['zonal']['u_bar'])) for d in st.session_state.diag_history]

            fig3 = make_subplots(rows=1, cols=2,
                subplot_titles=['Kinetic Energy', 'Max Zonal Wind'])

            fig3.add_trace(go.Scatter(x=times, y=ke_vals, mode='lines',
                                     line=dict(color='#1e88e5')), row=1, col=1)
            fig3.add_trace(go.Scatter(x=times, y=max_u, mode='lines',
                                     line=dict(color='#e53935')), row=1, col=2)

            fig3.update_xaxes(title='Day')
            fig3.update_yaxes(title='J/kg', row=1, col=1)
            fig3.update_yaxes(title='m/s', row=1, col=2)
            fig3.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Run a simulation to see circulation diagnostics")

# Tab 4: Energy Spectra
with tab4:
    st.header("Energy Spectra")

    if len(st.session_state.diag_history) > 0:
        diag = st.session_state.diag_history[-1]
        spectra = diag['spectral']

        wn = spectra['wavenumbers']
        ke_spec = spectra['kinetic_energy_spectrum']
        t_spec = spectra['temperature_variance_spectrum']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Kinetic Energy Spectrum")

            fig1 = go.Figure()

            # KE spectrum (avoid log of zero)
            valid = (wn > 0) & (ke_spec > 0)
            fig1.add_trace(go.Scatter(
                x=wn[valid], y=ke_spec[valid],
                mode='lines', name='KE spectrum',
                line=dict(color='#1e88e5', width=2)
            ))

            # Reference slopes
            if len(spectra['k_minus_3_reference']) > 0:
                ref_wn = wn[5:]
                k3_ref = spectra['k_minus_3_reference']
                k53_ref = spectra['k_minus_5_3_reference']

                fig1.add_trace(go.Scatter(
                    x=ref_wn, y=k3_ref,
                    mode='lines', name='k^-3',
                    line=dict(color='gray', dash='dash')
                ))
                fig1.add_trace(go.Scatter(
                    x=ref_wn, y=k53_ref,
                    mode='lines', name='k^-5/3',
                    line=dict(color='gray', dash='dot')
                ))

            fig1.update_xaxes(type='log', title='Wavenumber')
            fig1.update_yaxes(type='log', title='Energy')
            fig1.update_layout(height=400, title='Kinetic Energy Spectrum E(k)',
                             legend=dict(x=0.7, y=0.95))
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - k^-3 slope: characteristic of 2D turbulence (enstrophy cascade)
            - k^-5/3 slope: characteristic of 3D turbulence (energy cascade)
            - Synoptic scales (k ~ 5-15) typically show k^-3
            """)

        with col2:
            st.subheader("Temperature Variance Spectrum")

            fig2 = go.Figure()
            valid_t = (wn > 0) & (t_spec > 0)
            fig2.add_trace(go.Scatter(
                x=wn[valid_t], y=t_spec[valid_t],
                mode='lines', name='T variance',
                line=dict(color='#e53935', width=2)
            ))

            fig2.update_xaxes(type='log', title='Wavenumber')
            fig2.update_yaxes(type='log', title='Variance')
            fig2.update_layout(height=400, title='Temperature Variance Spectrum')
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - Related to Available Potential Energy (APE)
            - Large scales dominate temperature variance
            - Baroclinic instability peaks at synoptic scales
            """)
    else:
        st.info("Run a simulation to see energy spectra")

# Tab 5: Eddy Statistics
with tab5:
    st.header("Eddy Statistics")

    if len(st.session_state.diag_history) > 0:
        diag = st.session_state.diag_history[-1]
        eddy = diag['eddy']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Eddy Kinetic Energy")
            eke = eddy['eke']
            lat = eddy['latitude']
            p = eddy['pressure'][:, len(lat)//2]

            if eke is not None:
                fig1 = go.Figure()
                fig1.add_trace(go.Contour(
                    x=lat, y=p, z=eke,
                    colorscale='YlOrRd',
                    contours=dict(showlines=True),
                    colorbar=dict(title='m2/s2')
                ))
                fig1.update_yaxes(autorange='reversed', title='Pressure (hPa)')
                fig1.update_xaxes(title='Latitude (deg)')
                fig1.update_layout(height=350, title='Eddy Kinetic Energy')
                st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Eddy Momentum Flux")
            uv_stat = eddy['uv_stationary']

            if uv_stat is not None:
                fig2 = go.Figure()
                uv_clean = np.nan_to_num(uv_stat, nan=0.0)
                max_val = max(abs(np.min(uv_clean)), abs(np.max(uv_clean)), 0.1)
                fig2.add_trace(go.Contour(
                    x=lat, y=p, z=uv_clean,
                    colorscale='RdBu_r', zmid=0,
                    contours=dict(showlines=True),
                    colorbar=dict(title='m2/s2')
                ))
                fig2.update_yaxes(autorange='reversed', title='Pressure (hPa)')
                fig2.update_xaxes(title='Latitude (deg)')
                fig2.update_layout(height=350, title="[u'v'] Momentum Flux")
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Meridional Heat Transport")
        Q_total = eddy['heat_transport_total']
        Q_stat = eddy['heat_transport_stationary']
        Q_mean = eddy['heat_transport_mean']

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=lat, y=Q_total, mode='lines',
                                 name='Total', line=dict(color='black', width=2)))
        fig3.add_trace(go.Scatter(x=lat, y=Q_stat, mode='lines',
                                 name='Stationary Eddies', line=dict(color='#e53935', dash='dash')))
        fig3.add_trace(go.Scatter(x=lat, y=Q_mean, mode='lines',
                                 name='Mean Circulation', line=dict(color='#1e88e5', dash='dot')))

        fig3.add_hline(y=0, line_color='gray', line_dash='solid', opacity=0.5)
        fig3.update_xaxes(title='Latitude (deg)')
        fig3.update_yaxes(title='Heat Transport (PW)')
        fig3.update_layout(height=350, title='Meridional Heat Transport',
                          legend=dict(x=0.7, y=0.95))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        - **Stationary eddies**: Standing waves (e.g., forced by topography)
        - **Transient eddies**: Time-varying fluctuations (storms)
        - **Mean circulation**: Hadley cell contribution (dominant in tropics)
        """)
    else:
        st.info("Run a simulation to see eddy statistics")

# Realism check section
with st.expander("GCM Realism Check"):
    if len(st.session_state.diag_history) > 0 and st.session_state.gcm_diagnostics is not None:
        diag = st.session_state.diag_history[-1]
        diag_module = st.session_state.gcm_diagnostics

        report = diag_module.check_gcm_realism(diag)

        if report['passed']:
            st.success("GCM diagnostics appear reasonable!")
        else:
            st.warning("GCM diagnostics show potential issues")

        if report['warnings']:
            st.warning("Warnings:")
            for w in report['warnings']:
                st.markdown(f"- {w}")

        if report['errors']:
            st.error("Errors:")
            for e in report['errors']:
                st.markdown(f"- {e}")
    else:
        st.info("Run a simulation to check realism")

# Code reference
with st.expander("Code Reference"):
    st.code("""
# Using the GCM with comprehensive diagnostics
from gcm.core.model import GCM
from gcm.physics.held_suarez import HeldSuarezGCM
from gcm.diagnostics import ComprehensiveGCMDiagnostics

# Option 1: Held-Suarez benchmark (recommended for testing dynamics)
model = HeldSuarezGCM(nlon=64, nlat=32, nlev=20, dt=600)
model.initialize()

# Option 2: Full physics GCM
model = GCM(nlon=64, nlat=32, nlev=20, dt=600, co2_ppmv=400)
model.initialize(profile='tropical')

# Initialize diagnostics
diag_module = ComprehensiveGCMDiagnostics(model.grid, model.vgrid)

# Run simulation with diagnostics
for day in range(100):
    for step in range(144):  # 144 steps/day with dt=600s
        model.integrator.step(model.state, model.dt, model._compute_tendencies)

    # Compute comprehensive diagnostics
    diagnostics = diag_module.compute_all_diagnostics(model.state)

    # Access specific diagnostics
    ke_spectrum = diagnostics['spectral']['kinetic_energy_spectrum']
    zonal_wind = diagnostics['zonal']['u_bar']
    eddy_flux = diagnostics['eddy']['uv_stationary']
    hadley_strength = diagnostics['circulation']['hadley_strength_nh']

# Plot all diagnostics
diag_module.plot_diagnostics()
    """, language="python")
