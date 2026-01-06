"""
Graduate Atmospheric Dynamics Education Tools

Uses the actual GraduateAtmosphericDynamicsTool class from weatherflow/education/graduate_tool.py
This is an educational tool using idealized scenarios - ERA5 data is optional.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))

from weatherflow.education.graduate_tool import (
    GraduateAtmosphericDynamicsTool,
    SolutionStep,
    ProblemScenario,
    OMEGA, R_EARTH, GRAVITY, R_AIR
)

st.set_page_config(page_title="Atmospheric Dynamics Education", page_icon="📚", layout="wide")

st.title("📚 Graduate Atmospheric Dynamics")
st.markdown("""
Interactive learning tools for graduate-level atmospheric dynamics.
This runs the actual `GraduateAtmosphericDynamicsTool` class from the repository.
""")

# Note about data source
st.info("""
📊 **Data Note:** This educational module uses **idealized scenarios** based on analytical solutions 
and textbook examples. The calculations demonstrate fundamental atmospheric dynamics principles 
using physically realistic but simplified conditions.
""")

# Create tool instance
tool = GraduateAtmosphericDynamicsTool(reference_latitude=45.0)

# Sidebar
st.sidebar.header("Reference Parameters")
ref_latitude = st.sidebar.slider("Reference Latitude", 0.0, 90.0, 45.0)
tool = GraduateAtmosphericDynamicsTool(reference_latitude=ref_latitude)

st.sidebar.markdown("---")
st.sidebar.markdown("### Physical Constants")
st.sidebar.markdown(f"""
- Ω = {OMEGA:.4e} rad/s
- R_Earth = {R_EARTH/1e6:.3f} × 10⁶ m
- g = {GRAVITY:.2f} m/s²
- R_air = {R_AIR:.1f} J/(kg·K)
""")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Geostrophic Balance",
    "🌊 Rossby Waves",
    "🌀 Potential Vorticity",
    "📝 Practice Problems",
    "📖 Study Guide"
])

# Tab 1: Geostrophic Balance
with tab1:
    st.header("Geostrophic Wind Calculator")

    st.markdown("""
    The geostrophic wind is the theoretical wind that results from exact balance
    between the Coriolis force and the pressure gradient force:

    $$f \\mathbf{k} \\times \\mathbf{u}_g = -\\frac{1}{\\rho} \\nabla p = -g \\nabla Z$$

    where Z is the geopotential height.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")

        height_diff = st.slider("Height Difference (m)", 10.0, 200.0, 60.0,
                               help="Geopotential height difference across domain")
        distance = st.slider("Distance (km)", 100, 2000, 500) * 1000  # Convert to m
        latitude = st.slider("Latitude (°N)", 10.0, 80.0, 45.0)

        st.markdown("---")

        st.subheader("Thermal Wind")
        st.markdown("""
        The thermal wind relates the vertical wind shear to horizontal temperature gradients:

        $$\\frac{\\partial u_g}{\\partial \\ln p} = -\\frac{R}{f} \\frac{\\partial T}{\\partial y}$$
        """)

        temp_gradient = st.slider("Temperature Gradient (K/1000km)",
                                  0.0, 10.0, 4.0) * 1e-6  # K/m
        p_lower = st.number_input("Lower Pressure (hPa)", 500.0, 1000.0, 850.0) * 100
        p_upper = st.number_input("Upper Pressure (hPa)", 100.0, 500.0, 500.0) * 100

    with col2:
        st.subheader("Results")

        # Calculate geostrophic wind
        u_g, geo_steps = tool.geostrophic_wind_solution(height_diff, distance, latitude)

        st.markdown("### Geostrophic Wind Solution")

        for step in geo_steps:
            st.markdown(f"**{step.description}**: {step.value:.4e} {step.units}")

        st.success(f"**Geostrophic Wind: {u_g:.1f} m/s** {'(westerly)' if u_g > 0 else '(easterly)'}")

        st.markdown("---")

        # Calculate thermal wind
        thermal_shear, thermal_steps = tool.thermal_wind_solution(
            temp_gradient, p_lower, p_upper, latitude
        )

        st.markdown("### Thermal Wind Solution")

        for step in thermal_steps:
            st.markdown(f"**{step.description}**: {step.value:.4e} {step.units}")

        st.success(f"**Wind Shear: {thermal_shear:.1f} m/s** (between {p_upper/100:.0f}-{p_lower/100:.0f} hPa)")

        # Visualization
        st.markdown("### Geostrophic Balance Diagram")

        # Create height field
        lons = np.linspace(-10, 10, 50)
        lats = np.linspace(30, 60, 50)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Height increases northward
        height_field = 5500 + (lat_grid - 45) * height_diff / 30

        # Add some waviness
        height_field += 50 * np.sin(np.radians(lon_grid) * 5)

        fig = go.Figure()

        # Height contours
        fig.add_trace(go.Contour(
            z=height_field, x=lons, y=lats,
            colorscale='Viridis',
            contours=dict(showlabels=True, labelfont=dict(size=10)),
            colorbar=dict(title='Z (m)')
        ))

        # Add wind arrows
        skip = 5
        u_arrows = np.ones_like(height_field) * u_g
        v_arrows = np.zeros_like(height_field)

        fig.add_trace(go.Cone(
            x=lon_grid[::skip, ::skip].flatten(),
            y=lat_grid[::skip, ::skip].flatten(),
            z=np.zeros_like(lon_grid[::skip, ::skip].flatten()),
            u=u_arrows[::skip, ::skip].flatten() / 10,
            v=v_arrows[::skip, ::skip].flatten(),
            w=np.zeros_like(lon_grid[::skip, ::skip].flatten()),
            sizemode='absolute', sizeref=1,
            colorscale='Reds', showscale=False
        ))

        fig.update_layout(
            title='Geopotential Height (m) and Geostrophic Wind',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Rossby Waves
with tab2:
    st.header("Rossby Wave Dispersion")

    st.markdown("""
    Rossby waves arise from the conservation of potential vorticity on the β-plane.
    The dispersion relation for barotropic Rossby waves is:

    $$\\omega = \\bar{u} k - \\frac{\\beta k}{k^2 + l^2}$$

    where k and l are zonal and meridional wavenumbers.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")

        mean_flow = st.slider("Mean Zonal Flow (m/s)", 0.0, 30.0, 15.0)
        wavelength_km = st.slider("Zonal Wavelength (km)", 1000, 10000, 4000)

        beta = float(tool.beta_parameter(ref_latitude))
        st.metric("β Parameter", f"{beta:.2e} m⁻¹s⁻¹")

        # Calculate phase speed
        c_phase, phase_steps = tool.rossby_phase_speed_solution(
            wavelength_km * 1000, mean_flow, beta=beta
        )

        st.markdown("### Solution Steps")
        for step in phase_steps:
            st.markdown(f"**{step.description}**")
            st.markdown(f"   Value: {step.value:.4e} {step.units}")

        st.success(f"**Phase Speed: {c_phase:.1f} m/s**")

        if c_phase > 0:
            st.info("Wave propagates eastward relative to ground")
        else:
            st.info("Wave propagates westward relative to ground")

        if c_phase < mean_flow:
            st.info("Wave propagates westward relative to mean flow")

    with col2:
        st.subheader("Dispersion Surface")

        # Create 2D dispersion relation visualization
        k_vals = np.linspace(1e-7, 1e-5, 100)
        l_vals = np.linspace(0, 8e-6, 100)
        k_grid, l_grid = np.meshgrid(k_vals, l_vals)

        omega = tool.rossby_dispersion_relation(beta, mean_flow, k_grid, l_grid)
        c_phase_2d = np.where(k_grid != 0, omega / k_grid, 0)
        cg_x, cg_y = tool.rossby_group_velocity(beta, mean_flow, k_grid, l_grid)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Frequency ω', 'Phase Speed cₓ', 'Group Velocity cg,x'),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
        )

        # Frequency
        fig.add_trace(
            go.Heatmap(z=omega * 1e5, x=k_vals * 1e6, y=l_vals * 1e6,
                      colorscale='Viridis', showscale=True,
                      colorbar=dict(title='ω×10⁵ (s⁻¹)', x=0.28)),
            row=1, col=1
        )

        # Phase speed
        fig.add_trace(
            go.Heatmap(z=c_phase_2d, x=k_vals * 1e6, y=l_vals * 1e6,
                      colorscale='RdBu_r', zmid=0, showscale=True,
                      colorbar=dict(title='cₓ (m/s)', x=0.63)),
            row=1, col=2
        )

        # Group velocity
        fig.add_trace(
            go.Heatmap(z=cg_x, x=k_vals * 1e6, y=l_vals * 1e6,
                      colorscale='RdBu_r', zmid=mean_flow, showscale=True,
                      colorbar=dict(title='cg,x (m/s)', x=0.98)),
            row=1, col=3
        )

        fig.update_xaxes(title_text='k (×10⁻⁶ m⁻¹)')
        fig.update_yaxes(title_text='l (×10⁻⁶ m⁻¹)', row=1, col=1)
        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

        # Wave animation conceptual
        st.subheader("Wave Propagation Concept")

        x = np.linspace(0, 4 * np.pi, 200)
        t_anim = st.slider("Time (arbitrary units)", 0.0, 10.0, 0.0, 0.1)

        k_selected = 2 * np.pi / (wavelength_km * 1000)
        omega_selected = mean_flow * k_selected - beta * k_selected / (k_selected**2)

        wave = np.cos(k_selected * x * 1e6 - omega_selected * t_anim * 1e4)

        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=x / np.pi * wavelength_km / 2, y=wave,
            mode='lines', name='Wave',
            line=dict(color='#1e88e5', width=3)
        ))
        fig_wave.update_layout(
            title=f'Rossby Wave (λ = {wavelength_km} km)',
            xaxis_title='Distance (km)',
            yaxis_title='Amplitude',
            height=300
        )
        st.plotly_chart(fig_wave, use_container_width=True)

# Tab 3: Potential Vorticity
with tab3:
    st.header("Potential Vorticity")

    st.markdown("""
    Quasi-geostrophic potential vorticity (QGPV) is conserved following the flow:

    $$q = \\nabla^2\\psi + \\frac{f_0^2}{N^2}\\frac{\\partial^2 \\psi}{\\partial z^2} + f$$

    where ψ is the streamfunction and N is the buoyancy frequency.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("PV Diagnostics")

        f0 = float(tool.coriolis_parameter(ref_latitude))
        st.metric("f₀", f"{f0:.2e} s⁻¹")

        beta = float(tool.beta_parameter(ref_latitude))
        st.metric("β", f"{beta:.2e} m⁻¹s⁻¹")

        st.markdown("---")

        st.subheader("Generate PV Field")

        pv_type = st.selectbox(
            "PV Structure",
            ["Isolated Vortex", "PV Anomaly Pair", "PV Gradient"]
        )

        amplitude = st.slider("Amplitude (×10⁻⁵ s⁻¹)", 0.5, 5.0, 2.0)

    with col2:
        # Generate PV field
        x = np.linspace(-1000, 1000, 64) * 1000  # km to m
        y = np.linspace(-1000, 1000, 64) * 1000
        x_grid, y_grid = np.meshgrid(x, y)

        if pv_type == "Isolated Vortex":
            r = np.sqrt(x_grid**2 + y_grid**2)
            pv = amplitude * 1e-5 * np.exp(-(r / 300e3)**2)
        elif pv_type == "PV Anomaly Pair":
            r1 = np.sqrt((x_grid - 200e3)**2 + y_grid**2)
            r2 = np.sqrt((x_grid + 200e3)**2 + y_grid**2)
            pv = amplitude * 1e-5 * (np.exp(-(r1 / 200e3)**2) - np.exp(-(r2 / 200e3)**2))
        else:  # PV Gradient
            pv = amplitude * 1e-5 * (y_grid / 500e3) + beta * y_grid / f0

        # Add background PV gradient
        pv_total = pv + beta * y_grid / f0 * 0.1

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('PV Anomaly', 'Total PV (with β-effect)')
        )

        fig.add_trace(
            go.Contour(z=pv * 1e5, x=x/1000, y=y/1000,
                      colorscale='RdBu_r', contours_coloring='heatmap',
                      colorbar=dict(title='q×10⁵ (s⁻¹)', x=0.45)),
            row=1, col=1
        )

        fig.add_trace(
            go.Contour(z=pv_total * 1e5, x=x/1000, y=y/1000,
                      colorscale='Viridis', contours_coloring='heatmap',
                      colorbar=dict(title='q×10⁵ (s⁻¹)')),
            row=1, col=2
        )

        fig.update_xaxes(title_text='x (km)')
        fig.update_yaxes(title_text='y (km)', row=1, col=1)
        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

        # PV invertibility
        st.subheader("PV Inversion (Conceptual)")
        st.markdown("""
        **PV Invertibility Principle**: Given the PV distribution and boundary conditions,
        the balanced flow can be recovered:

        1. Solve ∇²ψ = q - f for streamfunction ψ
        2. Recover winds: u = -∂ψ/∂y, v = ∂ψ/∂x
        3. Recover pressure/height from geostrophy

        This is why PV is so powerful for understanding atmospheric dynamics!
        """)

# Tab 4: Practice Problems
with tab4:
    st.header("Practice Problems")

    st.markdown("""
    Test your understanding with these graduate-level problems.
    Each problem includes step-by-step solutions.
    """)

    # Generate problems
    problems = tool.generate_problem_scenarios()

    for i, problem in enumerate(problems):
        with st.expander(f"📝 Problem {i+1}: {problem.title}", expanded=(i==0)):
            st.markdown(f"**Problem:**\n{problem.problem}")

            show_solution = st.checkbox(f"Show Solution", key=f"sol_{i}")

            if show_solution:
                st.markdown("---")
                st.markdown("**Solution Steps:**")

                for j, step in enumerate(problem.solution_steps):
                    st.markdown(f"{j+1}. **{step.description}**")
                    st.latex(f"= {step.value:.4g} \\text{{ {step.units}}}")

                st.success(f"**Answer:** {problem.answer}")

    st.markdown("---")

    # Custom problem
    st.subheader("Create Custom Problem")

    custom_type = st.selectbox(
        "Problem Type",
        ["Geostrophic Wind", "Thermal Wind", "Rossby Phase Speed"]
    )

    if custom_type == "Geostrophic Wind":
        col1, col2 = st.columns(2)
        with col1:
            custom_dz = st.number_input("Height difference (m)", 10.0, 200.0, 50.0)
            custom_dist = st.number_input("Distance (km)", 100.0, 2000.0, 500.0) * 1000
            custom_lat = st.number_input("Latitude (°)", 10.0, 80.0, 40.0)
        with col2:
            if st.button("Solve Custom Problem"):
                u, steps = tool.geostrophic_wind_solution(custom_dz, custom_dist, custom_lat)
                for step in steps:
                    st.markdown(f"**{step.description}**: {step.value:.4e} {step.units}")
                st.success(f"Answer: u_g = {u:.1f} m/s")

    elif custom_type == "Thermal Wind":
        col1, col2 = st.columns(2)
        with col1:
            custom_dt = st.number_input("Temp gradient (K/1000km)", 1.0, 10.0, 4.0) * 1e-6
            custom_p1 = st.number_input("Lower pressure (hPa)", 500.0, 1000.0, 850.0) * 100
            custom_p2 = st.number_input("Upper pressure (hPa)", 100.0, 500.0, 500.0) * 100
            custom_lat = st.number_input("Latitude (°)", 10.0, 80.0, 45.0, key="thermal_lat")
        with col2:
            if st.button("Solve Thermal Wind"):
                shear, steps = tool.thermal_wind_solution(custom_dt, custom_p1, custom_p2, custom_lat)
                for step in steps:
                    st.markdown(f"**{step.description}**: {step.value:.4e} {step.units}")
                st.success(f"Answer: Δu_g = {shear:.1f} m/s")

# Tab 5: Study Guide
with tab5:
    st.header("Conceptual Study Guide")

    checklist = tool.conceptual_checklist()

    for topic, description in checklist.items():
        with st.expander(f"📖 {topic}", expanded=True):
            st.markdown(description)

    st.markdown("---")

    st.subheader("Key Equations Reference")

    equations = {
        "Geostrophic Balance": r"f \mathbf{k} \times \mathbf{u}_g = -\nabla \Phi",
        "Thermal Wind": r"\frac{\partial \mathbf{u}_g}{\partial \ln p} = -\frac{R}{f} \mathbf{k} \times \nabla T",
        "Rossby Dispersion": r"\omega = \bar{u}k - \frac{\beta k}{k^2 + l^2}",
        "QG-PV": r"q = \nabla^2\psi + \frac{f_0^2}{N^2}\frac{\partial^2\psi}{\partial z^2} + \beta y",
        "Ertel PV": r"P = \frac{1}{\rho}(\zeta + f) \cdot \nabla\theta",
        "Omega Equation": r"\left(\nabla^2 + \frac{f_0^2}{N^2}\frac{\partial^2}{\partial z^2}\right)\omega = \frac{f_0}{\sigma}\frac{\partial}{\partial z}\left[\mathbf{u}_g \cdot \nabla(\zeta + f)\right]"
    }

    for name, eq in equations.items():
        st.markdown(f"**{name}**")
        st.latex(eq)
        st.markdown("")

# Code reference
with st.expander("📝 Code Reference"):
    st.markdown("""
    This page uses the following code from the repository:

    ```python
    # From weatherflow/education/graduate_tool.py
    from weatherflow.education.graduate_tool import (
        GraduateAtmosphericDynamicsTool,
        SolutionStep,
        ProblemScenario
    )

    # Create tool instance
    tool = GraduateAtmosphericDynamicsTool(reference_latitude=45.0)

    # Calculate Coriolis parameter
    f = tool.coriolis_parameter(latitude)

    # Calculate beta parameter
    beta = tool.beta_parameter(latitude)

    # Geostrophic wind solution
    u_g, steps = tool.geostrophic_wind_solution(
        height_difference=60.0,  # m
        distance=500000.0,  # m
        latitude=45.0  # degrees
    )

    # Thermal wind solution
    shear, steps = tool.thermal_wind_solution(
        temperature_gradient=4e-6,  # K/m
        pressure_lower=85000.0,  # Pa
        pressure_upper=50000.0,  # Pa
        latitude=50.0
    )

    # Rossby wave phase speed
    c_phase, steps = tool.rossby_phase_speed_solution(
        wavelength_x=4e6,  # m
        mean_flow=20.0  # m/s
    )

    # Generate practice problems
    problems = tool.generate_problem_scenarios()

    # Get study guide
    checklist = tool.conceptual_checklist()
    ```
    """)
