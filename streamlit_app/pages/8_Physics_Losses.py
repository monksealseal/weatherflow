"""
Physics-Based Loss Functions

Uses the actual PhysicsLossCalculator class from weatherflow/physics/losses.py
Demonstrates physics constraints that can be applied to ERA5 or synthetic data.
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))

from weatherflow.physics.losses import PhysicsLossCalculator

# Import ERA5 utilities
try:
    from era5_utils import (
        get_era5_data_banner,
        has_era5_data,
        get_active_era5_data,
        get_era5_wind_data,
        get_era5_slice,
    )
    ERA5_UTILS_AVAILABLE = True
except ImportError:
    ERA5_UTILS_AVAILABLE = False

st.set_page_config(page_title="Physics Loss Functions", page_icon="⚛️", layout="wide")

st.title("⚛️ Physics-Based Loss Functions")
st.markdown("""
Explore the physics constraints used to regularize weather prediction models.
This uses the actual `PhysicsLossCalculator` class from the repository.
""")

# Show data source banner
if ERA5_UTILS_AVAILABLE:
    banner = get_era5_data_banner()
    st.info(f"""
    📊 **Data Note:** Physics constraints are demonstrated with synthetic wind fields.
    These same constraints can be applied to ERA5 data to ensure physical consistency
    in machine learning weather predictions.
    """)

# Create calculator
calculator = PhysicsLossCalculator()

# Sidebar
st.sidebar.header("Physical Constants")
st.sidebar.markdown(f"""
- **Earth Radius**: {calculator.earth_radius/1e6:.3f} × 10⁶ m
- **Gravity**: {calculator.gravity:.2f} m/s²
- **Ω (rotation)**: {calculator.omega:.2e} rad/s
- **f₀**: {calculator.f0:.2e} s⁻¹
- **β**: {calculator.beta:.2e} m⁻¹s⁻¹
- **N²**: {calculator.N_squared:.2e} s⁻²
""")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Divergence Loss",
    "🌀 PV Conservation",
    "📈 Energy Spectra",
    "⚖️ Geostrophic Balance",
    "🧮 Combined Losses",
    "🌍 ERA5 Physics"
])

# Tab 1: Divergence Loss
with tab1:
    st.header("Mass-Weighted Divergence Loss")

    st.markdown("""
    Mass conservation requires the vertically-integrated divergence to vanish:

    $$\\int \\nabla \\cdot (\\rho \\mathbf{u}) \\, dp = 0$$

    This loss penalizes non-zero column-integrated divergence.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Wind Field Parameters")

        div_type = st.selectbox(
            "Flow Type",
            ["Nondivergent (Rotational)", "Divergent (Radial)", "Mixed"]
        )

        amplitude = st.slider("Wind Amplitude (m/s)", 5.0, 50.0, 20.0)
        n_levels = st.slider("Vertical Levels", 1, 10, 5)
        grid_size = st.slider("Grid Size", 16, 64, 32, key="div_grid")

    with col2:
        # Generate wind field
        np.random.seed(42)
        lat_dim, lon_dim = grid_size, grid_size * 2
        batch_size = 1

        lats = np.linspace(-np.pi/2, np.pi/2, lat_dim)
        lons = np.linspace(0, 2*np.pi, lon_dim)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

        if div_type == "Nondivergent (Rotational)":
            # Pure rotational flow (geostrophic-like)
            # u = -dψ/dy, v = dψ/dx
            psi = amplitude * np.sin(lat_grid * 2) * np.cos(lon_grid * 2)
            u = -amplitude * 2 * np.cos(lat_grid * 2) * np.cos(lon_grid * 2)
            v = -amplitude * 2 * np.sin(lat_grid * 2) * np.sin(lon_grid * 2)
        elif div_type == "Divergent (Radial)":
            # Radial outflow (divergent)
            cx, cy = np.pi, 0
            dx = lon_grid - cx
            dy = lat_grid - cy
            r = np.sqrt(dx**2 + dy**2) + 0.1
            u = amplitude * dx / r * np.exp(-r)
            v = amplitude * dy / r * np.exp(-r)
        else:
            # Mixed flow
            psi = amplitude * np.sin(lat_grid * 2) * np.cos(lon_grid * 2)
            u = -amplitude * np.cos(lat_grid * 2) * np.cos(lon_grid * 2)
            v = -amplitude * np.sin(lat_grid * 2) * np.sin(lon_grid * 2)
            # Add divergent component
            u += amplitude * 0.3 * lat_grid
            v += amplitude * 0.3 * lon_grid / 2

        # Expand to batch and level dimensions
        u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        v_tensor = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        if n_levels > 1:
            u_tensor = u_tensor.repeat(1, n_levels, 1, 1)
            v_tensor = v_tensor.repeat(1, n_levels, 1, 1)
            # Add some vertical structure
            for k in range(n_levels):
                factor = 1 + 0.1 * k
                u_tensor[:, k] *= factor
                v_tensor[:, k] *= factor

        pressure_levels = torch.linspace(1000, 200, n_levels) if n_levels > 1 else None

        # Compute loss
        div_loss = calculator.compute_mass_weighted_divergence_loss(
            u_tensor, v_tensor, pressure_levels
        )

        # Compute divergence field for visualization
        du_dx = np.gradient(u, axis=1) / (calculator.earth_radius * np.cos(lat_grid))
        dv_dy = np.gradient(v, axis=0) / calculator.earth_radius
        divergence = du_dx + dv_dy

        # Display
        st.metric("Divergence Loss", f"{div_loss.item():.6f}")

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('U Wind (m/s)', 'V Wind (m/s)', 'Divergence (s⁻¹)')
        )

        fig.add_trace(
            go.Heatmap(z=u, colorscale='RdBu_r', zmid=0, showscale=True,
                      colorbar=dict(title='m/s', x=0.28)),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=v, colorscale='RdBu_r', zmid=0, showscale=True,
                      colorbar=dict(title='m/s', x=0.63)),
            row=1, col=2
        )
        fig.add_trace(
            go.Heatmap(z=divergence * 1e5, colorscale='RdBu_r', zmid=0, showscale=True,
                      colorbar=dict(title='×10⁻⁵ s⁻¹')),
            row=1, col=3
        )

        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        **Analysis:**
        - Mean Divergence: {np.mean(divergence):.2e} s⁻¹
        - Max |Divergence|: {np.max(np.abs(divergence)):.2e} s⁻¹
        - RMS Divergence: {np.sqrt(np.mean(divergence**2)):.2e} s⁻¹
        """)

# Tab 2: PV Conservation
with tab2:
    st.header("Potential Vorticity Conservation Loss")

    st.markdown("""
    Quasi-geostrophic PV is materially conserved:

    $$q = \\nabla^2\\psi + \\frac{f_0^2}{N^2}\\frac{\\partial^2\\psi}{\\partial p^2} + f$$

    $$\\frac{Dq}{Dt} = 0$$

    This loss penalizes PV variance and gradients.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")

        pv_flow_type = st.selectbox(
            "Vorticity Pattern",
            ["Uniform Vorticity", "Isolated Vortex", "Wave Pattern", "Random"]
        )

        vorticity_amplitude = st.slider("Vorticity Amplitude (×10⁻⁵ s⁻¹)", 0.5, 5.0, 2.0)
        pv_levels = st.slider("Vertical Levels", 2, 8, 4, key="pv_levels")
        pv_grid = st.slider("Grid Size", 16, 64, 32, key="pv_grid")

    with col2:
        np.random.seed(123)
        lat_dim, lon_dim = pv_grid, pv_grid * 2

        lats = np.linspace(-np.pi/2, np.pi/2, lat_dim)
        lons = np.linspace(0, 2*np.pi, lon_dim)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

        amp = vorticity_amplitude * 1e-5

        if pv_flow_type == "Uniform Vorticity":
            vorticity = amp * np.ones_like(lat_grid)
        elif pv_flow_type == "Isolated Vortex":
            cx, cy = np.pi, 0
            r = np.sqrt((lon_grid - cx)**2 + lat_grid**2)
            vorticity = amp * np.exp(-(r / 0.5)**2)
        elif pv_flow_type == "Wave Pattern":
            vorticity = amp * np.sin(lat_grid * 4) * np.cos(lon_grid * 3)
        else:
            vorticity = amp * np.random.randn(lat_dim, lon_dim)

        # Convert vorticity to u, v (approximately)
        # ζ = dv/dx - du/dy
        # Use streamfunction approach
        psi = np.zeros_like(vorticity)
        for _ in range(50):  # Simple Jacobi iteration for Poisson equation
            psi_new = np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) + \
                     np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1)
            psi_new = (psi_new - vorticity * (np.pi/lat_dim)**2) / 4
            psi = 0.5 * psi + 0.5 * psi_new

        u = -np.gradient(psi, axis=0) / (np.pi/lat_dim)
        v = np.gradient(psi, axis=1) / (np.pi/lat_dim)

        u = u * calculator.earth_radius
        v = v * calculator.earth_radius

        # Create tensors
        u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        v_tensor = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        u_tensor = u_tensor.repeat(1, pv_levels, 1, 1)
        v_tensor = v_tensor.repeat(1, pv_levels, 1, 1)

        pressure_levels = torch.linspace(1000, 200, pv_levels)

        # Compute PV loss
        pv_loss = calculator.compute_pv_conservation_loss(
            u_tensor, v_tensor, pressure_levels=pressure_levels
        )

        st.metric("PV Conservation Loss", f"{pv_loss.item():.6f}")

        # Visualize
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Streamfunction', 'Relative Vorticity', 'Abs. Vorticity + βy')
        )

        fig.add_trace(
            go.Heatmap(z=psi, colorscale='RdBu_r', showscale=True,
                      colorbar=dict(title='ψ', x=0.28)),
            row=1, col=1
        )

        fig.add_trace(
            go.Heatmap(z=vorticity * 1e5, colorscale='RdBu_r', zmid=0, showscale=True,
                      colorbar=dict(title='ζ×10⁵', x=0.63)),
            row=1, col=2
        )

        f = 2 * calculator.omega * np.sin(lat_grid)
        abs_vort = vorticity + f + calculator.beta * lat_grid * calculator.earth_radius
        fig.add_trace(
            go.Heatmap(z=abs_vort * 1e5, colorscale='Viridis', showscale=True,
                      colorbar=dict(title='q×10⁵')),
            row=1, col=3
        )

        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Energy Spectra
with tab3:
    st.header("Energy Spectra Regularization")

    st.markdown("""
    Atmospheric turbulence exhibits characteristic spectral slopes:
    - **Enstrophy cascade** (small scales): E(k) ~ k⁻³
    - **Energy cascade** (large scales): E(k) ~ k⁻⁵/³

    This loss penalizes deviation from the expected spectral slope.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")

        target_slope = st.slider("Target Spectral Slope", -5.0, -1.0, -3.0)

        spectrum_type = st.selectbox(
            "Flow Type",
            ["k⁻³ (Enstrophy Cascade)", "k⁻⁵/³ (Energy Cascade)",
             "White Noise", "Red Noise", "Custom"]
        )

        if spectrum_type == "Custom":
            custom_slope = st.slider("Custom Slope", -5.0, 0.0, -2.5)
        else:
            custom_slope = None

        spec_grid = st.slider("Grid Size", 32, 128, 64, key="spec_grid")

    with col2:
        np.random.seed(456)
        n = spec_grid

        # Generate field with specific spectral properties
        kx = np.fft.fftfreq(n)
        ky = np.fft.fftfreq(n)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
        k_mag[0, 0] = 1e-10  # Avoid division by zero

        if spectrum_type == "k⁻³ (Enstrophy Cascade)":
            power_spectrum = k_mag**(-3)
        elif spectrum_type == "k⁻⁵/³ (Energy Cascade)":
            power_spectrum = k_mag**(-5/3)
        elif spectrum_type == "White Noise":
            power_spectrum = np.ones_like(k_mag)
        elif spectrum_type == "Red Noise":
            power_spectrum = k_mag**(-2)
        else:
            power_spectrum = k_mag**(custom_slope)

        # Generate random phases
        phases = np.random.rand(n, n) * 2 * np.pi
        spectrum = np.sqrt(power_spectrum) * np.exp(1j * phases)
        spectrum[0, 0] = 0  # Zero mean

        # Inverse FFT to get field
        u_field = np.real(np.fft.ifft2(spectrum))
        v_field = np.real(np.fft.ifft2(np.sqrt(power_spectrum) * np.exp(1j * np.random.rand(n, n) * 2 * np.pi)))

        # Normalize
        u_field = u_field / u_field.std() * 20
        v_field = v_field / v_field.std() * 20

        # Create tensors
        u_tensor = torch.tensor(u_field, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        v_tensor = torch.tensor(v_field, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Compute loss
        spectra_loss = calculator.compute_energy_spectra_loss(
            u_tensor, v_tensor, target_slope=target_slope
        )

        st.metric("Spectra Loss", f"{spectra_loss.item():.6f}")

        # Compute actual spectrum
        ke = 0.5 * (u_field**2 + v_field**2)
        ke_fft = np.abs(np.fft.fft2(ke))**2

        # Radial averaging
        k_bins = np.linspace(1, n//4, 30)
        radial_spectrum = []
        for i in range(len(k_bins) - 1):
            mask = (k_mag >= k_bins[i]/n) & (k_mag < k_bins[i+1]/n)
            if mask.sum() > 0:
                radial_spectrum.append(ke_fft[mask].mean())
            else:
                radial_spectrum.append(np.nan)

        k_centers = (k_bins[:-1] + k_bins[1:]) / 2

        # Plotting
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Kinetic Energy Field', 'Energy Spectrum')
        )

        fig.add_trace(
            go.Heatmap(z=ke, colorscale='hot', showscale=True,
                      colorbar=dict(title='KE', x=0.45)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.log10(k_centers), y=np.log10(radial_spectrum),
                      mode='markers+lines', name='Actual',
                      marker=dict(color='#1e88e5')),
            row=1, col=2
        )

        # Add reference slopes
        x_ref = np.log10(k_centers)
        y_ref_3 = -3 * x_ref + np.log10(radial_spectrum[5]) + 3 * x_ref[5]
        y_ref_53 = -5/3 * x_ref + np.log10(radial_spectrum[5]) + 5/3 * x_ref[5]

        fig.add_trace(
            go.Scatter(x=x_ref, y=y_ref_3, mode='lines',
                      name='k⁻³', line=dict(dash='dash', color='red')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=x_ref, y=y_ref_53, mode='lines',
                      name='k⁻⁵/³', line=dict(dash='dash', color='green')),
            row=1, col=2
        )

        fig.update_xaxes(title_text='log₁₀(k)', row=1, col=2)
        fig.update_yaxes(title_text='log₁₀(E)', row=1, col=2)
        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Geostrophic Balance
with tab4:
    st.header("Geostrophic Balance Loss")

    st.markdown("""
    In quasi-geostrophic theory, the wind should be in approximate balance
    with the pressure gradient:

    $$f u_g = -\\frac{\\partial \\Phi}{\\partial y}$$
    $$f v_g = \\frac{\\partial \\Phi}{\\partial x}$$

    This loss penalizes departure from geostrophic balance.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")

        balance_type = st.selectbox(
            "Flow Type",
            ["Geostrophic (Balanced)", "Ageostrophic (Unbalanced)", "Mixed"]
        )

        geo_amplitude = st.slider("Flow Amplitude", 10.0, 50.0, 30.0)
        geo_grid = st.slider("Grid Size", 16, 64, 32, key="geo_grid")

    with col2:
        np.random.seed(789)
        lat_dim, lon_dim = geo_grid, geo_grid * 2

        lats = np.linspace(-np.pi/4, np.pi/4, lat_dim)  # Mid-latitudes
        lons = np.linspace(0, 2*np.pi, lon_dim)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

        # Geopotential field
        phi = 50000 + 2000 * np.sin(lat_grid * 2) * np.cos(lon_grid * 3)

        # Coriolis parameter
        f = 2 * calculator.omega * np.sin(lat_grid)

        # Compute geostrophic wind
        dphi_dy = np.gradient(phi, axis=0) / (calculator.earth_radius * np.pi / (2 * lat_dim))
        dphi_dx = np.gradient(phi, axis=1) / (calculator.earth_radius * np.cos(lat_grid) * 2 * np.pi / lon_dim)

        u_geo = -dphi_dy / (f + 1e-8)
        v_geo = dphi_dx / (f + 1e-8)

        if balance_type == "Geostrophic (Balanced)":
            u = u_geo
            v = v_geo
        elif balance_type == "Ageostrophic (Unbalanced)":
            u = u_geo + geo_amplitude * np.random.randn(lat_dim, lon_dim)
            v = v_geo + geo_amplitude * np.random.randn(lat_dim, lon_dim)
        else:
            u = u_geo + 0.3 * geo_amplitude * np.sin(lon_grid * 5)
            v = v_geo + 0.3 * geo_amplitude * np.cos(lat_grid * 5)

        # Create tensors
        u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        v_tensor = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        phi_tensor = torch.tensor(phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Compute loss
        geo_loss = calculator.compute_geostrophic_balance_loss(u_tensor, v_tensor, phi_tensor)

        st.metric("Geostrophic Balance Loss", f"{geo_loss.item():.4f}")

        # Ageostrophic component
        u_ageo = u - u_geo
        v_ageo = v - v_geo
        ageo_mag = np.sqrt(u_ageo**2 + v_ageo**2)

        st.metric("Mean Ageostrophic Wind", f"{np.mean(ageo_mag):.2f} m/s")

        # Visualize
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Geopotential (m²/s²)', 'Actual Wind Speed',
                          'Geostrophic Wind Speed', 'Ageostrophic Component')
        )

        fig.add_trace(
            go.Heatmap(z=phi, colorscale='viridis', showscale=True,
                      colorbar=dict(title='Φ', x=0.45, y=0.78, len=0.4)),
            row=1, col=1
        )

        wind_speed = np.sqrt(u**2 + v**2)
        fig.add_trace(
            go.Heatmap(z=wind_speed, colorscale='YlOrRd', showscale=True,
                      colorbar=dict(title='m/s', x=1.0, y=0.78, len=0.4)),
            row=1, col=2
        )

        geo_speed = np.sqrt(u_geo**2 + v_geo**2)
        fig.add_trace(
            go.Heatmap(z=geo_speed, colorscale='YlOrRd', showscale=True,
                      colorbar=dict(title='m/s', x=0.45, y=0.22, len=0.4)),
            row=2, col=1
        )

        fig.add_trace(
            go.Heatmap(z=ageo_mag, colorscale='Reds', showscale=True,
                      colorbar=dict(title='m/s', x=1.0, y=0.22, len=0.4)),
            row=2, col=2
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# Tab 5: Combined Losses
with tab5:
    st.header("Combined Physics Losses")

    st.markdown("""
    In practice, multiple physics constraints are combined with configurable weights.
    Explore how different weighting strategies affect the total loss.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Loss Weights")

        w_pv = st.slider("PV Conservation", 0.0, 1.0, 0.1)
        w_spectra = st.slider("Energy Spectra", 0.0, 0.1, 0.01)
        w_div = st.slider("Mass Divergence", 0.0, 2.0, 1.0)
        w_geo = st.slider("Geostrophic Balance", 0.0, 1.0, 0.1)

        weights = {
            'pv_conservation': w_pv,
            'energy_spectra': w_spectra,
            'mass_divergence': w_div,
            'geostrophic_balance': w_geo
        }

        st.markdown("---")
        st.subheader("Test Data")

        comb_grid = st.slider("Grid Size", 16, 64, 32, key="comb_grid")
        comb_levels = st.slider("Vertical Levels", 2, 8, 4, key="comb_levels")

    with col2:
        # Generate test data
        np.random.seed(111)

        lat_dim, lon_dim = comb_grid, comb_grid * 2

        u = 20 * np.sin(np.linspace(-np.pi/2, np.pi/2, lat_dim))[:, np.newaxis] * np.ones(lon_dim)
        u += 5 * np.random.randn(lat_dim, lon_dim)

        v = 5 * np.cos(np.linspace(0, 2*np.pi, lon_dim)) * np.ones((lat_dim, 1))
        v += 3 * np.random.randn(lat_dim, lon_dim)

        phi = 50000 + 1000 * np.random.randn(lat_dim, lon_dim)

        # Tensors
        u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        v_tensor = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        phi_tensor = torch.tensor(phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        u_tensor = u_tensor.repeat(1, comb_levels, 1, 1)
        v_tensor = v_tensor.repeat(1, comb_levels, 1, 1)
        phi_tensor = phi_tensor.repeat(1, comb_levels, 1, 1)

        pressure_levels = torch.linspace(1000, 200, comb_levels)

        # Compute all losses
        all_losses = calculator.compute_all_physics_losses(
            u_tensor, v_tensor,
            geopotential=phi_tensor,
            pressure_levels=pressure_levels,
            loss_weights=weights
        )

        # Display results
        st.subheader("Individual Losses")

        loss_data = []
        for name, value in all_losses.items():
            if name != 'physics_total':
                loss_data.append({'Loss': name, 'Value': value.item()})

        fig = go.Figure()

        names = [d['Loss'] for d in loss_data]
        values = [d['Value'] for d in loss_data]
        weighted_values = [d['Value'] * weights.get(d['Loss'], 1) for d in loss_data]

        fig.add_trace(go.Bar(
            x=names, y=values,
            name='Raw Loss',
            marker_color='#1e88e5'
        ))
        fig.add_trace(go.Bar(
            x=names, y=weighted_values,
            name='Weighted Loss',
            marker_color='#7c4dff'
        ))

        fig.update_layout(
            title='Physics Loss Components',
            xaxis_title='Loss Type',
            yaxis_title='Loss Value',
            barmode='group',
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

        # Total loss
        st.metric("Total Physics Loss", f"{all_losses['physics_total'].item():.6f}")

        # Pie chart of contributions
        contributions = [w * v for w, v in zip(
            [w_pv, w_spectra, w_div, w_geo],
            [all_losses.get('pv_conservation', torch.tensor(0)).item(),
             all_losses.get('energy_spectra', torch.tensor(0)).item(),
             all_losses.get('mass_divergence', torch.tensor(0)).item(),
             all_losses.get('geostrophic_balance', torch.tensor(0)).item()]
        )]

        fig_pie = go.Figure(go.Pie(
            labels=['PV', 'Spectra', 'Divergence', 'Geostrophic'],
            values=[max(0.001, c) for c in contributions],
            marker_colors=['#1e88e5', '#66bb6a', '#ef5350', '#7c4dff']
        ))
        fig_pie.update_layout(title='Loss Contribution', height=300)

        st.plotly_chart(fig_pie, use_container_width=True)

# Tab 6: ERA5 Physics Analysis
with tab6:
    st.header("🌍 Physics Analysis of ERA5 Data")
    
    st.markdown("""
    Apply physics loss functions to **real ERA5 reanalysis data**.
    This demonstrates how to evaluate physical consistency of actual atmospheric observations.
    """)
    
    if ERA5_UTILS_AVAILABLE and has_era5_data():
        data, metadata = get_active_era5_data()
        
        st.success(f"✅ Analyzing Real ERA5 Data: **{metadata.get('name', 'Unknown')}**")
        st.markdown(f"**Period:** {metadata.get('start_date', '?')} to {metadata.get('end_date', '?')}")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Analysis Configuration")
            
            # Time selection
            time_values = data.time.values
            time_options = [str(t)[:19] for t in time_values]
            era5_phys_time = st.selectbox(
                "Select Time",
                options=range(len(time_options)),
                format_func=lambda x: time_options[x],
                key="era5_phys_time"
            )
            
            # Level selection
            if "level" in data.coords:
                levels = sorted([int(l) for l in data.level.values])
                era5_phys_level = st.selectbox(
                    "Pressure Level (hPa)",
                    options=levels,
                    index=min(1, len(levels) - 1),
                    key="era5_phys_level"
                )
            else:
                era5_phys_level = None
            
            st.markdown("---")
            st.markdown("**Analysis Options:**")
            
            analyze_div = st.checkbox("Divergence", value=True, key="era5_analyze_div")
            analyze_vor = st.checkbox("Vorticity", value=True, key="era5_analyze_vor")
            analyze_ke = st.checkbox("Kinetic Energy", value=True, key="era5_analyze_ke")
            
            run_era5_analysis = st.button("🔍 Analyze ERA5 Physics", type="primary", key="era5_phys_btn")
        
        with col2:
            if run_era5_analysis:
                st.subheader("Physics Analysis Results")
                
                with st.spinner("Analyzing ERA5 wind field physics..."):
                    try:
                        # Get wind data
                        u_data, v_data, lats, lons = get_era5_wind_data(era5_phys_time, era5_phys_level)
                        
                        if u_data is not None and v_data is not None:
                            # Compute physics diagnostics
                            results = {}
                            
                            if analyze_div:
                                # Divergence
                                du_dx = np.gradient(u_data, axis=1) / (calculator.earth_radius * np.cos(np.radians(lats[:, np.newaxis])))
                                dv_dy = np.gradient(v_data, axis=0) / calculator.earth_radius
                                divergence = du_dx + dv_dy
                                results['Divergence'] = divergence
                            
                            if analyze_vor:
                                # Relative vorticity
                                dv_dx = np.gradient(v_data, axis=1) / (calculator.earth_radius * np.cos(np.radians(lats[:, np.newaxis])))
                                du_dy = np.gradient(u_data, axis=0) / calculator.earth_radius
                                vorticity = dv_dx - du_dy
                                results['Vorticity'] = vorticity
                            
                            if analyze_ke:
                                # Kinetic energy
                                kinetic_energy = 0.5 * (u_data**2 + v_data**2)
                                results['Kinetic Energy'] = kinetic_energy
                            
                            # Visualize results
                            n_plots = len(results)
                            if n_plots > 0:
                                n_cols = min(2, n_plots)
                                n_rows = (n_plots + n_cols - 1) // n_cols
                                
                                fig = make_subplots(
                                    rows=n_rows, cols=n_cols,
                                    subplot_titles=list(results.keys())
                                )
                                
                                colorscales = {
                                    'Divergence': 'RdBu_r',
                                    'Vorticity': 'RdBu_r',
                                    'Kinetic Energy': 'Viridis'
                                }
                                
                                for idx, (name, field) in enumerate(results.items()):
                                    row = idx // n_cols + 1
                                    col = idx % n_cols + 1
                                    
                                    # Scale for visualization
                                    if name in ['Divergence', 'Vorticity']:
                                        display_field = field * 1e5  # Scale to 10^-5
                                    else:
                                        display_field = field
                                    
                                    fig.add_trace(
                                        go.Heatmap(
                                            z=display_field,
                                            x=lons,
                                            y=lats,
                                            colorscale=colorscales.get(name, 'Viridis'),
                                            showscale=True,
                                            zmid=0 if name in ['Divergence', 'Vorticity'] else None
                                        ),
                                        row=row, col=col
                                    )
                                
                                title = f"ERA5 Physics Analysis - {time_options[era5_phys_time]}"
                                if era5_phys_level:
                                    title += f" at {era5_phys_level} hPa"
                                
                                fig.update_layout(
                                    title=title,
                                    height=300 * n_rows
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Statistics
                                st.subheader("📊 Physics Statistics")
                                
                                stat_cols = st.columns(len(results))
                                for idx, (name, field) in enumerate(results.items()):
                                    with stat_cols[idx]:
                                        st.markdown(f"**{name}**")
                                        if name in ['Divergence', 'Vorticity']:
                                            st.metric("Mean (×10⁻⁵)", f"{np.mean(field) * 1e5:.4f}")
                                            st.metric("RMS (×10⁻⁵)", f"{np.sqrt(np.mean(field**2)) * 1e5:.4f}")
                                            st.metric("Max |field| (×10⁻⁵)", f"{np.max(np.abs(field)) * 1e5:.4f}")
                                        else:
                                            st.metric("Mean", f"{np.mean(field):.2f}")
                                            st.metric("Max", f"{np.max(field):.2f}")
                                            st.metric("Total", f"{np.sum(field):.2e}")
                                
                                # Physical interpretation
                                st.markdown("---")
                                st.subheader("📝 Physical Interpretation")
                                
                                if analyze_div and 'Divergence' in results:
                                    div_rms = np.sqrt(np.mean(results['Divergence']**2)) * 1e5
                                    if div_rms < 0.5:
                                        st.success(f"✅ **Divergence**: Low RMS divergence ({div_rms:.3f} × 10⁻⁵ s⁻¹) indicates near-balanced flow")
                                    else:
                                        st.info(f"ℹ️ **Divergence**: RMS divergence ({div_rms:.3f} × 10⁻⁵ s⁻¹) indicates significant convergence/divergence regions")
                                
                                if analyze_vor and 'Vorticity' in results:
                                    vor_max = np.max(np.abs(results['Vorticity'])) * 1e5
                                    if vor_max > 5:
                                        st.info(f"🌀 **Vorticity**: High max vorticity ({vor_max:.2f} × 10⁻⁵ s⁻¹) suggests strong cyclonic/anticyclonic features")
                                    else:
                                        st.success(f"✅ **Vorticity**: Moderate vorticity ({vor_max:.2f} × 10⁻⁵ s⁻¹)")
                                
                                if analyze_ke and 'Kinetic Energy' in results:
                                    ke_mean = np.mean(results['Kinetic Energy'])
                                    st.info(f"⚡ **Kinetic Energy**: Mean KE = {ke_mean:.1f} m²/s² (wind speed ≈ {np.sqrt(2*ke_mean):.1f} m/s)")
                        
                        else:
                            st.warning("Wind components (u, v) not found in the dataset.")
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            else:
                st.info("""
                Click **'Analyze ERA5 Physics'** to compute:
                
                - **Divergence**: Measures mass convergence/divergence
                - **Vorticity**: Measures rotational motion
                - **Kinetic Energy**: Measures flow intensity
                
                These diagnostics help assess the physical consistency of the ERA5 data
                and can be used as loss functions during model training.
                """)
    
    else:
        st.warning("""
        ⚠️ **ERA5 Data Not Available**
        
        To analyze physics of real ERA5 data:
        1. Go to the **📊 Data Manager** page
        2. Download a sample dataset with wind components
        3. Click "Use This Dataset" to activate it
        4. Return here to analyze physical properties
        
        The other tabs demonstrate physics constraints using synthetic data.
        """)

# Code reference
with st.expander("📝 Code Reference"):
    st.markdown("""
    This page uses the following code from the repository:

    ```python
    # From weatherflow/physics/losses.py
    from weatherflow.physics.losses import PhysicsLossCalculator

    # Create calculator
    calculator = PhysicsLossCalculator(
        earth_radius=6371e3,
        gravity=9.81,
        omega=7.292e-5,
        f0=1e-4,
        beta=1.6e-11
    )

    # Individual losses
    div_loss = calculator.compute_mass_weighted_divergence_loss(u, v, pressure_levels)
    pv_loss = calculator.compute_pv_conservation_loss(u, v, geopotential, pressure_levels)
    spectra_loss = calculator.compute_energy_spectra_loss(u, v, target_slope=-3.0)
    geo_loss = calculator.compute_geostrophic_balance_loss(u, v, geopotential)

    # Combined losses
    all_losses = calculator.compute_all_physics_losses(
        u, v,
        geopotential=phi,
        pressure_levels=pressure_levels,
        loss_weights={
            'pv_conservation': 0.1,
            'energy_spectra': 0.01,
            'mass_divergence': 1.0,
            'geostrophic_balance': 0.1
        }
    )
    ```
    """)
