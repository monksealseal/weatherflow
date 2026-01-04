"""Enhanced physics-based loss functions for atmospheric flow matching.

This module implements advanced physics constraints beyond basic divergence:
- Potential vorticity (PV) conservation
- Energy spectra regularization (enstrophy cascade)
- Mass-weighted column divergence
- Geostrophic balance constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


class PhysicsLossCalculator(nn.Module):
    """Calculate physics-based loss terms for atmospheric flows.

    Implements constraints from fundamental atmospheric dynamics:
    - PV conservation (quasi-geostrophic approximation)
    - Energy spectra matching (k^-3 enstrophy cascade)
    - Mass conservation in vertical columns
    """

    def __init__(
        self,
        earth_radius: float = 6371e3,  # meters
        gravity: float = 9.81,         # m/s^2
        omega: float = 7.292e-5,       # Earth's angular velocity (rad/s)
        f0: float = 1e-4,              # Reference Coriolis parameter (s^-1)
        beta: float = 1.6e-11,         # Beta parameter (m^-1 s^-1)
        reference_pressure: float = 500.0,  # hPa for PV calculations
        stratification: float = 1e-4,  # N^2, Brunt-Väisälä frequency squared
    ):
        """Initialize physics calculator with Earth constants.

        Args:
            earth_radius: Earth's radius in meters
            gravity: Gravitational acceleration (m/s^2)
            omega: Earth's angular velocity (rad/s)
            f0: Reference Coriolis parameter at mid-latitudes
            beta: df/dy beta-plane approximation
            reference_pressure: Reference pressure level for QG-PV (hPa)
            stratification: Static stability N^2 (s^-2)
        """
        super().__init__()
        self.earth_radius = earth_radius
        self.gravity = gravity
        self.omega = omega
        self.f0 = f0
        self.beta = beta
        self.p_ref = reference_pressure
        self.N_squared = stratification

    def coriolis_parameter(self, lat: torch.Tensor) -> torch.Tensor:
        """Calculate Coriolis parameter f = 2Ω sin(φ).

        Args:
            lat: Latitude in radians, shape [..., lat_dim, ...]

        Returns:
            Coriolis parameter (s^-1)
        """
        return 2 * self.omega * torch.sin(lat)

    def compute_pv_conservation_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        geopotential: Optional[torch.Tensor] = None,
        pressure_levels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute potential vorticity conservation loss.

        Uses quasi-geostrophic PV for synoptic-scale flows:
        q = ∇²ψ + f₀²/N² ∂²ψ/∂p² + f

        Where ψ is the geostrophic streamfunction. For a conserved tracer,
        Dq/Dt should be small.

        Args:
            u: Zonal wind (m/s), shape [batch, level, lat, lon]
            v: Meridional wind (m/s), shape [batch, level, lat, lon]
            geopotential: Geopotential height (m), shape [batch, level, lat, lon]
            pressure_levels: Pressure levels (hPa), shape [level]

        Returns:
            Scalar PV conservation loss
        """
        batch, n_levels, lat_dim, lon_dim = u.shape
        device = u.device

        # Create latitude grid (radians)
        lat_grid = torch.linspace(-np.pi/2, np.pi/2, lat_dim, device=device, dtype=u.dtype)
        lon_grid = torch.linspace(0, 2*np.pi, lon_dim, device=device, dtype=u.dtype)

        # Spatial spacing
        dlat = np.pi / (lat_dim - 1)
        dlon = 2 * np.pi / lon_dim

        # Calculate relative vorticity (ζ = ∂v/∂x - ∂u/∂y)
        cos_lat = torch.cos(lat_grid).view(1, 1, lat_dim, 1).clamp(min=1e-8)

        # Metric factors (meters per radian)
        dx = self.earth_radius * cos_lat
        dy = self.earth_radius

        # Finite differences with periodic wrapping in longitude
        dvdx = torch.gradient(v, spacing=(dlon,), dim=3)[0] / dx
        dudy = torch.gradient(u, spacing=(dlat,), dim=2)[0] / dy

        vorticity = dvdx - dudy  # [batch, level, lat, lon]

        # Add planetary vorticity
        f = self.coriolis_parameter(lat_grid).view(1, 1, lat_dim, 1)
        abs_vorticity = vorticity + f

        # Compute stretching term if we have vertical levels
        if n_levels > 1 and pressure_levels is not None:
            # Convert pressure to log-pressure coordinates for vertical derivative
            if pressure_levels.dim() == 1:
                pressure_levels = pressure_levels.view(1, n_levels, 1, 1)

            # Vertical vorticity gradient (simplified stretching term)
            # In QG theory: f₀²/N² · ∂²ψ/∂p²
            # Approximate with vorticity vertical derivative
            dp = torch.diff(pressure_levels, dim=1).mean() * 100  # hPa to Pa

            if n_levels >= 3:
                # Second derivative in pressure coordinates
                vort_p = torch.gradient(vorticity, spacing=(dp,), dim=1)[0]
                vort_pp = torch.gradient(vort_p, spacing=(dp,), dim=1)[0]

                stretching = (self.f0**2 / self.N_squared) * vort_pp
                qg_pv = abs_vorticity + stretching
            else:
                qg_pv = abs_vorticity
        else:
            # Barotropic case: PV = ζ + f
            qg_pv = abs_vorticity

        # PV should be materially conserved: minimize variance in local PV tendency
        # Proxy: minimize spatial variance of PV (assumes smooth PV fields)
        pv_variance = torch.var(qg_pv, dim=(-2, -1)).mean()

        # Also penalize large PV magnitudes at small scales (non-physical)
        # Use spatial gradients as proxy for small-scale structure
        pv_grad_lat = torch.gradient(qg_pv, spacing=(dlat,), dim=2)[0] / dy
        pv_grad_lon = torch.gradient(qg_pv, spacing=(dlon,), dim=3)[0] / dx
        pv_gradient_penalty = (pv_grad_lat**2 + pv_grad_lon**2).mean()

        return pv_variance + 0.1 * pv_gradient_penalty

    def compute_energy_spectra_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        target_slope: float = -3.0,
    ) -> torch.Tensor:
        """Compute energy spectra regularization loss.

        Weather models should exhibit k^(-3) enstrophy cascade in the
        free troposphere. This loss penalizes deviation from the expected
        spectral slope in wavenumber space.

        Args:
            u: Zonal wind (m/s), shape [batch, level, lat, lon]
            v: Meridional wind (m/s), shape [batch, level, lat, lon]
            target_slope: Expected spectral slope (default -3 for enstrophy)

        Returns:
            Scalar spectra loss
        """
        batch, n_levels, lat_dim, lon_dim = u.shape

        # Compute kinetic energy for each level
        ke = 0.5 * (u**2 + v**2)  # [batch, level, lat, lon]

        # Take 2D FFT in spatial dimensions
        # Average over batch and vertical levels for robustness
        ke_mean = ke.mean(dim=(0, 1))  # [lat, lon]

        # 2D FFT
        ke_fft = torch.fft.rfft2(ke_mean, norm='ortho')
        power_spectrum = torch.abs(ke_fft)**2

        # Compute radial spectrum (azimuthally averaged)
        freq_y = torch.fft.fftfreq(lat_dim, d=1.0, device=u.device)
        freq_x = torch.fft.rfftfreq(lon_dim, d=1.0, device=u.device)

        ky = freq_y.view(-1, 1)
        kx = freq_x.view(1, -1)
        k_radial = torch.sqrt(ky**2 + kx**2)

        # Bin into radial wavenumber bins
        k_max = min(lat_dim, lon_dim) // 4  # Avoid aliasing
        k_bins = torch.linspace(1, k_max, 20, device=u.device)

        radial_spectrum = []
        valid_bins = []

        for i in range(len(k_bins) - 1):
            mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])
            if mask.sum() > 0:
                radial_spectrum.append(power_spectrum[mask].mean())
                valid_bins.append((k_bins[i] + k_bins[i+1]) / 2)

        if len(radial_spectrum) < 3:
            # Not enough bins for meaningful slope
            return torch.tensor(0.0, device=u.device)

        spectrum = torch.stack(radial_spectrum)
        k_values = torch.stack(valid_bins)

        # Compute slope in log-log space
        log_k = torch.log(k_values + 1e-8)
        log_spectrum = torch.log(spectrum + 1e-8)

        # Linear regression in log space
        # Slope = Cov(log_k, log_spectrum) / Var(log_k)
        log_k_mean = log_k.mean()
        log_spectrum_mean = log_spectrum.mean()

        numerator = ((log_k - log_k_mean) * (log_spectrum - log_spectrum_mean)).sum()
        denominator = ((log_k - log_k_mean)**2).sum()

        slope = numerator / (denominator + 1e-8)

        # Penalize deviation from target slope
        slope_loss = (slope - target_slope)**2

        return slope_loss

    def compute_mass_weighted_divergence_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        pressure_levels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mass-weighted column-integrated divergence loss.

        Mass conservation requires:
        ∫ ∇·(ρu) dp = 0

        where integration is over a vertical column. This is stronger than
        layer-by-layer divergence constraints.

        Args:
            u: Zonal wind (m/s), shape [batch, level, lat, lon]
            v: Meridional wind (m/s), shape [batch, level, lat, lon]
            pressure_levels: Pressure levels (hPa), shape [level]
                If None, assumes equal weighting

        Returns:
            Scalar mass-weighted divergence loss
        """
        batch, n_levels, lat_dim, lon_dim = u.shape
        device = u.device

        # Latitude/longitude spacing
        lat_grid = torch.linspace(-np.pi/2, np.pi/2, lat_dim, device=device, dtype=u.dtype)
        dlat = np.pi / (lat_dim - 1)
        dlon = 2 * np.pi / lon_dim

        # Metric terms (meters per radian)
        cos_lat = torch.cos(lat_grid).view(1, 1, lat_dim, 1).clamp(min=1e-8)
        dx = self.earth_radius * cos_lat
        dy = self.earth_radius

        # Compute divergence for each level
        dudx = torch.gradient(u, spacing=(dlon,), dim=3)[0] / dx
        dvdy = torch.gradient(v, spacing=(dlat,), dim=2)[0] / dy
        divergence = dudx + dvdy  # [batch, level, lat, lon]

        # Pressure-weighted vertical integration
        if pressure_levels is not None and n_levels > 1:
            # Convert to Pa and compute pressure weights
            if pressure_levels.dim() == 1:
                p_levels = pressure_levels.view(1, n_levels, 1, 1) * 100  # hPa to Pa
            else:
                p_levels = pressure_levels * 100

            # Trapezoidal integration weights
            dp = torch.diff(p_levels, dim=1)
            weights = torch.zeros_like(divergence)

            if n_levels == 1:
                weights[:, 0] = 1.0
            elif n_levels == 2:
                weights[:, 0] = dp[:, 0] / 2
                weights[:, 1] = dp[:, 0] / 2
            else:
                # Interior points
                weights[:, 0] = dp[:, 0] / 2
                for i in range(1, n_levels - 1):
                    weights[:, i] = (dp[:, i-1] + dp[:, i]) / 2
                weights[:, -1] = dp[:, -1] / 2

            # Normalize weights
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

            # Column-integrated divergence
            column_div = (divergence * weights).sum(dim=1)  # [batch, lat, lon]
        else:
            # Simple average if no pressure information
            column_div = divergence.mean(dim=1)  # [batch, lat, lon]

        # Penalize non-zero column divergence
        mass_loss = (column_div**2).mean()

        return mass_loss

    def compute_geostrophic_balance_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        geopotential: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geostrophic balance constraint loss.

        In quasi-geostrophic theory:
        f·u_g = -∂Φ/∂y
        f·v_g = ∂Φ/∂x

        This encourages the wind field to be in approximate geostrophic balance
        with the geopotential field.

        Args:
            u: Zonal wind (m/s), shape [batch, level, lat, lon]
            v: Meridional wind (m/s), shape [batch, level, lat, lon]
            geopotential: Geopotential (m²/s²), shape [batch, level, lat, lon]

        Returns:
            Scalar geostrophic balance loss
        """
        batch, n_levels, lat_dim, lon_dim = u.shape
        device = u.device

        # Latitude grid and Coriolis parameter
        lat_grid = torch.linspace(-np.pi/2, np.pi/2, lat_dim, device=device, dtype=u.dtype)
        f = self.coriolis_parameter(lat_grid).view(1, 1, lat_dim, 1)

        # Spatial derivatives
        dlat = np.pi / (lat_dim - 1)
        dlon = 2 * np.pi / lon_dim

        cos_lat = torch.cos(lat_grid).view(1, 1, lat_dim, 1).clamp(min=1e-8)
        dx = self.earth_radius * cos_lat
        dy = self.earth_radius

        dPhi_dx = torch.gradient(geopotential, spacing=(dlon,), dim=3)[0] / dx
        dPhi_dy = torch.gradient(geopotential, spacing=(dlat,), dim=2)[0] / dy

        # Geostrophic wind components
        u_g = -dPhi_dy / (f + 1e-8)
        v_g = dPhi_dx / (f + 1e-8)

        # Balance loss: actual wind should match geostrophic wind
        balance_loss = F.mse_loss(u, u_g) + F.mse_loss(v, v_g)

        return balance_loss

    def compute_all_physics_losses(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        geopotential: Optional[torch.Tensor] = None,
        pressure_levels: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all physics-based losses with configurable weights.

        Args:
            u: Zonal wind (m/s), shape [batch, level, lat, lon]
            v: Meridional wind (m/s), shape [batch, level, lat, lon]
            geopotential: Geopotential height (m), optional
            pressure_levels: Pressure levels (hPa), optional
            loss_weights: Dictionary of loss weights, defaults to unit weights

        Returns:
            Dictionary of individual and total physics losses
        """
        if loss_weights is None:
            loss_weights = {
                'pv_conservation': 0.1,
                'energy_spectra': 0.01,
                'mass_divergence': 1.0,
                'geostrophic_balance': 0.1,
            }

        losses = {}
        total_loss = torch.tensor(0.0, device=u.device, dtype=u.dtype)

        # PV conservation
        if loss_weights.get('pv_conservation', 0) > 0:
            pv_loss = self.compute_pv_conservation_loss(u, v, geopotential, pressure_levels)
            losses['pv_conservation'] = pv_loss
            total_loss = total_loss + loss_weights['pv_conservation'] * pv_loss

        # Energy spectra
        if loss_weights.get('energy_spectra', 0) > 0:
            spectra_loss = self.compute_energy_spectra_loss(u, v)
            losses['energy_spectra'] = spectra_loss
            total_loss = total_loss + loss_weights['energy_spectra'] * spectra_loss

        # Mass-weighted divergence
        if loss_weights.get('mass_divergence', 0) > 0:
            mass_loss = self.compute_mass_weighted_divergence_loss(u, v, pressure_levels)
            losses['mass_divergence'] = mass_loss
            total_loss = total_loss + loss_weights['mass_divergence'] * mass_loss

        # Geostrophic balance
        if geopotential is not None and loss_weights.get('geostrophic_balance', 0) > 0:
            balance_loss = self.compute_geostrophic_balance_loss(u, v, geopotential)
            losses['geostrophic_balance'] = balance_loss
            total_loss = total_loss + loss_weights['geostrophic_balance'] * balance_loss

        losses['physics_total'] = total_loss
        return losses
