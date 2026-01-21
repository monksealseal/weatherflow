"""
Spectral Diagnostics for GCM

Computes energy spectra in spherical harmonics:
- Kinetic energy spectrum E(n)
- Available potential energy spectrum
- Enstrophy spectrum
- Energy flux between scales
"""

import numpy as np
from scipy import fft


class SpectralDiagnostics:
    """
    Compute spectral diagnostics for atmospheric fields

    Uses FFT-based approximation for spherical harmonic analysis
    on lat-lon grids. For true spherical harmonic decomposition,
    one would use libraries like SHTns or pyshtools.
    """

    def __init__(self, grid, vgrid):
        """
        Initialize spectral diagnostics

        Parameters
        ----------
        grid : SphericalGrid
            Horizontal grid
        vgrid : VerticalGrid
            Vertical grid
        """
        self.grid = grid
        self.vgrid = vgrid
        self.nlat = grid.nlat
        self.nlon = grid.nlon
        self.nlev = vgrid.nlev

        # Precompute Gaussian weights for latitude integration
        # For proper spherical harmonics, these should be Gaussian quadrature weights
        self.lat_weights = np.cos(grid.lat)
        self.lat_weights /= np.sum(self.lat_weights)

        # Maximum total wavenumber
        self.n_max = min(self.nlat // 2, self.nlon // 2)

    def compute_kinetic_energy_spectrum(self, u, v, level=None):
        """
        Compute kinetic energy spectrum E(n) where n is total wavenumber

        For spherical harmonics: KE(n) = sum_m |u_nm|^2 + |v_nm|^2

        We approximate this using 2D FFT and binning by total wavenumber.

        Parameters
        ----------
        u : ndarray
            Zonal wind (nlev, nlat, nlon) or (nlat, nlon)
        v : ndarray
            Meridional wind, same shape as u
        level : int, optional
            Vertical level to analyze. If None, use vertically averaged.

        Returns
        -------
        wavenumbers : ndarray
            Total wavenumber array
        ke_spectrum : ndarray
            Kinetic energy spectrum E(n)
        """
        if u.ndim == 3:
            if level is not None:
                u_2d = u[level]
                v_2d = v[level]
            else:
                # Vertical average
                u_2d = np.mean(u, axis=0)
                v_2d = np.mean(v, axis=0)
        else:
            u_2d = u
            v_2d = v

        # Apply latitude weighting for proper spherical area weighting
        u_weighted = u_2d * np.sqrt(np.cos(self.grid.lat2d))
        v_weighted = v_2d * np.sqrt(np.cos(self.grid.lat2d))

        # 2D FFT
        u_hat = fft.fft2(u_weighted)
        v_hat = fft.fft2(v_weighted)

        # Power spectrum
        power_u = np.abs(u_hat)**2
        power_v = np.abs(v_hat)**2
        total_power = 0.5 * (power_u + power_v)

        # Bin by total wavenumber
        # Create wavenumber arrays
        k_lat = fft.fftfreq(self.nlat, d=1.0/self.nlat)
        k_lon = fft.fftfreq(self.nlon, d=1.0/self.nlon)
        kk_lon, kk_lat = np.meshgrid(k_lon, k_lat)
        k_total = np.sqrt(kk_lat**2 + kk_lon**2)

        # Bin into total wavenumber shells
        wavenumbers = np.arange(0, self.n_max + 1)
        ke_spectrum = np.zeros(len(wavenumbers))

        for i, n in enumerate(wavenumbers):
            mask = (k_total >= n - 0.5) & (k_total < n + 0.5)
            ke_spectrum[i] = np.sum(total_power[mask])

        # Normalize
        ke_spectrum /= (self.nlat * self.nlon)**2

        return wavenumbers, ke_spectrum

    def compute_enstrophy_spectrum(self, vorticity, level=None):
        """
        Compute enstrophy spectrum Z(n) = n(n+1) * E(n) / R^2

        Enstrophy is the integral of squared vorticity.

        Parameters
        ----------
        vorticity : ndarray
            Relative vorticity field (nlev, nlat, nlon) or (nlat, nlon)
        level : int, optional
            Vertical level to analyze

        Returns
        -------
        wavenumbers : ndarray
            Total wavenumber array
        enstrophy_spectrum : ndarray
            Enstrophy spectrum Z(n)
        """
        if vorticity.ndim == 3:
            if level is not None:
                vort_2d = vorticity[level]
            else:
                vort_2d = np.mean(vorticity, axis=0)
        else:
            vort_2d = vorticity

        # Weight by latitude
        vort_weighted = vort_2d * np.sqrt(np.cos(self.grid.lat2d))

        # 2D FFT
        vort_hat = fft.fft2(vort_weighted)
        power = np.abs(vort_hat)**2

        # Bin by total wavenumber
        k_lat = fft.fftfreq(self.nlat, d=1.0/self.nlat)
        k_lon = fft.fftfreq(self.nlon, d=1.0/self.nlon)
        kk_lon, kk_lat = np.meshgrid(k_lon, k_lat)
        k_total = np.sqrt(kk_lat**2 + kk_lon**2)

        wavenumbers = np.arange(0, self.n_max + 1)
        enstrophy_spectrum = np.zeros(len(wavenumbers))

        for i, n in enumerate(wavenumbers):
            mask = (k_total >= n - 0.5) & (k_total < n + 0.5)
            enstrophy_spectrum[i] = np.sum(power[mask])

        enstrophy_spectrum /= (self.nlat * self.nlon)**2

        return wavenumbers, enstrophy_spectrum

    def compute_temperature_variance_spectrum(self, T, level=None):
        """
        Compute temperature variance spectrum

        Related to available potential energy (APE) spectrum.

        Parameters
        ----------
        T : ndarray
            Temperature field
        level : int, optional
            Vertical level to analyze

        Returns
        -------
        wavenumbers : ndarray
        T_variance_spectrum : ndarray
        """
        if T.ndim == 3:
            if level is not None:
                T_2d = T[level]
            else:
                T_2d = np.mean(T, axis=0)
        else:
            T_2d = T

        # Remove zonal mean
        T_prime = T_2d - np.mean(T_2d, axis=1, keepdims=True)

        # Weight and FFT
        T_weighted = T_prime * np.sqrt(np.cos(self.grid.lat2d))
        T_hat = fft.fft2(T_weighted)
        power = np.abs(T_hat)**2

        # Bin by wavenumber
        k_lat = fft.fftfreq(self.nlat, d=1.0/self.nlat)
        k_lon = fft.fftfreq(self.nlon, d=1.0/self.nlon)
        kk_lon, kk_lat = np.meshgrid(k_lon, k_lat)
        k_total = np.sqrt(kk_lat**2 + kk_lon**2)

        wavenumbers = np.arange(0, self.n_max + 1)
        T_spectrum = np.zeros(len(wavenumbers))

        for i, n in enumerate(wavenumbers):
            mask = (k_total >= n - 0.5) & (k_total < n + 0.5)
            T_spectrum[i] = np.sum(power[mask])

        T_spectrum /= (self.nlat * self.nlon)**2

        return wavenumbers, T_spectrum

    def compute_all_spectra(self, state, level=None):
        """
        Compute all spectral diagnostics

        Parameters
        ----------
        state : ModelState
            Current model state
        level : int, optional
            Vertical level (default: mid-troposphere)

        Returns
        -------
        spectra : dict
            Dictionary containing all computed spectra
        """
        if level is None:
            level = self.nlev // 2

        # Compute vorticity
        vorticity = np.zeros_like(state.u)
        for k in range(self.nlev):
            vorticity[k] = self.grid.vorticity(state.u[k], state.v[k])

        # Compute spectra
        wn_ke, ke_spec = self.compute_kinetic_energy_spectrum(state.u, state.v, level)
        wn_z, z_spec = self.compute_enstrophy_spectrum(vorticity, level)
        wn_t, t_spec = self.compute_temperature_variance_spectrum(state.T, level)

        # Theoretical slopes for comparison
        # KE spectrum: k^-3 for 2D turbulence (enstrophy cascade)
        #              k^-5/3 for 3D turbulence (energy cascade)
        k_3_slope = ke_spec[5] * (wn_ke[5:] / wn_ke[5])**(-3)
        k_53_slope = ke_spec[5] * (wn_ke[5:] / wn_ke[5])**(-5/3)

        return {
            'wavenumbers': wn_ke,
            'kinetic_energy_spectrum': ke_spec,
            'enstrophy_spectrum': z_spec,
            'temperature_variance_spectrum': t_spec,
            'k_minus_3_reference': k_3_slope,
            'k_minus_5_3_reference': k_53_slope,
            'level': level
        }

    def compute_zonal_wavenumber_spectrum(self, field, lat_index=None):
        """
        Compute spectrum as function of zonal wavenumber at fixed latitude

        Parameters
        ----------
        field : ndarray
            2D field (nlat, nlon)
        lat_index : int, optional
            Latitude index. Default is equator.

        Returns
        -------
        zonal_wavenumbers : ndarray
        spectrum : ndarray
        """
        if lat_index is None:
            lat_index = self.nlat // 2

        # Extract latitude band
        field_1d = field[lat_index, :]

        # FFT
        field_hat = fft.fft(field_1d)
        power = np.abs(field_hat[:self.nlon//2])**2

        zonal_wavenumbers = np.arange(self.nlon // 2)

        return zonal_wavenumbers, power / self.nlon**2
