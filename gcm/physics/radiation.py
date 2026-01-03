"""
Radiation scheme with shortwave and longwave components

Implements a comprehensive radiation scheme including:
- Solar radiation (shortwave)
- Thermal radiation (longwave)
- Gas absorption (H2O, CO2, O3)
- Cloud-radiation interactions
- Rayleigh scattering
- Aerosol effects
"""

import numpy as np


class RadiationScheme:
    """
    Comprehensive radiation scheme for atmospheric modeling

    Based on two-stream approximation with multiple spectral bands
    """

    def __init__(self, grid, vgrid, co2_ppmv=400.0):
        """
        Initialize radiation scheme

        Parameters
        ----------
        grid : SphericalGrid
            Horizontal grid
        vgrid : VerticalGrid
            Vertical grid
        co2_ppmv : float
            CO2 concentration in ppmv
        """
        self.grid = grid
        self.vgrid = vgrid

        # Physical constants
        self.sigma_sb = 5.67e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
        self.solar_constant = 1361.0  # Solar constant (W/m^2)

        # Gas concentrations
        self.co2_ppmv = co2_ppmv
        self.o3_column = 300.0  # Dobson units

        # Spectral bands for radiation
        self.n_sw_bands = 4  # Shortwave bands
        self.n_lw_bands = 6  # Longwave bands

        # Initialize band properties
        self._init_spectral_bands()

    def _init_spectral_bands(self):
        """Initialize spectral band properties"""
        # Shortwave bands (wavelength ranges in μm)
        # Band 1: UV-visible (0.2-0.7 μm) - Rayleigh scattering, O3 absorption
        # Band 2: Near-IR (0.7-1.3 μm) - H2O absorption
        # Band 3: IR (1.3-2.5 μm) - H2O, CO2 absorption
        # Band 4: IR (2.5-4.0 μm) - H2O, CO2 absorption

        self.sw_band_weights = np.array([0.5, 0.3, 0.15, 0.05])  # Fraction of solar spectrum

        # Longwave bands (wavenumber ranges in cm^-1)
        # Optimized for atmospheric windows and absorption bands
        self.lw_band_centers = np.array([500, 667, 800, 1000, 1200, 1500])  # cm^-1
        self.lw_band_weights = np.array([0.2, 0.25, 0.2, 0.15, 0.1, 0.1])

    def compute_radiation(self, state, time):
        """
        Compute radiative heating rates

        Parameters
        ----------
        state : ModelState
            Current model state
        time : float
            Time in seconds since start

        Returns
        -------
        Modifies state.physics_tendencies['radiation']['T']
        """
        # Solar zenith angle
        zenith_angle = self._compute_solar_zenith(time)

        # Shortwave radiation
        sw_heating = self._shortwave_radiation(state, zenith_angle)

        # Longwave radiation
        lw_heating = self._longwave_radiation(state)

        # Total radiative heating
        state.physics_tendencies['radiation']['T'][:] = sw_heating + lw_heating

    def _compute_solar_zenith(self, time):
        """
        Compute solar zenith angle

        Parameters
        ----------
        time : float
            Time in seconds since start

        Returns
        -------
        zenith : ndarray
            Solar zenith angle (radians) at each grid point
        """
        # Simple diurnal and seasonal cycle
        day_of_year = (time / 86400.0) % 365.0
        hour_of_day = (time / 3600.0) % 24.0

        # Solar declination (simplified)
        declination = 23.45 * np.pi/180 * np.sin(2*np.pi * (day_of_year - 81) / 365)

        # Hour angle
        hour_angle = (hour_of_day - 12.0) * np.pi / 12.0

        # Zenith angle: cos(zenith) = sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(hour_angle)
        lat = self.grid.lat2d
        cos_zenith = (np.sin(lat) * np.sin(declination) +
                     np.cos(lat) * np.cos(declination) * np.cos(hour_angle))

        # Ensure cos_zenith is in valid range
        cos_zenith = np.maximum(0.0, cos_zenith)

        zenith = np.arccos(cos_zenith)

        return zenith

    def _shortwave_radiation(self, state, zenith):
        """
        Compute shortwave radiative heating

        Parameters
        ----------
        state : ModelState
            Current model state
        zenith : ndarray
            Solar zenith angle

        Returns
        -------
        heating : ndarray
            Shortwave heating rate (K/s)
        """
        heating = np.zeros_like(state.T)

        # Incoming solar radiation at TOA
        cos_zenith = np.cos(zenith)
        incoming = self.solar_constant * cos_zenith

        # Process each spectral band
        for band in range(self.n_sw_bands):
            band_flux = incoming * self.sw_band_weights[band]

            # Compute optical depths
            tau_rayleigh = self._rayleigh_optical_depth(state, band)
            tau_o3 = self._ozone_optical_depth(state, band)
            tau_h2o = self._water_vapor_optical_depth(state, band)
            tau_cloud = self._cloud_optical_depth(state, band)
            tau_co2 = self._co2_optical_depth(state, band)

            # Total optical depth
            tau_total = tau_rayleigh + tau_o3 + tau_h2o + tau_cloud + tau_co2

            # Two-stream approximation for radiative transfer
            heating_band = self._two_stream_sw(state, band_flux, tau_total, state.albedo)

            heating += heating_band

        return heating

    def _longwave_radiation(self, state):
        """
        Compute longwave radiative heating

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        heating : ndarray
            Longwave heating rate (K/s)
        """
        heating = np.zeros_like(state.T)

        # Process each spectral band
        for band in range(self.n_lw_bands):
            # Compute optical depths for longwave
            tau_h2o = self._water_vapor_optical_depth_lw(state, band)
            tau_co2 = self._co2_optical_depth_lw(state, band)
            tau_cloud = self._cloud_optical_depth(state, band)

            tau_total = tau_h2o + tau_co2 + tau_cloud

            # Two-stream approximation for longwave
            heating_band = self._two_stream_lw(state, tau_total, band)

            heating += heating_band * self.lw_band_weights[band]

        return heating

    def _rayleigh_optical_depth(self, state, band):
        """Compute Rayleigh scattering optical depth"""
        tau = np.zeros_like(state.T)

        if band == 0:  # UV-visible band
            # Rayleigh scattering strongest in UV-vis
            for k in range(self.vgrid.nlev):
                # Optical depth proportional to pressure
                tau[k] = 0.02 * (state.p[k] / 101325.0)

        return tau

    def _ozone_optical_depth(self, state, band):
        """Compute ozone absorption optical depth"""
        tau = np.zeros_like(state.T)

        if band == 0:  # UV band
            # Ozone absorption in UV
            # Assume ozone concentrated in stratosphere
            for k in range(self.vgrid.nlev):
                p_mb = state.p[k] / 100.0
                # Simple ozone profile (only in stratosphere)
                tau[k] = np.where(p_mb < 100.0,
                                 0.1 * np.exp(-(np.log(np.maximum(p_mb, 1.0) / 10.0))**2 / 2.0),
                                 0.0)

        return tau

    def _water_vapor_optical_depth(self, state, band):
        """Compute water vapor absorption optical depth (shortwave)"""
        tau = np.zeros_like(state.T)

        if band >= 1:  # Near-IR and IR bands
            for k in range(self.vgrid.nlev):
                # Optical depth proportional to water vapor path
                # tau ~ q * dp / g
                if k == 0:
                    dp = state.p[k]
                else:
                    dp = state.p[k] - state.p[k-1]

                # Absorption coefficient varies by band
                k_h2o = 0.01 * (band - 0.5)  # Stronger in far-IR
                tau[k] = k_h2o * state.q[k] * dp / 9.81

        return tau

    def _water_vapor_optical_depth_lw(self, state, band):
        """Compute water vapor optical depth for longwave"""
        tau = np.zeros_like(state.T)

        # Water vapor is major greenhouse gas
        for k in range(self.vgrid.nlev):
            if k == 0:
                dp = state.p[k]
            else:
                dp = state.p[k] - state.p[k-1]

            # Band-dependent absorption
            # Stronger in window regions and rotation-vibration bands
            if band in [1, 3, 5]:  # Strong absorption bands
                k_h2o = 0.1
            else:
                k_h2o = 0.03

            tau[k] = k_h2o * state.q[k] * dp / 9.81

        return tau

    def _co2_optical_depth(self, state, band):
        """Compute CO2 optical depth (shortwave)"""
        tau = np.zeros_like(state.T)

        if band >= 2:  # IR bands
            for k in range(self.vgrid.nlev):
                if k == 0:
                    dp = state.p[k]
                else:
                    dp = state.p[k] - state.p[k-1]

                # CO2 mixing ratio
                co2_mr = self.co2_ppmv * 1e-6

                # Weak absorption in shortwave
                k_co2 = 0.001
                tau[k] = k_co2 * co2_mr * dp / 9.81

        return tau

    def _co2_optical_depth_lw(self, state, band):
        """Compute CO2 optical depth (longwave)"""
        tau = np.zeros_like(state.T)

        # CO2 15 μm band (667 cm^-1)
        for k in range(self.vgrid.nlev):
            if k == 0:
                dp = state.p[k]
            else:
                dp = state.p[k] - state.p[k-1]

            co2_mr = self.co2_ppmv * 1e-6

            # Strong absorption near 667 cm^-1
            if band == 1:  # 667 cm^-1 band
                k_co2 = 0.5
            else:
                k_co2 = 0.01

            tau[k] = k_co2 * co2_mr * dp / 9.81

        return tau

    def _cloud_optical_depth(self, state, band):
        """Compute cloud optical depth"""
        tau = np.zeros_like(state.T)

        for k in range(self.vgrid.nlev):
            # Cloud optical depth from liquid and ice water content
            # tau ~ (qc + qi) * extinction_efficiency * dp/g

            if k == 0:
                dp = state.p[k]
            else:
                dp = state.p[k] - state.p[k-1]

            # Extinction efficiency (simplified)
            ext_eff = 100.0  # m^2/kg

            tau[k] = ext_eff * (state.qc[k] + state.qi[k]) * dp / 9.81

        return tau

    def _two_stream_sw(self, state, incoming_flux, tau, albedo):
        """
        Two-stream approximation for shortwave radiation

        Parameters
        ----------
        state : ModelState
            Model state
        incoming_flux : ndarray
            Incoming solar flux at TOA
        tau : ndarray
            Optical depth at each level
        albedo : ndarray
            Surface albedo

        Returns
        -------
        heating : ndarray
            Heating rate (K/s)
        """
        heating = np.zeros_like(state.T)

        # Cumulative optical depth from top
        tau_cum = np.cumsum(tau, axis=0)

        # Downward flux (Beer's law with single scattering)
        asymmetry = 0.7  # Asymmetry parameter
        flux_down = incoming_flux[None, :, :] * np.exp(-tau_cum / np.cos(0.5))

        # Surface reflection
        flux_up_surface = albedo * flux_down[-1]

        # Upward flux
        flux_up = np.zeros_like(flux_down)
        flux_up[-1] = flux_up_surface

        for k in range(self.vgrid.nlev - 2, -1, -1):
            # Scattering and absorption
            flux_up[k] = flux_up[k+1] * np.exp(-tau[k+1])

        # Net flux divergence
        for k in range(self.vgrid.nlev):
            if k == 0:
                flux_top = incoming_flux
                flux_bot = flux_down[k]
            elif k == self.vgrid.nlev - 1:
                flux_top = flux_down[k-1]
                flux_bot = flux_up_surface
            else:
                flux_top = flux_down[k-1]
                flux_bot = flux_down[k]

            # Net heating from flux divergence
            dp = state.p[k] - (state.p[k-1] if k > 0 else 0)
            net_flux = (flux_top - flux_bot) + (flux_up[k] - (flux_up[k+1] if k < self.vgrid.nlev-1 else 0))

            # Convert to heating rate: dT/dt = -g/(cp*dp) * dF/dz
            heating[k] = 9.81 / (1004.0 * dp) * net_flux

        return heating

    def _two_stream_lw(self, state, tau, band):
        """
        Two-stream approximation for longwave radiation

        Parameters
        ----------
        state : ModelState
            Model state
        tau : ndarray
            Optical depth
        band : int
            Spectral band index

        Returns
        -------
        heating : ndarray
            Heating rate (K/s)
        """
        heating = np.zeros_like(state.T)

        # Planck function for each level
        B = self.sigma_sb * state.T**4 / np.pi  # Simplified

        # Upward flux from surface
        flux_up = np.zeros_like(state.T)
        flux_up[-1] = self.sigma_sb * state.tsurf**4

        # Upward propagation
        for k in range(self.vgrid.nlev - 2, -1, -1):
            # Emission + transmitted flux from below
            transmissivity = np.exp(-tau[k])
            flux_up[k] = (1 - transmissivity) * B[k] + transmissivity * flux_up[k+1]

        # Downward flux from TOA
        flux_down = np.zeros_like(state.T)
        flux_down[0] = 0.0  # No downward flux at TOA

        # Downward propagation
        for k in range(1, self.vgrid.nlev):
            transmissivity = np.exp(-tau[k-1])
            flux_down[k] = (1 - transmissivity) * B[k-1] + transmissivity * flux_down[k-1]

        # Net flux divergence
        for k in range(self.vgrid.nlev):
            # Net flux
            if k == 0:
                net_flux_top = flux_up[k]
            else:
                net_flux_top = flux_up[k] - flux_down[k]

            if k == self.vgrid.nlev - 1:
                net_flux_bot = flux_down[k] - flux_up[k]
            else:
                net_flux_bot = flux_down[k+1] - flux_up[k+1]

            # Flux divergence
            dp = state.p[k] - (state.p[k-1] if k > 0 else 0)
            div_flux = net_flux_top - net_flux_bot

            # Heating rate
            heating[k] = -9.81 / (1004.0 * dp) * div_flux

        return heating
