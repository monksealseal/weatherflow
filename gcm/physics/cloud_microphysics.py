"""
Cloud microphysics parameterization

Implements two-moment microphysics with:
- Autoconversion (cloud droplets -> rain)
- Accretion (collection)
- Evaporation
- Ice processes (freezing, deposition, sublimation)
- Sedimentation
"""

import numpy as np


class CloudMicrophysics:
    """
    Two-moment cloud microphysics scheme
    """

    def __init__(self, grid, vgrid):
        """
        Initialize cloud microphysics

        Parameters
        ----------
        grid : SphericalGrid
            Horizontal grid
        vgrid : VerticalGrid
            Vertical grid
        """
        self.grid = grid
        self.vgrid = vgrid

        # Physical constants
        self.g = 9.81
        self.Rd = 287.0
        self.cp = 1004.0
        self.Lv = 2.5e6  # Latent heat of vaporization (J/kg)
        self.Ls = 2.834e6  # Latent heat of sublimation (J/kg)
        self.Lf = self.Ls - self.Lv  # Latent heat of fusion

        # Microphysical parameters
        self.rho_water = 1000.0  # Density of liquid water (kg/m^3)
        self.rho_ice = 917.0     # Density of ice (kg/m^3)

        # Autoconversion threshold
        self.qc_crit = 5e-4  # Critical cloud water for autoconversion (kg/kg)

        # Timescales
        self.tau_autoconv = 1800.0  # Autoconversion timescale (s)
        self.tau_evap = 600.0       # Evaporation timescale (s)

    def compute_microphysics(self, state, dt):
        """
        Compute cloud microphysical tendencies

        Parameters
        ----------
        state : ModelState
            Current model state
        dt : float
            Time step (s)

        Returns
        -------
        Modifies state.physics_tendencies['cloud_micro']
        """
        # Warm cloud processes
        self._warm_processes(state, dt)

        # Ice processes
        self._ice_processes(state, dt)

        # Sedimentation
        self._sedimentation(state, dt)

    def _warm_processes(self, state, dt):
        """
        Warm cloud microphysics (liquid phase only)

        Includes:
        - Condensation/Evaporation
        - Autoconversion
        - Accretion
        """
        for k in range(self.vgrid.nlev):
            T = state.T[k]
            q = state.q[k]
            qc = state.qc[k]
            p = state.p[k]

            # Saturation mixing ratio
            qsat = self._saturation_mixing_ratio(T, p)

            # Supersaturation
            supersat = q - qsat

            # Condensation (if supersaturated and above freezing)
            freezing_temp = 273.15
            is_warm = T > freezing_temp

            # Condensation rate (relaxation to saturation)
            if np.any(supersat > 0):
                # Condensation
                condensation_rate = supersat / self.tau_evap * is_warm
                condensation_rate = np.maximum(0, condensation_rate)

                # Update tendencies
                state.physics_tendencies['cloud_micro']['q'][k] -= condensation_rate
                state.physics_tendencies['cloud_micro']['qc'][k] += condensation_rate

                # Latent heating
                heating = (self.Lv / self.cp) * condensation_rate
                state.physics_tendencies['cloud_micro']['T'][k] += heating

            # Evaporation (if subsaturated and cloud water present)
            if np.any((supersat < 0) & (qc > 0)):
                # Evaporation limited by available cloud water
                evap_rate = np.minimum(qc, -supersat) / self.tau_evap * is_warm
                evap_rate = np.maximum(0, evap_rate)

                # Update tendencies
                state.physics_tendencies['cloud_micro']['q'][k] += evap_rate
                state.physics_tendencies['cloud_micro']['qc'][k] -= evap_rate

                # Latent cooling
                cooling = -(self.Lv / self.cp) * evap_rate
                state.physics_tendencies['cloud_micro']['T'][k] += cooling

            # Autoconversion: cloud droplets -> precipitation
            # Occurs when cloud water exceeds threshold
            exceed_threshold = qc > self.qc_crit
            if np.any(exceed_threshold):
                # Autoconversion rate (Kessler scheme)
                autoconv_rate = ((qc - self.qc_crit) / self.tau_autoconv *
                                exceed_threshold * is_warm)
                autoconv_rate = np.maximum(0, autoconv_rate)

                # Reduce cloud water (precipitation falls out immediately)
                state.physics_tendencies['cloud_micro']['qc'][k] -= autoconv_rate

            # Accretion: rain collecting cloud droplets
            # Simplified: proportional to qc * qr
            # For now, assume rain falls out, so accretion ~ autoconversion

    def _ice_processes(self, state, dt):
        """
        Ice microphysics

        Includes:
        - Homogeneous freezing
        - Deposition/Sublimation
        - Bergeron process
        - Melting
        """
        T_freeze = 273.15
        T_homogeneous = 233.15  # Homogeneous freezing temperature

        for k in range(self.vgrid.nlev):
            T = state.T[k]
            q = state.q[k]
            qc = state.qc[k]
            qi = state.qi[k]
            p = state.p[k]

            # Freezing of liquid water
            is_cold = T < T_freeze
            is_very_cold = T < T_homogeneous

            # Homogeneous freezing (all liquid freezes below -40°C)
            if np.any(is_very_cold & (qc > 0)):
                freeze_rate = qc / dt * is_very_cold
                freeze_rate = np.maximum(0, freeze_rate)

                state.physics_tendencies['cloud_micro']['qc'][k] -= freeze_rate
                state.physics_tendencies['cloud_micro']['qi'][k] += freeze_rate

                # Latent heating from freezing
                heating = (self.Lf / self.cp) * freeze_rate
                state.physics_tendencies['cloud_micro']['T'][k] += heating

            # Heterogeneous freezing (partial freezing between 0°C and -40°C)
            is_mixed_phase = is_cold & ~is_very_cold & (qc > 0)
            if np.any(is_mixed_phase):
                # Freezing rate increases as temperature decreases
                freeze_fraction = (T_freeze - T) / (T_freeze - T_homogeneous)
                freeze_fraction = np.clip(freeze_fraction, 0, 1)

                freeze_rate = qc * freeze_fraction / (10 * dt) * is_mixed_phase
                freeze_rate = np.maximum(0, freeze_rate)

                state.physics_tendencies['cloud_micro']['qc'][k] -= freeze_rate
                state.physics_tendencies['cloud_micro']['qi'][k] += freeze_rate

                heating = (self.Lf / self.cp) * freeze_rate
                state.physics_tendencies['cloud_micro']['T'][k] += heating

            # Deposition/Sublimation
            if np.any(is_cold):
                # Saturation over ice
                qsat_ice = self._saturation_mixing_ratio_ice(T, p)

                # Deposition (water vapor -> ice)
                supersat_ice = q - qsat_ice
                if np.any((supersat_ice > 0) & is_cold):
                    deposition_rate = supersat_ice / self.tau_evap * is_cold
                    deposition_rate = np.maximum(0, deposition_rate)

                    state.physics_tendencies['cloud_micro']['q'][k] -= deposition_rate
                    state.physics_tendencies['cloud_micro']['qi'][k] += deposition_rate

                    # Latent heating
                    heating = (self.Ls / self.cp) * deposition_rate
                    state.physics_tendencies['cloud_micro']['T'][k] += heating

                # Sublimation (ice -> water vapor)
                if np.any((supersat_ice < 0) & (qi > 0) & is_cold):
                    sublim_rate = np.minimum(qi, -supersat_ice) / self.tau_evap * is_cold
                    sublim_rate = np.maximum(0, sublim_rate)

                    state.physics_tendencies['cloud_micro']['q'][k] += sublim_rate
                    state.physics_tendencies['cloud_micro']['qi'][k] -= sublim_rate

                    # Latent cooling
                    cooling = -(self.Ls / self.cp) * sublim_rate
                    state.physics_tendencies['cloud_micro']['T'][k] += cooling

            # Melting of ice
            is_warm = T > T_freeze
            if np.any(is_warm & (qi > 0)):
                # Ice melts above 0°C
                melt_rate = qi / dt * is_warm
                melt_rate = np.maximum(0, melt_rate)

                state.physics_tendencies['cloud_micro']['qi'][k] -= melt_rate
                state.physics_tendencies['cloud_micro']['qc'][k] += melt_rate

                # Latent cooling from melting
                cooling = -(self.Lf / self.cp) * melt_rate
                state.physics_tendencies['cloud_micro']['T'][k] += cooling

            # Bergeron process: ice grows at expense of liquid in mixed-phase clouds
            # (ice has lower saturation vapor pressure than liquid)
            if np.any(is_mixed_phase & (qi > 0) & (qc > 0)):
                qsat_liquid = self._saturation_mixing_ratio(T, p)
                qsat_ice = self._saturation_mixing_ratio_ice(T, p)

                # Liquid evaporates, ice grows
                bergeron_rate = 0.1 * (qsat_liquid - qsat_ice) / self.tau_evap * is_mixed_phase
                bergeron_rate = np.minimum(bergeron_rate, qc)  # Limited by available liquid
                bergeron_rate = np.maximum(0, bergeron_rate)

                state.physics_tendencies['cloud_micro']['qc'][k] -= bergeron_rate
                state.physics_tendencies['cloud_micro']['qi'][k] += bergeron_rate

                # Net latent heat release (evaporation + deposition)
                # dQ = -Lv (evap) + Ls (deposition) = Lf
                heating = (self.Lf / self.cp) * bergeron_rate
                state.physics_tendencies['cloud_micro']['T'][k] += heating

    def _sedimentation(self, state, dt):
        """
        Gravitational settling of cloud particles

        Larger particles fall faster
        """
        # Fall speeds (simplified)
        v_fall_liquid = 0.01  # m/s for cloud droplets
        v_fall_ice = 0.5      # m/s for ice crystals

        # Sedimentation of cloud liquid
        for k in range(self.vgrid.nlev - 1, 0, -1):
            # Mass flux downward
            flux_down = state.qc[k] * v_fall_liquid

            # Update tendency
            # Loss from this level, gain at level below
            if k == 0:
                dz = 1000.0
            else:
                dz = state.z[k-1] - state.z[k]

            sed_rate = -flux_down / dz

            state.physics_tendencies['cloud_micro']['qc'][k] += sed_rate

            if k < self.vgrid.nlev - 1:
                state.physics_tendencies['cloud_micro']['qc'][k+1] -= sed_rate

        # Sedimentation of cloud ice
        for k in range(self.vgrid.nlev - 1, 0, -1):
            flux_down = state.qi[k] * v_fall_ice

            if k == 0:
                dz = 1000.0
            else:
                dz = state.z[k-1] - state.z[k]

            sed_rate = -flux_down / dz

            state.physics_tendencies['cloud_micro']['qi'][k] += sed_rate

            if k < self.vgrid.nlev - 1:
                state.physics_tendencies['cloud_micro']['qi'][k+1] -= sed_rate

    def _saturation_mixing_ratio(self, T, p):
        """Saturation mixing ratio over liquid water"""
        T0 = 273.15
        e0 = 611.2  # Pa
        Rv = 461.5

        es = e0 * np.exp((self.Lv / Rv) * (1/T0 - 1/T))

        epsilon = self.Rd / Rv
        qsat = epsilon * es / (p - es)

        return qsat

    def _saturation_mixing_ratio_ice(self, T, p):
        """Saturation mixing ratio over ice"""
        T0 = 273.15
        e0 = 611.2  # Pa
        Rv = 461.5

        # Use sublimation in Clausius-Clapeyron
        es_ice = e0 * np.exp((self.Ls / Rv) * (1/T0 - 1/T))

        epsilon = self.Rd / Rv
        qsat_ice = epsilon * es_ice / (p - es_ice)

        return qsat_ice
