"""
Convection parameterization scheme

Implements a mass-flux convection scheme with:
- Deep convection
- Shallow convection
- CAPE-based triggering
- Convective momentum transport
- Detrainment and entrainment
"""

import numpy as np


class ConvectionScheme:
    """
    Mass-flux convection scheme for deep and shallow convection
    """

    def __init__(self, grid, vgrid):
        """
        Initialize convection scheme

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
        self.Rv = 461.5
        self.cp = 1004.0
        self.Lv = 2.5e6  # Latent heat of vaporization
        self.Ls = 2.834e6  # Latent heat of sublimation

        # Convection parameters
        self.cape_threshold = 100.0  # J/kg - threshold for convection
        self.tau_conv = 3600.0  # Convective adjustment timescale (s)
        self.entrainment_rate = 1e-4  # m^-1
        self.detrainment_rate = 1e-4  # m^-1

    def compute_convection(self, state, dt):
        """
        Compute convective tendencies

        Parameters
        ----------
        state : ModelState
            Current model state
        dt : float
            Time step (s)

        Returns
        -------
        Modifies state.physics_tendencies['convection']
        """
        # Check for convective instability
        cape, cin, lcl = self._compute_cape(state)

        # Identify convective columns
        convective_mask = (cape > self.cape_threshold)

        # Deep convection
        deep_mask = convective_mask & (cape > 500.0)
        if np.any(deep_mask):
            self._deep_convection(state, dt, deep_mask, cape, lcl)

        # Shallow convection
        shallow_mask = convective_mask & ~deep_mask
        if np.any(shallow_mask):
            self._shallow_convection(state, dt, shallow_mask, lcl)

    def _compute_cape(self, state):
        """
        Compute CAPE (Convective Available Potential Energy)

        Returns
        -------
        cape : ndarray
            CAPE (J/kg) at each horizontal grid point
        cin : ndarray
            Convective inhibition (J/kg)
        lcl : ndarray
            Lifting condensation level index
        """
        nlat, nlon = state.ps.shape
        cape = np.zeros((nlat, nlon))
        cin = np.zeros((nlat, nlon))
        lcl = np.zeros((nlat, nlon), dtype=int)

        for i in range(nlat):
            for j in range(nlon):
                # Start from surface
                T_parcel = state.T[-1, i, j]
                q_parcel = state.q[-1, i, j]
                p_parcel = state.p[-1, i, j]

                # Lift parcel adiabatically
                cape_ij = 0.0
                cin_ij = 0.0
                lcl_ij = self.vgrid.nlev - 1

                for k in range(self.vgrid.nlev - 2, -1, -1):
                    # Environment
                    T_env = state.T[k, i, j]
                    p_env = state.p[k, i, j]

                    # Lift parcel to this level
                    T_parcel_lifted = self._lift_parcel(T_parcel, q_parcel,
                                                        p_parcel, p_env)

                    # Check for condensation
                    q_sat = self._saturation_mixing_ratio(T_parcel_lifted, p_env)

                    if q_parcel > q_sat and lcl_ij == self.vgrid.nlev - 1:
                        # Reached LCL
                        lcl_ij = k
                        # Release latent heat
                        T_parcel_lifted += (self.Lv / self.cp) * (q_parcel - q_sat)
                        q_parcel = q_sat

                    # Buoyancy
                    buoyancy = self.g * (T_parcel_lifted - T_env) / T_env

                    # Integrate CAPE/CIN
                    if buoyancy > 0:
                        if k == 0:
                            dz = 1000.0  # Approximate
                        else:
                            dz = (state.z[k, i, j] - state.z[k+1, i, j])
                        cape_ij += buoyancy * dz
                    else:
                        if k > lcl_ij:  # Below LCL
                            dz = (state.z[k, i, j] - state.z[k+1, i, j]) if k < self.vgrid.nlev - 1 else 1000.0
                            cin_ij -= buoyancy * dz

                    # Update parcel state for next level
                    T_parcel = T_parcel_lifted
                    p_parcel = p_env

                cape[i, j] = cape_ij
                cin[i, j] = cin_ij
                lcl[i, j] = lcl_ij

        return cape, cin, lcl

    def _lift_parcel(self, T, q, p_initial, p_final):
        """
        Adiabatically lift an air parcel

        Parameters
        ----------
        T : float
            Initial temperature (K)
        q : float
            Mixing ratio (kg/kg)
        p_initial : float
            Initial pressure (Pa)
        p_final : float
            Final pressure (Pa)

        Returns
        -------
        T_final : float
            Final temperature (K)
        """
        # Dry adiabatic lapse rate
        kappa = self.Rd / self.cp
        T_final = T * (p_final / p_initial)**kappa

        return T_final

    def _saturation_mixing_ratio(self, T, p):
        """
        Compute saturation mixing ratio

        Parameters
        ----------
        T : float or ndarray
            Temperature (K)
        p : float or ndarray
            Pressure (Pa)

        Returns
        -------
        qsat : float or ndarray
            Saturation mixing ratio (kg/kg)
        """
        # Clausius-Clapeyron equation
        T0 = 273.15
        e0 = 611.2  # Pa

        # Saturation vapor pressure
        es = e0 * np.exp((self.Lv / self.Rv) * (1/T0 - 1/T))

        # Mixing ratio
        epsilon = self.Rd / self.Rv
        qsat = epsilon * es / (p - es)

        return qsat

    def _deep_convection(self, state, dt, mask, cape, lcl):
        """
        Parameterize deep convection

        Parameters
        ----------
        state : ModelState
            Model state
        dt : float
            Time step
        mask : ndarray
            Boolean mask for convective columns
        cape : ndarray
            CAPE values
        lcl : ndarray
            LCL indices
        """
        # Mass flux proportional to CAPE
        w_star = np.sqrt(2 * cape)  # Convective velocity scale
        mass_flux = 0.01 * state.rho[-1] * w_star * mask

        # Vertical distribution of convection
        for k in range(self.vgrid.nlev):
            # Convection active above LCL
            active = (k <= lcl) & mask

            if not np.any(active):
                continue

            # Updraft properties
            # Simplification: assume updraft reaches environmental saturation
            T_env = state.T[k]
            q_env = state.q[k]
            p_env = state.p[k]

            qsat = self._saturation_mixing_ratio(T_env, p_env)

            # Condensation in updraft
            condensation = np.maximum(0, q_env - qsat) * active

            # Latent heating
            heating_rate = (self.Lv / self.cp) * condensation / self.tau_conv
            state.physics_tendencies['convection']['T'][k] += heating_rate

            # Moisture convergence
            drying_rate = -condensation / self.tau_conv
            state.physics_tendencies['convection']['q'][k] += drying_rate

            # Momentum transport (simplified)
            # Convection tends to mix momentum vertically
            if k < self.vgrid.nlev - 1:
                u_flux = 0.1 * (state.u[k+1] - state.u[k]) / self.tau_conv * active
                v_flux = 0.1 * (state.v[k+1] - state.v[k]) / self.tau_conv * active

                state.physics_tendencies['convection']['u'][k] += u_flux
                state.physics_tendencies['convection']['v'][k] += v_flux

    def _shallow_convection(self, state, dt, mask, lcl):
        """
        Parameterize shallow convection

        Parameters
        ----------
        state : ModelState
            Model state
        dt : float
            Time step
        mask : ndarray
            Boolean mask for shallow convective columns
        lcl : ndarray
            LCL indices
        """
        # Shallow convection: weaker mixing, confined to lower troposphere

        for k in range(int(self.vgrid.nlev * 0.7), self.vgrid.nlev):  # Lower 30% of atmosphere
            active = (k > lcl) & mask

            if not np.any(active):
                continue

            # Weak vertical mixing
            T_env = state.T[k]
            q_env = state.q[k]
            p_env = state.p[k]

            qsat = self._saturation_mixing_ratio(T_env, p_env)

            # Partial condensation
            condensation = 0.3 * np.maximum(0, q_env - qsat) * active

            # Latent heating (weaker than deep convection)
            heating_rate = (self.Lv / self.cp) * condensation / (2 * self.tau_conv)
            state.physics_tendencies['convection']['T'][k] += heating_rate

            # Moisture adjustment
            drying_rate = -condensation / (2 * self.tau_conv)
            state.physics_tendencies['convection']['q'][k] += drying_rate

    def compute_precipitation(self, state, dt):
        """
        Compute precipitation rate from convection

        Parameters
        ----------
        state : ModelState
            Model state
        dt : float
            Time step

        Returns
        -------
        precip : ndarray
            Precipitation rate (mm/hr)
        """
        # Integrate condensation over vertical column
        precip = np.zeros_like(state.ps)

        for k in range(self.vgrid.nlev):
            # Condensation rate
            condensation_rate = -state.physics_tendencies['convection']['q'][k]

            # Convert to precipitation
            # precip = integral of (rho * condensation_rate * dz)
            if k == 0:
                dp = state.p[k]
            else:
                dp = state.p[k] - state.p[k-1]

            # mm/hr conversion
            precip += condensation_rate * dp / self.g * 3600.0 * 1000.0

        return np.maximum(0, precip)
