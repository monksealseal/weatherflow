"""
Land Surface Model

Implements:
- Multi-layer soil temperature and moisture
- Vegetation effects
- Snow accumulation and melt
- Surface energy balance
- Evapotranspiration
"""

import numpy as np


class LandSurfaceModel:
    """
    Comprehensive land surface model with soil physics
    """

    def __init__(self, grid, vgrid, n_soil_layers=4):
        """
        Initialize land surface model

        Parameters
        ----------
        grid : SphericalGrid
            Horizontal grid
        vgrid : VerticalGrid
            Vertical grid
        n_soil_layers : int
            Number of soil layers
        """
        self.grid = grid
        self.vgrid = vgrid
        self.n_soil = n_soil_layers

        # Physical constants
        self.g = 9.81
        self.cp = 1004.0
        self.sigma_sb = 5.67e-8
        self.Lv = 2.5e6  # Latent heat of vaporization
        self.Lf = 3.34e5  # Latent heat of fusion

        # Soil parameters
        self.soil_depth = np.array([0.05, 0.15, 0.5, 1.5])  # Layer depths (m)
        self.soil_heat_capacity = 2.0e6  # J/(m^3*K)
        self.soil_thermal_cond = 1.5  # W/(m*K)
        self.soil_porosity = 0.4  # Fraction

        # Initialize soil state
        nlat, nlon = grid.nlat, grid.nlon
        self.soil_temp = np.zeros((self.n_soil, nlat, nlon))
        self.soil_moisture = np.zeros((self.n_soil, nlat, nlon))
        self.snow_depth = np.zeros((nlat, nlon))
        self.snow_water_equiv = np.zeros((nlat, nlon))

        # Vegetation parameters
        self.lai = np.ones((nlat, nlon))  # Leaf area index
        self.veg_fraction = np.ones((nlat, nlon)) * 0.5  # Vegetation fraction
        self.stomatal_resistance = np.ones((nlat, nlon)) * 100.0  # s/m

        # Initialize with reasonable values
        self._initialize_soil()

    def _initialize_soil(self):
        """Initialize soil state with climatological values"""
        # Temperature: decreases toward poles
        lat = self.grid.lat2d
        T_base = 288.0 - 30.0 * np.abs(np.sin(lat))

        for i in range(self.n_soil):
            self.soil_temp[i] = T_base

        # Soil moisture: wet in tropics, dry near 30°
        self.soil_moisture[:] = self.soil_porosity * 0.6  # 60% saturation

        # Adjust for latitude (simplified)
        lat_deg = np.rad2deg(lat)
        dry_regions = (np.abs(lat_deg) > 20) & (np.abs(lat_deg) < 40)
        self.soil_moisture[:, dry_regions] *= 0.3  # Dry subtropics

    def compute_land_surface(self, state, dt, net_radiation, precipitation):
        """
        Compute land surface tendencies

        Parameters
        ----------
        state : ModelState
            Current model state
        dt : float
            Time step (s)
        net_radiation : ndarray
            Net radiation at surface (W/m^2)
        precipitation : ndarray
            Precipitation rate (kg/m^2/s)

        Returns
        -------
        Modifies state.tsurf, state.qsurf, and soil state
        """
        # Surface energy balance
        self._surface_energy_balance(state, dt, net_radiation)

        # Snow processes
        self._snow_processes(state, dt, precipitation)

        # Soil heat diffusion
        self._soil_heat_diffusion(state, dt)

        # Soil moisture
        self._soil_moisture(state, dt, precipitation)

        # Evapotranspiration
        self._evapotranspiration(state, dt)

    def _surface_energy_balance(self, state, dt, net_radiation):
        """
        Solve surface energy balance

        R_net = H + LE + G

        where:
        R_net = net radiation
        H = sensible heat flux
        LE = latent heat flux
        G = ground heat flux
        """
        # Get atmospheric surface layer values
        k_sfc = self.vgrid.nlev - 1
        T_air = state.T[k_sfc]
        q_air = state.q[k_sfc]
        p_sfc = state.ps
        rho_air = state.rho[k_sfc]

        # Current surface temperature
        T_sfc = state.tsurf

        # Estimate fluxes
        # Sensible heat flux (simplified bulk formula)
        C_h = 0.01  # Heat transfer coefficient
        wind_speed = np.sqrt(state.u[k_sfc]**2 + state.v[k_sfc]**2) + 1.0
        H = rho_air * self.cp * C_h * wind_speed * (T_sfc - T_air)

        # Latent heat flux (evapotranspiration)
        q_sat_sfc = self._saturation_mixing_ratio(T_sfc, p_sfc)
        # Soil moisture limitation
        soil_moisture_factor = np.minimum(1.0, self.soil_moisture[0] / (0.75 * self.soil_porosity))
        evap_efficiency = 0.5 + 0.5 * soil_moisture_factor * self.veg_fraction

        LE = rho_air * self.Lv * C_h * wind_speed * evap_efficiency * (q_sat_sfc - q_air)

        # Ground heat flux (into soil)
        G = self.soil_thermal_cond * (T_sfc - self.soil_temp[0]) / (0.5 * self.soil_depth[0])

        # Update surface temperature
        # Heat capacity of thin surface layer
        C_surface = 1e5  # J/(m^2*K) - effective heat capacity

        dT_sfc = (net_radiation - H - LE - G) / C_surface * dt
        state.tsurf += dT_sfc

        # Update surface humidity (from evaporation)
        state.qsurf = q_sat_sfc * evap_efficiency

    def _snow_processes(self, state, dt, precipitation):
        """
        Handle snow accumulation and melt

        Parameters
        ----------
        state : ModelState
            Model state
        dt : float
            Time step
        precipitation : ndarray
            Precipitation rate (kg/m^2/s)
        """
        T_freeze = 273.15

        # Determine if precipitation is snow or rain
        T_sfc = state.tsurf
        is_snow = T_sfc < T_freeze

        # Snow accumulation
        snow_precip = precipitation * is_snow
        self.snow_water_equiv += snow_precip * dt

        # Snow density (simplified)
        rho_snow = 200.0  # kg/m^3
        self.snow_depth = self.snow_water_equiv / rho_snow

        # Snow melt
        is_warm = T_sfc > T_freeze
        if np.any(is_warm & (self.snow_water_equiv > 0)):
            # Melt rate proportional to excess temperature
            melt_rate = 0.001 * (T_sfc - T_freeze) * is_warm  # kg/m^2/s
            melt_amount = melt_rate * dt

            # Limit to available snow
            melt_amount = np.minimum(melt_amount, self.snow_water_equiv)

            self.snow_water_equiv -= melt_amount
            self.snow_depth = self.snow_water_equiv / rho_snow

            # Cooling from melt
            # Energy used for melting reduces surface temperature
            energy_for_melt = self.Lf * melt_amount
            C_surface = 1e5
            state.tsurf -= energy_for_melt / C_surface

        # Update surface albedo based on snow cover
        # Snow has high albedo
        albedo_snow = 0.8
        albedo_ground = 0.2
        albedo_veg = 0.15

        # Snow cover fraction (simple parameterization)
        snow_cover = np.minimum(1.0, self.snow_depth / 0.05)  # 5cm for full cover

        # Combined albedo
        state.albedo = (snow_cover * albedo_snow +
                       (1 - snow_cover) * (self.veg_fraction * albedo_veg +
                                          (1 - self.veg_fraction) * albedo_ground))

    def _soil_heat_diffusion(self, state, dt):
        """
        Solve heat diffusion equation in soil

        dT/dt = κ * d²T/dz²

        where κ is thermal diffusivity
        """
        kappa = self.soil_thermal_cond / self.soil_heat_capacity

        # Boundary condition at top: couple to surface temperature
        self.soil_temp[0] += (state.tsurf - self.soil_temp[0]) * 0.1

        # Diffusion between layers
        for i in range(1, self.n_soil):
            # Layer spacing
            dz_up = self.soil_depth[i] - self.soil_depth[i-1]
            if i < self.n_soil - 1:
                dz_down = self.soil_depth[i+1] - self.soil_depth[i]
            else:
                dz_down = dz_up  # Bottom boundary

            # Temperature gradients
            dT_dz_up = (self.soil_temp[i-1] - self.soil_temp[i]) / dz_up

            if i < self.n_soil - 1:
                dT_dz_down = (self.soil_temp[i] - self.soil_temp[i+1]) / dz_down
            else:
                dT_dz_down = 0.0  # No flux at bottom

            # Heat diffusion
            d2T_dz2 = (dT_dz_up - dT_dz_down) / (0.5 * (dz_up + dz_down))

            self.soil_temp[i] += kappa * d2T_dz2 * dt

    def _soil_moisture(self, state, dt, precipitation):
        """
        Soil moisture budget

        dθ/dt = P - E - R - D

        where:
        θ = soil moisture
        P = precipitation
        E = evapotranspiration
        R = runoff
        D = drainage
        """
        T_freeze = 273.15
        T_sfc = state.tsurf

        # Precipitation input (only rain, not snow)
        rain = precipitation * (T_sfc >= T_freeze)

        # Add to top soil layer
        # Convert precipitation rate to change in soil moisture
        # P [kg/m^2/s] * dt [s] / (rho_water * depth [m]) = dθ
        rho_water = 1000.0
        d_theta = rain * dt / (rho_water * self.soil_depth[0])
        self.soil_moisture[0] += d_theta

        # Drainage between layers (gravity + diffusion)
        for i in range(self.n_soil - 1):
            # Hydraulic conductivity (simplified)
            # K increases with moisture content
            theta_rel = self.soil_moisture[i] / self.soil_porosity
            K_hydraulic = 1e-6 * theta_rel**3  # m/s

            # Drainage to layer below
            drainage = K_hydraulic * dt / self.soil_depth[i]

            # Limit drainage
            drainage = np.minimum(drainage, self.soil_moisture[i] * 0.5)

            self.soil_moisture[i] -= drainage
            self.soil_moisture[i+1] += drainage * self.soil_depth[i] / self.soil_depth[i+1]

        # Runoff if soil saturated
        for i in range(self.n_soil):
            excess = np.maximum(0, self.soil_moisture[i] - self.soil_porosity)
            self.soil_moisture[i] -= excess

        # Ensure non-negative
        self.soil_moisture = np.maximum(0, self.soil_moisture)

    def _evapotranspiration(self, state, dt):
        """
        Remove water from soil due to evapotranspiration

        This couples to the latent heat flux computed earlier
        """
        # Get atmospheric surface values
        k_sfc = self.vgrid.nlev - 1
        T_air = state.T[k_sfc]
        q_air = state.q[k_sfc]
        rho_air = state.rho[k_sfc]

        # Evapotranspiration rate
        C_h = 0.01
        wind_speed = np.sqrt(state.u[k_sfc]**2 + state.v[k_sfc]**2) + 1.0

        q_sat_sfc = self._saturation_mixing_ratio(state.tsurf, state.ps)
        soil_moisture_factor = np.minimum(1.0, self.soil_moisture[0] / (0.75 * self.soil_porosity))
        evap_efficiency = 0.5 + 0.5 * soil_moisture_factor * self.veg_fraction

        E = rho_air * C_h * wind_speed * evap_efficiency * (q_sat_sfc - q_air)
        E = np.maximum(0, E)  # Only evaporation, not condensation

        # Remove water from top soil layer
        rho_water = 1000.0
        d_theta = -E * dt / (rho_water * self.soil_depth[0])

        # Don't evaporate more than available
        self.soil_moisture[0] = np.maximum(0, self.soil_moisture[0] + d_theta)

    def _saturation_mixing_ratio(self, T, p):
        """Compute saturation mixing ratio"""
        T0 = 273.15
        e0 = 611.2
        Rv = 461.5
        Lv = self.Lv

        es = e0 * np.exp((Lv / Rv) * (1/T0 - 1/T))

        Rd = 287.0
        epsilon = Rd / Rv
        qsat = epsilon * es / (p - es)

        return qsat
