"""
Held-Suarez Forcing for GCM Benchmark Testing

Implements the Held & Suarez (1994) forcing which provides:
- Newtonian relaxation of temperature toward a radiative equilibrium profile
- Rayleigh friction in the boundary layer
- No moisture, clouds, or other complications

This forcing produces a realistic-looking jet stream, Hadley cell,
and baroclinic eddies, making it ideal for testing GCM dynamics.

Reference:
Held, I. M., & Suarez, M. J. (1994). A proposal for the intercomparison
of the dynamical cores of atmospheric general circulation models.
Bulletin of the American Meteorological Society, 75(10), 1825-1830.
"""

import numpy as np


class HeldSuarezForcing:
    """
    Held-Suarez benchmark forcing for GCM testing

    Provides a simple, well-defined forcing that produces:
    - Subtropical jets around 30 degrees
    - Hadley cells in the tropics
    - Baroclinic eddies in mid-latitudes
    - Realistic temperature structure
    """

    def __init__(self, grid, vgrid):
        """
        Initialize Held-Suarez forcing

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
        self.kappa = self.Rd / self.cp
        self.p0 = 100000.0  # Reference pressure (Pa)

        # Held-Suarez parameters
        # Temperature relaxation timescales
        self.k_a = 1.0 / (40.0 * 86400.0)  # 40 days (free atmosphere)
        self.k_s = 1.0 / (4.0 * 86400.0)   # 4 days (near surface)

        # Boundary layer friction timescale
        self.k_f = 1.0 / (1.0 * 86400.0)   # 1 day (friction)

        # Temperature profile parameters
        self.T_0 = 315.0      # Global mean surface temperature (K)
        self.delta_T = 60.0   # Equator-to-pole temperature difference (K)
        self.delta_theta = 10.0  # Static stability parameter (K)

        # Pressure levels for boundary layer
        self.sigma_b = 0.7  # Boundary layer top (sigma coordinate)

        # Precompute latitude-dependent quantities
        self._precompute_coefficients()

    def _precompute_coefficients(self):
        """Precompute latitude-dependent coefficients"""
        lat = self.grid.lat2d
        cos_lat = self.grid.cos_lat
        sin_lat = self.grid.sin_lat

        # Equilibrium temperature profile depends on latitude
        # T_eq = max(200, (T_0 - delta_T * sin^2(lat) - delta_theta * log(p/p0) * cos^2(lat)) * (p/p0)^kappa)

        # Store for use in compute method
        self.sin_lat_sq = sin_lat**2
        self.cos_lat_sq = cos_lat**2

    def compute_forcing(self, state, dt):
        """
        Compute Held-Suarez forcing tendencies

        Parameters
        ----------
        state : ModelState
            Current model state
        dt : float
            Time step (s)

        Returns
        -------
        Modifies state.physics_tendencies
        """
        # Temperature relaxation
        self._temperature_relaxation(state)

        # Boundary layer friction
        self._boundary_layer_friction(state)

    def _temperature_relaxation(self, state):
        """
        Apply Newtonian temperature relaxation

        dT/dt = -k_T * (T - T_eq)

        where k_T varies with height (stronger near surface)
        and T_eq is the equilibrium temperature profile.
        """
        for k in range(self.vgrid.nlev):
            # Sigma coordinate at this level
            sigma = state.p[k] / state.ps

            # Equilibrium temperature at this level
            T_eq = self._equilibrium_temperature(sigma)

            # Relaxation coefficient (stronger near surface)
            # k_T = k_a + (k_s - k_a) * max(0, (sigma - sigma_b)/(1 - sigma_b)) * cos^4(lat)
            k_T = self.k_a + (self.k_s - self.k_a) * \
                  np.maximum(0, (sigma - self.sigma_b) / (1.0 - self.sigma_b)) * \
                  self.grid.cos_lat**4

            # Temperature tendency
            dT_dt = -k_T * (state.T[k] - T_eq)

            # Add to physics tendencies (radiation category since it's thermal forcing)
            state.physics_tendencies['radiation']['T'][k] += dT_dt

    def _equilibrium_temperature(self, sigma):
        """
        Compute equilibrium temperature profile

        T_eq(lat, sigma) = max(200, (T_0 - delta_T*sin^2(lat)
                              - delta_theta*log(sigma)*cos^2(lat)) * sigma^kappa)

        Parameters
        ----------
        sigma : ndarray
            Sigma coordinate (p/ps)

        Returns
        -------
        T_eq : ndarray
            Equilibrium temperature (K)
        """
        # Latitude-dependent base temperature
        T_y = self.T_0 - self.delta_T * self.sin_lat_sq

        # Static stability term (vertical temperature gradient)
        # Using log(sigma) to create temperature decreasing with height
        log_sigma = np.log(np.maximum(sigma, 1e-6))
        T_z = -self.delta_theta * log_sigma * self.cos_lat_sq

        # Potential temperature to actual temperature conversion
        T_eq = np.maximum(200.0, (T_y + T_z) * sigma**self.kappa)

        return T_eq

    def _boundary_layer_friction(self, state):
        """
        Apply Rayleigh friction in boundary layer

        du/dt = -k_v * u
        dv/dt = -k_v * v

        where k_v is zero above the boundary layer.
        """
        for k in range(self.vgrid.nlev):
            # Sigma coordinate at this level
            sigma = state.p[k] / state.ps

            # Friction coefficient (non-zero only in boundary layer)
            # k_v = k_f * max(0, (sigma - sigma_b)/(1 - sigma_b))
            k_v = self.k_f * np.maximum(0, (sigma - self.sigma_b) / (1.0 - self.sigma_b))

            # Momentum tendencies
            du_dt = -k_v * state.u[k]
            dv_dt = -k_v * state.v[k]

            # Add to boundary layer tendencies
            state.physics_tendencies['boundary_layer']['u'][k] += du_dt
            state.physics_tendencies['boundary_layer']['v'][k] += dv_dt

    def get_equilibrium_temperature_profile(self, ps=None):
        """
        Return the equilibrium temperature profile for visualization

        Parameters
        ----------
        ps : ndarray, optional
            Surface pressure. If None, use reference pressure.

        Returns
        -------
        sigma : ndarray
            Sigma levels
        T_eq : ndarray
            Equilibrium temperature at each sigma level (nlev, nlat)
        """
        if ps is None:
            ps = np.full((self.grid.nlat, self.grid.nlon), self.p0)

        sigma_levels = self.vgrid.sigma

        T_eq_profile = np.zeros((self.vgrid.nlev, self.grid.nlat))

        for k, sigma in enumerate(sigma_levels):
            sigma_2d = np.full_like(ps, sigma)
            T_eq = self._equilibrium_temperature(sigma_2d)
            T_eq_profile[k] = np.mean(T_eq, axis=1)  # Zonal mean

        return sigma_levels, T_eq_profile


class HeldSuarezGCM:
    """
    Simplified GCM using only Held-Suarez forcing

    This is useful for testing the dynamical core without
    the complexity of full physics parameterizations.
    """

    def __init__(self, nlon=64, nlat=32, nlev=20, dt=None,
                 integration_method='rk3'):
        """
        Initialize Held-Suarez GCM

        Parameters
        ----------
        nlon : int
            Number of longitude points
        nlat : int
            Number of latitude points
        nlev : int
            Number of vertical levels
        dt : float, optional
            Time step (seconds). If None, computed automatically from CFL.
        integration_method : str
            Time integration method
        """
        # Import here to avoid circular imports
        from ..grid import SphericalGrid, VerticalGrid
        from ..core.state import ModelState
        from ..core.dynamics import AtmosphericDynamics
        from ..numerics import TimeIntegrator

        print(f"Initializing Held-Suarez GCM with resolution: {nlon}x{nlat}x{nlev}")

        # Grid
        self.grid = SphericalGrid(nlon, nlat, nlev)
        self.vgrid = VerticalGrid(nlev, coord_type='sigma')

        # Model state
        self.state = ModelState(self.grid, self.vgrid)

        # Time step - compute from CFL if not specified
        if dt is None:
            # CFL-based time step calculation
            # dx at equator (minimum is actually near poles but we use equator as reference)
            dx_equator = self.grid.radius * self.grid.dlon
            # dx at 80 degrees latitude (where polar filter kicks in)
            dx_polar = self.grid.radius * self.grid.dlon * np.cos(np.deg2rad(80))
            # Assume max wind speed of 100 m/s
            u_max = 100.0
            # CFL number of 0.5 for safety
            cfl = 0.5
            dt = cfl * min(dx_equator, dx_polar * 2) / u_max  # Factor of 2 for polar filter help
            # Round to nice value and cap
            dt = min(dt, 300.0)  # Cap at 300 seconds
            print(f"  CFL-based dt = {dt:.0f} seconds")
        self.dt = dt

        # Dynamics
        self.dynamics = AtmosphericDynamics(self.grid, self.vgrid)

        # Held-Suarez forcing
        self.held_suarez = HeldSuarezForcing(self.grid, self.vgrid)

        # Time integrator
        self.integrator = TimeIntegrator(method=integration_method)

        # Diagnostics
        self.diagnostics = {
            'time': [],
            'global_mean_T': [],
            'global_mean_KE': [],
            'max_u': [],
        }

    def initialize(self, perturbation=True, with_thermal_wind=True):
        """
        Initialize model state

        Parameters
        ----------
        perturbation : bool
            Whether to add small random perturbations to seed instabilities
        with_thermal_wind : bool
            Whether to initialize with thermal wind balanced jet
        """
        Rd = 287.0
        cp = 1004.0
        g = 9.81
        kappa = Rd / cp
        p0 = 100000.0
        Omega = 7.292e-5

        # Initialize surface pressure
        self.state.ps[:] = p0

        # Compute pressure levels
        _, self.state.p = self.vgrid.compute_pressure(self.state.ps)

        # Initialize temperature to equilibrium profile
        sigma_levels, T_eq_profile = self.held_suarez.get_equilibrium_temperature_profile(self.state.ps)

        for k in range(self.vgrid.nlev):
            sigma = self.vgrid.sigma[k]
            T_eq = self.held_suarez._equilibrium_temperature(
                np.full_like(self.state.ps, sigma)
            )
            self.state.T[k] = T_eq

        # Initialize winds
        if with_thermal_wind:
            # Initialize with proper thermal wind balanced jet
            # Thermal wind balance: f * du/dz = (R/T) * dT/dy
            # or equivalently: du/d(ln p) = (R/f) * dT/dy
            # This ensures geostrophic/hydrostatic balance
            lat = self.grid.lat2d
            Rd = 287.0

            # Compute thermal wind from temperature gradient
            self.state.u[:] = 0.0
            self.state.v[:] = 0.0

            # Integrate thermal wind from surface upward
            for k in range(self.vgrid.nlev - 1, -1, -1):
                sigma = self.vgrid.sigma[k]
                T_eq = self.held_suarez._equilibrium_temperature(
                    np.full_like(self.state.ps, sigma)
                )

                # Meridional temperature gradient
                dT_dy = self.grid.gradient_y(T_eq)

                # Coriolis parameter
                f = self.grid.f_coriolis
                f_safe = np.where(np.abs(f) < 1e-6, 1e-6 * np.sign(f + 1e-10), f)

                if k < self.vgrid.nlev - 1:
                    # Thermal wind increment
                    dlnp = np.log(self.state.p[k+1] / np.maximum(self.state.p[k], 100.0))
                    du_thermal = (Rd / f_safe) * dT_dy * dlnp

                    # Limit thermal wind increment to prevent extremes
                    du_thermal = np.clip(du_thermal, -5.0, 5.0)

                    self.state.u[k] = self.state.u[k+1] + du_thermal
                else:
                    # Start with small surface winds
                    self.state.u[k] = 0.0

            # Limit total wind speed
            max_init_u = 40.0
            self.state.u = np.clip(self.state.u, -max_init_u, max_init_u)

        else:
            # Initialize winds to zero (they will develop from instabilities)
            self.state.u[:] = 0.0
            self.state.v[:] = 0.0

        # Add small perturbations to seed baroclinic instability
        if perturbation:
            np.random.seed(42)
            for k in range(self.vgrid.nlev):
                # Add small temperature perturbations
                self.state.T[k] += 0.1 * np.random.randn(self.grid.nlat, self.grid.nlon)

                # Add small wind perturbations
                self.state.u[k] += 0.1 * np.random.randn(self.grid.nlat, self.grid.nlon)
                self.state.v[k] += 0.1 * np.random.randn(self.grid.nlat, self.grid.nlon)

        # Surface temperature
        self.state.tsurf = self.state.T[-1].copy()

        # Initialize humidity to zero (dry dynamics)
        self.state.q[:] = 0.0
        self.state.qc[:] = 0.0
        self.state.qi[:] = 0.0

        # Update diagnostics
        self.state.update_diagnostics()

        print("Held-Suarez GCM initialized!")
        print(f"  Initial max|u| = {np.max(np.abs(self.state.u)):.1f} m/s")

    def run(self, duration_days=100, output_interval_days=1):
        """
        Run the simulation

        Parameters
        ----------
        duration_days : float
            Simulation duration in days
        output_interval_days : float
            Output interval in days
        """
        import time as systime

        total_seconds = duration_days * 86400.0
        output_interval = output_interval_days * 86400.0

        n_steps = int(total_seconds / self.dt)
        output_frequency = int(output_interval / self.dt)

        print(f"\nStarting Held-Suarez simulation:")
        print(f"  Duration: {duration_days} days")
        print(f"  Time step: {self.dt} seconds")
        print(f"  Total steps: {n_steps}")

        start_time = systime.time()

        for step in range(n_steps):
            # Time integration
            self.integrator.step(self.state, self.dt, self._compute_tendencies)

            # Apply state limits to prevent runaway values
            self.dynamics.apply_state_limits(self.state)

            # Output diagnostics
            if step % output_frequency == 0:
                self._output_diagnostics()
                elapsed = systime.time() - start_time
                progress = (step + 1) / n_steps * 100
                sim_days = self.state.time / 86400.0

                print(f"Day {sim_days:.1f} / {duration_days} ({progress:.1f}%) - "
                      f"max|u| = {np.max(np.abs(self.state.u)):.1f} m/s - "
                      f"Elapsed: {elapsed:.1f}s")

        total_elapsed = systime.time() - start_time
        print(f"\nSimulation complete!")
        print(f"Total time: {total_elapsed:.1f} seconds")

    def _compute_tendencies(self, state):
        """Compute all tendencies"""
        state.reset_tendencies()

        # Dynamics
        self.dynamics.compute_tendencies(state)

        # Held-Suarez forcing
        self.held_suarez.compute_forcing(state, self.dt)

        # Sum physics tendencies
        state.dT_dt += state.physics_tendencies['radiation']['T']
        state.du_dt += state.physics_tendencies['boundary_layer']['u']
        state.dv_dt += state.physics_tendencies['boundary_layer']['v']

    def _output_diagnostics(self):
        """Compute and store diagnostics"""
        T_mean = self.grid.global_mean(self.state.T[self.vgrid.nlev//2])
        KE = 0.5 * (self.state.u**2 + self.state.v**2)
        KE_mean = self.grid.global_mean(KE[self.vgrid.nlev//2])

        self.diagnostics['time'].append(self.state.time / 86400.0)
        self.diagnostics['global_mean_T'].append(T_mean)
        self.diagnostics['global_mean_KE'].append(KE_mean)
        self.diagnostics['max_u'].append(np.max(np.abs(self.state.u)))
