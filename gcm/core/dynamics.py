"""
Atmospheric dynamics engine

Implements the primitive equations for atmospheric flow on a rotating sphere.
Includes advection, pressure gradient force, Coriolis force, and gravity.
"""

import numpy as np


class AtmosphericDynamics:
    """
    Solver for atmospheric primitive equations

    Equations solved:
    - Zonal momentum: du/dt = -u*du/dx - v*du/dy - w*du/dp + f*v - (1/rho)*dp/dx
    - Meridional momentum: dv/dt = -u*dv/dx - v*dv/dy - w*dv/dp - f*u - (1/rho)*dp/dy
    - Thermodynamic: dT/dt = -u*dT/dx - v*dT/dy - w*dT/dp + Q/cp
    - Continuity: dps/dt = -div(u, v)
    - Hydrostatic balance: dΦ/dp = -RT/p
    """

    def __init__(self, grid, vgrid):
        """
        Initialize dynamics solver

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
        self.Rd = 287.0      # Gas constant for dry air (J/kg/K)
        self.cp = 1004.0     # Specific heat at constant pressure (J/kg/K)
        self.cv = 717.0      # Specific heat at constant volume (J/kg/K)
        self.g = 9.81        # Gravitational acceleration (m/s^2)
        self.kappa = self.Rd / self.cp

        # Numerical diffusion coefficients
        self.nu_horizontal = 1e5  # Horizontal diffusion (m^2/s)
        self.nu_vertical = 1.0    # Vertical diffusion (m^2/s)

    def compute_tendencies(self, state):
        """
        Compute dynamical tendencies for all prognostic variables

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        Modifies state.du_dt, state.dv_dt, state.dT_dt, state.dq_dt, state.dps_dt
        """
        # Advection
        self._advection(state)

        # Pressure gradient force
        self._pressure_gradient(state)

        # Coriolis force
        self._coriolis(state)

        # Vertical motion
        self._vertical_motion(state)

        # Adiabatic heating/cooling from vertical motion
        self._adiabatic_temperature(state)

        # Horizontal diffusion
        self._horizontal_diffusion(state)

        # Surface pressure tendency from mass continuity
        self._surface_pressure_tendency(state)

    def _advection(self, state):
        """Compute advection terms"""
        # Horizontal advection for each level
        for k in range(self.vgrid.nlev):
            u = state.u[k]
            v = state.v[k]

            # Advection of u
            du_dx = self.grid.gradient_x(u)
            du_dy = self.grid.gradient_y(u)
            state.du_dt[k] -= u * du_dx + v * du_dy

            # Advection of v
            dv_dx = self.grid.gradient_x(v)
            dv_dy = self.grid.gradient_y(v)
            state.dv_dt[k] -= u * dv_dx + v * dv_dy

            # Advection of T
            dT_dx = self.grid.gradient_x(state.T[k])
            dT_dy = self.grid.gradient_y(state.T[k])
            state.dT_dt[k] -= u * dT_dx + v * dT_dy

            # Advection of q
            dq_dx = self.grid.gradient_x(state.q[k])
            dq_dy = self.grid.gradient_y(state.q[k])
            state.dq_dt[k] -= u * dq_dx + v * dq_dy

            # Advection of cloud water
            dqc_dx = self.grid.gradient_x(state.qc[k])
            dqc_dy = self.grid.gradient_y(state.qc[k])
            state.dqc_dt[k] -= u * dqc_dx + v * dqc_dy

            # Advection of cloud ice
            dqi_dx = self.grid.gradient_x(state.qi[k])
            dqi_dy = self.grid.gradient_y(state.qi[k])
            state.dqi_dt[k] -= u * dqi_dx + v * dqi_dy

    def _pressure_gradient(self, state):
        """Compute pressure gradient force"""
        for k in range(self.vgrid.nlev):
            # Geopotential height at this level
            z = state.z[k]

            # Geopotential gradient = pressure gradient / rho
            dphi_dx = self.g * self.grid.gradient_x(z)
            dphi_dy = self.g * self.grid.gradient_y(z)

            # Add to momentum tendencies
            state.du_dt[k] -= dphi_dx
            state.dv_dt[k] -= dphi_dy

    def _coriolis(self, state):
        """Compute Coriolis force"""
        for k in range(self.vgrid.nlev):
            # f * v for zonal wind
            state.du_dt[k] += self.grid.f_coriolis * state.v[k]

            # -f * u for meridional wind
            state.dv_dt[k] -= self.grid.f_coriolis * state.u[k]

    def _vertical_motion(self, state):
        """Compute vertical velocity from continuity equation"""
        # omega = dp/dt from mass continuity
        # omega_k = omega_{k-1} - int_{p_{k-1}}^{p_k} div(V) dp

        for k in range(self.vgrid.nlev):
            # Horizontal divergence
            div = self.grid.divergence(state.u[k], state.v[k])

            if k == 0:
                # At top: omega = 0
                state.w[k] = 0.0
            else:
                # Integrate continuity equation
                dp = state.p[k] - state.p[k-1]
                state.w[k] = state.w[k-1] - state.rho[k] * div * dp

        # Vertical advection
        for k in range(1, self.vgrid.nlev - 1):
            w = state.w[k]

            # d/dp using centered differences
            dp_down = state.p[k+1] - state.p[k]
            dp_up = state.p[k] - state.p[k-1]

            # Advection of u
            du_dp = (state.u[k+1] - state.u[k-1]) / (dp_down + dp_up)
            state.du_dt[k] -= w * du_dp

            # Advection of v
            dv_dp = (state.v[k+1] - state.v[k-1]) / (dp_down + dp_up)
            state.dv_dt[k] -= w * dv_dp

            # Advection of T
            dT_dp = (state.T[k+1] - state.T[k-1]) / (dp_down + dp_up)
            state.dT_dt[k] -= w * dT_dp

            # Advection of q
            dq_dp = (state.q[k+1] - state.q[k-1]) / (dp_down + dp_up)
            state.dq_dt[k] -= w * dq_dp

    def _adiabatic_temperature(self, state):
        """Adiabatic temperature change from vertical motion"""
        for k in range(self.vgrid.nlev):
            # Adiabatic heating/cooling: dT/dt = (T/theta) * (dtheta/dt)
            # For adiabatic process: dtheta/dt = 0, but vertical motion causes
            # compression/expansion heating/cooling

            # dT/dt = (kappa * T / p) * omega
            # where omega = dp/dt
            adiabatic_heating = (self.kappa * state.T[k] / state.p[k]) * state.w[k]
            state.dT_dt[k] += adiabatic_heating

    def _horizontal_diffusion(self, state):
        """Add horizontal diffusion for numerical stability"""
        for k in range(self.vgrid.nlev):
            # Diffusion = nu * Laplacian
            state.du_dt[k] += self.nu_horizontal * self.grid.laplacian(state.u[k])
            state.dv_dt[k] += self.nu_horizontal * self.grid.laplacian(state.v[k])
            state.dT_dt[k] += self.nu_horizontal * self.grid.laplacian(state.T[k])
            state.dq_dt[k] += self.nu_horizontal * self.grid.laplacian(state.q[k])

    def _surface_pressure_tendency(self, state):
        """Compute surface pressure tendency from mass continuity"""
        # dps/dt = -∫ div(ρ*V) dp from surface to top
        # Approximation: sum over all layers

        state.dps_dt[:] = 0.0

        for k in range(self.vgrid.nlev):
            div = self.grid.divergence(state.u[k], state.v[k])

            # Mass in this layer
            if k == 0:
                dp = state.p[k]
            else:
                dp = state.p[k] - state.p[k-1]

            state.dps_dt -= div * dp / self.g

    def compute_energy_diagnostics(self, state):
        """
        Compute energy diagnostics

        Returns
        -------
        diagnostics : dict
            Dictionary containing energy diagnostics
        """
        # Kinetic energy
        KE = 0.5 * (state.u**2 + state.v**2)

        # Potential energy
        PE = self.g * state.z

        # Internal energy
        IE = self.cv * state.T

        # Total energy per unit mass
        total_energy = KE + PE + IE

        # Global averages (mass-weighted)
        diagnostics = {
            'kinetic_energy': self._global_mass_weighted_mean(KE, state),
            'potential_energy': self._global_mass_weighted_mean(PE, state),
            'internal_energy': self._global_mass_weighted_mean(IE, state),
            'total_energy': self._global_mass_weighted_mean(total_energy, state)
        }

        return diagnostics

    def _global_mass_weighted_mean(self, field, state):
        """Compute global mass-weighted mean of a 3D field"""
        total_mass = 0.0
        weighted_sum = 0.0

        for k in range(self.vgrid.nlev):
            # Mass in this layer
            if k == 0:
                dp = state.p[k]
            else:
                dp = state.p[k] - state.p[k-1]

            mass = dp / self.g * self.grid.cell_area

            weighted_sum += np.sum(field[k] * mass)
            total_mass += np.sum(mass)

        return weighted_sum / total_mass

    def apply_polar_filter(self, state):
        """
        Apply polar filter to avoid CFL issues near poles

        Smooths fields near poles using zonal averaging
        """
        # Threshold latitude for filtering (degrees)
        filter_lat = 80.0  # degrees

        lat_deg = np.rad2deg(self.grid.lat)

        for i, lat in enumerate(lat_deg):
            if abs(lat) > filter_lat:
                # Strength of filtering increases toward pole
                strength = (abs(lat) - filter_lat) / (90.0 - filter_lat)

                # Apply zonal averaging
                for k in range(self.vgrid.nlev):
                    u_mean = np.mean(state.u[k, i, :])
                    v_mean = np.mean(state.v[k, i, :])
                    T_mean = np.mean(state.T[k, i, :])
                    q_mean = np.mean(state.q[k, i, :])

                    state.u[k, i, :] = (1 - strength) * state.u[k, i, :] + strength * u_mean
                    state.v[k, i, :] = (1 - strength) * state.v[k, i, :] + strength * v_mean
                    state.T[k, i, :] = (1 - strength) * state.T[k, i, :] + strength * T_mean
                    state.q[k, i, :] = (1 - strength) * state.q[k, i, :] + strength * q_mean
