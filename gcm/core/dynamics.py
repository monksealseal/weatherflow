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
        # Higher diffusion needed for numerical stability, especially near poles
        # Scale with resolution: higher resolution needs less diffusion
        # For coarse resolution (32x16), we need strong diffusion
        self.nu_horizontal = 1e6  # Horizontal diffusion (m^2/s) - very strong for stability
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
        # Apply polar filter first to smooth fields near poles
        self.apply_polar_filter(state)

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

        # Horizontal diffusion (increased for stability)
        self._horizontal_diffusion(state)

        # Additional polar damping for stability
        self._polar_damping(state)

        # Surface pressure tendency from mass continuity
        self._surface_pressure_tendency(state)

        # Safety: limit tendencies to prevent runaway instabilities
        self._limit_tendencies(state)

    def _advection(self, state):
        """
        Compute advection terms including metric (curvature) terms

        On a sphere, the momentum equations include additional terms
        from the curvature of the coordinate system:

        du/dt = ... + u*v*tan(lat)/R  (metric term)
        dv/dt = ... - u^2*tan(lat)/R  (metric term)

        These are essential for proper angular momentum conservation.
        """
        # Horizontal advection for each level
        for k in range(self.vgrid.nlev):
            u = state.u[k]
            v = state.v[k]

            # Advection of u
            du_dx = self.grid.gradient_x(u)
            du_dy = self.grid.gradient_y(u)
            state.du_dt[k] -= u * du_dx + v * du_dy

            # Metric term for u: +u*v*tan(lat)/R
            # This term arises from the curvature of the sphere
            tan_lat = self.grid.tan_lat
            metric_u = u * v * tan_lat / self.grid.radius
            state.du_dt[k] += metric_u

            # Advection of v
            dv_dx = self.grid.gradient_x(v)
            dv_dy = self.grid.gradient_y(v)
            state.dv_dt[k] -= u * dv_dx + v * dv_dy

            # Metric term for v: -u^2*tan(lat)/R
            metric_v = -u**2 * tan_lat / self.grid.radius
            state.dv_dt[k] += metric_v

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
        """
        Compute vertical pressure velocity (omega = dp/dt) from continuity equation

        The vertical velocity omega is computed from mass continuity:
        d(omega)/dp = -div(V)

        With boundary condition omega = 0 at top of atmosphere.

        Note: state.w stores omega (Pa/s), not vertical velocity w (m/s).
        The relationship is: omega = -rho * g * w
        """
        # Compute omega from continuity equation
        # Starting from top where omega = 0
        omega = np.zeros_like(state.u)

        for k in range(self.vgrid.nlev):
            # Horizontal divergence
            div = self.grid.divergence(state.u[k], state.v[k])

            # Limit divergence to prevent blowup
            # Typical divergence should be O(1e-5) to O(1e-4) s^-1
            div = np.clip(div, -1e-3, 1e-3)

            if k == 0:
                # At top: omega = 0
                omega[k] = 0.0
            else:
                # Integrate continuity equation downward
                # omega_k = omega_{k-1} + integral(div) dp
                dp = state.p[k] - state.p[k-1]
                # Ensure dp is positive and reasonable
                dp = np.maximum(dp, 100.0)  # At least 100 Pa layers
                # Average divergence over the layer
                div_prev = self.grid.divergence(state.u[k-1], state.v[k-1])
                div_prev = np.clip(div_prev, -1e-3, 1e-3)
                div_avg = 0.5 * (div + div_prev)
                omega[k] = omega[k-1] + div_avg * dp

        # Limit omega to reasonable values
        # Typical omega should be O(0.1-1) Pa/s for synoptic systems
        omega = np.clip(omega, -10.0, 10.0)

        # Store omega in state.w (Pa/s)
        state.w[:] = omega

        # Vertical advection using omega (d/dp terms)
        for k in range(1, self.vgrid.nlev - 1):
            omega_k = omega[k]

            # d/dp using centered differences
            dp_down = np.maximum(state.p[k+1] - state.p[k], 100.0)
            dp_up = np.maximum(state.p[k] - state.p[k-1], 100.0)
            dp_total = dp_down + dp_up

            # Vertical advection: omega * d(field)/dp
            # Advection of u
            du_dp = (state.u[k+1] - state.u[k-1]) / dp_total
            state.du_dt[k] -= omega_k * du_dp

            # Advection of v
            dv_dp = (state.v[k+1] - state.v[k-1]) / dp_total
            state.dv_dt[k] -= omega_k * dv_dp

            # Advection of T
            dT_dp = (state.T[k+1] - state.T[k-1]) / dp_total
            state.dT_dt[k] -= omega_k * dT_dp

            # Advection of q
            dq_dp = (state.q[k+1] - state.q[k-1]) / dp_total
            state.dq_dt[k] -= omega_k * dq_dp

            # Advection of cloud water
            dqc_dp = (state.qc[k+1] - state.qc[k-1]) / dp_total
            state.dqc_dt[k] -= omega_k * dqc_dp

            # Advection of cloud ice
            dqi_dp = (state.qi[k+1] - state.qi[k-1]) / dp_total
            state.dqi_dt[k] -= omega_k * dqi_dp

    def _adiabatic_temperature(self, state):
        """
        Adiabatic temperature change from vertical motion

        The thermodynamic equation in pressure coordinates:
        dT/dt = (kappa * T / p) * omega

        where kappa = R/cp and omega = dp/dt (Pa/s)

        This represents compression warming (omega > 0, descending)
        and expansion cooling (omega < 0, ascending).
        """
        for k in range(self.vgrid.nlev):
            # Adiabatic heating/cooling from vertical motion
            # dT/dt = (kappa * T / p) * omega
            # where omega is stored in state.w (Pa/s)
            omega = state.w[k]
            T = state.T[k]
            p = state.p[k]

            # Prevent division by very small pressure at top levels
            p_safe = np.maximum(p, 100.0)

            adiabatic_heating = (self.kappa * T / p_safe) * omega
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

    def _polar_damping(self, state):
        """
        Apply explicit damping near poles for numerical stability

        Near poles, the CFL condition becomes restrictive due to:
        1. Convergence of meridians (small dx)
        2. Large metric terms (tan_lat -> infinity)

        This adds Rayleigh damping to tendencies near poles.
        """
        # Threshold latitude for damping (degrees)
        damp_lat = 75.0  # Start damping at 75 degrees

        lat_deg = np.rad2deg(self.grid.lat)

        for i, lat in enumerate(lat_deg):
            if abs(lat) > damp_lat:
                # Damping strength increases toward pole
                # Use smooth function to avoid discontinuities
                dist_from_thresh = (abs(lat) - damp_lat) / (90.0 - damp_lat)
                # Damping timescale: 1 hour at pole, infinite at threshold
                tau = 3600.0  # 1 hour at pole
                damp_rate = dist_from_thresh**2 / tau

                for k in range(self.vgrid.nlev):
                    # Damp momentum tendencies toward zonal mean
                    u_mean = np.mean(state.u[k, i, :])
                    v_mean = np.mean(state.v[k, i, :])

                    # Add damping to tendencies
                    state.du_dt[k, i, :] -= damp_rate * (state.u[k, i, :] - u_mean)
                    state.dv_dt[k, i, :] -= damp_rate * (state.v[k, i, :] - v_mean)

    def _limit_tendencies(self, state, max_du_dt=0.01, max_dT_dt=0.01):
        """
        Limit tendencies to prevent numerical blowup

        This is a safety measure to prevent runaway instabilities.

        Parameters
        ----------
        max_du_dt : float
            Maximum wind tendency (m/s per second)
        max_dT_dt : float
            Maximum temperature tendency (K per second)
        """
        # Limit wind tendencies
        state.du_dt = np.clip(state.du_dt, -max_du_dt, max_du_dt)
        state.dv_dt = np.clip(state.dv_dt, -max_du_dt, max_du_dt)

        # Limit temperature tendencies
        state.dT_dt = np.clip(state.dT_dt, -max_dT_dt, max_dT_dt)

    def apply_state_limits(self, state, max_wind=150.0, T_min=150.0, T_max=350.0):
        """
        Apply physical limits to model state

        This prevents unrealistic values from developing during spin-up
        or due to numerical issues. Limits are applied gently using
        damping rather than hard clipping for momentum.

        Parameters
        ----------
        state : ModelState
            Model state to limit
        max_wind : float
            Maximum allowed wind speed (m/s)
        T_min, T_max : float
            Temperature bounds (K)
        """
        # Wind speed damping for excessive winds
        wind_speed = np.sqrt(state.u**2 + state.v**2)

        # Apply damping to excessive winds (scale down to max_wind)
        mask = wind_speed > max_wind
        if np.any(mask):
            # Damping factor: 1 where wind <= max_wind, <1 where wind > max_wind
            damp_factor = np.where(
                mask,
                max_wind / np.maximum(wind_speed, 1e-10),
                1.0
            )
            state.u *= damp_factor
            state.v *= damp_factor

        # Temperature limits
        state.T = np.clip(state.T, T_min, T_max)
