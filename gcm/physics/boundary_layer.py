"""
Planetary Boundary Layer (PBL) parameterization

Implements:
- Turbulent Kinetic Energy (TKE) scheme
- Non-local mixing
- Surface fluxes (momentum, heat, moisture)
- Monin-Obukhov similarity theory
- Vertical diffusion
"""

import numpy as np


class BoundaryLayerScheme:
    """
    Comprehensive boundary layer scheme with TKE-based turbulence
    """

    def __init__(self, grid, vgrid):
        """
        Initialize boundary layer scheme

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
        self.kappa = 0.4  # von Karman constant
        self.cp = 1004.0
        self.Rd = 287.0

        # PBL parameters
        self.Pr = 0.74  # Turbulent Prandtl number
        self.z_roughness_default = 0.1  # Default roughness length (m)

        # TKE parameters
        self.tke_min = 1e-6  # Minimum TKE (m^2/s^2)
        self.length_scale_max = 1000.0  # Maximum mixing length (m)

    def compute_boundary_layer(self, state, dt):
        """
        Compute boundary layer tendencies

        Parameters
        ----------
        state : ModelState
            Current model state
        dt : float
            Time step (s)

        Returns
        -------
        Modifies state.physics_tendencies['boundary_layer']
        """
        # Compute surface fluxes
        u_star, T_star, q_star = self._surface_fluxes(state)

        # Compute TKE
        tke = self._compute_tke(state, u_star)

        # Compute eddy diffusivities
        K_m, K_h = self._eddy_diffusivity(state, tke)

        # Vertical mixing
        self._vertical_mixing(state, K_m, K_h, u_star, T_star, q_star)

    def _surface_fluxes(self, state):
        """
        Compute surface fluxes using Monin-Obukhov similarity theory

        Returns
        -------
        u_star : ndarray
            Friction velocity (m/s)
        T_star : ndarray
            Temperature scale (K)
        q_star : ndarray
            Moisture scale (kg/kg)
        """
        # Surface layer (lowest model level)
        k_sfc = self.vgrid.nlev - 1

        u_sfc = state.u[k_sfc]
        v_sfc = state.v[k_sfc]
        T_sfc = state.T[k_sfc]
        q_sfc = state.q[k_sfc]

        # Surface values
        T_ground = state.tsurf
        q_ground = state.qsurf
        z0 = np.maximum(state.z0, 1e-4)  # Avoid zero roughness

        # Wind speed at lowest level
        wind_speed = np.sqrt(u_sfc**2 + v_sfc**2) + 1e-3  # Avoid division by zero

        # Height of lowest level (approximate)
        z_sfc = state.z[k_sfc]

        # Neutral transfer coefficients
        Cd_n = (self.kappa / np.log(z_sfc / z0))**2
        Ch_n = Cd_n / self.Pr

        # Stability correction (simplified Richardson number approach)
        Ri = self.g * z_sfc * (T_sfc - T_ground) / (T_sfc * wind_speed**2 + 1e-6)

        # Stability functions (simple parameterization)
        psi_m = np.zeros_like(Ri)
        psi_h = np.zeros_like(Ri)

        # Stable conditions (Ri > 0)
        stable = Ri > 0
        psi_m[stable] = -5 * Ri[stable]
        psi_h[stable] = -5 * Ri[stable]

        # Unstable conditions (Ri < 0)
        unstable = Ri < 0
        x = (1 - 16 * Ri[unstable])**0.25
        psi_m[unstable] = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi/2
        psi_h[unstable] = 2 * np.log((1 + x**2) / 2)

        # Corrected transfer coefficients
        Cd = Cd_n / (1 + psi_m)**2
        Ch = Ch_n / ((1 + psi_m) * (1 + psi_h))

        # Surface fluxes
        # Momentum flux: τ = ρ * Cd * U^2
        rho_sfc = state.rho[k_sfc]
        tau = rho_sfc * Cd * wind_speed**2

        # Friction velocity
        u_star = np.sqrt(tau / rho_sfc)

        # Heat flux: H = ρ * cp * Ch * U * (θ_ground - θ_sfc)
        theta_sfc = T_sfc * (100000.0 / state.p[k_sfc])**(self.Rd / self.cp)
        theta_ground = T_ground * (100000.0 / state.ps)**(self.Rd / self.cp)

        H = rho_sfc * self.cp * Ch * wind_speed * (theta_ground - theta_sfc)

        # Temperature scale
        T_star = -H / (rho_sfc * self.cp * u_star + 1e-10)

        # Moisture flux: E = ρ * Ch * U * (q_ground - q_sfc)
        E = rho_sfc * Ch * wind_speed * (q_ground - q_sfc)

        # Moisture scale
        q_star = -E / (rho_sfc * u_star + 1e-10)

        return u_star, T_star, q_star

    def _compute_tke(self, state, u_star):
        """
        Compute turbulent kinetic energy

        Parameters
        ----------
        state : ModelState
            Model state
        u_star : ndarray
            Surface friction velocity

        Returns
        -------
        tke : ndarray
            TKE at each level (m^2/s^2)
        """
        tke = np.zeros_like(state.T)

        # Surface layer TKE proportional to u_star^2
        k_sfc = self.vgrid.nlev - 1
        tke[k_sfc] = 3.75 * u_star**2

        # TKE decreases with height in boundary layer
        # Simple exponential profile
        pbl_height = 1000.0  # Approximate PBL height (m)

        for k in range(self.vgrid.nlev - 2, -1, -1):
            z = state.z[k]
            z_sfc = state.z[k_sfc]
            height_above_sfc = z - z_sfc

            # Exponential decay
            tke[k] = tke[k_sfc] * np.exp(-height_above_sfc / pbl_height)

        # Ensure minimum TKE
        tke = np.maximum(tke, self.tke_min)

        return tke

    def _eddy_diffusivity(self, state, tke):
        """
        Compute eddy diffusivity from TKE

        Parameters
        ----------
        state : ModelState
            Model state
        tke : ndarray
            Turbulent kinetic energy

        Returns
        -------
        K_m : ndarray
            Eddy diffusivity for momentum (m^2/s)
        K_h : ndarray
            Eddy diffusivity for heat (m^2/s)
        """
        # Mixing length
        l_mix = np.zeros_like(tke)

        pbl_height = 1000.0  # m

        for k in range(self.vgrid.nlev):
            # Height above surface
            z = state.z[k] - state.z[-1]

            # Mixing length: l = kappa * z / (1 + kappa * z / lambda)
            # where lambda is asymptotic length scale
            l_mix[k] = self.kappa * z / (1 + self.kappa * z / pbl_height)

        # Cap mixing length
        l_mix = np.minimum(l_mix, self.length_scale_max)

        # Eddy diffusivity: K = c_k * l * sqrt(TKE)
        c_k = 0.1  # Constant from TKE theory
        K_m = c_k * l_mix * np.sqrt(tke)

        # Heat diffusivity related to momentum diffusivity
        K_h = K_m / self.Pr

        return K_m, K_h

    def _vertical_mixing(self, state, K_m, K_h, u_star, T_star, q_star):
        """
        Apply vertical diffusion

        Parameters
        ----------
        state : ModelState
            Model state
        K_m : ndarray
            Eddy diffusivity for momentum
        K_h : ndarray
            Eddy diffusivity for heat
        u_star : ndarray
            Friction velocity
        T_star : ndarray
            Temperature scale
        q_star : ndarray
            Moisture scale
        """
        # Vertical diffusion using implicit scheme
        # d(phi)/dt = d/dz(K * d(phi)/dz)

        # For simplicity, use explicit forward differencing
        # (Should be implicit for stability, but explicit for clarity here)

        for k in range(1, self.vgrid.nlev - 1):
            # Layer spacing
            dz_up = state.z[k-1] - state.z[k]
            dz_down = state.z[k] - state.z[k+1]
            dz_avg = 0.5 * (dz_up + dz_down)

            # Diffusivity at interfaces (average)
            K_m_up = 0.5 * (K_m[k-1] + K_m[k])
            K_m_down = 0.5 * (K_m[k] + K_m[k+1])
            K_h_up = 0.5 * (K_h[k-1] + K_h[k])
            K_h_down = 0.5 * (K_h[k] + K_h[k+1])

            # Momentum diffusion
            du_dz_up = (state.u[k-1] - state.u[k]) / dz_up
            du_dz_down = (state.u[k] - state.u[k+1]) / dz_down
            flux_u_up = K_m_up * du_dz_up
            flux_u_down = K_m_down * du_dz_down
            state.physics_tendencies['boundary_layer']['u'][k] += (flux_u_up - flux_u_down) / dz_avg

            dv_dz_up = (state.v[k-1] - state.v[k]) / dz_up
            dv_dz_down = (state.v[k] - state.v[k+1]) / dz_down
            flux_v_up = K_m_up * dv_dz_up
            flux_v_down = K_m_down * dv_dz_down
            state.physics_tendencies['boundary_layer']['v'][k] += (flux_v_up - flux_v_down) / dz_avg

            # Heat diffusion
            dT_dz_up = (state.T[k-1] - state.T[k]) / dz_up
            dT_dz_down = (state.T[k] - state.T[k+1]) / dz_down
            flux_T_up = K_h_up * dT_dz_up
            flux_T_down = K_h_down * dT_dz_down
            state.physics_tendencies['boundary_layer']['T'][k] += (flux_T_up - flux_T_down) / dz_avg

            # Moisture diffusion
            dq_dz_up = (state.q[k-1] - state.q[k]) / dz_up
            dq_dz_down = (state.q[k] - state.q[k+1]) / dz_down
            flux_q_up = K_h_up * dq_dz_up
            flux_q_down = K_h_down * dq_dz_down
            state.physics_tendencies['boundary_layer']['q'][k] += (flux_q_up - flux_q_down) / dz_avg

        # Surface layer (apply surface fluxes)
        k_sfc = self.vgrid.nlev - 1
        dz_sfc = state.z[k_sfc-1] - state.z[k_sfc]

        # Momentum flux from surface
        state.physics_tendencies['boundary_layer']['u'][k_sfc] -= u_star**2 / dz_sfc * np.sign(state.u[k_sfc])
        state.physics_tendencies['boundary_layer']['v'][k_sfc] -= u_star**2 / dz_sfc * np.sign(state.v[k_sfc])

        # Heat flux from surface
        state.physics_tendencies['boundary_layer']['T'][k_sfc] += u_star * T_star / dz_sfc

        # Moisture flux from surface
        state.physics_tendencies['boundary_layer']['q'][k_sfc] += u_star * q_star / dz_sfc
