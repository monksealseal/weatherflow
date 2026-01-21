"""
Zonal Mean Diagnostics for GCM

Computes zonal mean quantities:
- Mean zonal wind [u]
- Mean meridional wind [v]
- Mean temperature [T]
- Meridional mass streamfunction (Hadley cell)
- Zonal mean moisture budget
"""

import numpy as np


class ZonalMeanDiagnostics:
    """
    Compute zonal mean diagnostics for atmospheric circulation
    """

    def __init__(self, grid, vgrid):
        """
        Initialize zonal mean diagnostics

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
        self.radius = grid.radius
        self.g = 9.81

    def zonal_mean(self, field):
        """
        Compute zonal mean of a field

        Parameters
        ----------
        field : ndarray
            3D field (nlev, nlat, nlon) or 2D field (nlat, nlon)

        Returns
        -------
        zm : ndarray
            Zonal mean (nlev, nlat) or (nlat,)
        """
        return np.mean(field, axis=-1)

    def compute_zonal_mean_winds(self, state):
        """
        Compute zonal mean wind components

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        u_bar : ndarray
            Zonal mean zonal wind [u] (nlev, nlat)
        v_bar : ndarray
            Zonal mean meridional wind [v] (nlev, nlat)
        """
        u_bar = self.zonal_mean(state.u)
        v_bar = self.zonal_mean(state.v)
        return u_bar, v_bar

    def compute_zonal_mean_temperature(self, state):
        """
        Compute zonal mean temperature

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        T_bar : ndarray
            Zonal mean temperature [T] (nlev, nlat)
        theta_bar : ndarray
            Zonal mean potential temperature [theta] (nlev, nlat)
        """
        T_bar = self.zonal_mean(state.T)
        theta_bar = self.zonal_mean(state.theta)
        return T_bar, theta_bar

    def compute_meridional_streamfunction(self, state):
        """
        Compute meridional mass streamfunction

        Psi = (2 * pi * a * cos(phi) / g) * integral_0^p [v] dp

        This shows the Hadley, Ferrel, and Polar cells.

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        psi : ndarray
            Meridional streamfunction (nlev, nlat) in kg/s
        """
        v_bar = self.zonal_mean(state.v)
        p_bar = self.zonal_mean(state.p)

        # Streamfunction: integrate v from top of atmosphere
        psi = np.zeros((self.nlev, self.nlat))

        # Coefficient: 2 * pi * a * cos(phi) / g
        cos_lat = np.cos(self.grid.lat)
        coeff = 2 * np.pi * self.radius * cos_lat / self.g

        for j in range(self.nlat):
            # Integrate from top downward
            for k in range(1, self.nlev):
                dp = p_bar[k, j] - p_bar[k-1, j]
                psi[k, j] = psi[k-1, j] + coeff[j] * v_bar[k, j] * dp

        # Convert to more intuitive units (10^9 kg/s = Sv equivalent)
        psi = psi / 1e9

        return psi

    def compute_angular_momentum(self, state):
        """
        Compute angular momentum diagnostics

        M = (u + Omega * a * cos(phi)) * a * cos(phi)

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        M_bar : ndarray
            Zonal mean angular momentum (nlev, nlat)
        """
        Omega = self.grid.rotation_rate
        a = self.radius

        # Use 1D cos_lat for zonal mean calculations
        cos_lat = np.cos(self.grid.lat)  # Shape: (nlat,)

        # Full angular momentum per unit mass
        u_bar = self.zonal_mean(state.u)  # Shape: (nlev, nlat)

        # Broadcast cos_lat to match u_bar shape
        M = (u_bar + Omega * a * cos_lat[np.newaxis, :]) * a * cos_lat[np.newaxis, :]

        return M

    def compute_thermal_wind(self, state):
        """
        Compute thermal wind shear

        From thermal wind balance:
        du/dz = -(g/f) * (1/T) * dT/dy

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        du_dz_thermal : ndarray
            Thermal wind shear (nlev-1, nlat)
        """
        f = self.grid.f_coriolis[self.nlat//2]  # Representative f
        T_bar = self.zonal_mean(state.T)
        z_bar = self.zonal_mean(state.z)

        # Avoid division by zero at equator
        f = np.where(np.abs(self.grid.f_coriolis) < 1e-10, 1e-10, self.grid.f_coriolis)

        # dT/dy
        dlat = self.grid.dlat
        dT_dy = np.gradient(T_bar, dlat * self.radius, axis=1)

        # Thermal wind
        du_dz_thermal = -self.g / f * dT_dy / T_bar

        return du_dz_thermal

    def compute_ep_flux(self, state):
        """
        Compute Eliassen-Palm (EP) flux diagnostics

        EP flux measures wave activity propagation and is crucial for
        understanding stratosphere-troposphere coupling.

        F_phi = -rho * [u'v']
        F_p = rho * f * [v'theta'] / (d[theta]/dz)

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        F_phi : ndarray
            Meridional component of EP flux (nlev, nlat)
        F_p : ndarray
            Vertical component of EP flux (nlev, nlat)
        ep_div : ndarray
            EP flux divergence (acceleration, nlev, nlat)
        """
        # Eddy components
        u_bar = self.zonal_mean(state.u)
        v_bar = self.zonal_mean(state.v)
        theta_bar = self.zonal_mean(state.theta)
        rho_bar = self.zonal_mean(state.rho)

        u_prime = state.u - u_bar[:, :, np.newaxis]
        v_prime = state.v - v_bar[:, :, np.newaxis]
        theta_prime = state.theta - theta_bar[:, :, np.newaxis]

        # Eddy covariances
        uv_bar = self.zonal_mean(u_prime * v_prime)
        vtheta_bar = self.zonal_mean(v_prime * theta_prime)

        # Meridional EP flux component
        F_phi = -rho_bar * uv_bar

        # Vertical EP flux component
        # Need d[theta]/dp
        p_bar = self.zonal_mean(state.p)
        dtheta_dp = np.zeros((self.nlev, self.nlat))
        for j in range(self.nlat):
            dtheta_dp[:, j] = np.gradient(theta_bar[:, j], p_bar[:, j])

        # Avoid division by zero
        dtheta_dp = np.where(np.abs(dtheta_dp) < 1e-10, 1e-10, dtheta_dp)

        # Use 1D Coriolis for zonal mean
        f = 2 * self.grid.rotation_rate * np.sin(self.grid.lat)  # Shape: (nlat,)
        F_p = rho_bar * f[np.newaxis, :] * vtheta_bar / dtheta_dp

        # EP flux divergence (gives zonal acceleration)
        # div(F) = (1/(a*cos(phi))) * d(F_phi*cos(phi))/dphi + dF_p/dp
        cos_lat = np.cos(self.grid.lat)  # Shape: (nlat,)

        # Meridional divergence
        d_Fphi_dphi = np.gradient(F_phi * cos_lat[np.newaxis, :], self.grid.dlat, axis=1)
        div_phi = d_Fphi_dphi / (self.radius * cos_lat[np.newaxis, :])

        # Vertical divergence
        div_p = np.zeros((self.nlev, self.nlat))
        for j in range(self.nlat):
            div_p[:, j] = np.gradient(F_p[:, j], p_bar[:, j])

        ep_div = div_phi + div_p

        return F_phi, F_p, ep_div

    def compute_all_zonal_diagnostics(self, state):
        """
        Compute comprehensive zonal mean diagnostics

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        diagnostics : dict
            Dictionary containing all zonal mean diagnostics
        """
        u_bar, v_bar = self.compute_zonal_mean_winds(state)
        T_bar, theta_bar = self.compute_zonal_mean_temperature(state)
        psi = self.compute_meridional_streamfunction(state)
        M = self.compute_angular_momentum(state)

        # Pressure levels for plotting
        p_bar = self.zonal_mean(state.p)

        # Jet stream indices
        u_max_idx = np.unravel_index(np.argmax(np.abs(u_bar)), u_bar.shape)
        jet_lat = np.rad2deg(self.grid.lat[u_max_idx[1]])
        jet_pressure = p_bar[u_max_idx] / 100  # hPa
        jet_speed = u_bar[u_max_idx]

        return {
            'latitude': np.rad2deg(self.grid.lat),
            'pressure': p_bar / 100,  # hPa
            'u_bar': u_bar,
            'v_bar': v_bar,
            'T_bar': T_bar,
            'theta_bar': theta_bar,
            'streamfunction': psi,
            'angular_momentum': M,
            'jet_latitude': jet_lat,
            'jet_pressure': jet_pressure,
            'jet_speed': jet_speed
        }
