"""
Eddy Diagnostics for GCM

Computes eddy statistics:
- Stationary eddies: [u*v*], [v*T*] where * = deviation from zonal mean
- Transient eddies: [u'v'], [v'T'] where ' = deviation from time mean
- Eddy kinetic energy
- Eddy heat transport
- Eddy momentum transport
"""

import numpy as np
from collections import deque


class EddyDiagnostics:
    """
    Compute eddy diagnostics for atmospheric circulation

    Eddies are decomposed into:
    - Stationary eddies: standing waves, time-mean deviations from zonal mean
    - Transient eddies: time-varying fluctuations

    Total eddy flux = stationary + transient
    """

    def __init__(self, grid, vgrid, time_mean_window=10):
        """
        Initialize eddy diagnostics

        Parameters
        ----------
        grid : SphericalGrid
            Horizontal grid
        vgrid : VerticalGrid
            Vertical grid
        time_mean_window : int
            Number of time steps for computing time mean (for transient eddies)
        """
        self.grid = grid
        self.vgrid = vgrid
        self.nlat = grid.nlat
        self.nlon = grid.nlon
        self.nlev = vgrid.nlev

        # Storage for time series (for transient eddy calculation)
        self.time_mean_window = time_mean_window
        self.state_history = deque(maxlen=time_mean_window)

        # Running sums for time mean
        self.n_samples = 0
        self.u_sum = None
        self.v_sum = None
        self.T_sum = None
        self.uv_sum = None
        self.vT_sum = None

    def zonal_mean(self, field):
        """Compute zonal mean"""
        return np.mean(field, axis=-1)

    def zonal_deviation(self, field):
        """Compute deviation from zonal mean"""
        zm = self.zonal_mean(field)
        return field - zm[:, :, np.newaxis] if field.ndim == 3 else field - zm[:, np.newaxis]

    def add_state(self, state):
        """
        Add state to time series for transient eddy calculation

        Parameters
        ----------
        state : ModelState
            Current model state
        """
        # Store copy of relevant fields
        state_copy = {
            'u': state.u.copy(),
            'v': state.v.copy(),
            'T': state.T.copy(),
            'theta': state.theta.copy()
        }
        self.state_history.append(state_copy)

        # Update running sums
        if self.u_sum is None:
            self.u_sum = np.zeros_like(state.u)
            self.v_sum = np.zeros_like(state.v)
            self.T_sum = np.zeros_like(state.T)
            self.uv_sum = np.zeros_like(state.u)
            self.vT_sum = np.zeros_like(state.T)

        self.u_sum += state.u
        self.v_sum += state.v
        self.T_sum += state.T
        self.uv_sum += state.u * state.v
        self.vT_sum += state.v * state.T
        self.n_samples += 1

    def compute_stationary_eddy_fluxes(self, state):
        """
        Compute stationary eddy fluxes

        Stationary eddies: u* = u - [u], v* = v - [v], T* = T - [T]
        Stationary momentum flux: [u*v*]
        Stationary heat flux: [v*T*]

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        uv_stationary : ndarray
            Stationary eddy momentum flux [u*v*] (nlev, nlat)
        vT_stationary : ndarray
            Stationary eddy heat flux [v*T*] (nlev, nlat)
        """
        # Deviations from zonal mean
        u_star = self.zonal_deviation(state.u)
        v_star = self.zonal_deviation(state.v)
        T_star = self.zonal_deviation(state.T)

        # Zonal mean of products
        uv_stationary = self.zonal_mean(u_star * v_star)
        vT_stationary = self.zonal_mean(v_star * T_star)

        return uv_stationary, vT_stationary

    def compute_transient_eddy_fluxes(self):
        """
        Compute transient eddy fluxes from accumulated time series

        Transient eddies: u' = u - u_bar, v' = v - v_bar where bar = time mean
        Transient momentum flux: [u'v']
        Transient heat flux: [v'T']

        Returns
        -------
        uv_transient : ndarray
            Transient eddy momentum flux (nlev, nlat), or None if insufficient data
        vT_transient : ndarray
            Transient eddy heat flux (nlev, nlat)
        """
        if self.n_samples < 2:
            return None, None

        # Time mean
        u_mean = self.u_sum / self.n_samples
        v_mean = self.v_sum / self.n_samples
        T_mean = self.T_sum / self.n_samples

        # Time mean of products
        uv_mean = self.uv_sum / self.n_samples
        vT_mean = self.vT_sum / self.n_samples

        # Transient covariance: [u'v'] = [uv] - [u][v]
        uv_transient_3d = uv_mean - u_mean * v_mean
        vT_transient_3d = vT_mean - v_mean * T_mean

        # Take zonal mean
        uv_transient = self.zonal_mean(uv_transient_3d)
        vT_transient = self.zonal_mean(vT_transient_3d)

        return uv_transient, vT_transient

    def compute_eddy_kinetic_energy(self, state):
        """
        Compute eddy kinetic energy

        EKE = 0.5 * ([u*^2] + [v*^2])

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        eke : ndarray
            Eddy kinetic energy (nlev, nlat)
        eke_stationary : ndarray
            Stationary EKE contribution
        """
        u_star = self.zonal_deviation(state.u)
        v_star = self.zonal_deviation(state.v)

        eke = 0.5 * self.zonal_mean(u_star**2 + v_star**2)

        # Can also decompose into stationary vs transient
        # For now just return total EKE
        return eke

    def compute_transient_eke(self):
        """
        Compute transient eddy kinetic energy from time series

        Transient EKE = 0.5 * ([u'^2] + [v'^2])

        Returns
        -------
        eke_transient : ndarray
            Transient EKE (nlev, nlat), or None if insufficient data
        """
        if self.n_samples < 2:
            return None

        # Time mean
        u_mean = self.u_sum / self.n_samples
        v_mean = self.v_sum / self.n_samples

        # Compute variance
        u2_sum = np.zeros_like(u_mean)
        v2_sum = np.zeros_like(v_mean)

        for state in self.state_history:
            u_prime = state['u'] - u_mean
            v_prime = state['v'] - v_mean
            u2_sum += u_prime**2
            v2_sum += v_prime**2

        u2_mean = u2_sum / len(self.state_history)
        v2_mean = v2_sum / len(self.state_history)

        eke_transient = 0.5 * self.zonal_mean(u2_mean + v2_mean)

        return eke_transient

    def compute_heat_transport(self, state):
        """
        Compute meridional heat transport

        Q = cp * integral([v*T*] + [v'T']) dp/g

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        total_heat_transport : ndarray
            Total heat transport (nlat,) in PW
        stationary_component : ndarray
            Stationary eddy contribution
        mean_component : ndarray
            Mean meridional circulation contribution
        """
        cp = 1004.0
        g = 9.81
        a = self.grid.radius

        # Stationary eddy heat flux
        _, vT_stat = self.compute_stationary_eddy_fluxes(state)

        # Mean circulation heat flux
        v_bar = self.zonal_mean(state.v)
        T_bar = self.zonal_mean(state.T)
        vT_mean = v_bar * T_bar

        # Total heat flux
        v = state.v
        T = state.T
        vT_total = self.zonal_mean(v * T)

        # Integrate vertically
        p_bar = self.zonal_mean(state.p)
        cos_lat = np.cos(self.grid.lat)

        # Heat transport = 2*pi*a*cos(phi) * cp * integral(vT) dp/g
        stationary_transport = np.zeros(self.nlat)
        mean_transport = np.zeros(self.nlat)
        total_transport = np.zeros(self.nlat)

        for j in range(self.nlat):
            for k in range(1, self.nlev):
                dp = p_bar[k, j] - p_bar[k-1, j]
                stationary_transport[j] += vT_stat[k, j] * dp
                mean_transport[j] += vT_mean[k, j] * dp
                total_transport[j] += vT_total[k, j] * dp

        # Convert to PW
        coeff = 2 * np.pi * a * cos_lat * cp / g / 1e15
        stationary_transport *= coeff
        mean_transport *= coeff
        total_transport *= coeff

        return total_transport, stationary_transport, mean_transport

    def compute_momentum_transport(self, state):
        """
        Compute meridional momentum transport

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        uv_total : ndarray
            Total momentum flux [uv] (nlev, nlat)
        uv_stationary : ndarray
            Stationary component [u*v*]
        uv_mean : ndarray
            Mean component [u][v]
        """
        uv_stationary, _ = self.compute_stationary_eddy_fluxes(state)

        v_bar = self.zonal_mean(state.v)
        u_bar = self.zonal_mean(state.u)
        uv_mean = u_bar * v_bar

        uv_total = self.zonal_mean(state.u * state.v)

        return uv_total, uv_stationary, uv_mean

    def compute_all_eddy_diagnostics(self, state):
        """
        Compute comprehensive eddy diagnostics

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        diagnostics : dict
            Dictionary containing all eddy diagnostics
        """
        # Add to time series
        self.add_state(state)

        # Stationary eddy fluxes
        uv_stat, vT_stat = self.compute_stationary_eddy_fluxes(state)

        # Transient eddy fluxes (if enough data)
        uv_trans, vT_trans = self.compute_transient_eddy_fluxes()

        # EKE
        eke = self.compute_eddy_kinetic_energy(state)
        eke_trans = self.compute_transient_eke()

        # Heat transport
        Q_total, Q_stat, Q_mean = self.compute_heat_transport(state)

        # Momentum transport
        M_total, M_stat, M_mean = self.compute_momentum_transport(state)

        # Pressure levels
        p_bar = self.zonal_mean(state.p) / 100  # hPa

        return {
            'latitude': np.rad2deg(self.grid.lat),
            'pressure': p_bar,
            'uv_stationary': uv_stat,
            'vT_stationary': vT_stat,
            'uv_transient': uv_trans,
            'vT_transient': vT_trans,
            'eke': eke,
            'eke_transient': eke_trans,
            'heat_transport_total': Q_total,
            'heat_transport_stationary': Q_stat,
            'heat_transport_mean': Q_mean,
            'momentum_transport_total': M_total,
            'momentum_transport_stationary': M_stat,
            'momentum_transport_mean': M_mean,
            'n_samples': self.n_samples
        }

    def reset(self):
        """Reset accumulated statistics"""
        self.state_history.clear()
        self.n_samples = 0
        self.u_sum = None
        self.v_sum = None
        self.T_sum = None
        self.uv_sum = None
        self.vT_sum = None
