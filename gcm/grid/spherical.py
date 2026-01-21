"""
Spherical grid for atmospheric GCM

Implements a latitude-longitude grid on a sphere with proper
metric terms and geometric factors.
"""

import numpy as np


class SphericalGrid:
    """Spherical grid for atmospheric modeling"""

    def __init__(self, nlon, nlat, nlev, planet_radius=6.371e6, rotation_rate=7.292e-5):
        """
        Initialize spherical grid

        Parameters
        ----------
        nlon : int
            Number of longitude points
        nlat : int
            Number of latitude points
        nlev : int
            Number of vertical levels
        planet_radius : float
            Planet radius in meters (default: Earth)
        rotation_rate : float
            Planetary rotation rate in rad/s (default: Earth's rotation)
            Set to 0 for non-rotating planet (e.g., Tropic World)
        """
        self.nlon = nlon
        self.nlat = nlat
        self.nlev = nlev
        self.radius = planet_radius
        self.rotation_rate = rotation_rate

        # Create coordinate arrays
        self.lon = np.linspace(0, 2*np.pi, nlon, endpoint=False)
        self.lat = np.linspace(-np.pi/2, np.pi/2, nlat)

        # Grid spacing
        self.dlon = 2 * np.pi / nlon
        self.dlat = np.pi / (nlat - 1)

        # 2D grid
        self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat, indexing='xy')

        # Metric terms
        self._compute_metric_terms()

        # Coriolis parameter
        self._compute_coriolis()

    def _compute_metric_terms(self):
        """Compute metric terms for spherical geometry"""
        # Map scale factors
        self.cos_lat = np.cos(self.lat2d)
        self.sin_lat = np.sin(self.lat2d)
        self.tan_lat = np.tan(self.lat2d)

        # Grid cell areas (per unit solid angle)
        # Area = R^2 * dlon * cos(lat) * dlat
        self.cell_area = (self.radius**2 * self.dlon *
                         np.abs(np.cos(self.lat2d)) * self.dlat)

        # Total surface area for normalization
        self.total_area = 4 * np.pi * self.radius**2

    def _compute_coriolis(self):
        """Compute Coriolis parameter

        Uses the rotation_rate set during initialization.
        For non-rotating planets (Tropic World), rotation_rate = 0.
        """
        self.f_coriolis = 2 * self.rotation_rate * self.sin_lat

    def gradient_x(self, field):
        """
        Compute zonal gradient on sphere

        Parameters
        ----------
        field : ndarray
            Field to differentiate (nlat, nlon)

        Returns
        -------
        grad : ndarray
            Zonal gradient
        """
        grad = np.zeros_like(field)
        grad[:, :] = np.gradient(field, self.dlon, axis=1)
        # Divide by R*cos(lat) for proper metric
        grad /= (self.radius * self.cos_lat)
        return grad

    def gradient_y(self, field):
        """
        Compute meridional gradient on sphere

        Parameters
        ----------
        field : ndarray
            Field to differentiate (nlat, nlon)

        Returns
        -------
        grad : ndarray
            Meridional gradient
        """
        grad = np.gradient(field, self.dlat, axis=0)
        # Divide by R for proper metric
        grad /= self.radius
        return grad

    def divergence(self, u, v):
        """
        Compute horizontal divergence on sphere

        Parameters
        ----------
        u : ndarray
            Zonal wind component
        v : ndarray
            Meridional wind component

        Returns
        -------
        div : ndarray
            Horizontal divergence
        """
        # d(u*cos(lat))/dlon
        du_dlon = np.gradient(u * self.cos_lat, self.dlon, axis=1)

        # d(v*cos(lat))/dlat
        dv_dlat = np.gradient(v * self.cos_lat, self.dlat, axis=0)

        # Divergence = 1/(R*cos(lat)) * [du/dlon + d(v*cos(lat))/dlat]
        div = (du_dlon + dv_dlat) / (self.radius * self.cos_lat)

        return div

    def vorticity(self, u, v):
        """
        Compute vertical component of vorticity on sphere

        Parameters
        ----------
        u : ndarray
            Zonal wind component
        v : ndarray
            Meridional wind component

        Returns
        -------
        vort : ndarray
            Vertical vorticity
        """
        # d(v*cos(lat))/dlon
        dv_dlon = np.gradient(v * self.cos_lat, self.dlon, axis=1)

        # d(u*cos(lat))/dlat
        du_dlat = np.gradient(u * self.cos_lat, self.dlat, axis=0)

        # Vorticity = 1/(R*cos(lat)) * [dv/dlon - d(u*cos(lat))/dlat]
        vort = (dv_dlon - du_dlat) / (self.radius * self.cos_lat)

        return vort

    def laplacian(self, field):
        """
        Compute Laplacian on sphere

        Parameters
        ----------
        field : ndarray
            Field to compute Laplacian

        Returns
        -------
        lap : ndarray
            Laplacian of field
        """
        # First derivatives
        df_dlon = np.gradient(field, self.dlon, axis=1)
        df_dlat = np.gradient(field, self.dlat, axis=0)

        # Second derivatives
        d2f_dlon2 = np.gradient(df_dlon, self.dlon, axis=1)

        # d/dlat(cos(lat) * df/dlat)
        d_coslat_dfdlat = np.gradient(self.cos_lat * df_dlat, self.dlat, axis=0)

        # Laplacian = 1/(R^2*cos^2(lat)) * d2f/dlon2 +
        #             1/(R^2*cos(lat)) * d/dlat(cos(lat)*df/dlat)
        lap = (d2f_dlon2 / (self.radius**2 * self.cos_lat**2) +
               d_coslat_dfdlat / (self.radius**2 * self.cos_lat))

        return lap

    def global_mean(self, field):
        """
        Compute area-weighted global mean

        Parameters
        ----------
        field : ndarray
            Field to average

        Returns
        -------
        mean : float
            Global mean value
        """
        weighted_sum = np.sum(field * self.cell_area)
        mean = weighted_sum / self.total_area
        return mean

    def zonal_mean(self, field):
        """
        Compute zonal mean

        Parameters
        ----------
        field : ndarray
            Field to average

        Returns
        -------
        zmean : ndarray
            Zonal mean (nlat,)
        """
        return np.mean(field, axis=1)
