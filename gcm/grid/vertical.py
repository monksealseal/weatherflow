"""
Vertical grid and coordinate systems

Implements sigma and hybrid sigma-pressure coordinates
"""

import numpy as np


class VerticalGrid:
    """Vertical coordinate system for atmospheric models"""

    def __init__(self, nlev, coord_type='sigma', ptop=100.0, psurf=101325.0):
        """
        Initialize vertical grid

        Parameters
        ----------
        nlev : int
            Number of vertical levels
        coord_type : str
            Coordinate type: 'sigma' or 'hybrid'
        ptop : float
            Pressure at model top (Pa). Default 100 Pa (~0.1 hPa, upper stratosphere)
            Must be > 0 to avoid numerical issues with log(0).
        psurf : float
            Reference surface pressure (Pa)
        """
        self.nlev = nlev
        self.coord_type = coord_type
        # Ensure ptop > 0 to avoid log(0) issues
        self.ptop = max(ptop, 1.0)  # At least 1 Pa
        self.psurf = psurf

        if coord_type == 'sigma':
            self._init_sigma_coords()
        elif coord_type == 'hybrid':
            self._init_hybrid_coords()
        else:
            raise ValueError(f"Unknown coordinate type: {coord_type}")

    def _init_sigma_coords(self):
        """Initialize sigma coordinates (terrain-following)"""
        # Sigma at layer interfaces (0 = top, 1 = surface)
        self.sigma_interface = np.linspace(0, 1, self.nlev + 1)

        # Sigma at layer centers
        self.sigma = 0.5 * (self.sigma_interface[:-1] + self.sigma_interface[1:])

        # Sigma layer thickness
        self.dsigma = np.diff(self.sigma_interface)

        # For pure sigma: p = ptop + sigma * (ps - ptop)
        self.ak = np.full(self.nlev + 1, self.ptop)
        self.bk = self.sigma_interface

    def _init_hybrid_coords(self):
        """Initialize hybrid sigma-pressure coordinates"""
        # Transition from pressure to sigma coordinates
        # Generally: p = ak + bk * ps
        # where ak -> constant in stratosphere, bk -> 0
        #       ak -> 0 in troposphere, bk -> sigma

        self.sigma_interface = np.linspace(0, 1, self.nlev + 1)
        self.sigma = 0.5 * (self.sigma_interface[:-1] + self.sigma_interface[1:])

        # Define hybrid coefficients
        # Smooth transition around sigma = 0.3
        sigma_trans = 0.3
        transition = 0.5 * (1 + np.tanh((self.sigma_interface - sigma_trans) / 0.1))

        # ak: pressure contribution (high in stratosphere)
        self.ak = self.ptop + (self.psurf - self.ptop) * (1 - transition) * self.sigma_interface

        # bk: sigma contribution (high in troposphere)
        self.bk = transition * self.sigma_interface

    def compute_pressure(self, ps):
        """
        Compute pressure at all levels

        Parameters
        ----------
        ps : ndarray or float
            Surface pressure (Pa)

        Returns
        -------
        p_interface : ndarray
            Pressure at layer interfaces
        p_center : ndarray
            Pressure at layer centers
        """
        # Pressure at interfaces: p = ak + bk * ps
        p_interface = self.ak[:, None, None] + self.bk[:, None, None] * ps[None, :, :]

        # Ensure positive pressures for log calculation
        p_interface = np.maximum(p_interface, 1.0)  # At least 1 Pa

        # Pressure at centers (log-pressure average)
        # This is more accurate than arithmetic average for hydrostatic atmosphere
        p_center = np.exp(0.5 * (np.log(p_interface[:-1]) + np.log(p_interface[1:])))

        return p_interface, p_center

    def compute_geopotential_height(self, T, ps, q=None):
        """
        Compute geopotential height using hydrostatic equation

        Parameters
        ----------
        T : ndarray
            Temperature (K) with shape (nlev, nlat, nlon)
        ps : ndarray
            Surface pressure (Pa) with shape (nlat, nlon)
        q : ndarray, optional
            Specific humidity (kg/kg)

        Returns
        -------
        z : ndarray
            Geopotential height (m) at layer centers
        """
        Rd = 287.0  # Gas constant for dry air (J/kg/K)
        Rv = 461.5  # Gas constant for water vapor (J/kg/K)
        g = 9.81    # Gravity (m/s^2)

        # Compute pressure at interfaces and centers
        p_interface, p_center = self.compute_pressure(ps)

        # Virtual temperature
        if q is not None:
            Tv = T * (1 + 0.61 * q)  # Virtual temperature correction
        else:
            Tv = T

        # Initialize geopotential height
        z = np.zeros_like(T)

        # Integrate hydrostatic equation from surface upward
        # phi_{k} = phi_{k+1} + R*Tv/g * ln(p_{k+1}/p_k)
        # Start from surface (z = 0)
        for k in range(self.nlev - 1, -1, -1):
            if k == self.nlev - 1:
                # Surface layer
                z[k] = (Rd * Tv[k] / g *
                       np.log(p_interface[k+1] / p_center[k]))
            else:
                # Upper layers
                z[k] = z[k+1] + (Rd * Tv[k] / g *
                                np.log(p_interface[k+1] / p_center[k]))

        return z

    def vertical_derivative(self, field, ps):
        """
        Compute vertical derivative in pressure coordinates

        Parameters
        ----------
        field : ndarray
            Field to differentiate (nlev, nlat, nlon)
        ps : ndarray
            Surface pressure (Pa)

        Returns
        -------
        dfield_dp : ndarray
            Vertical derivative
        """
        _, p_center = self.compute_pressure(ps)

        # Compute d(field)/dp using centered differences
        dfield_dp = np.zeros_like(field)

        # Interior points
        for k in range(1, self.nlev - 1):
            dp = p_center[k+1] - p_center[k-1]
            dfield_dp[k] = (field[k+1] - field[k-1]) / dp

        # Boundaries (one-sided differences)
        dfield_dp[0] = (field[1] - field[0]) / (p_center[1] - p_center[0])
        dfield_dp[-1] = (field[-1] - field[-2]) / (p_center[-1] - p_center[-2])

        return dfield_dp

    def pressure_to_sigma(self, p, ps):
        """
        Convert pressure to sigma coordinate

        Parameters
        ----------
        p : float or ndarray
            Pressure (Pa)
        ps : float or ndarray
            Surface pressure (Pa)

        Returns
        -------
        sigma : float or ndarray
            Sigma coordinate value
        """
        if self.coord_type == 'sigma':
            return (p - self.ptop) / (ps - self.ptop)
        else:
            # For hybrid coords, need to solve: p = ak + bk*ps for sigma
            # This is an approximation
            return (p - self.ptop) / (ps - self.ptop)
