"""
Model state container

Holds all prognostic and diagnostic variables for the GCM
"""

import numpy as np


class ModelState:
    """Container for model state variables"""

    def __init__(self, grid, vgrid):
        """
        Initialize model state

        Parameters
        ----------
        grid : SphericalGrid
            Horizontal grid
        vgrid : VerticalGrid
            Vertical grid
        """
        self.grid = grid
        self.vgrid = vgrid

        nlat = grid.nlat
        nlon = grid.nlon
        nlev = vgrid.nlev

        # Prognostic variables (time-evolved)
        self.u = np.zeros((nlev, nlat, nlon))      # Zonal wind (m/s)
        self.v = np.zeros((nlev, nlat, nlon))      # Meridional wind (m/s)
        self.w = np.zeros((nlev, nlat, nlon))      # Vertical velocity (Pa/s)
        self.T = np.zeros((nlev, nlat, nlon))      # Temperature (K)
        self.q = np.zeros((nlev, nlat, nlon))      # Specific humidity (kg/kg)
        self.qc = np.zeros((nlev, nlat, nlon))     # Cloud liquid water (kg/kg)
        self.qi = np.zeros((nlev, nlat, nlon))     # Cloud ice (kg/kg)
        self.ps = np.zeros((nlat, nlon))           # Surface pressure (Pa)

        # Diagnostic variables
        self.z = np.zeros((nlev, nlat, nlon))      # Geopotential height (m)
        self.p = np.zeros((nlev, nlat, nlon))      # Pressure (Pa)
        self.rho = np.zeros((nlev, nlat, nlon))    # Density (kg/m^3)
        self.theta = np.zeros((nlev, nlat, nlon))  # Potential temperature (K)

        # Tendencies (time derivatives)
        self.du_dt = np.zeros((nlev, nlat, nlon))
        self.dv_dt = np.zeros((nlev, nlat, nlon))
        self.dT_dt = np.zeros((nlev, nlat, nlon))
        self.dq_dt = np.zeros((nlev, nlat, nlon))
        self.dqc_dt = np.zeros((nlev, nlat, nlon))
        self.dqi_dt = np.zeros((nlev, nlat, nlon))
        self.dps_dt = np.zeros((nlat, nlon))

        # Surface fields
        self.tsurf = np.zeros((nlat, nlon))        # Surface temperature (K)
        self.qsurf = np.zeros((nlat, nlon))        # Surface humidity (kg/kg)
        self.albedo = np.zeros((nlat, nlon))       # Surface albedo
        self.z0 = np.zeros((nlat, nlon))           # Surface roughness (m)

        # Physics tendencies (separated for diagnostics)
        self.physics_tendencies = {
            'radiation': {'T': np.zeros((nlev, nlat, nlon))},
            'convection': {
                'T': np.zeros((nlev, nlat, nlon)),
                'q': np.zeros((nlev, nlat, nlon)),
                'u': np.zeros((nlev, nlat, nlon)),
                'v': np.zeros((nlev, nlat, nlon))
            },
            'cloud_micro': {
                'T': np.zeros((nlev, nlat, nlon)),
                'q': np.zeros((nlev, nlat, nlon)),
                'qc': np.zeros((nlev, nlat, nlon)),
                'qi': np.zeros((nlev, nlat, nlon))
            },
            'boundary_layer': {
                'T': np.zeros((nlev, nlat, nlon)),
                'q': np.zeros((nlev, nlat, nlon)),
                'u': np.zeros((nlev, nlat, nlon)),
                'v': np.zeros((nlev, nlat, nlon))
            },
            'surface': {
                'T': np.zeros((nlev, nlat, nlon)),
                'q': np.zeros((nlev, nlat, nlon))
            }
        }

        # Time
        self.time = 0.0  # seconds since start

    def initialize_atmosphere(self, profile_type='tropical'):
        """
        Initialize atmospheric state with realistic profiles

        Parameters
        ----------
        profile_type : str
            Type of atmospheric profile: 'tropical', 'midlatitude', 'polar'
        """
        Rd = 287.0  # Gas constant for dry air
        cp = 1004.0  # Specific heat at constant pressure
        g = 9.81

        # Initialize surface pressure
        self.ps[:] = 101325.0  # Standard sea level pressure

        # Compute pressure levels
        _, self.p = self.vgrid.compute_pressure(self.ps)

        if profile_type == 'tropical':
            # Warm tropical profile
            T_surf = 300.0  # K
            lapse_rate = 6.5e-3  # K/m
            q_surf = 0.018  # kg/kg (humid)

        elif profile_type == 'midlatitude':
            # Mid-latitude profile
            T_surf = 288.0  # K
            lapse_rate = 6.5e-3  # K/m
            q_surf = 0.010  # kg/kg

        elif profile_type == 'polar':
            # Cold polar profile
            T_surf = 260.0  # K
            lapse_rate = 5.0e-3  # K/m
            q_surf = 0.001  # kg/kg (dry)

        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

        # Set surface temperature with latitude variation
        lat_factor = np.cos(self.grid.lat2d)
        self.tsurf = T_surf * (0.7 + 0.3 * lat_factor)

        # Initialize temperature profile
        for k in range(self.vgrid.nlev):
            # Approximate height from pressure
            z_approx = -Rd * T_surf / g * np.log(self.p[k] / self.ps)

            # Temperature decreases with height
            self.T[k] = self.tsurf - lapse_rate * z_approx

            # Add stratospheric temperature inversion above tropopause
            tropopause_p = 20000.0  # Pa
            if np.mean(self.p[k]) < tropopause_p:
                # Isothermal or slight warming in stratosphere
                T_tropo = self.tsurf - lapse_rate * (-Rd * T_surf / g *
                                                    np.log(tropopause_p / self.ps))
                self.T[k] = T_tropo

        # Initialize humidity (decreases exponentially with height)
        for k in range(self.vgrid.nlev):
            scale_height = 2000.0  # m
            z_approx = -Rd * T_surf / g * np.log(self.p[k] / self.ps)
            self.q[k] = q_surf * np.exp(-z_approx / scale_height)

        # Initialize winds (simple jet stream)
        lat_deg = np.rad2deg(self.grid.lat2d)
        for k in range(self.vgrid.nlev):
            # Subtropical jet around 30 degrees
            jet_strength = 30.0 * (self.p[k] / self.ps[0, 0])**0.5  # Stronger aloft
            self.u[k] = jet_strength * np.exp(-((lat_deg - 30)**2) / 400.0)

            # Weak meridional circulation
            self.v[k] = 2.0 * np.sin(self.grid.lat2d * 2)

        # Update diagnostic variables
        self.update_diagnostics()

    def update_diagnostics(self):
        """Update diagnostic variables from prognostic state"""
        Rd = 287.0
        cp = 1004.0
        p0 = 100000.0  # Reference pressure for potential temperature

        # Pressure
        _, self.p = self.vgrid.compute_pressure(self.ps)

        # Density from ideal gas law
        self.rho = self.p / (Rd * self.T)

        # Potential temperature
        self.theta = self.T * (p0 / self.p)**(Rd / cp)

        # Geopotential height
        self.z = self.vgrid.compute_geopotential_height(self.T, self.ps, self.q)

    def reset_tendencies(self):
        """Reset all tendency fields to zero"""
        self.du_dt[:] = 0.0
        self.dv_dt[:] = 0.0
        self.dT_dt[:] = 0.0
        self.dq_dt[:] = 0.0
        self.dqc_dt[:] = 0.0
        self.dqi_dt[:] = 0.0
        self.dps_dt[:] = 0.0

        # Reset physics tendencies
        for process in self.physics_tendencies.values():
            for field in process.values():
                field[:] = 0.0

    def copy(self):
        """Create a deep copy of the state"""
        import copy
        return copy.deepcopy(self)
