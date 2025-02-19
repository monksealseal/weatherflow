
import numpy as np

class AtmosphericPhysics:
    """Core atmospheric physics calculations from our successful experiments."""
    
    def __init__(self, grid_spacing=2.5, pressure_levels=None):
        self.dx = grid_spacing  # degrees
        self.dy = grid_spacing
        self.pressure_levels = pressure_levels or np.array([1000, 850, 500, 250])
        self.earth_radius = 6371e3  # meters
        self.omega = 7.292e-5  # Earth's angular velocity
        self.g = 9.81  # gravitational acceleration
        
    def coriolis_parameter(self, lat):
        """Calculate Coriolis parameter (f)."""
        return 2 * self.omega * np.sin(np.deg2rad(lat))
    
    def potential_temperature(self, temperature, pressure):
        """Calculate potential temperature (theta)."""
        p0 = 1000  # reference pressure (hPa)
        R = 287.0  # gas constant for dry air
        cp = 1004.0  # specific heat at constant pressure
        return temperature * (p0/pressure)**(R/cp)
    
    def static_stability(self, temperature, pressure):
        """Calculate static stability."""
        theta = self.potential_temperature(temperature, pressure)
        return np.gradient(theta, pressure, axis=0)
    
    def vorticity(self, u, v, lat):
        """Calculate relative vorticity."""
        dx = self.dx * np.pi/180 * self.earth_radius * np.cos(np.deg2rad(lat))
        dy = self.dy * np.pi/180 * self.earth_radius
        dudy = np.gradient(u, dy, axis=0)
        dvdx = np.gradient(v, dx, axis=1)
        return dvdx - dudy
