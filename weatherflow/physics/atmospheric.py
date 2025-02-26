"""Atmospheric physics utility functions."""

import numpy as np
import torch
from typing import Optional, Union, Tuple

class AtmosphericPhysics:
    """Atmospheric physics calculations for Earth's atmosphere."""
    
    def __init__(
        self, 
        earth_radius: float = 6371e3,  # meters
        gravity: float = 9.81,         # m/s^2
        r_gas: float = 287.0,          # gas constant for dry air
        cp: float = 1004.0,            # specific heat at constant pressure
        omega: float = 7.292e-5,       # Earth's angular velocity
        grid_spacing: float = 2.5      # degrees
    ):
        """Initialize with physical constants.
        
        Args:
            earth_radius: Earth's radius in meters
            gravity: Gravitational acceleration in m/s^2
            r_gas: Gas constant for dry air in J/(kg·K)
            cp: Specific heat capacity at constant pressure in J/(kg·K)
            omega: Earth's angular velocity in rad/s
            grid_spacing: Grid spacing in degrees
        """
        self.earth_radius = earth_radius
        self.gravity = gravity
        self.r_gas = r_gas
        self.cp = cp
        self.omega = omega
        self.dx = grid_spacing
        self.dy = grid_spacing
        
    def coriolis_parameter(self, lat: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Calculate Coriolis parameter f = 2Ω·sin(φ).
        
        Args:
            lat: Latitude in degrees
            
        Returns:
            Coriolis parameter in s^-1
        """
        if isinstance(lat, torch.Tensor):
            return 2 * self.omega * torch.sin(torch.deg2rad(lat))
        else:
            return 2 * self.omega * np.sin(np.deg2rad(lat))
    
    def potential_temperature(
        self, 
        temperature: Union[np.ndarray, torch.Tensor], 
        pressure: Union[np.ndarray, torch.Tensor], 
        p0: float = 1000.0  # hPa
    ) -> Union[np.ndarray, torch.Tensor]:
        """Calculate potential temperature.
        
        Args:
            temperature: Temperature in K
            pressure: Pressure in hPa
            p0: Reference pressure in hPa
            
        Returns:
            Potential temperature in K
        """
        kappa = self.r_gas / self.cp
        
        if isinstance(temperature, torch.Tensor):
            return temperature * (p0 / pressure) ** kappa
        else:
            return temperature * (p0 / pressure) ** kappa
    
    def divergence(
        self, 
        u: Union[np.ndarray, torch.Tensor], 
        v: Union[np.ndarray, torch.Tensor], 
        lat: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Calculate horizontal divergence.
        
        Args:
            u: Zonal wind component (m/s)
            v: Meridional wind component (m/s)
            lat: Latitude in degrees
            
        Returns:
            Horizontal divergence (1/s)
        """
        if isinstance(u, torch.Tensor):
            cos_lat = torch.cos(torch.deg2rad(lat))
            dx_scaled = self.dx * np.pi/180 * self.earth_radius * cos_lat
            dy_scaled = self.dy * np.pi/180 * self.earth_radius
            
            dudx = torch.gradient(u, spacing=dx_scaled, dim=-1)[0]
            dvdy = torch.gradient(v, spacing=dy_scaled, dim=-2)[0]
        else:
            cos_lat = np.cos(np.deg2rad(lat))
            dx_scaled = self.dx * np.pi/180 * self.earth_radius * cos_lat
            dy_scaled = self.dy * np.pi/180 * self.earth_radius
            
            dudx = np.gradient(u, dx_scaled, axis=-1)
            dvdy = np.gradient(v, dy_scaled, axis=-2)
        
        return dudx + dvdy
    
    def vorticity(
        self, 
        u: Union[np.ndarray, torch.Tensor], 
        v: Union[np.ndarray, torch.Tensor], 
        lat: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Calculate relative vorticity.
        
        Args:
            u: Zonal wind component (m/s)
            v: Meridional wind component (m/s)
            lat: Latitude in degrees
            
        Returns:
            Relative vorticity (1/s)
        """
        if isinstance(u, torch.Tensor):
            cos_lat = torch.cos(torch.deg2rad(lat))
            dx_scaled = self.dx * np.pi/180 * self.earth_radius * cos_lat
            dy_scaled = self.dy * np.pi/180 * self.earth_radius
            
            dudy = torch.gradient(u, spacing=dy_scaled, dim=-2)[0]
            dvdx = torch.gradient(v, spacing=dx_scaled, dim=-1)[0]
        else:
            cos_lat = np.cos(np.deg2rad(lat))
            dx_scaled = self.dx * np.pi/180 * self.earth_radius * cos_lat
            dy_scaled = self.dy * np.pi/180 * self.earth_radius
            
            dudy = np.gradient(u, dy_scaled, axis=-2)
            dvdx = np.gradient(v, dx_scaled, axis=-1)
        
        return dvdx - dudy