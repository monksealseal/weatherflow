"""
Ocean Mixed Layer Model

Implements:
- Slab ocean with heat capacity
- Ocean heat transport
- Sea ice thermodynamics
- Ocean-atmosphere coupling
"""

import numpy as np


class OceanMixedLayerModel:
    """
    Simple slab ocean model with thermodynamics
    """

    def __init__(self, grid):
        """
        Initialize ocean mixed layer model

        Parameters
        ----------
        grid : SphericalGrid
            Horizontal grid
        """
        self.grid = grid

        # Physical constants
        self.rho_ocean = 1025.0  # Ocean density (kg/m^3)
        self.cp_ocean = 4000.0   # Ocean heat capacity (J/kg/K)
        self.rho_ice = 917.0     # Sea ice density (kg/m^3)
        self.Lf = 3.34e5         # Latent heat of fusion (J/kg)

        # Ocean parameters
        self.mixed_layer_depth = 50.0  # m
        self.q_transport = 30.0  # Ocean heat transport magnitude (W/m^2)

        # Ocean state
        nlat, nlon = grid.nlat, grid.nlon
        self.sst = np.zeros((nlat, nlon))  # Sea surface temperature (K)
        self.ice_thickness = np.zeros((nlat, nlon))  # Sea ice thickness (m)
        self.ice_fraction = np.zeros((nlat, nlon))  # Sea ice fraction

        # Ocean mask (1 = ocean, 0 = land)
        # Simplified: all points are ocean
        self.ocean_mask = np.ones((nlat, nlon))

        # Initialize SST
        self._initialize_sst()

    def _initialize_sst(self):
        """Initialize SST with realistic distribution"""
        # Temperature decreases from equator to poles
        lat = self.grid.lat2d

        # Tropical SST ~ 300K, polar ~ 271K (freezing point)
        T_equator = 300.0
        T_pole = 271.0

        self.sst = T_equator - (T_equator - T_pole) * np.abs(np.sin(lat))**2

        # Initialize sea ice in polar regions
        T_freeze = 271.4  # Freezing point of seawater (K)
        ice_regions = self.sst < T_freeze

        self.ice_thickness[ice_regions] = 2.0  # 2m ice thickness
        self.ice_fraction[ice_regions] = 1.0
        self.sst[ice_regions] = T_freeze  # Ice-covered ocean at freezing point

    def compute_ocean(self, state, dt, net_radiation, sensible_flux, latent_flux):
        """
        Compute ocean mixed layer tendencies

        Parameters
        ----------
        state : ModelState
            Current model state
        dt : float
            Time step (s)
        net_radiation : ndarray
            Net radiation at surface (W/m^2)
        sensible_flux : ndarray
            Sensible heat flux (W/m^2)
        latent_flux : ndarray
            Latent heat flux (W/m^2)

        Returns
        -------
        Modifies self.sst, self.ice_thickness, self.ice_fraction
        Updates state.tsurf over ocean
        """
        # Ocean heat budget
        self._ocean_heat_budget(dt, net_radiation, sensible_flux, latent_flux)

        # Ocean heat transport
        self._ocean_heat_transport(dt)

        # Sea ice thermodynamics
        self._sea_ice_thermodynamics(state, dt, net_radiation, sensible_flux, latent_flux)

        # Update surface temperature over ocean
        state.tsurf = np.where(self.ocean_mask > 0.5, self.sst, state.tsurf)

    def _ocean_heat_budget(self, dt, net_radiation, sensible_flux, latent_flux):
        """
        Update SST from surface heat fluxes

        Heat budget:
        ρ * cp * h * dT/dt = F_net

        where h is mixed layer depth, F_net is net heat flux
        """
        # Net heat flux into ocean (positive downward)
        F_net = net_radiation - sensible_flux - latent_flux

        # Heat capacity of mixed layer
        C_ocean = self.rho_ocean * self.cp_ocean * self.mixed_layer_depth

        # Temperature change
        # Only update ice-free ocean
        ice_free = (self.ice_fraction < 0.01) & (self.ocean_mask > 0.5)

        dSST = F_net * dt / C_ocean * ice_free
        self.sst += dSST

        # Prevent SST from falling below freezing
        T_freeze = 271.4
        self.sst = np.maximum(self.sst, T_freeze)

    def _ocean_heat_transport(self, dt):
        """
        Parameterize ocean heat transport

        Simple diffusive representation:
        Q = -k * ∇T

        Transports heat from warm equator to cold poles
        """
        # Meridional heat transport (simplified)
        # Use Laplacian to diffuse heat

        # Diffusivity (large for ocean)
        kappa = 2000.0  # m^2/s

        # Compute Laplacian of SST
        lap_sst = self.grid.laplacian(self.sst)

        # Update SST from heat transport
        dSST_transport = kappa * lap_sst * dt / (self.mixed_layer_depth * self.cp_ocean)

        self.sst += dSST_transport * self.ocean_mask

    def _sea_ice_thermodynamics(self, state, dt, net_radiation, sensible_flux, latent_flux):
        """
        Sea ice formation and melt

        If SST tries to fall below freezing, form ice
        If ice exists and heat is available, melt ice
        """
        T_freeze = 271.4  # K

        # Ice formation (when SST at freezing and heat loss continues)
        F_net = net_radiation - sensible_flux - latent_flux

        # Points at freezing with heat loss
        freezing_points = (self.sst <= T_freeze) & (F_net < 0) & (self.ocean_mask > 0.5)

        if np.any(freezing_points):
            # Energy available for freezing
            energy_for_freezing = -F_net * dt * freezing_points

            # Mass of ice formed: m = E / Lf
            ice_mass_formed = energy_for_freezing / self.Lf

            # Ice thickness increase
            d_ice_thickness = ice_mass_formed / self.rho_ice

            self.ice_thickness += d_ice_thickness
            self.ice_fraction[freezing_points] = np.minimum(1.0,
                                                            self.ice_fraction[freezing_points] + 0.1)

        # Ice melt (when heat flux is positive and ice exists)
        melting_points = (self.ice_thickness > 0) & (F_net > 0) & (self.ocean_mask > 0.5)

        if np.any(melting_points):
            # Energy available for melting
            energy_for_melting = F_net * dt * melting_points

            # Mass of ice melted
            ice_mass_melted = energy_for_melting / self.Lf

            # Ice thickness decrease
            d_ice_thickness = ice_mass_melted / self.rho_ice

            self.ice_thickness -= d_ice_thickness
            self.ice_thickness = np.maximum(0, self.ice_thickness)

            # Update ice fraction
            # Fraction related to thickness (simplified)
            self.ice_fraction = np.minimum(1.0, self.ice_thickness / 1.0)  # 1m for full cover
            self.ice_fraction = np.maximum(0, self.ice_fraction)

        # Ice-covered points stay at freezing
        ice_covered = (self.ice_fraction > 0.01) & (self.ocean_mask > 0.5)
        self.sst[ice_covered] = T_freeze

    def get_surface_properties(self):
        """
        Get surface properties for atmosphere coupling

        Returns
        -------
        surface_temp : ndarray
            Surface temperature (K)
        albedo : ndarray
            Surface albedo
        roughness : ndarray
            Surface roughness length (m)
        """
        # Temperature
        surface_temp = self.sst.copy()

        # Albedo: open ocean has low albedo, ice has high albedo
        albedo_ocean = 0.06
        albedo_ice = 0.7

        albedo = (self.ice_fraction * albedo_ice +
                 (1 - self.ice_fraction) * albedo_ocean)

        # Roughness: ocean is smooth, ice is rougher
        z0_ocean = 1e-4  # m
        z0_ice = 1e-3    # m

        roughness = (self.ice_fraction * z0_ice +
                    (1 - self.ice_fraction) * z0_ocean)

        return surface_temp, albedo, roughness
