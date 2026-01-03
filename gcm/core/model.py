"""
Main GCM Model Class

Integrates all components:
- Grid system
- Dynamics
- Physics parameterizations
- Time integration
- I/O and diagnostics
"""

import numpy as np
import time as systime
from ..grid import SphericalGrid, VerticalGrid
from ..core.state import ModelState
from ..core.dynamics import AtmosphericDynamics
from ..physics import (RadiationScheme, ConvectionScheme, CloudMicrophysics,
                       BoundaryLayerScheme, LandSurfaceModel)
from ..physics.ocean import OceanMixedLayerModel
from ..numerics import TimeIntegrator


class GCM:
    """
    General Circulation Model

    Main model class that coordinates all components
    """

    def __init__(self, nlon=64, nlat=32, nlev=20, dt=600.0,
                 integration_method='rk3', co2_ppmv=400.0):
        """
        Initialize GCM

        Parameters
        ----------
        nlon : int
            Number of longitude points
        nlat : int
            Number of latitude points
        nlev : int
            Number of vertical levels
        dt : float
            Time step (seconds)
        integration_method : str
            Time integration method ('euler', 'rk3', 'leapfrog', 'ab2')
        co2_ppmv : float
            CO2 concentration (ppmv)
        """
        print(f"Initializing GCM with resolution: {nlon}x{nlat}x{nlev}")

        # Grid
        self.grid = SphericalGrid(nlon, nlat, nlev)
        self.vgrid = VerticalGrid(nlev, coord_type='sigma')

        # Model state
        self.state = ModelState(self.grid, self.vgrid)

        # Time step
        self.dt = dt

        # Components
        print("Initializing dynamics...")
        self.dynamics = AtmosphericDynamics(self.grid, self.vgrid)

        print("Initializing physics schemes...")
        self.radiation = RadiationScheme(self.grid, self.vgrid, co2_ppmv=co2_ppmv)
        self.convection = ConvectionScheme(self.grid, self.vgrid)
        self.cloud_micro = CloudMicrophysics(self.grid, self.vgrid)
        self.boundary_layer = BoundaryLayerScheme(self.grid, self.vgrid)
        self.land_surface = LandSurfaceModel(self.grid, self.vgrid)
        self.ocean = OceanMixedLayerModel(self.grid)

        # Time integrator
        print(f"Initializing time integrator ({integration_method})...")
        self.integrator = TimeIntegrator(method=integration_method)

        # Diagnostics storage
        self.diagnostics = {
            'time': [],
            'global_mean_T': [],
            'global_mean_precip': [],
            'total_energy': [],
            'kinetic_energy': [],
        }

        # Configuration
        self.co2_ppmv = co2_ppmv

        print("GCM initialization complete!")

    def initialize(self, profile='tropical', sst_pattern='realistic'):
        """
        Initialize model state

        Parameters
        ----------
        profile : str
            Atmospheric profile type: 'tropical', 'midlatitude', 'polar'
        sst_pattern : str
            SST initialization: 'realistic', 'uniform'
        """
        print(f"Initializing atmosphere with {profile} profile...")
        self.state.initialize_atmosphere(profile)

        print(f"Initializing ocean with {sst_pattern} SST...")
        # Ocean already initialized in __init__

        # Set surface properties
        sst, albedo_ocean, z0_ocean = self.ocean.get_surface_properties()

        # Combine land and ocean properties
        # For simplicity, assume all ocean
        self.state.tsurf = sst
        self.state.albedo = albedo_ocean
        self.state.z0 = z0_ocean

        print("Initialization complete!")

    def run(self, duration_days=10, output_interval_hours=6):
        """
        Run the GCM simulation

        Parameters
        ----------
        duration_days : float
            Simulation duration in days
        output_interval_hours : float
            Interval for diagnostic output (hours)
        """
        total_seconds = duration_days * 86400.0
        output_interval = output_interval_hours * 3600.0

        n_steps = int(total_seconds / self.dt)
        output_frequency = int(output_interval / self.dt)

        print(f"\nStarting simulation:")
        print(f"  Duration: {duration_days} days")
        print(f"  Time step: {self.dt} seconds")
        print(f"  Total steps: {n_steps}")
        print(f"  Output interval: {output_interval_hours} hours")

        start_time = systime.time()
        last_output_step = 0

        for step in range(n_steps):
            # Time integration
            self.integrator.step(self.state, self.dt, self._compute_tendencies)

            # Diagnostics
            if step % output_frequency == 0:
                self._output_diagnostics(step)
                elapsed = systime.time() - start_time
                progress = (step + 1) / n_steps * 100
                sim_days = self.state.time / 86400.0

                print(f"Step {step+1}/{n_steps} ({progress:.1f}%) - "
                      f"Day {sim_days:.2f} - "
                      f"Elapsed: {elapsed:.1f}s")

        total_elapsed = systime.time() - start_time
        print(f"\nSimulation complete!")
        print(f"Total time: {total_elapsed:.1f} seconds")
        print(f"Performance: {n_steps/total_elapsed:.1f} steps/second")

    def _compute_tendencies(self, state):
        """
        Compute all tendencies (dynamics + physics)

        This is the main tendency computation routine called by the integrator

        Parameters
        ----------
        state : ModelState
            Current model state
        """
        # Reset all tendencies
        state.reset_tendencies()

        # Dynamics
        self.dynamics.compute_tendencies(state)

        # Physics
        self._compute_physics(state)

        # Combine tendencies
        self._sum_tendencies(state)

    def _compute_physics(self, state):
        """Compute all physics parameterization tendencies"""

        # Radiation
        self.radiation.compute_radiation(state, state.time)

        # Convection
        self.convection.compute_convection(state, self.dt)

        # Cloud microphysics
        self.cloud_micro.compute_microphysics(state, self.dt)

        # Boundary layer
        self.boundary_layer.compute_boundary_layer(state, self.dt)

        # Compute net radiation for surface schemes
        # Simplified: assume some net radiation
        lat = self.grid.lat2d
        zenith = np.arccos(np.maximum(0, np.cos(state.time * 2*np.pi / 86400.0) * np.cos(lat)))
        solar_forcing = self.radiation.solar_constant * np.cos(zenith)
        solar_forcing = np.maximum(0, solar_forcing)

        net_radiation = solar_forcing * (1 - state.albedo) - 200.0  # Simplified

        # Precipitation from convection
        precipitation = self.convection.compute_precipitation(state, self.dt) / 3600.0 / 1000.0  # kg/m^2/s

        # Surface fluxes (simplified estimates for land/ocean)
        sensible_flux = 50.0  # W/m^2
        latent_flux = 100.0   # W/m^2

        # Land surface (where ocean_mask = 0)
        # For simplicity, assume all ocean for now
        # self.land_surface.compute_land_surface(state, self.dt, net_radiation, precipitation)

        # Ocean
        self.ocean.compute_ocean(state, self.dt, net_radiation, sensible_flux, latent_flux)

        # Update surface temperature and properties
        sst, albedo_ocean, z0_ocean = self.ocean.get_surface_properties()
        state.tsurf = sst
        state.albedo = albedo_ocean
        state.z0 = z0_ocean

    def _sum_tendencies(self, state):
        """Sum physics tendencies into total tendencies"""

        # Add radiation heating
        state.dT_dt += state.physics_tendencies['radiation']['T']

        # Add convection
        state.dT_dt += state.physics_tendencies['convection']['T']
        state.dq_dt += state.physics_tendencies['convection']['q']
        state.du_dt += state.physics_tendencies['convection']['u']
        state.dv_dt += state.physics_tendencies['convection']['v']

        # Add cloud microphysics
        state.dT_dt += state.physics_tendencies['cloud_micro']['T']
        state.dq_dt += state.physics_tendencies['cloud_micro']['q']
        state.dqc_dt += state.physics_tendencies['cloud_micro']['qc']
        state.dqi_dt += state.physics_tendencies['cloud_micro']['qi']

        # Add boundary layer
        state.dT_dt += state.physics_tendencies['boundary_layer']['T']
        state.dq_dt += state.physics_tendencies['boundary_layer']['q']
        state.du_dt += state.physics_tendencies['boundary_layer']['u']
        state.dv_dt += state.physics_tendencies['boundary_layer']['v']

    def _output_diagnostics(self, step):
        """Compute and store diagnostic quantities"""

        # Global means
        T_mean = self.grid.global_mean(self.state.T[self.vgrid.nlev//2])  # Mid-level temp
        precip = self.convection.compute_precipitation(self.state, self.dt)
        precip_mean = np.mean(precip)

        # Energy diagnostics
        energy_diag = self.dynamics.compute_energy_diagnostics(self.state)

        # Store
        self.diagnostics['time'].append(self.state.time / 86400.0)  # days
        self.diagnostics['global_mean_T'].append(T_mean)
        self.diagnostics['global_mean_precip'].append(precip_mean)
        self.diagnostics['total_energy'].append(energy_diag['total_energy'])
        self.diagnostics['kinetic_energy'].append(energy_diag['kinetic_energy'])

    def get_state(self):
        """Return current model state"""
        return self.state

    def plot_diagnostics(self, filename=None):
        """
        Plot diagnostic time series

        Parameters
        ----------
        filename : str, optional
            If provided, save figure to file
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(10, 10))

            # Temperature
            axes[0].plot(self.diagnostics['time'], self.diagnostics['global_mean_T'])
            axes[0].set_ylabel('Global Mean T (K)')
            axes[0].set_title('GCM Diagnostics')
            axes[0].grid(True)

            # Precipitation
            axes[1].plot(self.diagnostics['time'], self.diagnostics['global_mean_precip'])
            axes[1].set_ylabel('Mean Precip (mm/hr)')
            axes[1].grid(True)

            # Energy
            axes[2].plot(self.diagnostics['time'], self.diagnostics['total_energy'], label='Total')
            axes[2].plot(self.diagnostics['time'], self.diagnostics['kinetic_energy'], label='Kinetic')
            axes[2].set_ylabel('Energy (J/kg)')
            axes[2].set_xlabel('Time (days)')
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=150)
                print(f"Diagnostics saved to {filename}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")

    def plot_state(self, filename=None):
        """
        Plot current model state

        Parameters
        ----------
        filename : str, optional
            If provided, save figure to file
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Surface temperature
            im1 = axes[0, 0].contourf(np.rad2deg(self.grid.lon), np.rad2deg(self.grid.lat),
                                     self.state.tsurf, levels=20, cmap='RdBu_r')
            axes[0, 0].set_title('Surface Temperature (K)')
            axes[0, 0].set_ylabel('Latitude')
            plt.colorbar(im1, ax=axes[0, 0])

            # Zonal wind at mid-level
            k_mid = self.vgrid.nlev // 2
            im2 = axes[0, 1].contourf(np.rad2deg(self.grid.lon), np.rad2deg(self.grid.lat),
                                     self.state.u[k_mid], levels=20, cmap='RdBu_r')
            axes[0, 1].set_title(f'Zonal Wind at level {k_mid} (m/s)')
            plt.colorbar(im2, ax=axes[0, 1])

            # Specific humidity
            im3 = axes[1, 0].contourf(np.rad2deg(self.grid.lon), np.rad2deg(self.grid.lat),
                                     self.state.q[k_mid]*1000, levels=20, cmap='YlGnBu')
            axes[1, 0].set_title(f'Specific Humidity at level {k_mid} (g/kg)')
            axes[1, 0].set_ylabel('Latitude')
            axes[1, 0].set_xlabel('Longitude')
            plt.colorbar(im3, ax=axes[1, 0])

            # Cloud water
            im4 = axes[1, 1].contourf(np.rad2deg(self.grid.lon), np.rad2deg(self.grid.lat),
                                     (self.state.qc[k_mid] + self.state.qi[k_mid])*1000,
                                     levels=20, cmap='Greys')
            axes[1, 1].set_title(f'Cloud Water at level {k_mid} (g/kg)')
            axes[1, 1].set_xlabel('Longitude')
            plt.colorbar(im4, ax=axes[1, 1])

            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=150)
                print(f"State plot saved to {filename}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")
