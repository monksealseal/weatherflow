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

    Main model class that coordinates all components.

    Supports both Earth-like and Tropic World configurations:
    - Earth-like: Rotating planet with latitudinal temperature gradients
    - Tropic World: Non-rotating planet with uniform solar heating and
      spontaneous convective organization (based on Section 2.4 of
      "Heuristic Models of the General Circulation")
    """

    def __init__(self, nlon=64, nlat=32, nlev=20, dt=600.0,
                 integration_method='rk3', co2_ppmv=400.0, tropic_world=False,
                 tropic_world_sst=300.0, sst_perturbation=0.5,
                 mixed_layer_depth=50.0):
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
        tropic_world : bool
            If True, configure the model for Tropic World simulation:
            - Non-rotating planet (no Coriolis force)
            - Uniform incoming solar radiation over entire surface
            - Slab ocean with no horizontal heat transport
            - Uniform initial SST with small perturbations
            - No sea ice
        tropic_world_sst : float
            Base SST for Tropic World mode (K). Default: 300K (27C)
        sst_perturbation : float
            Amplitude of random SST perturbations in Tropic World mode (K)
        mixed_layer_depth : float
            Depth of slab ocean mixed layer (m). Default: 50m
        """
        self.tropic_world = tropic_world
        mode_str = "Tropic World" if tropic_world else "Earth-like"
        print(f"Initializing GCM ({mode_str} mode) with resolution: {nlon}x{nlat}x{nlev}")

        # Grid - non-rotating for Tropic World
        rotation_rate = 0.0 if tropic_world else 7.292e-5
        self.grid = SphericalGrid(nlon, nlat, nlev, rotation_rate=rotation_rate)
        self.vgrid = VerticalGrid(nlev, coord_type='sigma')

        # Model state
        self.state = ModelState(self.grid, self.vgrid)

        # Time step
        self.dt = dt

        # Components
        print("Initializing dynamics...")
        self.dynamics = AtmosphericDynamics(self.grid, self.vgrid)

        print("Initializing physics schemes...")
        # Radiation - uniform solar for Tropic World
        self.radiation = RadiationScheme(self.grid, self.vgrid, co2_ppmv=co2_ppmv,
                                         uniform_solar=tropic_world)
        self.convection = ConvectionScheme(self.grid, self.vgrid)
        self.cloud_micro = CloudMicrophysics(self.grid, self.vgrid)
        self.boundary_layer = BoundaryLayerScheme(self.grid, self.vgrid)
        self.land_surface = LandSurfaceModel(self.grid, self.vgrid)

        # Ocean - configured for Tropic World with slab ocean
        self.ocean = OceanMixedLayerModel(
            self.grid,
            mixed_layer_depth=mixed_layer_depth,
            tropic_world=tropic_world,
            tropic_world_sst=tropic_world_sst,
            sst_perturbation=sst_perturbation
        )

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

        # Tropic World specific diagnostics
        if tropic_world:
            self.diagnostics['sst_contrast'] = []  # Max - min SST
            self.diagnostics['sst_warm_fraction'] = []  # Fraction of warm regions
            self.diagnostics['global_mean_evap'] = []  # Global evaporation
            self.diagnostics['global_mean_sst'] = []  # Global mean SST

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

    def run_with_visualization(self, duration_days=10, day_callback=None, output_interval_hours=6):
        """
        Run the GCM simulation with visualization updates after each day.

        This method pauses the simulation after each simulated day to allow
        visualization updates. The callback is called with the model instance
        after each day completes.

        Parameters
        ----------
        duration_days : float
            Simulation duration in days
        day_callback : callable, optional
            Function called after each day with signature: callback(model, day)
            The simulation pauses until this callback returns.
        output_interval_hours : float
            Interval for diagnostic output (hours)

        Examples
        --------
        >>> from gcm.visualization import Interactive3DVisualizer
        >>> viz = Interactive3DVisualizer()
        >>> model.run_with_visualization(
        ...     duration_days=10,
        ...     day_callback=lambda m, d: viz.update(m, d)
        ... )
        """
        total_seconds = duration_days * 86400.0
        output_interval = output_interval_hours * 3600.0

        n_steps = int(total_seconds / self.dt)
        output_frequency = int(output_interval / self.dt)
        steps_per_day = int(86400.0 / self.dt)

        print(f"\nStarting simulation with visualization:")
        print(f"  Duration: {duration_days} days")
        print(f"  Time step: {self.dt} seconds")
        print(f"  Total steps: {n_steps}")
        print(f"  Steps per day: {steps_per_day}")
        print(f"  Visualization callback: {'Yes' if day_callback else 'No'}")

        start_time = systime.time()
        last_day = 0

        for step in range(n_steps):
            # Time integration
            self.integrator.step(self.state, self.dt, self._compute_tendencies)

            # Diagnostics at output interval
            if step % output_frequency == 0:
                self._output_diagnostics(step)

            # Check if a full day has passed
            current_day = int(self.state.time / 86400.0)
            if current_day > last_day:
                elapsed = systime.time() - start_time
                progress = (step + 1) / n_steps * 100
                sim_days = self.state.time / 86400.0

                print(f"\nDay {current_day} complete ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")

                # Call visualization callback
                if day_callback is not None:
                    print(f"  Rendering visualization for day {current_day}...")
                    day_callback(self, current_day)
                    print(f"  Visualization complete. Continuing simulation...")

                last_day = current_day

        # Final callback at end of simulation
        if day_callback is not None:
            final_day = self.state.time / 86400.0
            print(f"\nRendering final visualization for day {final_day:.1f}...")
            day_callback(self, final_day)

        total_elapsed = systime.time() - start_time
        print(f"\nSimulation complete!")
        print(f"Total time: {total_elapsed:.1f} seconds")
        print(f"Performance: {n_steps/total_elapsed:.1f} steps/second")

    def step_one_day(self):
        """
        Advance the simulation by exactly one day.

        This method is useful for manual day-by-day control of the simulation
        with visualization updates between days.

        Returns
        -------
        day : float
            The current simulation day after stepping
        """
        steps_per_day = int(86400.0 / self.dt)
        output_frequency = max(1, steps_per_day // 4)  # Output 4 times per day

        for step in range(steps_per_day):
            self.integrator.step(self.state, self.dt, self._compute_tendencies)

            if step % output_frequency == 0:
                self._output_diagnostics(step)

        return self.state.time / 86400.0

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
        if self.tropic_world:
            # Tropic World: uniform solar radiation over entire surface
            # Solar constant / 4 is the average over a sphere
            solar_forcing = np.full_like(state.albedo, self.radiation.solar_constant / 4.0)
        else:
            # Earth-like: diurnal cycle with latitude dependence
            lat = self.grid.lat2d
            zenith = np.arccos(np.maximum(0, np.cos(state.time * 2*np.pi / 86400.0) * np.cos(lat)))
            solar_forcing = self.radiation.solar_constant * np.cos(zenith)
            solar_forcing = np.maximum(0, solar_forcing)

        net_radiation = solar_forcing * (1 - state.albedo) - 200.0  # Simplified LW cooling

        # Precipitation from convection
        precipitation = self.convection.compute_precipitation(state, self.dt) / 3600.0 / 1000.0  # kg/m^2/s

        # Surface fluxes - more sophisticated for Tropic World
        if self.tropic_world:
            # Compute bulk aerodynamic surface fluxes for Tropic World
            # This allows SST contrasts to drive circulation
            sensible_flux, latent_flux = self._compute_surface_fluxes(state)
        else:
            # Simplified estimates for Earth-like
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

    def _compute_surface_fluxes(self, state):
        """
        Compute bulk aerodynamic surface fluxes for Tropic World

        Uses bulk formulas to compute sensible and latent heat fluxes
        that depend on local SST and wind speed. This is important for
        Tropic World because:
        - Warm regions have enhanced evaporation
        - Strong SST contrasts drive surface winds
        - Surface fluxes provide the feedback mechanism for SST evolution

        Returns
        -------
        sensible_flux : ndarray
            Sensible heat flux (W/m^2, positive upward)
        latent_flux : ndarray
            Latent heat flux (W/m^2, positive upward)
        """
        # Surface properties
        T_surf = state.tsurf
        T_air = state.T[-1]  # Lowest atmospheric level
        q_air = state.q[-1]

        # Wind speed at surface
        u_surf = state.u[-1]
        v_surf = state.v[-1]
        wind_speed = np.sqrt(u_surf**2 + v_surf**2)
        wind_speed = np.maximum(wind_speed, 1.0)  # Minimum wind speed

        # Air density
        rho_air = state.rho[-1]

        # Bulk transfer coefficients (neutral stability)
        C_H = 1.0e-3   # Heat transfer coefficient
        C_E = 1.2e-3   # Moisture transfer coefficient

        # Sensible heat flux: H = rho * cp * C_H * U * (T_surf - T_air)
        cp = 1004.0
        sensible_flux = rho_air * cp * C_H * wind_speed * (T_surf - T_air)

        # Latent heat flux: E = rho * Lv * C_E * U * (q_sat(T_surf) - q_air)
        Lv = 2.5e6
        T0, e0 = 273.15, 611.2
        Rv = 461.5
        es = e0 * np.exp((Lv / Rv) * (1/T0 - 1/T_surf))
        qsat = 0.622 * es / state.ps
        latent_flux = rho_air * Lv * C_E * wind_speed * (qsat - q_air)
        latent_flux = np.maximum(0, latent_flux)  # No negative evaporation

        return sensible_flux, latent_flux

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

        # Tropic World specific diagnostics
        if self.tropic_world:
            self._output_tropic_world_diagnostics()

    def _output_tropic_world_diagnostics(self):
        """
        Compute Tropic World specific diagnostics

        Tracks:
        - SST contrast: difference between warmest and coolest regions
        - Warm fraction: fraction of planet with SST above global mean
        - Global mean evaporation and SST

        These diagnostics help visualize the 2-3 year oscillation between
        high and low SST contrast regimes characteristic of Tropic World.
        """
        sst = self.ocean.sst

        # SST statistics
        sst_max = np.max(sst)
        sst_min = np.min(sst)
        sst_contrast = sst_max - sst_min

        # Global mean SST (area-weighted)
        sst_mean = self.grid.global_mean(sst)

        # Fraction of planet with above-average SST
        warm_fraction = np.sum((sst > sst_mean) * self.grid.cell_area) / self.grid.total_area

        # Store diagnostics
        self.diagnostics['sst_contrast'].append(sst_contrast)
        self.diagnostics['sst_warm_fraction'].append(warm_fraction)
        self.diagnostics['global_mean_sst'].append(sst_mean)

        # Estimate evaporation from latent heat flux (simplified)
        # Use bulk formula: E ~ C_E * U * (q_sat(SST) - q_a)
        # This is a rough estimate based on surface conditions
        T_surf = self.state.tsurf
        T0, e0 = 273.15, 611.2
        Lv, Rv = 2.5e6, 461.5
        es = e0 * np.exp((Lv / Rv) * (1/T0 - 1/T_surf))
        qsat = 0.622 * es / self.state.ps
        q_surf = self.state.q[-1]  # Near-surface humidity
        wind_speed = np.sqrt(self.state.u[-1]**2 + self.state.v[-1]**2)
        C_E = 1.5e-3  # Exchange coefficient
        rho_air = 1.2  # kg/m^3 approximate
        evap = C_E * rho_air * wind_speed * (qsat - q_surf) * 3600 * 24  # kg/m^2/day
        evap_mean = self.grid.global_mean(np.maximum(0, evap))
        self.diagnostics['global_mean_evap'].append(evap_mean)

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

    def plot_tropic_world(self, filename=None):
        """
        Plot Tropic World specific visualization

        Shows SST and surface wind vectors, similar to Figure 2.5 in
        "Heuristic Models of the General Circulation" (Section 2.4).

        Parameters
        ----------
        filename : str, optional
            If provided, save figure to file
        """
        if not self.tropic_world:
            print("Warning: plot_tropic_world is designed for Tropic World simulations")

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Get SST and surface winds
            sst = self.ocean.sst
            u_surf = self.state.u[-1]  # Lowest level
            v_surf = self.state.v[-1]

            # Coordinates
            lon = np.rad2deg(self.grid.lon)
            lat = np.rad2deg(self.grid.lat)

            # Panel 1: SST with wind vectors
            ax = axes[0, 0]
            im = ax.contourf(lon, lat, sst, levels=20, cmap='RdBu_r')
            # Subsample wind vectors for clarity
            skip = max(1, self.grid.nlon // 16)
            ax.quiver(lon[::skip], lat[::skip], u_surf[::skip, ::skip],
                     v_surf[::skip, ::skip], scale=100, alpha=0.7)
            ax.set_title(f'SST and Surface Winds (Day {self.state.time/86400:.1f})')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='SST (K)')

            # Panel 2: SST anomaly from global mean
            ax = axes[0, 1]
            sst_mean = self.grid.global_mean(sst)
            sst_anom = sst - sst_mean
            vmax = max(abs(sst_anom.min()), abs(sst_anom.max()))
            im = ax.contourf(lon, lat, sst_anom, levels=20,
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'SST Anomaly (Mean: {sst_mean:.1f} K)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='SST Anomaly (K)')

            # Panel 3: Wind speed
            ax = axes[1, 0]
            wind_speed = np.sqrt(u_surf**2 + v_surf**2)
            im = ax.contourf(lon, lat, wind_speed, levels=20, cmap='viridis')
            ax.set_title('Surface Wind Speed')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='Wind Speed (m/s)')

            # Panel 4: Mid-tropospheric relative humidity
            ax = axes[1, 1]
            k_mid = self.vgrid.nlev // 2
            T_mid = self.state.T[k_mid]
            q_mid = self.state.q[k_mid]
            p_mid = self.state.p[k_mid]
            # Saturation specific humidity
            T0, e0 = 273.15, 611.2
            Lv, Rv = 2.5e6, 461.5
            es = e0 * np.exp((Lv / Rv) * (1/T0 - 1/T_mid))
            qsat = 0.622 * es / p_mid
            rh = 100 * q_mid / qsat
            rh = np.clip(rh, 0, 100)
            im = ax.contourf(lon, lat, rh, levels=np.linspace(0, 100, 21), cmap='YlGnBu')
            ax.set_title(f'Mid-level Relative Humidity (level {k_mid})')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='RH (%)')

            plt.suptitle(f'Tropic World Simulation - Day {self.state.time/86400:.1f}', fontsize=14)
            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=150)
                print(f"Tropic World plot saved to {filename}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")

    def plot_tropic_world_diagnostics(self, filename=None):
        """
        Plot Tropic World diagnostic time series

        Shows the evolution of SST contrast and other quantities that
        demonstrate the 2-3 year oscillation characteristic of Tropic World.

        Parameters
        ----------
        filename : str, optional
            If provided, save figure to file
        """
        if not self.tropic_world:
            print("Warning: plot_tropic_world_diagnostics requires Tropic World mode")
            return

        if 'sst_contrast' not in self.diagnostics or len(self.diagnostics['sst_contrast']) == 0:
            print("No Tropic World diagnostics available yet. Run simulation first.")
            return

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1, figsize=(12, 12))

            time = self.diagnostics['time']

            # SST contrast
            ax = axes[0]
            ax.plot(time, self.diagnostics['sst_contrast'], 'b-', linewidth=1.5)
            ax.set_ylabel('SST Contrast (K)')
            ax.set_title('Tropic World Diagnostics: SST Contrast Evolution')
            ax.grid(True, alpha=0.3)

            # Global mean SST
            ax = axes[1]
            ax.plot(time, self.diagnostics['global_mean_sst'], 'r-', linewidth=1.5)
            ax.set_ylabel('Global Mean SST (K)')
            ax.grid(True, alpha=0.3)

            # Warm fraction
            ax = axes[2]
            ax.plot(time, self.diagnostics['sst_warm_fraction'], 'g-', linewidth=1.5)
            ax.set_ylabel('Warm Area Fraction')
            ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # Precipitation and evaporation
            ax = axes[3]
            ax.plot(time, self.diagnostics['global_mean_precip'], 'b-',
                   linewidth=1.5, label='Precipitation')
            if 'global_mean_evap' in self.diagnostics:
                ax.plot(time, self.diagnostics['global_mean_evap'], 'r--',
                       linewidth=1.5, label='Evaporation')
            ax.set_ylabel('Rate (mm/hr or kg/m2/day)')
            ax.set_xlabel('Time (days)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=150)
                print(f"Tropic World diagnostics saved to {filename}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")
