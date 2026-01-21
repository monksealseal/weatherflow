"""
Comprehensive GCM Diagnostics

Combines all diagnostic modules and provides plotting utilities
for validating GCM physics.
"""

import numpy as np
from .spectral import SpectralDiagnostics
from .zonal_mean import ZonalMeanDiagnostics
from .eddy_diagnostics import EddyDiagnostics
from .circulation import CirculationDiagnostics


class ComprehensiveGCMDiagnostics:
    """
    Comprehensive diagnostics for GCM validation

    Combines spectral, zonal mean, eddy, and circulation diagnostics
    into a single interface with plotting utilities.
    """

    def __init__(self, grid, vgrid):
        """
        Initialize comprehensive diagnostics

        Parameters
        ----------
        grid : SphericalGrid
            Horizontal grid
        vgrid : VerticalGrid
            Vertical grid
        """
        self.grid = grid
        self.vgrid = vgrid

        # Initialize sub-modules
        self.spectral = SpectralDiagnostics(grid, vgrid)
        self.zonal = ZonalMeanDiagnostics(grid, vgrid)
        self.eddy = EddyDiagnostics(grid, vgrid)
        self.circulation = CirculationDiagnostics(grid, vgrid)

        # Storage for time series
        self.diagnostic_history = []
        self.times = []

    def compute_all_diagnostics(self, state, store=True):
        """
        Compute all diagnostics for current state

        Parameters
        ----------
        state : ModelState
            Current model state
        store : bool
            Whether to store diagnostics in history

        Returns
        -------
        diagnostics : dict
            Complete set of diagnostics
        """
        # Spectral diagnostics
        spectra = self.spectral.compute_all_spectra(state)

        # Zonal mean diagnostics
        zonal = self.zonal.compute_all_zonal_diagnostics(state)

        # Eddy diagnostics
        eddy = self.eddy.compute_all_eddy_diagnostics(state)

        # Circulation diagnostics
        circ = self.circulation.compute_all_circulation_diagnostics(
            state, self.zonal, self.eddy
        )

        # Energy budget
        energy = self._compute_energy_budget(state)

        # Mass budget
        mass = self._compute_mass_budget(state)

        diagnostics = {
            'time': state.time / 86400.0,  # days
            'spectral': spectra,
            'zonal': zonal,
            'eddy': eddy,
            'circulation': circ,
            'energy': energy,
            'mass': mass
        }

        if store:
            self.diagnostic_history.append(diagnostics)
            self.times.append(state.time / 86400.0)

        return diagnostics

    def _compute_energy_budget(self, state):
        """
        Compute energy budget diagnostics

        Returns
        -------
        energy : dict
            Energy budget components
        """
        Rd = 287.0
        cp = 1004.0
        cv = cp - Rd
        g = 9.81

        # Kinetic energy
        ke = 0.5 * (state.u**2 + state.v**2)

        # Potential energy (geopotential)
        pe = g * state.z

        # Internal energy
        ie = cv * state.T

        # Latent energy
        Lv = 2.5e6
        le = Lv * state.q

        # Mass weighted global means
        p = state.p
        dp = np.zeros_like(p)
        for k in range(self.vgrid.nlev):
            if k == 0:
                dp[k] = p[k]
            else:
                dp[k] = p[k] - p[k-1]

        # Use 1D cos_lat for proper broadcasting: (nlat,) -> (1, nlat, 1)
        cos_lat_1d = np.cos(self.grid.lat)  # Shape: (nlat,)
        cos_lat_3d = cos_lat_1d[np.newaxis, :, np.newaxis]  # Shape: (1, nlat, 1)

        total_mass = np.sum(dp * cos_lat_3d) / g

        ke_global = np.sum(ke * dp * cos_lat_3d) / g / total_mass
        pe_global = np.sum(pe * dp * cos_lat_3d) / g / total_mass
        ie_global = np.sum(ie * dp * cos_lat_3d) / g / total_mass
        le_global = np.sum(le * dp * cos_lat_3d) / g / total_mass

        total_energy = ke_global + pe_global + ie_global + le_global

        # Available potential energy (approximate)
        T_bar = np.mean(state.T, axis=(1, 2), keepdims=True)
        T_prime = state.T - T_bar
        gamma = Rd / cp
        ape = 0.5 * cp * np.sum(T_prime**2 / T_bar * dp * cos_lat_3d) / g / total_mass

        return {
            'kinetic_energy': ke_global,
            'potential_energy': pe_global,
            'internal_energy': ie_global,
            'latent_energy': le_global,
            'total_energy': total_energy,
            'available_potential_energy': ape,
            'ke_field': ke,
            'ape_approx_field': 0.5 * cp * T_prime**2 / T_bar
        }

    def _compute_mass_budget(self, state):
        """
        Compute mass budget diagnostics

        Returns
        -------
        mass : dict
            Mass budget components
        """
        g = 9.81

        # Total atmospheric mass
        ps_mean = self.grid.global_mean(state.ps)
        total_mass = ps_mean / g

        # Water mass
        water_mass = 0.0
        p = state.p
        for k in range(self.vgrid.nlev):
            if k == 0:
                dp = p[k]
            else:
                dp = p[k] - p[k-1]
            q_mean = self.grid.global_mean(state.q[k])
            water_mass += q_mean * np.mean(dp) / g

        # Cloud water mass
        cloud_mass = 0.0
        for k in range(self.vgrid.nlev):
            if k == 0:
                dp = p[k]
            else:
                dp = p[k] - p[k-1]
            qc_mean = self.grid.global_mean(state.qc[k])
            qi_mean = self.grid.global_mean(state.qi[k])
            cloud_mass += (qc_mean + qi_mean) * np.mean(dp) / g

        return {
            'total_mass': total_mass,
            'water_vapor_mass': water_mass,
            'cloud_mass': cloud_mass,
            'surface_pressure_mean': ps_mean
        }

    def check_gcm_realism(self, diagnostics):
        """
        Check if GCM results are realistic

        Compares diagnostics against expected values for Earth-like climate.

        Parameters
        ----------
        diagnostics : dict
            Computed diagnostics

        Returns
        -------
        report : dict
            Realism check report with pass/fail flags
        """
        report = {'passed': True, 'warnings': [], 'errors': []}

        # Check jet stream
        jet = diagnostics['circulation']['jet_diagnostics']
        if jet['subtropical_jet_nh_speed'] < 5:
            report['warnings'].append(
                f"NH subtropical jet weak ({jet['subtropical_jet_nh_speed']:.1f} m/s, expected >15 m/s)"
            )
        if jet['subtropical_jet_nh_speed'] > 100:
            report['errors'].append(
                f"NH subtropical jet unrealistically strong ({jet['subtropical_jet_nh_speed']:.1f} m/s)"
            )
            report['passed'] = False

        # Check Hadley cell
        circ = diagnostics['circulation']
        if circ['hadley_strength_nh'] < 10:
            report['warnings'].append(
                f"NH Hadley cell weak ({circ['hadley_strength_nh']:.1f} x10^9 kg/s, expected >50)"
            )

        # Check temperatures
        zonal = diagnostics['zonal']
        T_equator = zonal['T_bar'][self.vgrid.nlev-1, self.grid.nlat//2]
        T_pole = zonal['T_bar'][self.vgrid.nlev-1, 0]

        if T_equator < 280 or T_equator > 320:
            report['warnings'].append(
                f"Equatorial surface T unusual ({T_equator:.1f} K, expected 290-305 K)"
            )

        if T_pole < 200 or T_pole > 290:
            report['warnings'].append(
                f"Polar surface T unusual ({T_pole:.1f} K, expected 230-280 K)"
            )

        # Check energy spectra slope
        spectra = diagnostics['spectral']
        ke = spectra['kinetic_energy_spectrum']
        wn = spectra['wavenumbers']

        # Check if spectrum has reasonable slope
        if len(ke) > 10:
            # Fit power law to wavenumbers 5-15
            idx = (wn >= 5) & (wn <= 15)
            if np.sum(idx) > 3:
                log_ke = np.log(ke[idx] + 1e-20)
                log_wn = np.log(wn[idx])
                slope = np.polyfit(log_wn, log_ke, 1)[0]

                if slope > -1.5:
                    report['warnings'].append(
                        f"KE spectrum slope too shallow ({slope:.1f}, expected < -2)"
                    )
                if slope < -5:
                    report['warnings'].append(
                        f"KE spectrum slope too steep ({slope:.1f}, expected > -4)"
                    )

        # Check energy conservation
        energy = diagnostics['energy']
        if len(self.diagnostic_history) > 5:
            e_init = self.diagnostic_history[0]['energy']['total_energy']
            e_now = energy['total_energy']
            drift = abs(e_now - e_init) / abs(e_init)

            if drift > 0.1:
                report['warnings'].append(
                    f"Energy drift {drift*100:.1f}% (should be <10%)"
                )

        return report

    def get_summary_statistics(self):
        """
        Get summary statistics from diagnostic history

        Returns
        -------
        summary : dict
            Summary statistics
        """
        if len(self.diagnostic_history) < 2:
            return None

        # Time series of key quantities
        times = np.array(self.times)
        ke = np.array([d['energy']['kinetic_energy'] for d in self.diagnostic_history])
        ape = np.array([d['energy']['available_potential_energy'] for d in self.diagnostic_history])
        total_e = np.array([d['energy']['total_energy'] for d in self.diagnostic_history])

        jet_lat = np.array([d['circulation']['jet_diagnostics']['subtropical_jet_nh_lat']
                          for d in self.diagnostic_history])
        jet_speed = np.array([d['circulation']['jet_diagnostics']['subtropical_jet_nh_speed']
                             for d in self.diagnostic_history])

        return {
            'times': times,
            'kinetic_energy': ke,
            'available_potential_energy': ape,
            'total_energy': total_e,
            'jet_latitude': jet_lat,
            'jet_speed': jet_speed,
            'ke_mean': np.mean(ke),
            'ke_std': np.std(ke),
            'ape_mean': np.mean(ape),
            'ape_std': np.std(ape),
            'energy_drift': (total_e[-1] - total_e[0]) / total_e[0] if total_e[0] != 0 else 0
        }

    def plot_diagnostics(self, diagnostics=None, filename=None):
        """
        Create comprehensive diagnostic plots

        Parameters
        ----------
        diagnostics : dict, optional
            Diagnostics to plot. If None, use latest from history.
        filename : str, optional
            If provided, save figure to file

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("Matplotlib required for plotting")
            return None

        if diagnostics is None:
            if len(self.diagnostic_history) == 0:
                print("No diagnostics available")
                return None
            diagnostics = self.diagnostic_history[-1]

        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Zonal mean zonal wind
        ax1 = fig.add_subplot(gs[0, 0:2])
        zonal = diagnostics['zonal']
        lat = zonal['latitude']
        p = zonal['pressure']
        u_bar = zonal['u_bar']

        # Use mid-latitude pressure profile
        p_plot = p[:, len(lat)//2]
        levels = np.linspace(-50, 50, 21)
        cf = ax1.contourf(lat, p_plot, u_bar, levels=levels, cmap='RdBu_r', extend='both')
        ax1.contour(lat, p_plot, u_bar, levels=[0], colors='k', linewidths=0.5)
        ax1.set_ylim([1000, 100])
        ax1.set_xlabel('Latitude')
        ax1.set_ylabel('Pressure (hPa)')
        ax1.set_title('Zonal Mean Zonal Wind [u] (m/s)')
        plt.colorbar(cf, ax=ax1)

        # 2. Meridional streamfunction
        ax2 = fig.add_subplot(gs[0, 2:4])
        circ = diagnostics['circulation']
        psi = circ['streamfunction']
        if psi is not None:
            levels = np.linspace(-200, 200, 21)
            cf = ax2.contourf(lat, p_plot, psi, levels=levels, cmap='RdBu_r', extend='both')
            ax2.contour(lat, p_plot, psi, levels=[0], colors='k', linewidths=0.5)
            ax2.set_ylim([1000, 100])
            ax2.set_xlabel('Latitude')
            ax2.set_ylabel('Pressure (hPa)')
            ax2.set_title('Meridional Streamfunction (10⁹ kg/s)')
            plt.colorbar(cf, ax=ax2)

        # 3. Zonal mean temperature
        ax3 = fig.add_subplot(gs[1, 0:2])
        T_bar = zonal['T_bar']
        levels = np.arange(180, 320, 10)
        cf = ax3.contourf(lat, p_plot, T_bar, levels=levels, cmap='Spectral_r')
        ax3.contour(lat, p_plot, T_bar, levels=levels[::2], colors='k', linewidths=0.5)
        ax3.set_ylim([1000, 100])
        ax3.set_xlabel('Latitude')
        ax3.set_ylabel('Pressure (hPa)')
        ax3.set_title('Zonal Mean Temperature [T] (K)')
        plt.colorbar(cf, ax=ax3)

        # 4. Eddy kinetic energy
        ax4 = fig.add_subplot(gs[1, 2:4])
        eddy = diagnostics['eddy']
        eke = eddy['eke']
        if eke is not None:
            levels = np.linspace(0, np.percentile(eke, 99), 15)
            cf = ax4.contourf(lat, p_plot, eke, levels=levels, cmap='YlOrRd')
            ax4.set_ylim([1000, 100])
            ax4.set_xlabel('Latitude')
            ax4.set_ylabel('Pressure (hPa)')
            ax4.set_title('Eddy Kinetic Energy (m²/s²)')
            plt.colorbar(cf, ax=ax4)

        # 5. KE spectrum
        ax5 = fig.add_subplot(gs[2, 0:2])
        spectra = diagnostics['spectral']
        wn = spectra['wavenumbers']
        ke_spec = spectra['kinetic_energy_spectrum']

        ax5.loglog(wn[1:], ke_spec[1:], 'b-', linewidth=2, label='KE spectrum')
        # Reference slopes
        ref_wn = wn[5:]
        ax5.loglog(ref_wn, spectra['k_minus_3_reference'], 'k--', alpha=0.5, label='k⁻³')
        ax5.loglog(ref_wn, spectra['k_minus_5_3_reference'], 'k:', alpha=0.5, label='k⁻⁵/³')
        ax5.set_xlabel('Wavenumber')
        ax5.set_ylabel('Energy')
        ax5.set_title('Kinetic Energy Spectrum')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Heat transport
        ax6 = fig.add_subplot(gs[2, 2:4])
        Q_total = eddy['heat_transport_total']
        Q_stat = eddy['heat_transport_stationary']
        Q_mean = eddy['heat_transport_mean']

        ax6.plot(lat, Q_total, 'k-', linewidth=2, label='Total')
        ax6.plot(lat, Q_stat, 'r--', label='Stationary')
        ax6.plot(lat, Q_mean, 'b:', label='Mean')
        ax6.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax6.set_xlabel('Latitude')
        ax6.set_ylabel('Heat Transport (PW)')
        ax6.set_title('Meridional Heat Transport')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Time series (if history available)
        if len(self.diagnostic_history) > 1:
            ax7 = fig.add_subplot(gs[3, 0:2])
            summary = self.get_summary_statistics()
            ax7.plot(summary['times'], summary['kinetic_energy'], 'b-', label='KE')
            ax7.plot(summary['times'], summary['available_potential_energy'], 'r-', label='APE')
            ax7.set_xlabel('Time (days)')
            ax7.set_ylabel('Energy (J/kg)')
            ax7.set_title('Energy Time Series')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

            ax8 = fig.add_subplot(gs[3, 2:4])
            ax8.plot(summary['times'], summary['jet_speed'], 'g-', linewidth=2)
            ax8.set_xlabel('Time (days)')
            ax8.set_ylabel('Speed (m/s)')
            ax8.set_title('NH Subtropical Jet Speed')
            ax8.grid(True, alpha=0.3)

        plt.suptitle(f'GCM Diagnostics - Day {diagnostics["time"]:.1f}', fontsize=14, y=1.02)

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Diagnostics plot saved to {filename}")

        return fig

    def reset(self):
        """Reset all stored diagnostics"""
        self.diagnostic_history = []
        self.times = []
        self.eddy.reset()
