"""
Circulation Diagnostics for GCM

Computes general circulation diagnostics:
- Hadley cell strength and extent
- Jet stream position and intensity
- Storm tracks
- Potential vorticity
- Rossby wave activity
"""

import numpy as np


class CirculationDiagnostics:
    """
    Compute circulation diagnostics for atmospheric dynamics
    """

    def __init__(self, grid, vgrid):
        """
        Initialize circulation diagnostics

        Parameters
        ----------
        grid : SphericalGrid
            Horizontal grid
        vgrid : VerticalGrid
            Vertical grid
        """
        self.grid = grid
        self.vgrid = vgrid
        self.nlat = grid.nlat
        self.nlon = grid.nlon
        self.nlev = vgrid.nlev

        # Physical constants
        self.g = 9.81
        self.Rd = 287.0
        self.cp = 1004.0
        self.Omega = grid.rotation_rate
        self.a = grid.radius

    def compute_potential_vorticity(self, state):
        """
        Compute Ertel's potential vorticity

        PV = (f + zeta) * (-g * dtheta/dp)

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        pv : ndarray
            Potential vorticity (nlev, nlat, nlon) in PVU
        """
        # Relative vorticity
        zeta = np.zeros_like(state.u)
        for k in range(self.nlev):
            zeta[k] = self.grid.vorticity(state.u[k], state.v[k])

        # Absolute vorticity
        f = self.grid.f_coriolis
        abs_vort = zeta + f

        # Static stability: -dtheta/dp
        theta = state.theta
        p = state.p
        dtheta_dp = np.zeros_like(theta)

        for i in range(self.nlat):
            for j in range(self.nlon):
                dtheta_dp[:, i, j] = np.gradient(theta[:, i, j], p[:, i, j])

        # PV = absolute_vorticity * static_stability * g
        pv = abs_vort * (-self.g * dtheta_dp)

        # Convert to PVU (1 PVU = 10^-6 K m^2 / kg / s)
        pv = pv * 1e6

        return pv

    def compute_hadley_cell_diagnostics(self, psi):
        """
        Analyze Hadley cell from meridional streamfunction

        Parameters
        ----------
        psi : ndarray
            Meridional streamfunction (nlev, nlat)

        Returns
        -------
        hadley_strength_nh : float
            Northern Hemisphere Hadley cell strength (10^9 kg/s)
        hadley_strength_sh : float
            Southern Hemisphere Hadley cell strength
        hadley_edge_nh : float
            Northern edge of NH Hadley cell (degrees)
        hadley_edge_sh : float
            Southern edge of SH Hadley cell (degrees)
        """
        lat_deg = np.rad2deg(self.grid.lat)

        # Find NH Hadley cell (typically positive psi in lower troposphere)
        # Look in subtropical region of NH
        nh_mask = lat_deg > 0
        sh_mask = lat_deg < 0

        # Lower troposphere (bottom 30% of atmosphere)
        lower_levels = slice(int(0.7 * self.nlev), None)

        # NH: Find max positive streamfunction
        psi_nh = psi[lower_levels, nh_mask]
        hadley_strength_nh = np.max(psi_nh) if psi_nh.size > 0 else 0

        # SH: Find min (most negative) streamfunction
        psi_sh = psi[lower_levels, sh_mask]
        hadley_strength_sh = -np.min(psi_sh) if psi_sh.size > 0 else 0

        # Find Hadley cell edges (where psi changes sign)
        # Use 500 hPa level (roughly level nlev//2)
        mid_level = self.nlev // 2
        psi_mid = psi[mid_level, :]

        # NH edge: northernmost point where psi > 0.1 * max
        threshold = 0.1 * hadley_strength_nh
        nh_extent = lat_deg[nh_mask][psi_mid[nh_mask] > threshold]
        hadley_edge_nh = np.max(nh_extent) if len(nh_extent) > 0 else 30.0

        # SH edge: southernmost point where psi < -0.1 * |min|
        threshold = -0.1 * hadley_strength_sh
        sh_extent = lat_deg[sh_mask][psi_mid[sh_mask] < threshold]
        hadley_edge_sh = np.min(sh_extent) if len(sh_extent) > 0 else -30.0

        return hadley_strength_nh, hadley_strength_sh, hadley_edge_nh, hadley_edge_sh

    def compute_jet_diagnostics(self, state):
        """
        Analyze jet stream characteristics

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        jet_diagnostics : dict
            Jet stream diagnostics including position, strength, and structure
        """
        u_bar = np.mean(state.u, axis=-1)  # Zonal mean
        p_bar = np.mean(state.p, axis=-1) / 100  # hPa

        lat_deg = np.rad2deg(self.grid.lat)

        # Find subtropical jet (upper troposphere, ~200 hPa)
        upper_levels = p_bar[:, self.nlat//2] < 300
        u_upper = np.mean(u_bar[upper_levels, :], axis=0) if np.any(upper_levels) else u_bar[0, :]

        # NH subtropical jet
        nh_mask = lat_deg > 0
        if np.any(nh_mask):
            nh_lat = lat_deg[nh_mask]
            nh_u = u_upper[nh_mask]
            stj_nh_idx = np.argmax(nh_u)
            stj_nh_lat = nh_lat[stj_nh_idx]
            stj_nh_speed = nh_u[stj_nh_idx]
        else:
            stj_nh_lat, stj_nh_speed = 30.0, 0.0

        # SH subtropical jet
        if np.any(~nh_mask):
            sh_lat = lat_deg[~nh_mask]
            sh_u = u_upper[~nh_mask]
            stj_sh_idx = np.argmax(sh_u)
            stj_sh_lat = sh_lat[stj_sh_idx]
            stj_sh_speed = sh_u[stj_sh_idx]
        else:
            stj_sh_lat, stj_sh_speed = -30.0, 0.0

        # Find polar/eddy-driven jet (mid-troposphere, ~500 hPa)
        mid_levels = (p_bar[:, self.nlat//2] > 400) & (p_bar[:, self.nlat//2] < 700)
        u_mid = np.mean(u_bar[mid_levels, :], axis=0) if np.any(mid_levels) else u_bar[self.nlev//2, :]

        # Look for secondary jet in mid-latitudes
        midlat_nh = (lat_deg > 40) & (lat_deg < 70)
        if np.any(midlat_nh):
            edj_nh_idx = np.argmax(u_mid[midlat_nh])
            edj_nh_lat = lat_deg[midlat_nh][edj_nh_idx]
            edj_nh_speed = u_mid[midlat_nh][edj_nh_idx]
        else:
            edj_nh_lat, edj_nh_speed = 50.0, 0.0

        return {
            'subtropical_jet_nh_lat': stj_nh_lat,
            'subtropical_jet_nh_speed': stj_nh_speed,
            'subtropical_jet_sh_lat': stj_sh_lat,
            'subtropical_jet_sh_speed': stj_sh_speed,
            'eddy_driven_jet_nh_lat': edj_nh_lat,
            'eddy_driven_jet_nh_speed': edj_nh_speed,
            'u_zonal_mean': u_bar,
            'latitude': lat_deg,
            'pressure': p_bar
        }

    def compute_storm_track(self, state, eddy_diag):
        """
        Compute storm track diagnostics

        Storm tracks are regions of high transient eddy activity.

        Parameters
        ----------
        state : ModelState
            Current model state
        eddy_diag : EddyDiagnostics
            Eddy diagnostics object with accumulated statistics

        Returns
        -------
        storm_track : dict
            Storm track diagnostics
        """
        # Use transient EKE as proxy for storm track
        eke_trans = eddy_diag.compute_transient_eke()

        if eke_trans is None:
            return {
                'storm_track_eke': None,
                'storm_track_lat_nh': None,
                'storm_track_lat_sh': None
            }

        # Find storm track latitude (latitude of max EKE)
        # Use lower-mid troposphere
        mid_level = int(0.6 * self.nlev)
        eke_mid = eke_trans[mid_level, :]

        lat_deg = np.rad2deg(self.grid.lat)

        # NH storm track
        nh_mask = lat_deg > 20
        if np.any(nh_mask):
            nh_idx = np.argmax(eke_mid[nh_mask])
            storm_track_lat_nh = lat_deg[nh_mask][nh_idx]
        else:
            storm_track_lat_nh = 45.0

        # SH storm track
        sh_mask = lat_deg < -20
        if np.any(sh_mask):
            sh_idx = np.argmax(eke_mid[sh_mask])
            storm_track_lat_sh = lat_deg[sh_mask][sh_idx]
        else:
            storm_track_lat_sh = -45.0

        return {
            'storm_track_eke': eke_trans,
            'storm_track_lat_nh': storm_track_lat_nh,
            'storm_track_lat_sh': storm_track_lat_sh
        }

    def compute_wave_activity_flux(self, state):
        """
        Compute wave activity flux (Plumb flux)

        Measures the propagation of stationary wave activity.

        Parameters
        ----------
        state : ModelState
            Current model state

        Returns
        -------
        Fx : ndarray
            Zonal component of wave activity flux
        Fy : ndarray
            Meridional component
        """
        # Simplified Plumb flux computation at a single level
        # Use geostrophic streamfunction anomaly at mid-troposphere

        mid_level = self.nlev // 2

        # Deviation from zonal mean at this level
        z = state.z[mid_level]  # Shape: (nlat, nlon)
        z_bar = np.mean(z, axis=-1, keepdims=True)  # Shape: (nlat, 1)
        z_star = z - z_bar  # Shape: (nlat, nlon)

        # Geostrophic streamfunction: psi = g*z/f
        # Use 1D Coriolis parameter
        f = 2 * self.Omega * np.sin(self.grid.lat)  # Shape: (nlat,)
        f = np.where(np.abs(f) < 1e-10, 1e-10, f)  # Avoid division by zero

        # Broadcast f to match z_star shape
        psi = self.g * z_star / f[:, np.newaxis]  # Shape: (nlat, nlon)

        # Wave activity flux (simplified)
        # Fx ~ -d(psi)/dy * d(psi)/dx
        # Fy ~ (d(psi)/dy)^2

        dpsi_dx = self.grid.gradient_x(psi)
        dpsi_dy = self.grid.gradient_y(psi)

        Fx = -dpsi_dy * dpsi_dx
        Fy = dpsi_dy**2

        return Fx, Fy

    def compute_all_circulation_diagnostics(self, state, zonal_diag=None, eddy_diag=None):
        """
        Compute comprehensive circulation diagnostics

        Parameters
        ----------
        state : ModelState
            Current model state
        zonal_diag : ZonalMeanDiagnostics, optional
            Zonal mean diagnostics object
        eddy_diag : EddyDiagnostics, optional
            Eddy diagnostics object

        Returns
        -------
        diagnostics : dict
            Dictionary containing all circulation diagnostics
        """
        # PV
        pv = self.compute_potential_vorticity(state)

        # Jet diagnostics
        jet = self.compute_jet_diagnostics(state)

        # Hadley cell (need streamfunction from zonal diag)
        if zonal_diag is not None:
            psi = zonal_diag.compute_meridional_streamfunction(state)
            h_nh, h_sh, h_edge_nh, h_edge_sh = self.compute_hadley_cell_diagnostics(psi)
        else:
            psi = None
            h_nh, h_sh, h_edge_nh, h_edge_sh = 0, 0, 30, -30

        # Storm track
        if eddy_diag is not None:
            storm = self.compute_storm_track(state, eddy_diag)
        else:
            storm = {'storm_track_eke': None, 'storm_track_lat_nh': None, 'storm_track_lat_sh': None}

        # Wave activity
        Fx, Fy = self.compute_wave_activity_flux(state)

        return {
            'potential_vorticity': pv,
            'pv_zonal_mean': np.mean(pv, axis=-1),
            'streamfunction': psi,
            'hadley_strength_nh': h_nh,
            'hadley_strength_sh': h_sh,
            'hadley_edge_nh': h_edge_nh,
            'hadley_edge_sh': h_edge_sh,
            'jet_diagnostics': jet,
            'storm_track': storm,
            'wave_activity_flux_x': Fx,
            'wave_activity_flux_y': Fy
        }
