"""Interactive graduate-level atmospheric dynamics learning tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import guard exercised indirectly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover - handled at runtime
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]
    _PLOTLY_IMPORT_ERROR = exc
else:  # pragma: no cover - simple assignment
    _PLOTLY_IMPORT_ERROR = None

OMEGA = 7.2921159e-5  # Earth's rotation rate [s^-1]
R_EARTH = 6.371e6  # Earth's radius [m]
GRAVITY = 9.80665  # Gravity [m s^-2]
R_AIR = 287.0  # Gas constant for dry air [J kg^-1 K^-1]


@dataclass
class SolutionStep:
    """Represents a single instructional step in a worked solution."""

    description: str
    value: float
    units: str


@dataclass
class ProblemScenario:
    """Container for a complete educational problem with solution."""

    title: str
    problem: str
    solution_steps: List[SolutionStep]
    answer: str


def _require_plotly() -> None:
    """Ensure Plotly is available before creating visualizations."""

    if go is None or make_subplots is None:  # pragma: no cover - simple guard
        message = (
            "Plotly is required for the educational dashboards. Install it via"
            " `pip install plotly>=5.18.0` or install WeatherFlow's optional"
            " extras."
        )
        raise ImportError(message) from _PLOTLY_IMPORT_ERROR


class GraduateAtmosphericDynamicsTool:
    """Comprehensive toolkit for graduate atmospheric dynamics education.

    The class bundles together diagnostic calculators, physically based
    simulations, and high-quality Plotly visualizations.  It is designed
    to provide students with intuition for the governing equations while
    simultaneously walking them through detailed problem-solving steps.
    """

    OMEGA = OMEGA
    R_EARTH = R_EARTH
    GRAVITY = GRAVITY
    R_AIR = R_AIR

    def __init__(self, reference_latitude: float = 45.0) -> None:
        """Initialize the tool with a reference latitude."""

        self.reference_latitude = reference_latitude

    # ------------------------------------------------------------------
    # Fundamental diagnostic utilities
    # ------------------------------------------------------------------
    @staticmethod
    def coriolis_parameter(latitude: np.ndarray | float) -> np.ndarray:
        """Return the Coriolis parameter for the supplied latitude(s)."""

        lat_rad = np.deg2rad(latitude)
        return 2.0 * OMEGA * np.sin(lat_rad)

    @staticmethod
    def beta_parameter(latitude: np.ndarray | float) -> np.ndarray:
        """Return the beta-plane parameter at the supplied latitude(s)."""

        lat_rad = np.deg2rad(latitude)
        return 2.0 * OMEGA * np.cos(lat_rad) / R_EARTH

    @staticmethod
    def rossby_dispersion_relation(
        beta: float, mean_flow: float, k: np.ndarray, l: np.ndarray
    ) -> np.ndarray:
        """Return the Rossby wave frequency ω for given wavenumbers."""

        denom = k**2 + l**2
        return mean_flow * k - beta * k / denom

    @staticmethod
    def rossby_group_velocity(
        beta: float, mean_flow: float, k: np.ndarray, l: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the group velocity components for Rossby waves."""

        denom = k**2 + l**2
        cg_x = mean_flow - beta * (denom - 2.0 * k**2) / denom**2
        cg_y = 2.0 * beta * k * l / denom**2
        return cg_x, cg_y

    @staticmethod
    def compute_geostrophic_wind(
        height_field: np.ndarray,
        latitudes: Sequence[float],
        longitudes: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute geostrophic wind components from a height field."""

        latitudes = np.asarray(latitudes)
        longitudes = np.asarray(longitudes)
        field = np.asarray(height_field, dtype=float)

        if field.shape != (latitudes.size, longitudes.size):
            raise ValueError(
                "Height field must have shape (n_lat, n_lon) matching coordinate lengths."
            )

        # Metric distances for differentiation
        y_coords = R_EARTH * np.deg2rad(latitudes)

        dphidy = np.empty_like(field)
        for j in range(field.shape[1]):
            dphidy[:, j] = np.gradient(field[:, j], y_coords, edge_order=2)

        dphidx = np.empty_like(field)
        lon_radians = np.deg2rad(longitudes)
        for i, lat in enumerate(latitudes):
            x_coords = R_EARTH * np.cos(np.deg2rad(lat)) * lon_radians
            dphidx[i, :] = np.gradient(field[i, :], x_coords, edge_order=2)

        f = GraduateAtmosphericDynamicsTool.coriolis_parameter(latitudes)[:, None]
        threshold = 1.0e-5
        sign = np.where(f >= 0.0, 1.0, -1.0)
        f = np.where(np.abs(f) < threshold, threshold * sign, f)

        u_g = -GRAVITY * dphidy / f
        v_g = GRAVITY * dphidx / f
        return u_g, v_g

    @staticmethod
    def compute_quasigeostrophic_pv(
        streamfunction: np.ndarray,
        z_levels: Sequence[float],
        y_coords: Sequence[float],
        x_coords: Sequence[float],
        f0: float,
        beta: float,
        stratification: Sequence[float],
    ) -> np.ndarray:
        """Compute quasi-geostrophic potential vorticity from ψ."""

        psi = np.asarray(streamfunction, dtype=float)
        z = np.asarray(z_levels, dtype=float)
        y = np.asarray(y_coords, dtype=float)
        x = np.asarray(x_coords, dtype=float)
        N = np.asarray(stratification, dtype=float)

        if psi.shape != (z.size, y.size, x.size):
            raise ValueError(
                "Streamfunction must have shape (n_z, n_y, n_x) matching coordinate lengths."
            )
        if N.size != z.size:
            raise ValueError("Stratification profile must match the number of vertical levels.")

        psi_z, psi_y, psi_x = np.gradient(psi, z, y, x, edge_order=2)
        psi_zz = np.gradient(psi_z, z, axis=0, edge_order=2)
        psi_yy = np.gradient(psi_y, y, axis=1, edge_order=2)
        psi_xx = np.gradient(psi_x, x, axis=2, edge_order=2)

        laplacian = psi_xx + psi_yy
        n_squared = N**2
        stretched_term = (f0**2 / n_squared[:, None, None]) * psi_zz

        y_grid = np.meshgrid(z, y, x, indexing="ij")[1]
        pv = laplacian + stretched_term + beta * y_grid
        return pv

    # ------------------------------------------------------------------
    # Visualization suites
    # ------------------------------------------------------------------
    def create_balanced_flow_dashboard(
        self,
        height_field: np.ndarray,
        latitudes: Sequence[float],
        longitudes: Sequence[float],
    ) -> go.Figure:
        """Create a 3-D visualization of geostrophic balance."""

        _require_plotly()
        u_g, v_g = self.compute_geostrophic_wind(height_field, latitudes, longitudes)
        speed = np.sqrt(u_g**2 + v_g**2)

        lons, lats = np.meshgrid(longitudes, latitudes)

        # Downsample for cone plot clarity
        skip_lat = max(1, len(latitudes) // 20)
        skip_lon = max(1, len(longitudes) // 20)
        ds = (slice(None, None, skip_lat), slice(None, None, skip_lon))

        cones = go.Cone(
            x=lons[ds],
            y=lats[ds],
            z=height_field[ds],
            u=u_g[ds],
            v=v_g[ds],
            w=np.zeros_like(u_g[ds]),
            sizemode="absolute",
            sizeref=np.nanmax(speed) / 8.0 if np.any(speed) else 1.0,
            anchor="tail",
            colorscale="Portland",
            showscale=False,
        )

        surface = go.Surface(
            x=lons,
            y=lats,
            z=height_field,
            surfacecolor=speed,
            colorscale="Turbo",
            colorbar=dict(title="|V_g| (m s⁻¹)"),
            opacity=0.92,
        )

        fig = go.Figure(data=[surface, cones])
        fig.update_layout(
            title="Balanced Flow Explorer",
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                zaxis_title="Geopotential Height (m)",
                aspectmode="cube",
            ),
            template="plotly_dark",
        )
        return fig

    def create_rossby_wave_lab(
        self,
        beta: float | None = None,
        mean_flow: float = 15.0,
        k_range: Tuple[float, float, int] = (1e-7, 8e-6, 60),
        l_range: Tuple[float, float, int] = (0.0, 6e-6, 60),
    ) -> go.Figure:
        """Create an interactive Rossby wave dispersion laboratory."""

        _require_plotly()
        beta_value = (
            beta
            if beta is not None
            else float(self.beta_parameter(self.reference_latitude))
        )
        k = np.linspace(*k_range)
        l = np.linspace(*l_range)
        kk, ll = np.meshgrid(k, l)

        omega = self.rossby_dispersion_relation(beta_value, mean_flow, kk, ll)
        c_phase = np.where(kk != 0.0, omega / kk, 0.0)
        cg_x, cg_y = self.rossby_group_velocity(beta_value, mean_flow, kk, ll)

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "surface"}, {"type": "heatmap"}, {"type": "heatmap"}]],
            subplot_titles=(
                "Frequency Surface",
                "Zonal Phase Speed",
                "Meridional Group Velocity",
            ),
        )

        fig.add_trace(
            go.Surface(
                x=kk,
                y=ll,
                z=omega,
                colorscale="Viridis",
                colorbar=dict(title="ω (s⁻¹)"),
                opacity=0.95,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=k,
                y=l,
                z=c_phase,
                colorscale="Balance",
                colorbar=dict(title="cₓ (m s⁻¹)"),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Heatmap(
                x=k,
                y=l,
                z=cg_y,
                colorscale="Temps",
                colorbar=dict(title="c_{gy} (m s⁻¹)"),
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            title="Rossby Wave Dispersion Studio",
            scene=dict(
                xaxis_title="Zonal Wavenumber (m⁻¹)",
                yaxis_title="Meridional Wavenumber (m⁻¹)",
                zaxis_title="Frequency (s⁻¹)",
            ),
            xaxis2_title="Zonal Wavenumber (m⁻¹)",
            yaxis2_title="Meridional Wavenumber (m⁻¹)",
            xaxis3_title="Zonal Wavenumber (m⁻¹)",
            yaxis3_title="Meridional Wavenumber (m⁻¹)",
            template="plotly_dark",
        )
        return fig

    def create_pv_atelier(
        self,
        streamfunction: np.ndarray,
        z_levels: Sequence[float],
        y_coords: Sequence[float],
        x_coords: Sequence[float],
        f0: float | None = None,
        beta: float | None = None,
        stratification: Sequence[float] | None = None,
    ) -> go.Figure:
        """Create a volumetric potential vorticity visualization."""

        _require_plotly()
        f_value = f0 if f0 is not None else float(self.coriolis_parameter(self.reference_latitude))
        beta_value = (
            beta if beta is not None else float(self.beta_parameter(self.reference_latitude))
        )
        if stratification is None:
            stratification = np.full(len(z_levels), 0.012)

        pv = self.compute_quasigeostrophic_pv(
            streamfunction,
            z_levels,
            y_coords,
            x_coords,
            f_value,
            beta_value,
            stratification,
        )

        z_vals_km = np.asarray(z_levels) / 1000.0
        y_vals_km = np.asarray(y_coords) / 1000.0
        x_vals_km = np.asarray(x_coords) / 1000.0

        z_grid, y_grid, x_grid = np.meshgrid(
            z_vals_km, y_vals_km, x_vals_km, indexing="ij"
        )

        pv_norm = (pv - pv.min()) / (pv.max() - pv.min() + 1.0e-12)
        volume = go.Volume(
            x=x_grid.flatten(),
            y=y_grid.flatten(),
            z=z_grid.flatten(),
            value=pv_norm.flatten(),
            isomin=0.1,
            isomax=0.9,
            opacity=0.1,
            surface_count=12,
            colorscale="Plasma",
            caps=dict(x_show=False, y_show=False),
        )

        middle_level = pv[pv.shape[0] // 2]
        slice_surface = go.Surface(
            x=np.tile(x_vals_km, (len(y_coords), 1)),
            y=np.tile(y_vals_km[:, None], (1, len(x_coords))),
            z=np.full((len(y_coords), len(x_coords)), z_vals_km[len(z_levels) // 2]),
            surfacecolor=middle_level,
            colorscale="curl",
            colorbar=dict(title="PV (s⁻¹)"),
            showscale=True,
            opacity=0.95,
        )

        fig = go.Figure(data=[volume, slice_surface])
        fig.update_layout(
            title="Potential Vorticity Atelier",
            scene=dict(
                xaxis_title="x (km)",
                yaxis_title="y (km)",
                zaxis_title="Height (km)",
                aspectmode="cube",
            ),
            template="plotly_dark",
        )
        return fig

    # ------------------------------------------------------------------
    # Worked problem solvers and study aids
    # ------------------------------------------------------------------
    def geostrophic_wind_solution(
        self, height_difference: float, distance: float, latitude: float
    ) -> Tuple[float, List[SolutionStep]]:
        """Return the geostrophic wind for a height gradient problem."""

        gradient = height_difference / distance
        f = float(self.coriolis_parameter(latitude))
        ug = -GRAVITY * gradient / f
        steps = [
            SolutionStep("Compute Coriolis parameter", f, "s⁻¹"),
            SolutionStep("Form height gradient dZ/dy", gradient, "m m⁻¹"),
            SolutionStep("Solve for zonal geostrophic wind", ug, "m s⁻¹"),
        ]
        return ug, steps

    def thermal_wind_solution(
        self,
        temperature_gradient: float,
        pressure_lower: float,
        pressure_upper: float,
        latitude: float,
    ) -> Tuple[float, List[SolutionStep]]:
        """Return shear implied by the thermal wind relation."""

        f = float(self.coriolis_parameter(latitude))
        delta_ln_p = np.log(pressure_lower / pressure_upper)
        shear = R_AIR * temperature_gradient * delta_ln_p / f
        steps = [
            SolutionStep("Compute Coriolis parameter", f, "s⁻¹"),
            SolutionStep(
                "Evaluate Δln(p) between the two pressure levels",
                delta_ln_p,
                "",
            ),
            SolutionStep(
                "Apply thermal wind relation Δu_g = (R/f) ∂T/∂y Δln p",
                shear,
                "m s⁻¹",
            ),
        ]
        return shear, steps

    def rossby_phase_speed_solution(
        self,
        wavelength_x: float,
        mean_flow: float,
        beta: float | None = None,
        meridional_wavenumber: float = 0.0,
    ) -> Tuple[float, List[SolutionStep]]:
        """Return Rossby wave phase speed for a canonical setup."""

        beta_value = beta if beta is not None else float(self.beta_parameter(self.reference_latitude))
        k = 2.0 * np.pi / wavelength_x
        l = meridional_wavenumber
        omega = self.rossby_dispersion_relation(beta_value, mean_flow, k, l)
        c_phase = omega / k
        steps = [
            SolutionStep("Convert wavelength to zonal wavenumber", k, "m⁻¹"),
            SolutionStep("Evaluate dispersion relation ω", omega, "s⁻¹"),
            SolutionStep("Compute phase speed c = ω/k", c_phase, "m s⁻¹"),
        ]
        return c_phase, steps

    def generate_problem_scenarios(self) -> List[ProblemScenario]:
        """Generate curated graduate-level practice problems."""

        ug, geo_steps = self.geostrophic_wind_solution(
            height_difference=60.0, distance=500_000.0, latitude=42.0
        )
        thermal_shear, thermal_steps = self.thermal_wind_solution(
            temperature_gradient=4.0e-6,
            pressure_lower=85000.0,
            pressure_upper=50000.0,
            latitude=50.0,
        )
        phase_speed, phase_steps = self.rossby_phase_speed_solution(
            wavelength_x=4.0e6,
            mean_flow=20.0,
            meridional_wavenumber=2.0e-6,
        )

        problems = [
            ProblemScenario(
                title="High-Latitude Jet Diagnosis",
                problem=(
                    "A 500-hPa geopotential height field increases by 60 m over 500 km to the north."
                    " Estimate the zonal geostrophic wind at 42°N and interpret its direction."
                ),
                solution_steps=geo_steps,
                answer=f"u_g ≈ {ug:.1f} m s⁻¹ (westerly)",
            ),
            ProblemScenario(
                title="Thermal Wind Shear Analysis",
                problem=(
                    "Given a meridional temperature gradient of 4 K over 1000 km between 850 hPa"
                    " and 500 hPa at 50°N, quantify the vertical shear of the geostrophic wind."
                ),
                solution_steps=thermal_steps,
                answer=f"Δu_g ≈ {thermal_shear:.1f} m s⁻¹ (stronger aloft)",
            ),
            ProblemScenario(
                title="Oblique Rossby Wave Packet",
                problem=(
                    "For a Rossby wave with a 4000 km zonal wavelength propagating on a 20 m s⁻¹"
                    " mean flow and meridional wavenumber 2×10⁻⁶ m⁻¹, find the zonal phase speed."
                ),
                solution_steps=phase_steps,
                answer=f"cₓ ≈ {phase_speed:.1f} m s⁻¹",
            ),
        ]
        return problems

    def conceptual_checklist(self) -> Dict[str, str]:
        """Return a high-level conceptual study guide."""

        return {
            "Potential Vorticity Conservation": (
                "Track how PV couples vorticity and stratification.  The atelier visualization"
                " highlights column stretching and the β-effect simultaneously."
            ),
            "Wave-Mean Flow Interaction": (
                "Rossby wave group velocity arrows reveal how energy propagates relative"
                " to the phase speed, a critical ingredient in understanding baroclinic eddies."
            ),
            "Thermal Wind Balance": (
                "Use the shear calculator to connect observed temperature gradients with jet"
                " structure across pressure levels."
            ),
        }

