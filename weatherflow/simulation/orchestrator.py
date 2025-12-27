"""Orchestration utilities for selecting simulation cores and LOD streaming."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SimulationCoreSpec:
    """Description of a supported simulation core."""

    name: str
    dimensionality: str
    vertical_coordinate: str
    supports_moisture: bool
    recommended_timestep: int
    description: str


@dataclass
class ResolutionTier:
    """Resolution tiers used to zoom from global to local domains."""

    name: str
    lat: int
    lon: int
    vertical_levels: int
    time_step_seconds: int
    description: str


class MoistureScheme:
    """Lightweight moisture and phase-change proxy for real-time budgets."""

    def apply(self, state: torch.Tensor, config: object) -> torch.Tensor:
        if not getattr(config, "enable", True):
            return state

        condensation_threshold = getattr(config, "condensation_threshold", 0.6)
        condensation_rate = getattr(config, "condensation_rate", 0.15)
        evaporation_rate = getattr(config, "evaporation_rate", 0.05)
        cloud_entrainment = getattr(config, "cloud_entrainment", 0.1)

        humidity = torch.sigmoid(state)
        supersaturation = torch.clamp(humidity - condensation_threshold, min=0.0)
        condensation = supersaturation * condensation_rate
        evaporation = (1.0 - humidity) * evaporation_rate
        cloud_boost = condensation * cloud_entrainment

        return state + condensation - evaporation + cloud_boost


class SurfaceFluxScheme:
    """Surface flux proxy tuned for responsiveness."""

    def apply(self, state: torch.Tensor, config: object) -> torch.Tensor:
        latent_coeff = getattr(config, "latent_coeff", 0.4)
        sensible_coeff = getattr(config, "sensible_coeff", 0.2)
        drag_coeff = getattr(config, "drag_coeff", 0.05)

        mean_surface = torch.mean(state, dim=(-2, -1), keepdim=True)
        latent_flux = latent_coeff * torch.tanh(mean_surface)
        sensible_flux = sensible_coeff * mean_surface
        drag = drag_coeff * state

        return state + latent_flux + sensible_flux - drag


class LODStreamer:
    """Chunk-based level-of-detail streamer for atmospheric fields."""

    def describe_field(self, field: torch.Tensor, config: object) -> Dict[str, object]:
        if field.dim() != 3:
            raise ValueError("Expected a [channels, lat, lon] field for LOD streaming")

        min_chunk = max(1, int(getattr(config, "min_chunk", 8)))
        max_chunk = max(min_chunk, int(getattr(config, "max_chunk", 64)))
        overlap = max(0, int(getattr(config, "overlap", 2)))
        max_zoom = max(0, int(getattr(config, "max_zoom", 3)))

        _, lat, lon = field.shape
        chunk_lat = min(max_chunk, max(min_chunk, lat))
        chunk_lon = min(max_chunk, max(min_chunk, lon))

        tiles: List[Dict[str, object]] = []
        for level in range(max_zoom + 1):
            lat_step = max(1, chunk_lat // (2 ** level))
            lon_step = max(1, chunk_lon // (2 ** level))
            tile_idx = 0
            stride_lat = max(1, lat_step - overlap)
            stride_lon = max(1, lon_step - overlap)

            for lat_start in range(0, lat, stride_lat):
                lat_end = min(lat, lat_start + lat_step)
                for lon_start in range(0, lon, stride_lon):
                    lon_end = min(lon, lon_start + lon_step)
                    tile = field[:, lat_start:lat_end, lon_start:lon_end]
                    tiles.append(
                        {
                            "level": level,
                            "tile": f"L{level}-{tile_idx}",
                            "latStart": lat_start,
                            "latEnd": lat_end,
                            "lonStart": lon_start,
                            "lonEnd": lon_end,
                            "mean": float(tile.mean()),
                            "std": float(tile.std(unbiased=False)),
                        }
                    )
                    tile_idx += 1

        return {
            "chunkShape": [chunk_lat, chunk_lon],
            "tiles": tiles,
        }


class SimulationOrchestrator:
    """Coordinates simulation core choice, resolution tiers, and LOD streaming."""

    def __init__(
        self,
        cores: Optional[Dict[str, SimulationCoreSpec]] = None,
        resolution_tiers: Optional[Dict[str, ResolutionTier]] = None,
    ) -> None:
        self.cores = cores or self._default_cores()
        self.resolution_tiers = resolution_tiers or self._default_resolution_tiers()
        self.moisture_scheme = MoistureScheme()
        self.surface_flux_scheme = SurfaceFluxScheme()
        self.lod_streamer = LODStreamer()

    def _default_cores(self) -> Dict[str, SimulationCoreSpec]:
        return {
            "shallow-water": SimulationCoreSpec(
                name="shallow-water",
                dimensionality="2D",
                vertical_coordinate="single-layer",
                supports_moisture=True,
                recommended_timestep=300,
                description="Fast barotropic core for real-time previews.",
            ),
            "hydrostatic-3d": SimulationCoreSpec(
                name="hydrostatic-3d",
                dimensionality="3D",
                vertical_coordinate="pressure",
                supports_moisture=True,
                recommended_timestep=120,
                description="3D hydrostatic core with stratified layers.",
            ),
            "anelastic-3d": SimulationCoreSpec(
                name="anelastic-3d",
                dimensionality="3D",
                vertical_coordinate="height",
                supports_moisture=True,
                recommended_timestep=60,
                description="Compressible-friendly anelastic core for storms.",
            ),
        }

    def _default_resolution_tiers(self) -> Dict[str, ResolutionTier]:
        return {
            "custom": ResolutionTier(
                name="custom",
                lat=0,
                lon=0,
                vertical_levels=0,
                time_step_seconds=300,
                description="Use user-provided grid settings.",
            ),
            "global-low": ResolutionTier(
                name="global-low",
                lat=32,
                lon=64,
                vertical_levels=20,
                time_step_seconds=600,
                description="Coarse global coverage for overview maps.",
            ),
            "regional": ResolutionTier(
                name="regional",
                lat=64,
                lon=128,
                vertical_levels=35,
                time_step_seconds=300,
                description="Mid-resolution nests for continental zooms.",
            ),
            "local-high": ResolutionTier(
                name="local-high",
                lat=96,
                lon=192,
                vertical_levels=40,
                time_step_seconds=120,
                description="High fidelity close to the player camera.",
            ),
        }

    def resolve_grid_size(
        self, lat: int, lon: int, resolution_tier: str
    ) -> Tuple[int, int, int]:
        tier = self.resolution_tiers.get(
            resolution_tier, self.resolution_tiers["custom"]
        )
        if tier.name == "custom" or tier.lat == 0 or tier.lon == 0:
            return lat, lon, tier.time_step_seconds
        return tier.lat, tier.lon, tier.time_step_seconds

    def seed_initial_state(
        self,
        channels: int,
        grid: Tuple[int, int],
        initial_source: str,
        boundary_source: str,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        lat, lon = grid
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        base = torch.randn((channels, lat, lon), generator=generator, device=device)
        boundary_gradient = torch.linspace(-0.5, 0.5, steps=lon, device=device)
        base = base + boundary_gradient.view(1, 1, lon)

        if initial_source == "parametric":
            base = base * 0.5
        if boundary_source == "reanalysis":
            base = base + 0.1 * torch.sin(
                torch.linspace(0, 3.14, lat, device=device)
            ).view(1, lat, 1)

        return base

    def apply_boundary_conditions(
        self, state: torch.Tensor, boundary_source: str, step_fraction: float
    ) -> torch.Tensor:
        if boundary_source not in {"reanalysis", "parametric"}:
            return state

        ramp = torch.tanh(torch.tensor(step_fraction, device=state.device))
        edge = torch.mean(state, dim=(-2, -1), keepdim=True)
        adjustment = ramp * edge * 0.05
        return state + adjustment

    def simulate_time_step(
        self,
        state: torch.Tensor,
        core: str,
        time_step_seconds: int,
        dynamics_scale: float,
        replay_window_seconds: int,
        boundary_source: str,
        boundary_update_seconds: int,
        step_idx: int,
    ) -> torch.Tensor:
        core_spec = self.cores.get(core, self.cores["shallow-water"])
        dt_scale = time_step_seconds / max(1.0, float(core_spec.recommended_timestep))
        advected = state + dynamics_scale * torch.tanh(state) * dt_scale

        replay_span = max(1, replay_window_seconds // time_step_seconds)
        phase = (step_idx % replay_span) / float(replay_span)
        advected = advected + 0.05 * torch.sin(
            torch.tensor(phase * 3.14, device=state.device)
        )

        boundary_phase = (step_idx * time_step_seconds) % max(
            boundary_update_seconds, 1
        )
        boundary_fraction = boundary_phase / max(boundary_update_seconds, 1)
        advected = self.apply_boundary_conditions(
            advected, boundary_source, boundary_fraction
        )
        return advected

    def apply_moisture_and_surface_flux(
        self, state: torch.Tensor, moisture_config: object, flux_config: object
    ) -> torch.Tensor:
        moist_state = self.moisture_scheme.apply(state, moisture_config)
        return self.surface_flux_scheme.apply(moist_state, flux_config)

    def stream_level_of_detail(
        self, field: torch.Tensor, lod_config: object
    ) -> Dict[str, object]:
        return self.lod_streamer.describe_field(field, lod_config)

    def available_resolution_tiers(self) -> List[ResolutionTier]:
        return list(self.resolution_tiers.values())

    def available_cores(self) -> List[SimulationCoreSpec]:
        return list(self.cores.values())
