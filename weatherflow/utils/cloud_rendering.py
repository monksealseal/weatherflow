"""Volumetric cloud rendering utilities.

This module implements compact, differentiable-style utilities for rendering
volumetric clouds.  The focus is on CPU-friendly torch operations that make it
easy to experiment with atmospheric effects in notebooks or lightweight
services rather than on real-time GPU performance.  The implementation includes
ray marching with temporal reprojection, simple god-ray and shadowing support,
dual camera pipelines for pilot/orbital views, adaptive level of detail, and a
vertical cross-section renderer for cinematic diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RayMarchSettings:
    """Configuration for volumetric ray marching."""

    step_size: float = 0.02
    max_distance: float = 2.0
    density_scale: float = 1.0
    phase_g: float = 0.2
    shadow_step_size: float = 0.05
    transmittance_floor: float = 1e-3
    jitter: float = 0.0
    multi_scatter: float = 0.35


@dataclass
class CameraModel:
    """Simple pinhole camera description with curvature awareness."""

    position: torch.Tensor
    forward: torch.Tensor
    up: torch.Tensor
    fov: float = 60.0
    near: float = 0.01
    far: float = 5.0
    aspect: float = 1.0
    planet_radius: float = 1.0
    atmosphere_height: float = 0.25

    def generate_rays(self, resolution: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create ray origins and directions for the camera.

        Returns:
            origins: (H, W, 3) tensor of origins
            directions: (H, W, 3) tensor of normalized directions
        """

        height, width = resolution
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=self.position.device),
            torch.linspace(-1.0, 1.0, width, device=self.position.device),
            indexing="ij",
        )

        fov_rad = torch.deg2rad(torch.tensor(self.fov, device=self.position.device))
        xs = xs * torch.tan(fov_rad * 0.5) * self.aspect
        ys = ys * torch.tan(fov_rad * 0.5)

        forward = F.normalize(self.forward, dim=-1)
        right = F.normalize(torch.cross(forward, self.up), dim=-1)
        up = F.normalize(torch.cross(right, forward), dim=-1)

        dirs = (
            forward
            + xs[..., None] * right
            + ys[..., None] * up
        )
        dirs = F.normalize(dirs, dim=-1)

        origins = self.position.expand_as(dirs)
        return origins, dirs

    def horizon_falloff(self, directions: torch.Tensor) -> torch.Tensor:
        """Estimate horizon dimming based on curvature."""

        horizon_cos = torch.clamp(-directions[..., 1], 0.0, 1.0)
        curvature = torch.exp(
            -(self.planet_radius / (self.planet_radius + self.atmosphere_height))
            * horizon_cos
        )
        return curvature


@dataclass
class LightingModel:
    """Defines incoming light for volumetric scattering."""

    light_direction: torch.Tensor
    light_color: torch.Tensor = torch.tensor([1.0, 1.0, 1.0])
    ambient: float = 0.1


class TemporalReprojectionState:
    """Tracks history buffers for temporal accumulation."""

    def __init__(self) -> None:
        self._history_color: Optional[torch.Tensor] = None
        self._history_depth: Optional[torch.Tensor] = None

    def reproject(
        self,
        current_color: torch.Tensor,
        current_depth: torch.Tensor,
        motion_vectors: torch.Tensor,
        blend: float = 0.85,
        depth_tolerance: float = 0.05,
    ) -> torch.Tensor:
        """Combine the current frame with history using motion vectors."""

        if self._history_color is None or self._history_depth is None:
            self._history_color = current_color.detach()
            self._history_depth = current_depth.detach()
            return current_color

        height, width, _ = current_color.shape
        device = current_color.device

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device),
            torch.linspace(-1.0, 1.0, width, device=device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1) + motion_vectors
        grid = grid.unsqueeze(0)

        history_color = self._history_color.permute(2, 0, 1).unsqueeze(0)
        history_depth = self._history_depth.unsqueeze(0).unsqueeze(0)

        reproj_color = F.grid_sample(history_color, grid, align_corners=True)
        reproj_color = reproj_color.squeeze(0).permute(1, 2, 0)
        reproj_depth = F.grid_sample(history_depth, grid, align_corners=True)
        reproj_depth = reproj_depth.squeeze(0).squeeze(0)

        depth_valid = torch.abs(reproj_depth - current_depth) < depth_tolerance
        blended = torch.where(
            depth_valid[..., None],
            blend * reproj_color + (1.0 - blend) * current_color,
            current_color,
        )

        self._history_color = blended.detach()
        self._history_depth = current_depth.detach()
        return blended


class AdaptiveCloudLod:
    """Adaptive level-of-detail heuristics for volumetric clouds."""

    def __init__(
        self,
        base_resolution: Tuple[int, int, int] = (32, 64, 64),
        vram_budget_mb: float = 512.0,
        latency_budget_ms: float = 8.0,
    ) -> None:
        self.base_resolution = base_resolution
        self.vram_budget_mb = vram_budget_mb
        self.latency_budget_ms = latency_budget_ms

    def select_resolution(self, distances: torch.Tensor) -> Tuple[int, int, int]:
        """Choose a resolution given view distances."""

        median_distance = distances.median().item()
        scale = torch.clamp(1.0 / (1.0 + median_distance), 0.25, 1.0).item()

        depth, height, width = self.base_resolution
        scaled = (
            max(4, int(depth * scale)),
            max(8, int(height * scale)),
            max(8, int(width * scale)),
        )

        bytes_per_voxel = 4.0
        estimated_mb = bytes_per_voxel * np.prod(scaled) / (1024 * 1024)
        if estimated_mb > self.vram_budget_mb:
            shrink = (self.vram_budget_mb / estimated_mb) ** (1 / 3)
            scaled = (
                max(4, int(scaled[0] * shrink)),
                max(8, int(scaled[1] * shrink)),
                max(8, int(scaled[2] * shrink)),
            )

        return scaled

    def impostor_from_density(self, density: torch.Tensor) -> torch.Tensor:
        """Create a billboard-style impostor alpha mask."""

        alpha = torch.clamp(density.sum(dim=0), 0.0, 1.0)
        return alpha

    def decimate_density(self, density: torch.Tensor, resolution: Tuple[int, int, int]) -> torch.Tensor:
        """Downsample density to match the selected resolution."""

        density = density.unsqueeze(0).unsqueeze(0)
        pooled = F.interpolate(
            density,
            size=resolution,
            mode="trilinear",
            align_corners=True,
        )
        return pooled.squeeze(0).squeeze(0)


class VolumetricCloudRenderer:
    """CPU-friendly volumetric renderer for clouds."""

    def __init__(self, settings: Optional[RayMarchSettings] = None) -> None:
        self.settings = settings or RayMarchSettings()

    def generate_density_field(
        self, grid_shape: Tuple[int, int, int], seed: int = 0
    ) -> torch.Tensor:
        """Procedurally generate a cloud density field."""

        torch.manual_seed(seed)
        low_freq = torch.randn(1, 1, *grid_shape) * 0.5
        high_freq = F.interpolate(low_freq, scale_factor=0.5, mode="trilinear")
        density = low_freq + 0.5 * F.interpolate(
            high_freq, size=grid_shape, mode="trilinear", align_corners=True
        )
        density = torch.clamp(density.squeeze(0).squeeze(0), 0.0, 1.0)
        return density

    def _henyey_greenstein(self, cos_theta: torch.Tensor) -> torch.Tensor:
        g = self.settings.phase_g
        denom = 1.0 + g**2 - 2.0 * g * cos_theta
        return (1.0 - g**2) / torch.pow(denom, 1.5)

    def _sample_density(self, density: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        points = points.clamp(-1.0, 1.0)
        grid = points.view(1, 1, -1, 1, 3)
        samples = F.grid_sample(
            density.unsqueeze(0).unsqueeze(0), grid, align_corners=True
        )
        samples = samples.view(-1)
        return samples

    def _shadow_transmittance(
        self, density: torch.Tensor, points: torch.Tensor, light_dir: torch.Tensor
    ) -> torch.Tensor:
        step = self.settings.shadow_step_size
        num_steps = int(self.settings.max_distance / step)
        dirs = light_dir.expand_as(points)
        trans = torch.ones(points.shape[0], device=density.device)
        for i in range(num_steps):
            offsets = dirs * (i * step)
            sample_points = points + offsets
            alpha = self._sample_density(density, sample_points) * self.settings.density_scale
            trans = trans * torch.exp(-alpha * step)
        return trans

    def raymarch(
        self,
        density: torch.Tensor,
        camera: CameraModel,
        lighting: LightingModel,
        resolution: Tuple[int, int] = (64, 64),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render the density field from a camera."""

        origins, directions = camera.generate_rays(resolution)
        if self.settings.jitter > 0.0:
            jitter = torch.randn_like(origins[..., :1]) * self.settings.jitter
            origins = origins + directions * jitter

        step = self.settings.step_size
        num_steps = int(self.settings.max_distance / step)

        positions = []
        for i in range(num_steps):
            positions.append(origins + directions * (camera.near + i * step))
        positions_tensor = torch.stack(positions, dim=-2)  # (..., steps, 3)

        flat_pos = positions_tensor.reshape(-1, 3)
        densities = self._sample_density(density, flat_pos)
        densities = densities.view(*resolution, num_steps)

        cos_theta = torch.clamp(torch.sum(directions[..., None, :] * lighting.light_direction, dim=-1), -1.0, 1.0)
        phase = self._henyey_greenstein(cos_theta)

        shadow = self._shadow_transmittance(density, flat_pos, lighting.light_direction)
        shadow = shadow.view(*resolution, num_steps)

        sigma_s = densities * self.settings.density_scale
        trans = torch.exp(-sigma_s * step).cumprod(dim=-1)
        trans = torch.cat(
            [torch.ones((*resolution, 1), device=density.device), trans[..., :-1]],
            dim=-1,
        )

        scatter = sigma_s * step * (lighting.ambient + shadow * phase[..., None])
        color = trans[..., None] * scatter[..., None]
        color = color * lighting.light_color.view(1, 1, 1, 3)
        color = color.sum(dim=-2)

        multi_scatter_term = self.settings.multi_scatter * torch.exp(-densities.mean(dim=-1, keepdim=True))
        color = color + multi_scatter_term * lighting.light_color

        horizon = camera.horizon_falloff(directions)[..., None]
        color = color * horizon

        depth = torch.full(resolution, camera.far, device=density.device)
        occlusion_mask = densities > self.settings.transmittance_floor
        if occlusion_mask.any():
            first_hit = occlusion_mask.float().argmax(dim=-1)
            depth = camera.near + first_hit.float() * step

        return color.clamp(0.0, 1.0), depth


class DualCameraPipeline:
    """Render pilot and orbital views with atmospheric scattering."""

    def __init__(
        self,
        pilot_camera: CameraModel,
        orbital_camera: CameraModel,
        renderer: VolumetricCloudRenderer,
    ) -> None:
        self.pilot_camera = pilot_camera
        self.orbital_camera = orbital_camera
        self.renderer = renderer
        self.reprojection = TemporalReprojectionState()

    def _atmospheric_scatter(
        self, color: torch.Tensor, directions: torch.Tensor, altitude: float
    ) -> torch.Tensor:
        up_dot = torch.clamp(directions[..., 1], -1.0, 1.0)
        rayleigh = torch.exp(-altitude) * (1.0 + up_dot**2)
        mie = torch.exp(-altitude * 0.5) * (1.0 - up_dot**2)
        scattering = 0.65 * rayleigh[..., None] + 0.35 * mie[..., None]
        return color * scattering

    def render_dual(
        self,
        density: torch.Tensor,
        lighting: LightingModel,
        resolution: Tuple[int, int] = (64, 64),
    ) -> Dict[str, torch.Tensor]:
        pilot_color, pilot_depth = self.renderer.raymarch(
            density, self.pilot_camera, lighting, resolution
        )
        orbital_color, orbital_depth = self.renderer.raymarch(
            density, self.orbital_camera, lighting, resolution
        )

        _, pilot_dirs = self.pilot_camera.generate_rays(resolution)
        pilot_color = self._atmospheric_scatter(pilot_color, pilot_dirs, altitude=0.02)

        _, orbital_dirs = self.orbital_camera.generate_rays(resolution)
        orbital_color = self._atmospheric_scatter(orbital_color, orbital_dirs, altitude=0.2)

        motion = torch.zeros((*resolution, 2), device=density.device)
        orbital_color = self.reprojection.reproject(
            orbital_color, orbital_depth, motion
        )

        return {
            "pilot_color": pilot_color.clamp(0.0, 1.0),
            "pilot_depth": pilot_depth,
            "orbital_color": orbital_color.clamp(0.0, 1.0),
            "orbital_depth": orbital_depth,
        }


class VerticalCrossSectionRenderer:
    """Renders vertical slices with annotations and lighting hints."""

    def __init__(self, colormap: str = "Blues", translucency: float = 0.75) -> None:
        self.colormap = cm.get_cmap(colormap)
        self.translucency = translucency

    def slice_field(
        self,
        field: torch.Tensor,
        path: Iterable[Tuple[float, float]],
        vertical_levels: int = 32,
        samples_along_path: int = 64,
    ) -> torch.Tensor:
        """Extract a vertical cross-section along a path."""

        path_points = torch.tensor(list(path), dtype=torch.float32, device=field.device)
        if path_points.shape[0] < 2:
            raise ValueError("Path must contain at least two points")

        distances = torch.linspace(0, 1, samples_along_path, device=field.device)
        segment = path_points[-1] - path_points[0]
        points = path_points[0] + distances[:, None] * segment[None, :]

        zs = torch.linspace(0, 1, vertical_levels, device=field.device)
        grid_y, grid_x, grid_z = torch.meshgrid(points[:, 1], points[:, 0], zs, indexing="ij")
        grid = torch.stack([grid_z * 2 - 1, grid_y * 2 - 1, grid_x * 2 - 1], dim=-1)

        samples = F.grid_sample(
            field.unsqueeze(0).unsqueeze(0),
            grid.unsqueeze(0),
            align_corners=True,
        )
        return samples.squeeze(0).squeeze(0)

    def render_slice(
        self,
        field: torch.Tensor,
        path: Iterable[Tuple[float, float]],
        annotations: Optional[List[str]] = None,
        lighting: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """Convert a field slice into RGBA with annotations."""

        slice_field = self.slice_field(field, path)
        normalized = (slice_field - slice_field.min()) / (
            slice_field.max() - slice_field.min() + 1e-6
        )

        colors = torch.tensor(self.colormap(normalized.cpu().numpy()), device=field.device)
        colors = colors[..., :3]
        alphas = normalized * self.translucency
        lit = colors * (lighting + (1 - lighting) * normalized[..., None])

        result = torch.cat([lit, alphas[..., None]], dim=-1)
        metadata = {
            "annotations": annotations or [],
            "slice_min": slice_field.min().item(),
            "slice_max": slice_field.max().item(),
        }

        return {"rgba": result, "metadata": metadata}

