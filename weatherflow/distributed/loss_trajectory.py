"""Trajectory-aware flow matching loss for multi-step supervision."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..manifolds.sphere import Sphere


@dataclass
class TrajectoryLossConfig:
    alpha: float = 1.0
    divergence_weight: float = 0.1
    max_pairs: int = 4


class TrajectoryFlowLoss:
    """Multi-step flow matching objective."""

    def __init__(self, config: Optional[TrajectoryLossConfig] = None):
        self.config = config or TrajectoryLossConfig()
        self.sphere = Sphere()

    def _divergence(self, v: torch.Tensor) -> torch.Tensor:
        if v.shape[1] < 2:
            return torch.tensor(0.0, device=v.device, dtype=v.dtype)
        u = v[:, 0:1]
        v_comp = v[:, 1:2]
        _, _, lat_size, lon_size = u.shape
        lat_grid = torch.linspace(-torch.pi / 2, torch.pi / 2, steps=lat_size, device=v.device, dtype=v.dtype)
        eps = self.sphere._get_eps(v.dtype)
        cos_lat = torch.cos(lat_grid).clamp(min=eps).view(1, 1, lat_size, 1)
        dlon = (2 * torch.pi) / max(lon_size, 1)
        dphi = torch.pi / max(lat_size - 1, 1)
        du_dlambda = torch.gradient(u, spacing=(dlon,), dim=(3,))[0]
        dvcos_dphi = torch.gradient(v_comp * cos_lat, spacing=(dphi,), dim=(2,))[0]
        radius = torch.tensor(self.sphere.radius, device=v.device, dtype=v.dtype)
        return (du_dlambda / (radius * cos_lat)) + (dvcos_dphi / radius)

    def __call__(
        self,
        trajectory: torch.Tensor,
        dt: torch.Tensor,
        model: torch.nn.Module,
        targets: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
        forcing: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the trajectory flow loss.

        Args:
            trajectory: Tensor [T, B, C, H, W] or [B, T, C, H, W]
            dt: Time difference between consecutive steps (hours or seconds)
            model: Flow model with forward signature (x, t, static, forcing)
        """
        if trajectory.dim() != 5:
            raise ValueError("Trajectory must include time dimension")
        if trajectory.shape[0] < trajectory.shape[1]:
            traj = trajectory  # assume [B, T, C, H, W]
        else:
            traj = trajectory.permute(1, 0, 2, 3, 4)  # convert [T, B, C, H, W] -> [B, T, C, H, W]
        b, t_steps, c, h, w = traj.shape
        if dt.numel() == 1:
            dt_per_sample = dt.to(traj.device)
        else:
            dt_per_sample = dt.to(traj.device).view(-1)

        indices = [(i, j) for i in range(t_steps - 1) for j in range(i + 1, t_steps)]
        random.shuffle(indices)
        indices = indices[: self.config.max_pairs]

        total_flow = torch.tensor(0.0, device=traj.device)
        total_div = torch.tensor(0.0, device=traj.device)
        total_weight = torch.tensor(0.0, device=traj.device)
        horizons = []
        for i, j in indices:
            xi = traj[:, i]
            xj = traj[:, j]
            horizon = (j - i)
            horizons.append(horizon)
            weight = float(horizon**self.config.alpha)
            t_rand = torch.rand(b, device=traj.device)
            if dt_per_sample.numel() == 1:
                dt_factor = dt_per_sample
                dt_view = dt_per_sample
            else:
                dt_factor = dt_per_sample.view(b)
                dt_view = dt_factor.view(b, 1, 1, 1)
            t_scaled = t_rand * horizon * dt_factor
            x_t = torch.lerp(xi, xj, t_rand.view(b, 1, 1, 1))
            v_pred = model(x_t, t_scaled, static=static, forcing=forcing)
            v_target = (xj - xi) / (horizon * dt_view)
            flow_loss = F.mse_loss(v_pred, v_target)
            div = self._divergence(v_pred)
            div_loss = torch.mean(div**2)
            total_flow = total_flow + weight * flow_loss
            total_div = total_div + weight * div_loss
            total_weight = total_weight + weight

        total_weight = torch.clamp(total_weight, min=1e-6)
        flow_loss = total_flow / total_weight
        div_loss = total_div / total_weight
        total_loss = flow_loss + self.config.divergence_weight * div_loss
        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "div_loss": div_loss,
            "avg_horizon": torch.tensor(horizons, device=traj.device, dtype=torch.float32).mean()
            if horizons
            else torch.tensor(0.0, device=traj.device),
        }


__all__ = ["TrajectoryFlowLoss", "TrajectoryLossConfig"]
