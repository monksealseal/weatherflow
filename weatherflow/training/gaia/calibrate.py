"""Calibration utilities for probabilistic GAIA rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from weatherflow.training.gaia.losses import crps_ensemble, spectral_crps


@dataclass
class CalibrationBatch:
    """Container for GAIA calibration batches."""

    context: torch.Tensor
    targets: torch.Tensor


def calibration_step(
    model: torch.nn.Module,
    batch: CalibrationBatch,
    spectral_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute CRPS-based calibration loss with optional spectral term."""
    ensemble = model(batch.context)
    crps_loss = crps_ensemble(ensemble, batch.targets, member_dim=0)
    total_loss = crps_loss
    metrics = {"crps": crps_loss.item()}
    if spectral_weight is not None and spectral_weight > 0.0:
        spectral_loss = spectral_crps(ensemble, batch.targets, member_dim=0, time_dim=-2)
        total_loss = total_loss + spectral_weight * spectral_loss
        metrics["spectral_crps"] = spectral_loss.item()
    metrics["total_loss"] = total_loss.item()
    return total_loss, metrics
