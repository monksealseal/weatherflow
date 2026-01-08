"""Fine-tuning utilities for deterministic autoregressive rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from weatherflow.training.gaia.losses import rmse
from weatherflow.training.gaia.schedules import RolloutCurriculum


@dataclass
class FinetuneBatch:
    """Container for GAIA fine-tuning batches."""

    context: torch.Tensor
    targets: torch.Tensor


def autoregressive_rollout_loss(
    model: torch.nn.Module,
    context: torch.Tensor,
    targets: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Run deterministic autoregressive rollout for a fixed horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if targets.shape[1] < horizon:
        raise ValueError("targets do not provide enough steps for the horizon.")
    current = context
    loss = torch.zeros((), device=context.device)
    for step in range(horizon):
        prediction = model(current)
        loss = loss + rmse(prediction, targets[:, step])
        current = torch.cat([current[:, 1:], prediction[:, None, :]], dim=1)
    return loss / horizon


def finetune_step(
    model: torch.nn.Module,
    batch: FinetuneBatch,
    curriculum: RolloutCurriculum,
    global_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute fine-tuning loss with curriculum rollout scheduling."""
    horizon = curriculum.horizon_for_step(global_step)
    loss = autoregressive_rollout_loss(model, batch.context, batch.targets, horizon)
    metrics = {"loss": loss.item(), "horizon": float(horizon)}
    return loss, metrics
