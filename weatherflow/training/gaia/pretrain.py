"""Pretraining utilities for masked reconstruction and temporal ordering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from weatherflow.training.gaia.losses import rmse


@dataclass
class PretrainBatch:
    """Container for GAIA pretraining batches."""

    inputs: torch.Tensor


def apply_variable_mask(inputs: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a random mask to variables within the input tensor."""
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be between 0 and 1.")
    mask = torch.rand_like(inputs) < mask_ratio
    masked_inputs = inputs.clone()
    masked_inputs[mask] = 0.0
    return masked_inputs, mask


def masked_reconstruction_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked reconstruction loss using RMSE."""
    reconstruction = model(inputs)
    masked_values = reconstruction[mask]
    target_values = inputs[mask]
    return rmse(masked_values, target_values)


def temporal_ordering_loss(
    model: torch.nn.Module,
    sequence: torch.Tensor,
) -> torch.Tensor:
    """Binary temporal ordering objective using a model-provided classifier."""
    batch_size = sequence.shape[0]
    reordered = sequence.clone()
    labels = torch.zeros(batch_size, dtype=torch.long, device=sequence.device)
    reverse_mask = torch.rand(batch_size, device=sequence.device) < 0.5
    reordered[reverse_mask] = torch.flip(reordered[reverse_mask], dims=(1,))
    labels[reverse_mask] = 1
    logits = model.temporal_ordering_logits(reordered)
    return F.cross_entropy(logits, labels)


def pretrain_step(
    model: torch.nn.Module,
    batch: PretrainBatch,
    mask_ratio: float = 0.15,
    ordering_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute pretraining loss and metrics for a batch."""
    masked_inputs, mask = apply_variable_mask(batch.inputs, mask_ratio)
    reconstruction_loss = masked_reconstruction_loss(model, masked_inputs, mask)
    ordering_loss = temporal_ordering_loss(model, batch.inputs)
    total_loss = reconstruction_loss + ordering_weight * ordering_loss
    metrics = {
        "reconstruction_loss": reconstruction_loss.item(),
        "ordering_loss": ordering_loss.item(),
        "total_loss": total_loss.item(),
    }
    return total_loss, metrics
