"""Loss functions for GAIA training workflows."""

from __future__ import annotations

from typing import Iterable

import torch


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute root mean squared error."""
    return torch.sqrt(torch.mean((prediction - target) ** 2))


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean absolute error."""
    return torch.mean(torch.abs(prediction - target))


def crps_ensemble(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    member_dim: int = 0,
) -> torch.Tensor:
    """Compute ensemble CRPS using the empirical formula.

    Args:
        ensemble: Tensor with ensemble members along ``member_dim``.
        target: Observations broadcastable to ``ensemble`` without ``member_dim``.
        member_dim: Dimension containing ensemble members.
    """
    ens = torch.movedim(ensemble, member_dim, 0)
    term1 = torch.mean(torch.abs(ens - target), dim=0)
    pairwise_diff = torch.abs(ens[:, None, ...] - ens[None, :, ...])
    term2 = 0.5 * torch.mean(pairwise_diff, dim=(0, 1))
    crps = term1 - term2
    return torch.mean(crps)


def spectral_crps(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    member_dim: int = 0,
    time_dim: int = -2,
) -> torch.Tensor:
    """Compute CRPS on the magnitude spectrum of a time series."""
    ens = torch.movedim(ensemble, member_dim, 0)
    target_expanded = target
    if target.dim() == ens.dim() - 1:
        target_expanded = target.unsqueeze(0)
    ens_spectrum = torch.fft.rfft(ens, dim=time_dim)
    target_spectrum = torch.fft.rfft(target_expanded, dim=time_dim)
    ens_mag = torch.abs(ens_spectrum)
    target_mag = torch.abs(target_spectrum)
    return crps_ensemble(ens_mag, target_mag, member_dim=0)


def weighted_loss_sum(losses: Iterable[torch.Tensor], weights: Iterable[float]) -> torch.Tensor:
    """Combine losses with weights."""
    loss_list = list(losses)
    weight_list = list(weights)
    if len(loss_list) != len(weight_list):
        raise ValueError("Loss and weight lists must be the same length.")
    total = torch.zeros((), device=loss_list[0].device)
    for loss, weight in zip(loss_list, weight_list):
        total = total + loss * weight
    return total
