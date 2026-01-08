"""Post-processing utilities for Gaia inference."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch

from .pipeline import SchemaDefinition


def _build_stats_tensor(
    stats: Mapping[str, Mapping[str, float]],
    variables: Sequence[str],
    field: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = []
    for name in variables:
        entry = stats.get(name)
        if entry is None or field not in entry:
            raise ValueError(f"Missing {field} for variable '{name}'.")
        values.append(float(entry[field]))
    tensor = torch.tensor(values, device=device, dtype=dtype)
    return tensor.view(1, -1, 1, 1)


def _apply_constraints(
    data: torch.Tensor,
    schema: SchemaDefinition,
    constraints: Mapping[str, float],
) -> torch.Tensor:
    constrained = data.clone()
    for idx, variable in enumerate(schema.names):
        minimum = constraints.get(f"{variable}_min")
        maximum = constraints.get(f"{variable}_max")
        if minimum is not None:
            constrained[:, idx] = torch.clamp(constrained[:, idx], min=minimum)
        if maximum is not None:
            constrained[:, idx] = torch.clamp(constrained[:, idx], max=maximum)
    return constrained


def denormalize_and_constrain(
    data: torch.Tensor,
    normalization_stats: Mapping[str, Mapping[str, float]],
    schema: SchemaDefinition,
    constraints: Optional[Mapping[str, float]] = None,
) -> torch.Tensor:
    """Apply denormalization and physical constraints to model output."""
    if data.dim() != 4:
        raise ValueError("Expected data with shape (batch, channels, lat, lon).")

    if data.shape[1] != len(schema.variables):
        raise ValueError(
            "Channel count does not match schema variables: "
            f"{data.shape[1]} vs {len(schema.variables)}."
        )

    mean = _build_stats_tensor(
        normalization_stats, schema.names, "mean", data.device, data.dtype
    )
    std = _build_stats_tensor(
        normalization_stats, schema.names, "std", data.device, data.dtype
    )

    denormalized = data * std + mean

    if constraints:
        denormalized = _apply_constraints(denormalized, schema, constraints)

    return denormalized
