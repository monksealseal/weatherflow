"""Sampling helpers for GAIA inference with physical constraints."""
from __future__ import annotations

from typing import Optional

import torch

from weatherflow.gaia.constraints import ConstraintApplier
from weatherflow.gaia.model import GaiaModel


def gaia_sample(
    model: GaiaModel,
    inputs: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    constraints: Optional[ConstraintApplier] = None,
) -> torch.Tensor:
    """Run GAIA inference and apply optional physical constraints."""
    with torch.no_grad():
        outputs = model(inputs, lat, lon)
    if constraints is None:
        return outputs
    if outputs.dim() == 5:
        batch, members, channels, height, width = outputs.shape
        reshaped = outputs.view(batch * members, channels, height, width)
        constrained = constraints.apply(reshaped)
        return constrained.view(batch, members, channels, height, width)
    return constraints.apply(outputs)
