"""Physical constraint application utilities for GAIA outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch


class Constraint:
    """Base class for physical constraints."""

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class NonNegativeConstraint(Constraint):
    """Clamp selected channels to be non-negative."""

    channels: Sequence[int]

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        constrained = tensor.clone()
        constrained[:, self.channels] = torch.clamp(constrained[:, self.channels], min=0.0)
        return constrained


@dataclass
class RangeConstraint(Constraint):
    """Clamp selected channels to a min/max range."""

    channels: Sequence[int]
    min_value: float
    max_value: float

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        constrained = tensor.clone()
        constrained[:, self.channels] = torch.clamp(
            constrained[:, self.channels], min=self.min_value, max=self.max_value
        )
        return constrained


@dataclass
class MeanPreservingConstraint(Constraint):
    """Preserve the global mean for selected channels."""

    channels: Sequence[int]
    target_mean: Optional[torch.Tensor] = None

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        constrained = tensor.clone()
        for channel in self.channels:
            current = constrained[:, channel]
            if self.target_mean is None:
                target = current.mean(dim=(-2, -1), keepdim=True)
            else:
                target = self.target_mean[channel].to(current.device)
            delta = target - current.mean(dim=(-2, -1), keepdim=True)
            constrained[:, channel] = current + delta
        return constrained


class ConstraintApplier:
    """Apply a sequence of constraints to tensors."""

    def __init__(self, constraints: Optional[Iterable[Constraint]] = None) -> None:
        self.constraints = list(constraints or [])

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        output = tensor
        for constraint in self.constraints:
            output = constraint.apply(output)
        return output
