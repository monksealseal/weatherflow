"""Curriculum schedules for GAIA rollout training."""

from __future__ import annotations

from dataclasses import dataclass


def linear_rollout_schedule(step: int, start: int, end: int, total_steps: int) -> int:
    """Linearly increase rollout horizon with training steps."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    fraction = min(max(step / total_steps, 0.0), 1.0)
    horizon = int(round(start + (end - start) * fraction))
    return max(1, horizon)


@dataclass(frozen=True)
class RolloutCurriculum:
    """Simple curriculum for deterministic autoregressive rollout length."""

    start_horizon: int
    end_horizon: int
    total_steps: int

    def horizon_for_step(self, step: int) -> int:
        return linear_rollout_schedule(step, self.start_horizon, self.end_horizon, self.total_steps)
