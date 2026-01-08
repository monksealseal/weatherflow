"""Autoregressive rollout utilities for Gaia inference."""

from __future__ import annotations

from typing import Callable, List

import torch


PredictFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def autoregressive_rollout(
    predict_fn: PredictFn,
    analysis_state_t0: torch.Tensor,
    analysis_state_t1: torch.Tensor,
    steps: int,
) -> List[torch.Tensor]:
    """Perform an autoregressive rollout from two consecutive states.

    Args:
        predict_fn: Callable that maps (state_t0, state_t1) -> state_t2.
        analysis_state_t0: Tensor for the first state.
        analysis_state_t1: Tensor for the second state.
        steps: Number of rollout steps to predict (>= 1).

    Returns:
        List of states including the initial two and predicted steps.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1 for rollout.")

    states: List[torch.Tensor] = [analysis_state_t0, analysis_state_t1]
    prev_state = analysis_state_t0
    current_state = analysis_state_t1

    for _ in range(steps):
        next_state = predict_fn(prev_state, current_state)
        states.append(next_state)
        prev_state, current_state = current_state, next_state

    return states
