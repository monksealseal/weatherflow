"""Evaluation metrics and utilities for GAIA-style workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class EvaluationSelectionConfig:
    """Configuration for selecting variables and levels from arrays.

    Axes are optional and must be provided if corresponding indices/names are set.
    """

    variable_axis: Optional[int] = None
    level_axis: Optional[int] = None
    variable_indices: Optional[Sequence[int]] = None
    level_indices: Optional[Sequence[int]] = None
    variable_names: Optional[Sequence[str]] = None
    level_values: Optional[Sequence[int]] = None


@dataclass(frozen=True)
class EvaluationSelectionMetadata:
    """Metadata used to resolve names/values into indices for selection."""

    variable_names: Optional[Sequence[str]] = None
    level_values: Optional[Sequence[int]] = None


@dataclass(frozen=True)
class ReliabilityResult:
    """Reliability curve data for probabilistic forecasts."""

    bin_edges: NDArray[np.float64]
    forecast_probabilities: NDArray[np.float64]
    observed_frequencies: NDArray[np.float64]
    counts: NDArray[np.int64]


def _to_numpy(data: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(data, dtype=float)


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    return axis


def _validate_axis(axis: Optional[int], ndim: int, label: str) -> None:
    if axis is None:
        return
    axis_norm = _normalize_axis(axis, ndim)
    if axis_norm < 0 or axis_norm >= ndim:
        raise ValueError(f"{label} axis {axis} is out of bounds for ndim={ndim}.")


def _resolve_indices(
    indices: Optional[Sequence[int]],
    names: Optional[Sequence[object]],
    metadata_values: Optional[Sequence[object]],
    label: str,
) -> Optional[Sequence[int]]:
    if indices is not None and names is not None:
        raise ValueError(f"Provide either {label} indices or {label} names, not both.")
    if names is None:
        return indices
    if metadata_values is None:
        raise ValueError(f"{label} metadata is required to resolve names.")
    lookup = {value: idx for idx, value in enumerate(metadata_values)}
    resolved = []
    for name in names:
        if name not in lookup:
            raise ValueError(f"{label} '{name}' not found in metadata.")
        resolved.append(lookup[name])
    return resolved


def apply_selection(
    data: ArrayLike,
    config: Optional[EvaluationSelectionConfig] = None,
    metadata: Optional[EvaluationSelectionMetadata] = None,
) -> NDArray[np.float64]:
    """Select variables/levels from an array using explicit config.

    The caller must supply axes if indices or names are provided.
    """

    array = _to_numpy(data)
    if config is None:
        return array

    _validate_axis(config.variable_axis, array.ndim, "variable")
    _validate_axis(config.level_axis, array.ndim, "level")

    variable_indices = _resolve_indices(
        config.variable_indices,
        config.variable_names,
        metadata.variable_names if metadata else None,
        "variable",
    )
    level_indices = _resolve_indices(
        config.level_indices,
        config.level_values,
        metadata.level_values if metadata else None,
        "level",
    )

    if variable_indices is not None:
        if config.variable_axis is None:
            raise ValueError("variable_axis must be provided when selecting variables.")
        array = np.take(array, variable_indices, axis=config.variable_axis)

    if level_indices is not None:
        if config.level_axis is None:
            raise ValueError("level_axis must be provided when selecting levels.")
        array = np.take(array, level_indices, axis=config.level_axis)

    return array


def _apply_selection_pair(
    pred: ArrayLike,
    target: ArrayLike,
    selection: Optional[EvaluationSelectionConfig],
    metadata: Optional[EvaluationSelectionMetadata],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        apply_selection(pred, selection, metadata),
        apply_selection(target, selection, metadata),
    )


def rmse(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    selection: Optional[EvaluationSelectionConfig] = None,
    metadata: Optional[EvaluationSelectionMetadata] = None,
    axis: Optional[Tuple[int, ...]] = None,
) -> float:
    """Root-mean-square error."""

    pred_sel, target_sel = _apply_selection_pair(pred, target, selection, metadata)
    error = pred_sel - target_sel
    return float(np.sqrt(np.mean(error**2, axis=axis)))


def mae(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    selection: Optional[EvaluationSelectionConfig] = None,
    metadata: Optional[EvaluationSelectionMetadata] = None,
    axis: Optional[Tuple[int, ...]] = None,
) -> float:
    """Mean absolute error."""

    pred_sel, target_sel = _apply_selection_pair(pred, target, selection, metadata)
    return float(np.mean(np.abs(pred_sel - target_sel), axis=axis))


def acc(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    selection: Optional[EvaluationSelectionConfig] = None,
    metadata: Optional[EvaluationSelectionMetadata] = None,
    axis: Optional[Tuple[int, ...]] = None,
    epsilon: float = 1e-8,
) -> float:
    """Anomaly correlation coefficient."""

    pred_sel, target_sel = _apply_selection_pair(pred, target, selection, metadata)
    pred_anom = pred_sel - np.mean(pred_sel, axis=axis, keepdims=True)
    target_anom = target_sel - np.mean(target_sel, axis=axis, keepdims=True)
    numerator = np.sum(pred_anom * target_anom, axis=axis)
    denom = np.sqrt(
        np.sum(pred_anom**2, axis=axis) * np.sum(target_anom**2, axis=axis)
    )
    return float(numerator / (denom + epsilon))


def crps_ensemble(
    ensemble: ArrayLike,
    observations: ArrayLike,
    *,
    ensemble_axis: int = 0,
    selection: Optional[EvaluationSelectionConfig] = None,
    metadata: Optional[EvaluationSelectionMetadata] = None,
) -> float:
    """Continuous ranked probability score for ensemble forecasts."""

    ensemble_sel, obs_sel = _apply_selection_pair(
        ensemble, observations, selection, metadata
    )
    ensemble_np = _to_numpy(ensemble_sel)
    obs_np = _to_numpy(obs_sel)

    ensemble_axis = _normalize_axis(ensemble_axis, ensemble_np.ndim)
    ensemble_np = np.moveaxis(ensemble_np, ensemble_axis, 0)
    term1 = np.mean(np.abs(ensemble_np - obs_np), axis=0)
    pairwise = np.abs(ensemble_np[:, None, ...] - ensemble_np[None, :, ...])
    term2 = 0.5 * np.mean(pairwise, axis=(0, 1))
    return float(np.mean(term1 - term2))


def reliability_curve(
    probabilities: ArrayLike,
    observations: ArrayLike,
    *,
    bins: int | Iterable[float] = 10,
    selection: Optional[EvaluationSelectionConfig] = None,
    metadata: Optional[EvaluationSelectionMetadata] = None,
) -> ReliabilityResult:
    """Compute reliability curve data for binary event forecasts."""

    probs_sel, obs_sel = _apply_selection_pair(
        probabilities, observations, selection, metadata
    )
    probs = _to_numpy(probs_sel).ravel()
    obs = _to_numpy(obs_sel).ravel()

    if isinstance(bins, int):
        if bins <= 0:
            raise ValueError("bins must be a positive integer.")
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
    else:
        bin_edges = np.asarray(list(bins), dtype=float)
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise ValueError("bins must provide at least two bin edges.")

    bin_count = bin_edges.size - 1
    forecast_probabilities = np.full(bin_count, np.nan, dtype=float)
    observed_frequencies = np.full(bin_count, np.nan, dtype=float)
    counts = np.zeros(bin_count, dtype=int)

    indices = np.digitize(probs, bin_edges, right=False) - 1
    indices = np.clip(indices, 0, bin_count - 1)

    for bin_idx in range(bin_count):
        mask = indices == bin_idx
        counts[bin_idx] = int(np.sum(mask))
        if counts[bin_idx] == 0:
            continue
        forecast_probabilities[bin_idx] = float(np.mean(probs[mask]))
        observed_frequencies[bin_idx] = float(np.mean(obs[mask]))

    return ReliabilityResult(
        bin_edges=bin_edges,
        forecast_probabilities=forecast_probabilities,
        observed_frequencies=observed_frequencies,
        counts=counts,
    )


def brier_score(
    probabilities: ArrayLike,
    observations: ArrayLike,
    *,
    selection: Optional[EvaluationSelectionConfig] = None,
    metadata: Optional[EvaluationSelectionMetadata] = None,
) -> float:
    """Brier score for binary event forecasts."""

    probs_sel, obs_sel = _apply_selection_pair(
        probabilities, observations, selection, metadata
    )
    probs = _to_numpy(probs_sel)
    obs = _to_numpy(obs_sel)
    return float(np.mean((probs - obs) ** 2))
