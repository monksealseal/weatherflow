"""Regridding utilities with explicit strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import xarray as xr


class RegridStrategy(str, Enum):
    """Supported regridding strategies."""

    BILINEAR = "bilinear"
    CONSERVATIVE = "conservative"


@dataclass(frozen=True)
class TargetGrid:
    """Definition of a target latitude/longitude grid."""

    latitude: np.ndarray
    longitude: np.ndarray


def regrid_dataset(
    dataset: xr.Dataset,
    target_grid: TargetGrid,
    strategy: RegridStrategy,
    variable_names: Iterable[str] | None = None,
) -> xr.Dataset:
    """Regrid a dataset to the target grid."""
    if variable_names is None:
        variable_names = list(dataset.data_vars)

    if strategy == RegridStrategy.BILINEAR:
        return _bilinear_regrid(dataset, target_grid, variable_names)
    if strategy == RegridStrategy.CONSERVATIVE:
        return _conservative_regrid(dataset, target_grid, variable_names)

    raise ValueError(f"Unsupported regrid strategy: {strategy}")


def _bilinear_regrid(
    dataset: xr.Dataset,
    target_grid: TargetGrid,
    variable_names: Iterable[str],
) -> xr.Dataset:
    return dataset[tuple(variable_names)].interp(
        latitude=target_grid.latitude,
        longitude=target_grid.longitude,
        method="linear",
    )


def _conservative_regrid(
    dataset: xr.Dataset,
    target_grid: TargetGrid,
    variable_names: Iterable[str],
) -> xr.Dataset:
    lat = dataset["latitude"].values
    lon = dataset["longitude"].values
    target_lat = target_grid.latitude
    target_lon = target_grid.longitude

    lat_factor = _coarsen_factor(lat, target_lat)
    lon_factor = _coarsen_factor(lon, target_lon)

    _validate_alignment(lat, target_lat, lat_factor, "latitude", allow_block_mean=True)
    _validate_alignment(lon, target_lon, lon_factor, "longitude", allow_block_mean=True)

    weights = _latitude_weights(dataset["latitude"])

    regridded = xr.Dataset()
    for name in variable_names:
        data = dataset[name]
        weighted_sum = (data * weights).coarsen(
            latitude=lat_factor,
            longitude=lon_factor,
            boundary="trim",
        ).sum()
        weight_sum = weights.coarsen(
            latitude=lat_factor,
            longitude=lon_factor,
            boundary="trim",
        ).sum()
        regridded[name] = weighted_sum / weight_sum

    regridded = regridded.assign_coords(
        latitude=dataset["latitude"].coarsen(
            latitude=lat_factor,
            boundary="trim",
        ).mean(),
        longitude=dataset["longitude"].coarsen(
            longitude=lon_factor,
            boundary="trim",
        ).mean(),
    )

    _validate_alignment(regridded["latitude"].values, target_lat, 1, "latitude")
    _validate_alignment(regridded["longitude"].values, target_lon, 1, "longitude")

    return regridded


def _coarsen_factor(source: np.ndarray, target: np.ndarray) -> int:
    source_step = _grid_step(source)
    target_step = _grid_step(target)
    if target_step < source_step:
        raise ValueError("Target grid must be coarser for conservative regridding.")
    factor = int(round(target_step / source_step))
    if not np.isclose(source_step * factor, target_step):
        raise ValueError("Target grid spacing must be an integer multiple.")
    return factor


def _grid_step(values: np.ndarray) -> float:
    if len(values) < 2:
        raise ValueError("Grid must contain at least two points.")
    return float(np.diff(values).mean())


def _validate_alignment(
    source: np.ndarray,
    target: np.ndarray,
    factor: int,
    label: str,
    allow_block_mean: bool = False,
) -> None:
    if len(source) // factor != len(target):
        raise ValueError(f"Target {label} grid length mismatch.")
    aligned = np.allclose(source[::factor], target, rtol=1e-5, atol=1e-6)
    if allow_block_mean:
        block_mean = source[: len(target) * factor].reshape(-1, factor).mean(axis=1)
        aligned = aligned or np.allclose(block_mean, target, rtol=1e-5, atol=1e-6)
    if not aligned:
        raise ValueError(f"Target {label} grid is not aligned with source grid.")


def _latitude_weights(latitude: xr.DataArray) -> xr.DataArray:
    lat_radians = np.deg2rad(latitude)
    weights = np.cos(lat_radians)
    return xr.DataArray(weights, coords={"latitude": latitude}, dims=("latitude",))
