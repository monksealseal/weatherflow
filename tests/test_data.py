"""Tests for the data loading utilities."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import h5py
import numpy as np
import pytest
import torch
import xarray as xr

from weatherflow.data import ERA5Dataset, WeatherDataset, create_data_loaders


@pytest.fixture
def synthetic_hdf5_dataset(tmp_path: Path) -> Path:
    """Create a tiny HDF5 dataset on disk for :class:`WeatherDataset`."""

    root = tmp_path / "hdf5"
    root.mkdir()

    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 3, 4)).astype("float32")

    for variable in ("temperature", "geopotential"):
        with h5py.File(root / f"{variable}_train.h5", "w") as handle:
            handle.create_dataset(variable, data=data)

    return root


@pytest.fixture
def synthetic_era5_file(tmp_path: Path) -> Path:
    """Create a minimal ERA5-like NetCDF file for unit tests."""

    rng = np.random.default_rng(0)

    times = np.array(
        [
            np.datetime64("2000-01-01T00"),
            np.datetime64("2000-01-01T06"),
            np.datetime64("2000-01-01T12"),
            np.datetime64("2000-01-01T18"),
            np.datetime64("2000-01-02T00"),
        ]
    )
    levels = np.array([500], dtype="int32")
    latitudes = np.linspace(-30, 30, 3, dtype="float32")
    longitudes = np.linspace(0, 180, 4, dtype="float32")

    shape = (len(times), len(levels), len(latitudes), len(longitudes))
    geopotential = rng.standard_normal(shape).astype("float32")
    temperature = rng.standard_normal(shape).astype("float32") + 250.0

    dataset = xr.Dataset(
        {
            "geopotential": (
                ("time", "level", "latitude", "longitude"),
                geopotential,
            ),
            "temperature": (
                ("time", "level", "latitude", "longitude"),
                temperature,
            ),
        },
        coords={
            "time": times,
            "level": levels,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )

    path = tmp_path / "synthetic_era5.nc"
    dataset.to_netcdf(path)
    return path


def test_weather_dataset_reads_hdf5_files(synthetic_hdf5_dataset: Path) -> None:
    dataset = WeatherDataset(str(synthetic_hdf5_dataset), ["temperature", "geopotential"])

    assert len(dataset) == 8
    sample = dataset[0]

    assert set(sample) == {"temperature", "geopotential"}
    assert sample["temperature"].shape == (3, 4)


def test_era5_dataset_local_file_loading(synthetic_era5_file: Path) -> None:
    dataset = ERA5Dataset(
        variables=["z", "t"],
        pressure_levels=[500],
        data_path=str(synthetic_era5_file),
        time_slice=("2000-01-01", "2000-01-02"),
        normalize=False,
        cache_data=True,
        verbose=False,
    )

    # The dataset has five timestamps which yield four consecutive pairs.
    assert len(dataset) == 4

    sample = dataset[0]
    assert sample["input"].shape == (2, 1, 3, 4)
    assert sample["target"].shape == (2, 1, 3, 4)
    assert sample["metadata"]["variables"] == ["geopotential", "temperature"]

    # Cached access should return the exact same tensors.
    cached = dataset[0]
    assert torch.equal(sample["input"], cached["input"])
    assert torch.equal(sample["target"], cached["target"])


def test_create_data_loaders_from_local_file(synthetic_era5_file: Path) -> None:
    train_loader, val_loader = create_data_loaders(
        variables=["z", "t"],
        pressure_levels=[500],
        data_path=str(synthetic_era5_file),
        train_slice=("2000-01-01", "2000-01-01T18"),
        val_slice=("2000-01-01T12", "2000-01-02"),
        batch_size=2,
        num_workers=0,
        normalize=False,
    )

    train_batch = next(iter(train_loader))
    assert train_batch["input"].shape == (2, 2, 1, 3, 4)
    assert train_batch["target"].shape == (2, 2, 1, 3, 4)

    val_batch = next(iter(val_loader))
    assert val_batch["input"].shape[1:] == (2, 1, 3, 4)
    assert 1 <= val_batch["input"].shape[0] <= 2  # final batch may be smaller
