import json
import os
from typing import Dict, Iterable, Sequence, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset


def _coerce_years(years: Iterable[int]) -> Sequence[int]:
    """Return a sorted, unique list of integer years."""
    year_list = sorted({int(year) for year in years})
    if not year_list:
        raise ValueError("At least one year must be provided.")
    return year_list


def _coerce_levels(levels: Iterable[int]) -> Sequence[int]:
    """Return a sorted, unique list of integer pressure levels."""
    level_list = sorted({int(level) for level in levels})
    if not level_list:
        raise ValueError("At least one pressure level must be provided.")
    return level_list


class ERA5Dataset(Dataset):
    """
    A production-ready ERA5 data loader for Flow Matching models.
    Handles auto-downloading (CDSAPI), lazy-loading (Xarray), and normalization.
    """

    def __init__(
        self,
        root_dir: str,
        years: Iterable[int],
        variables: Sequence[str],
        levels: Iterable[int],
        download: bool = False,
    ):
        """
        Args:
            root_dir: Local cache folder for .nc files.
            years: Years to load (e.g. [2018, 2019]).
            variables: ERA5 variable names (e.g. ['u_component_of_wind']).
            levels: Pressure levels (e.g. [500, 850]).
            download: If True, auto-fetch missing data via CDSAPI.
        """
        self.root_dir = os.fspath(root_dir)
        self.variables = list(variables)
        self.levels = _coerce_levels(levels)
        self.years = _coerce_years(years)

        if download:
            self._download_data()

        try:
            self.ds = xr.open_mfdataset(
                os.path.join(self.root_dir, "era5_*.nc"),
                combine="by_coords",
                parallel=True,
            )
        except OSError as exc:
            raise FileNotFoundError(
                f"No NetCDF files found in {self.root_dir}. Set download=True to fetch them."
            ) from exc

        self.ds = self.ds[self.variables].sel(
            level=self.levels,
            time=slice(str(self.years[0]), str(self.years[-1])),
        )

        self.stats = self._load_or_compute_stats()

    def __len__(self) -> int:
        return self.ds.sizes["time"]

    def __getitem__(self, idx: int) -> torch.Tensor:
        data_slice = self.ds.isel(time=idx).load()
        numpy_data = data_slice.to_array(dim="variable").values
        tensor = torch.from_numpy(numpy_data).float()

        std = torch.clamp(self.stats["std"], min=1e-6)
        norm_tensor = (tensor - self.stats["mean"]) / std
        return norm_tensor

    def _download_data(self) -> None:
        """Downloads ERA5 data year-by-year using cdsapi."""
        import cdsapi

        try:
            client = cdsapi.Client()
        except Exception as exc:
            config_path = Path.home() / ".cdsapirc"
            raise RuntimeError(
                f"Could not initialize CDSAPI. Ensure {config_path} is configured with your API key. "
                f"See: https://cds.climate.copernicus.eu/api-how-to"
            ) from exc

        os.makedirs(self.root_dir, exist_ok=True)

        for year in self.years:
            fname = f"era5_{year}.nc"
            path = os.path.join(self.root_dir, fname)

            if os.path.exists(path):
                print(f"✅ Found {fname}, skipping.")
                continue

            print(f"⬇️ Requesting ERA5 data for {year}...")
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": self.variables,
                    "pressure_level": [str(level) for level in self.levels],
                    "year": str(year),
                    "month": [f"{month:02d}" for month in range(1, 13)],
                    "day": [f"{day:02d}" for day in range(1, 32)],
                    "time": [f"{hour:02d}:00" for hour in range(0, 24, 6)],
                },
                path,
            )

    def _load_or_compute_stats(self) -> Dict[str, torch.Tensor]:
        """Computes mean/std once and caches them to JSON."""
        stats_path = os.path.join(self.root_dir, "stats.json")

        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as stats_file:
                stats = json.load(stats_file)
            return {
                key: torch.tensor(value, dtype=torch.float32)
                .unsqueeze(-1)
                .unsqueeze(-1)
                for key, value in stats.items()
            }

        print("⏳ Computing normalization stats (one-time setup)...")
        mean = self.ds.mean(dim=["time", "latitude", "longitude"]).to_array().values
        std = self.ds.std(dim=["time", "latitude", "longitude"]).to_array().values
        std = np.where(std == 0, 1e-6, std)

        stats: Dict[str, Sequence[Sequence[float]]] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        with open(stats_path, "w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file)

        return {
            "mean": torch.tensor(mean, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1),
            "std": torch.tensor(std, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1),
        }

    def visualize(self, idx: int = 0) -> None:
        """Sanity check: denormalizes and plots the first variable."""
        tensor = self.__getitem__(idx)
        denorm = tensor * torch.clamp(self.stats["std"], min=1e-6) + self.stats["mean"]
        img = denorm[0, :, :].numpy()

        plt.figure(figsize=(10, 6))
        plt.imshow(img, cmap="RdBu_r", origin="upper")
        plt.colorbar(label="Physical Units")
        plt.title(
            f"Sample Visualization (Index {idx})\nVar: {self.variables[0]}, Level: {self.levels[0]}"
        )
        plt.show()


def create_data_loaders(
    root_dir: str,
    train_years: Iterable[int],
    val_years: Iterable[int],
    variables: Sequence[str],
    levels: Iterable[int],
    batch_size: int = 32,
    num_workers: int = 0,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience helper to build train/validation dataloaders.

    Args:
        root_dir: Directory containing (or to store) ERA5 NetCDF files.
        train_years: Years for training data.
        val_years: Years for validation data.
        variables: ERA5 variable names.
        levels: Pressure levels to include.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker processes.
        download: If True, fetch any missing files before loading.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_ds = ERA5Dataset(
        root_dir=root_dir,
        years=train_years,
        variables=variables,
        levels=levels,
        download=download,
    )
    val_ds = ERA5Dataset(
        root_dir=root_dir,
        years=val_years,
        variables=variables,
        levels=levels,
        download=download,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
