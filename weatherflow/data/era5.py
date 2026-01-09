import json
import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .cache import DatasetCache, get_default_cache, get_persistent_era5_path


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

    By default, uses a persistent cache directory (~/.weatherflow/datasets/era5/)
    so you don't need to re-download data every time you log in.

    Example:
        # Uses persistent cache automatically
        dataset = ERA5Dataset(
            years=[2020, 2021],
            variables=['temperature', 'u_component_of_wind'],
            levels=[500, 850],
            download=True
        )

        # Or specify a custom directory
        dataset = ERA5Dataset(
            root_dir="/custom/path",
            years=[2020],
            variables=['temperature'],
            levels=[500],
        )
    """

    def __init__(
        self,
        years: Iterable[int],
        variables: Sequence[str],
        levels: Iterable[int],
        root_dir: Optional[str] = None,
        download: bool = False,
        cache: Optional[DatasetCache] = None,
    ):
        """
        Args:
            years: Years to load (e.g. [2018, 2019]).
            variables: ERA5 variable names (e.g. ['u_component_of_wind']).
            levels: Pressure levels (e.g. [500, 850]).
            root_dir: Local cache folder for .nc files. If None, uses the
                persistent cache at ~/.weatherflow/datasets/era5/
            download: If True, auto-fetch missing data via CDSAPI.
            cache: Optional DatasetCache instance. If None, uses the default.
        """
        self.variables = list(variables)
        self.levels = _coerce_levels(levels)
        self.years = _coerce_years(years)

        # Use persistent cache if no root_dir specified
        if root_dir is None:
            self._cache = cache or get_default_cache()
            self.root_dir = str(self._cache.get_era5_path())
            self._using_persistent_cache = True
        else:
            self.root_dir = os.fspath(root_dir)
            self._cache = cache
            self._using_persistent_cache = False

        if download:
            self._download_data()

        try:
            self.ds = xr.open_mfdataset(
                os.path.join(self.root_dir, "era5_*.nc"),
                combine="by_coords",
                parallel=True,
            )
        except OSError as exc:
            cached_years = []
            if self._using_persistent_cache and self._cache:
                cached_years = self._cache.get_cached_era5_years()

            if cached_years:
                raise FileNotFoundError(
                    f"No NetCDF files found in {self.root_dir} for years {self.years}.\n"
                    f"Available cached years: {cached_years}\n"
                    f"Set download=True to fetch missing data."
                ) from exc
            else:
                raise FileNotFoundError(
                    f"No NetCDF files found in {self.root_dir}.\n"
                    f"Set download=True to fetch them, or check your cache with:\n"
                    f"  from weatherflow.data.cache import print_cache_info\n"
                    f"  print_cache_info()"
                ) from exc

        self.ds = self.ds[self.variables].sel(
            level=self.levels,
            time=slice(str(self.years[0]), str(self.years[-1])),
        )

        self.stats = self._load_or_compute_stats()

        # Update cache access time
        if self._using_persistent_cache and self._cache:
            self._cache.update_access_time("era5")

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
            raise RuntimeError(
                "Could not initialize CDSAPI. Ensure ~/.cdsapirc is configured."
            ) from exc

        os.makedirs(self.root_dir, exist_ok=True)

        if self._using_persistent_cache:
            print(f"📁 Using persistent cache: {self.root_dir}")

        downloaded_any = False
        for year in self.years:
            fname = f"era5_{year}.nc"
            path = os.path.join(self.root_dir, fname)

            if os.path.exists(path):
                print(f"✅ Found {fname} in cache, skipping download.")
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
            downloaded_any = True
            print(f"✅ Downloaded {fname} to persistent cache.")

        # Register the dataset in the cache
        if self._using_persistent_cache and self._cache and downloaded_any:
            self._cache.register_dataset(
                "era5",
                {
                    "years": list(self.years),
                    "variables": self.variables,
                    "levels": list(self.levels),
                },
            )
            print(f"📦 Dataset registered in cache. View with: print_cache_info()")

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
    train_years: Iterable[int],
    val_years: Iterable[int],
    variables: Sequence[str],
    levels: Iterable[int],
    root_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience helper to build train/validation dataloaders.

    By default, uses a persistent cache directory (~/.weatherflow/datasets/era5/)
    so you don't need to re-download data every time you log in.

    Args:
        train_years: Years for training data.
        val_years: Years for validation data.
        variables: ERA5 variable names.
        levels: Pressure levels to include.
        root_dir: Directory containing (or to store) ERA5 NetCDF files.
            If None, uses the persistent cache at ~/.weatherflow/datasets/era5/
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker processes.
        download: If True, fetch any missing files before loading.

    Returns:
        Tuple of (train_loader, val_loader).

    Example:
        # Uses persistent cache automatically
        train_loader, val_loader = create_data_loaders(
            train_years=[2018, 2019, 2020],
            val_years=[2021],
            variables=['temperature', 'u_component_of_wind'],
            levels=[500, 850],
            download=True,
        )
    """
    train_ds = ERA5Dataset(
        years=train_years,
        variables=variables,
        levels=levels,
        root_dir=root_dir,
        download=download,
    )
    val_ds = ERA5Dataset(
        years=val_years,
        variables=variables,
        levels=levels,
        root_dir=root_dir,
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
