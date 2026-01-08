"""
Mini Dataset Utilities for Quick Weather AI Experiments

Provides tools for creating small, fast-loading datasets ideal for:
- Quick prototyping and debugging
- Architecture comparison
- Hyperparameter tuning
- Educational demonstrations

Features:
- Synthetic weather data generation
- Mini subsets of ERA5 data
- Precomputed sample datasets
- Memory-efficient loading
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class MiniDataset(Dataset):
    """
    Small synthetic dataset for quick experiments.

    Generates weather-like data with realistic properties:
    - Spatial correlations (smooth fields)
    - Temporal evolution
    - Multiple variables with different characteristics

    Example:
        >>> dataset = MiniDataset(num_samples=100, img_size=(32, 64))
        >>> x, y = dataset[0]
        >>> print(x.shape)  # [channels, 32, 64]
    """

    def __init__(
        self,
        num_samples: int = 100,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        img_size: Tuple[int, int] = (32, 64),
        forecast_steps: int = 1,
        add_noise: bool = True,
        noise_std: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize mini dataset.

        Args:
            num_samples: Number of samples to generate
            in_channels: Number of input channels (variables)
            out_channels: Number of output channels (defaults to in_channels)
            img_size: Spatial dimensions (height, width)
            forecast_steps: Number of forecast steps (temporal offset)
            add_noise: Add realistic noise to data
            noise_std: Standard deviation of noise
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.img_size = img_size
        self.forecast_steps = forecast_steps
        self.add_noise = add_noise
        self.noise_std = noise_std

        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate data
        self.data = self._generate_data()

    def _generate_data(self) -> torch.Tensor:
        """Generate synthetic weather-like data."""
        h, w = self.img_size
        total_timesteps = self.num_samples + self.forecast_steps

        # Create coordinate grids
        lat = np.linspace(-90, 90, h)
        lon = np.linspace(0, 360, w)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Generate base patterns for each channel
        data = np.zeros((total_timesteps, self.in_channels, h, w), dtype=np.float32)

        for c in range(self.in_channels):
            # Different wave patterns for different variables
            freq_lat = 2 + c * 0.5
            freq_lon = 3 + c * 0.3
            phase = c * np.pi / 4

            for t in range(total_timesteps):
                # Base sinusoidal pattern
                pattern = (
                    np.sin(np.radians(lat_grid) * freq_lat + phase) *
                    np.cos(np.radians(lon_grid - t * 5) * freq_lon / 180 * np.pi)
                )

                # Add latitude-dependent amplitude (stronger in midlatitudes)
                amplitude = np.cos(np.radians(lat_grid))
                pattern = pattern * amplitude

                # Add temporal evolution
                pattern = pattern * (1 + 0.2 * np.sin(t * 0.1))

                # Add small-scale structure
                small_scale = np.random.randn(h, w) * 0.1
                # Smooth it
                from scipy.ndimage import gaussian_filter
                small_scale = gaussian_filter(small_scale, sigma=2)
                pattern = pattern + small_scale

                data[t, c] = pattern

        # Normalize each channel
        for c in range(self.in_channels):
            mean = data[:, c].mean()
            std = data[:, c].std() + 1e-6
            data[:, c] = (data[:, c] - mean) / std

        # Add noise if requested
        if self.add_noise:
            noise = np.random.randn(*data.shape).astype(np.float32) * self.noise_std
            data = data + noise

        return torch.from_numpy(data)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input-target pair.

        Returns:
            Tuple of (input, target) tensors
        """
        x = self.data[idx]
        y = self.data[idx + self.forecast_steps]

        # Handle different in/out channels
        if self.out_channels != self.in_channels:
            y = y[:self.out_channels]

        return x, y


class MiniERA5Dataset(Dataset):
    """
    Mini version of ERA5 dataset for quick experiments.

    Uses downsampled or subset ERA5 data for fast iteration.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        num_samples: int = 100,
        variables: List[str] = ["z_500", "t_850", "u_500", "v_500"],
        img_size: Tuple[int, int] = (32, 64),
        normalize: bool = True,
        seed: int = 42,
    ):
        """
        Initialize mini ERA5 dataset.

        Args:
            data_path: Path to ERA5 data (or None to generate synthetic)
            num_samples: Number of samples
            variables: Variables to include
            img_size: Target spatial size (will downsample if needed)
            normalize: Whether to normalize data
            seed: Random seed
        """
        self.num_samples = num_samples
        self.variables = variables
        self.img_size = img_size
        self.normalize = normalize

        np.random.seed(seed)

        if data_path is not None and os.path.exists(data_path):
            self.data = self._load_era5(data_path)
        else:
            # Generate synthetic ERA5-like data
            self.data = self._generate_synthetic()

    def _load_era5(self, data_path: str) -> torch.Tensor:
        """Load and downsample real ERA5 data."""
        try:
            import xarray as xr

            ds = xr.open_dataset(data_path)

            # Select variables
            data_vars = []
            for var in self.variables:
                if var in ds:
                    data_vars.append(ds[var].values)

            if not data_vars:
                raise ValueError(f"No matching variables found in {data_path}")

            # Stack and subsample
            data = np.stack(data_vars, axis=1)[:self.num_samples + 1]

            # Downsample spatially if needed
            if data.shape[-2:] != self.img_size:
                from scipy.ndimage import zoom
                zoom_factors = (1, 1, self.img_size[0] / data.shape[-2], self.img_size[1] / data.shape[-1])
                data = zoom(data, zoom_factors, order=1)

            # Normalize
            if self.normalize:
                for c in range(data.shape[1]):
                    mean = data[:, c].mean()
                    std = data[:, c].std() + 1e-6
                    data[:, c] = (data[:, c] - mean) / std

            return torch.from_numpy(data.astype(np.float32))

        except Exception as e:
            print(f"Failed to load ERA5 data: {e}. Using synthetic data.")
            return self._generate_synthetic()

    def _generate_synthetic(self) -> torch.Tensor:
        """Generate synthetic ERA5-like data."""
        mini_ds = MiniDataset(
            num_samples=self.num_samples + 1,
            in_channels=len(self.variables),
            img_size=self.img_size,
        )
        return mini_ds.data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.data[idx + 1]
        return x, y


def create_mini_era5(
    num_train: int = 100,
    num_val: int = 20,
    in_channels: int = 4,
    img_size: Tuple[int, int] = (32, 64),
    batch_size: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create mini ERA5-like train and validation data loaders.

    This is the primary function for quick experiments.

    Args:
        num_train: Number of training samples
        num_val: Number of validation samples
        in_channels: Number of variables/channels
        img_size: Spatial dimensions
        batch_size: Batch size for data loaders
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> train_loader, val_loader = create_mini_era5(num_train=100, num_val=20)
        >>> for x, y in train_loader:
        ...     print(x.shape, y.shape)
        ...     break
    """
    # Create datasets
    train_dataset = MiniDataset(
        num_samples=num_train,
        in_channels=in_channels,
        img_size=img_size,
        seed=seed,
    )

    val_dataset = MiniDataset(
        num_samples=num_val,
        in_channels=in_channels,
        img_size=img_size,
        seed=seed + 1000,  # Different seed for validation
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_synthetic_weather_data(
    num_samples: int = 100,
    variables: List[str] = ["temperature", "pressure", "u_wind", "v_wind"],
    img_size: Tuple[int, int] = (64, 128),
    include_surface: bool = True,
    include_pressure_levels: bool = True,
    pressure_levels: List[int] = [500, 700, 850],
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Create synthetic weather data with named variables.

    Args:
        num_samples: Number of time steps
        variables: Variable names
        img_size: Spatial dimensions
        include_surface: Include surface variables
        include_pressure_levels: Include pressure level variables
        pressure_levels: Pressure levels to include
        seed: Random seed

    Returns:
        Dictionary mapping variable names to tensors

    Example:
        >>> data = create_synthetic_weather_data(num_samples=100)
        >>> print(data["temperature"].shape)
    """
    np.random.seed(seed)

    h, w = img_size
    data = {}

    # Create coordinate grids
    lat = np.linspace(-90, 90, h)
    lon = np.linspace(0, 360, w)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Generate each variable
    for i, var in enumerate(variables):
        var_data = np.zeros((num_samples, h, w), dtype=np.float32)

        for t in range(num_samples):
            # Base pattern varies by variable type
            if "temperature" in var.lower() or "t_" in var.lower():
                # Temperature: latitude-dependent with seasonal variation
                base = 288 - 50 * np.abs(np.sin(np.radians(lat_grid)))
                seasonal = 10 * np.sin(2 * np.pi * t / 365)
                pattern = base + seasonal
            elif "pressure" in var.lower() or "z_" in var.lower():
                # Geopotential: smooth large-scale pattern
                pattern = 5500 + 500 * np.sin(np.radians(lat_grid * 2))
                pattern += 100 * np.cos(np.radians(lon_grid - t * 5))
            elif "u_wind" in var.lower() or "u_" in var.lower():
                # Zonal wind: jet stream pattern
                jet_lat = 45
                pattern = 30 * np.exp(-((lat_grid - jet_lat) / 10) ** 2)
                pattern -= 30 * np.exp(-((lat_grid + jet_lat) / 10) ** 2)
            elif "v_wind" in var.lower() or "v_" in var.lower():
                # Meridional wind: wave pattern
                pattern = 5 * np.sin(np.radians(lon_grid * 3 - t * 10))
                pattern *= np.cos(np.radians(lat_grid))
            else:
                # Generic variable
                pattern = np.sin(np.radians(lat_grid * (i + 1)))
                pattern *= np.cos(np.radians(lon_grid - t * 5))

            # Add small-scale noise
            noise = np.random.randn(h, w) * 0.05 * np.abs(pattern).max()
            var_data[t] = pattern + noise

        data[var] = torch.from_numpy(var_data)

    return data


def get_sample_data(
    dataset_type: str = "mini",
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get sample data for experiments.

    Convenience function for quickly getting train/val data.

    Args:
        dataset_type: Type of dataset ("mini", "synthetic", "flow_matching")
        **kwargs: Additional arguments for dataset creation

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> train_loader, val_loader = get_sample_data("mini", num_train=50)
    """
    if dataset_type == "mini":
        return create_mini_era5(**kwargs)

    elif dataset_type == "synthetic":
        data = create_synthetic_weather_data(**kwargs)
        # Convert to dataset format
        variables = list(data.keys())
        stacked = torch.stack([data[v] for v in variables], dim=1)

        # Split into train/val
        num_samples = stacked.shape[0]
        train_size = int(0.8 * num_samples)

        train_data = stacked[:train_size]
        val_data = stacked[train_size:]

        # Create simple datasets
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data) - 1

            def __getitem__(self, idx):
                return self.data[idx], self.data[idx + 1]

        batch_size = kwargs.get("batch_size", 4)

        train_loader = DataLoader(SimpleDataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SimpleDataset(val_data), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    elif dataset_type == "flow_matching":
        # Data format for flow matching: (x0, x1) pairs
        num_train = kwargs.get("num_train", 100)
        num_val = kwargs.get("num_val", 20)
        img_size = kwargs.get("img_size", (32, 64))
        in_channels = kwargs.get("in_channels", 4)
        batch_size = kwargs.get("batch_size", 4)

        train_ds = MiniDataset(num_train, in_channels, img_size=img_size)
        val_ds = MiniDataset(num_val, in_channels, img_size=img_size, seed=1000)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


class WeatherBenchMiniDataset(Dataset):
    """
    Mini dataset compatible with WeatherBench2 format.

    For use with real WeatherBench2 data when available.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        variables: List[str] = ["geopotential_500", "temperature_850"],
        years: List[int] = [2020],
        subsample_factor: int = 8,
        num_samples: Optional[int] = 100,
        normalize: bool = True,
    ):
        """
        Initialize WeatherBench mini dataset.

        Args:
            data_path: Path to WeatherBench2 data
            variables: Variables to include
            years: Years to use
            subsample_factor: Spatial subsampling factor
            num_samples: Maximum number of samples
            normalize: Whether to normalize data
        """
        self.variables = variables
        self.subsample_factor = subsample_factor
        self.normalize = normalize

        if data_path and os.path.exists(data_path):
            self.data = self._load_weatherbench(data_path, years, num_samples)
        else:
            # Generate synthetic data in WeatherBench format
            num_samples = num_samples or 100
            h = 721 // subsample_factor
            w = 1440 // subsample_factor
            self.data = MiniDataset(
                num_samples=num_samples + 1,
                in_channels=len(variables),
                img_size=(h, w),
            ).data

    def _load_weatherbench(
        self,
        data_path: str,
        years: List[int],
        num_samples: Optional[int],
    ) -> torch.Tensor:
        """Load WeatherBench2 data."""
        try:
            import xarray as xr

            # Load data
            ds = xr.open_zarr(data_path)

            # Select variables and years
            data_list = []
            for var in self.variables:
                if var in ds:
                    var_data = ds[var].sel(time=ds.time.dt.year.isin(years))
                    data_list.append(var_data.values)

            if not data_list:
                raise ValueError("No matching variables found")

            # Stack and process
            data = np.stack(data_list, axis=1)

            # Subsample spatially
            if self.subsample_factor > 1:
                data = data[:, :, ::self.subsample_factor, ::self.subsample_factor]

            # Limit samples
            if num_samples:
                data = data[:num_samples + 1]

            # Normalize
            if self.normalize:
                for c in range(data.shape[1]):
                    mean = data[:, c].mean()
                    std = data[:, c].std() + 1e-6
                    data[:, c] = (data[:, c] - mean) / std

            return torch.from_numpy(data.astype(np.float32))

        except Exception as e:
            print(f"Failed to load WeatherBench data: {e}")
            # Return synthetic data
            h = 721 // self.subsample_factor
            w = 1440 // self.subsample_factor
            return MiniDataset(
                num_samples=(num_samples or 100) + 1,
                in_channels=len(self.variables),
                img_size=(h, w),
            ).data

    def __len__(self) -> int:
        return len(self.data) - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.data[idx + 1]
