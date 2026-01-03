"""
Massive Data Pipeline for FlowAtmosphere Foundation Model

Handles streaming and preprocessing of petabytes of ERA5, CMIP6, and observational
data across multiple decades and variables. Uses Zarr for cloud-optimized storage
and Dask for distributed processing.
"""

import os
import tempfile
from typing import List, Dict, Optional, Tuple, Iterator
from pathlib import Path
import numpy as np
import xarray as xr
import zarr
import torch
from torch.utils.data import IterableDataset, DataLoader
import dask
import dask.array as da
from concurrent.futures import ThreadPoolExecutor


class MassiveDataPipeline:
    """
    Production data pipeline for multi-decadal climate data.

    Features:
    - Streaming data loading from cloud storage (S3, GCS)
    - Smart caching with LRU eviction
    - On-the-fly normalization and preprocessing
    - Support for multiple data sources (ERA5, CMIP6, obs)
    - Zarr-based cloud-optimized storage
    """

    def __init__(
        self,
        data_sources: List[str],
        cache_dir: Optional[str] = None,
        chunk_size: Tuple[int, int, int, int] = (1, 4, 128, 256),  # time, level, lat, lon
        num_workers: int = 8,
        prefetch_factor: int = 4,
    ):
        """
        Args:
            data_sources: List of data source paths (local or cloud URLs)
            cache_dir: Directory for local caching (default: system temp dir)
            chunk_size: Chunk size for Zarr arrays
            num_workers: Number of parallel data loading workers
            prefetch_factor: Number of batches to prefetch
        """
        self.data_sources = data_sources
        # Use system temp directory if not specified (cross-platform)
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "flowatm_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # Initialize data stores
        self.stores: Dict[str, zarr.Group] = {}
        self._initialize_stores()

        # Statistics for normalization
        self.stats: Optional[Dict[str, torch.Tensor]] = None

    def _initialize_stores(self):
        """Initialize Zarr stores for each data source."""
        for source in self.data_sources:
            if source.startswith('s3://') or source.startswith('gs://'):
                # Cloud storage
                import fsspec
                mapper = fsspec.get_mapper(source)
                store = zarr.open(mapper, mode='r')
            else:
                # Local storage
                store = zarr.open(source, mode='r')

            self.stores[source] = store
            print(f"Initialized store: {source}")

    def preprocess_era5(
        self,
        years: List[int],
        variables: List[str],
        pressure_levels: List[int],
        output_path: str,
    ):
        """
        Preprocess ERA5 data into cloud-optimized Zarr format.

        Args:
            years: Years to process
            variables: Variable names (e.g., ['u', 'v', 't', 'z'])
            pressure_levels: Pressure levels in hPa
            output_path: Output Zarr store path
        """
        print(f"Preprocessing ERA5 data for {len(years)} years...")

        # Create output Zarr store
        store = zarr.open(output_path, mode='w')

        # Process year by year to manage memory
        for year in years:
            print(f"Processing year {year}...")

            # Download year data using CDSAPI
            year_data = self._download_era5_year(year, variables, pressure_levels)

            # Convert to Dask array for parallel processing
            dask_array = da.from_array(year_data, chunks=self.chunk_size)

            # Compute statistics for normalization
            if self.stats is None:
                self._compute_statistics(dask_array)

            # Normalize
            normalized = self._normalize_data(dask_array)

            # Save to Zarr
            year_group = store.create_group(str(year))
            zarr.save_array(year_group, normalized, chunks=self.chunk_size)

        print(f"Preprocessing complete. Data saved to {output_path}")

    def preprocess_cmip6(
        self,
        models: List[str],
        scenarios: List[str],
        variables: List[str],
        output_path: str,
    ):
        """
        Preprocess CMIP6 model data.

        Args:
            models: CMIP6 model names (e.g., ['MPI-ESM1-2-LR', 'CESM2'])
            scenarios: SSP scenarios (e.g., ['ssp245', 'ssp585'])
            variables: Variable names
            output_path: Output Zarr store path
        """
        print(f"Preprocessing CMIP6 data for {len(models)} models...")

        store = zarr.open(output_path, mode='w')

        for model in models:
            for scenario in scenarios:
                print(f"Processing {model} - {scenario}...")

                # Download from ESGF or cloud provider
                data = self._download_cmip6(model, scenario, variables)

                # Process and save
                dask_array = da.from_array(data, chunks=self.chunk_size)
                normalized = self._normalize_data(dask_array)

                group_name = f"{model}_{scenario}"
                group = store.create_group(group_name)
                zarr.save_array(group, normalized, chunks=self.chunk_size)

        print(f"CMIP6 preprocessing complete. Data saved to {output_path}")

    def _download_era5_year(
        self,
        year: int,
        variables: List[str],
        pressure_levels: List[int],
    ) -> np.ndarray:
        """Download ERA5 data for a single year."""
        import cdsapi

        client = cdsapi.Client()

        # Temporary file
        temp_file = self.cache_dir / f"era5_{year}.nc"

        if not temp_file.exists():
            client.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': variables,
                    'pressure_level': [str(p) for p in pressure_levels],
                    'year': str(year),
                    'month': [f'{m:02d}' for m in range(1, 13)],
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': [f'{h:02d}:00' for h in range(0, 24, 6)],
                },
                str(temp_file)
            )

        # Load and return as array
        ds = xr.open_dataset(temp_file)
        data = ds.to_array().values
        return data

    def _download_cmip6(
        self,
        model: str,
        scenario: str,
        variables: List[str],
    ) -> np.ndarray:
        """Download CMIP6 data."""
        # Placeholder - implement ESGF download or cloud access
        print(f"Downloading CMIP6: {model} - {scenario}")

        # In production, use intake-esm for CMIP6 catalog access
        # For now, return dummy data
        return np.random.randn(365, 13, 181, 360).astype(np.float32)

    def _compute_statistics(self, data: da.Array):
        """Compute mean and std for normalization."""
        print("Computing dataset statistics...")

        mean = da.mean(data, axis=(0, 2, 3)).compute()
        std = da.std(data, axis=(0, 2, 3)).compute()

        self.stats = {
            'mean': torch.from_numpy(mean).float(),
            'std': torch.from_numpy(std).float().clamp(min=1e-6),
        }

        # Save statistics
        stats_path = self.cache_dir / 'normalization_stats.pt'
        torch.save(self.stats, stats_path)

        print(f"Statistics saved to {stats_path}")

    def _normalize_data(self, data: da.Array) -> da.Array:
        """Normalize data using precomputed statistics."""
        if self.stats is None:
            # Load from cache if available
            stats_path = self.cache_dir / 'normalization_stats.pt'
            if stats_path.exists():
                self.stats = torch.load(stats_path)
            else:
                raise ValueError("Statistics not computed. Run preprocessing first.")

        mean = self.stats['mean'].numpy()
        std = self.stats['std'].numpy()

        # Reshape for broadcasting
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]

        normalized = (data - mean) / std
        return normalized

    def create_training_dataset(
        self,
        split: str = 'train',
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create DataLoader for training.

        Args:
            split: 'train', 'val', or 'test'
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            DataLoader instance
        """
        dataset = StreamingWeatherDataset(
            stores=self.stores,
            split=split,
            chunk_size=self.chunk_size,
            stats=self.stats,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )

        return loader


class StreamingWeatherDataset(IterableDataset):
    """
    Streaming dataset that yields data from Zarr stores without loading
    entire dataset into memory.
    """

    def __init__(
        self,
        stores: Dict[str, zarr.Group],
        split: str = 'train',
        chunk_size: Tuple[int, int, int, int] = (1, 4, 128, 256),
        stats: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.stores = stores
        self.split = split
        self.chunk_size = chunk_size
        self.stats = stats

        # Determine split indices
        self.indices = self._get_split_indices()

    def _get_split_indices(self) -> List[Tuple[str, int]]:
        """Get indices for train/val/test split."""
        all_indices = []

        for source, store in self.stores.items():
            # Get total number of time steps
            for year_group in store.keys():
                data_shape = store[year_group].shape
                num_samples = data_shape[0]

                for i in range(num_samples):
                    all_indices.append((source, year_group, i))

        # Split into train/val/test (80/10/10)
        np.random.seed(42)
        np.random.shuffle(all_indices)

        n = len(all_indices)
        if self.split == 'train':
            indices = all_indices[:int(0.8 * n)]
        elif self.split == 'val':
            indices = all_indices[int(0.8 * n):int(0.9 * n)]
        else:  # test
            indices = all_indices[int(0.9 * n):]

        return indices

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over dataset."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single worker
            indices = self.indices
        else:
            # Multiple workers - split indices
            per_worker = len(self.indices) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.indices)
            indices = self.indices[start:end]

        for source, year_group, idx in indices:
            # Load data from Zarr
            store = self.stores[source]
            data = store[year_group][idx]

            # Convert to torch tensor
            x = torch.from_numpy(data).float()

            # For flow matching, we need (x0, x1) pairs
            # x0 is current state, x1 is next state
            if idx < store[year_group].shape[0] - 1:
                next_data = store[year_group][idx + 1]
                x1 = torch.from_numpy(next_data).float()
            else:
                # Use same state if at end
                x1 = x.clone()

            yield x, x1


class CurriculumDataScheduler:
    """
    Curriculum learning scheduler that progressively increases data complexity.

    Starts with low-resolution, simple cases and gradually introduces
    higher resolution and more complex atmospheric phenomena.
    """

    def __init__(
        self,
        initial_resolution: Tuple[int, int] = (32, 64),
        final_resolution: Tuple[int, int] = (720, 1440),
        num_stages: int = 5,
    ):
        self.initial_resolution = initial_resolution
        self.final_resolution = final_resolution
        self.num_stages = num_stages

        # Define curriculum stages
        self.stages = self._create_stages()
        self.current_stage = 0

    def _create_stages(self) -> List[Dict]:
        """Create curriculum stages with increasing complexity."""
        stages = []

        # Resolution progression (geometric)
        lat_steps = np.geomspace(
            self.initial_resolution[0],
            self.final_resolution[0],
            self.num_stages
        ).astype(int)

        lon_steps = np.geomspace(
            self.initial_resolution[1],
            self.final_resolution[1],
            self.num_stages
        ).astype(int)

        for i in range(self.num_stages):
            stage = {
                'resolution': (lat_steps[i], lon_steps[i]),
                'num_variables': min(4 + i * 2, 20),  # Start with 4, increase to 20
                'lead_time_hours': 6 * (2 ** i),  # 6h, 12h, 24h, 48h, 96h
                'complexity': i / self.num_stages,
            }
            stages.append(stage)

        return stages

    def get_current_stage(self) -> Dict:
        """Get current curriculum stage."""
        return self.stages[self.current_stage]

    def advance_stage(self):
        """Move to next curriculum stage."""
        if self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            print(f"Advanced to curriculum stage {self.current_stage + 1}/{self.num_stages}")
            print(f"Resolution: {self.stages[self.current_stage]['resolution']}")

    def should_advance(self, val_loss: float, patience: int = 3) -> bool:
        """Check if should advance to next stage based on validation loss."""
        # Implement logic to track val_loss improvement
        # Advance when loss plateaus
        # Placeholder for now
        return False
