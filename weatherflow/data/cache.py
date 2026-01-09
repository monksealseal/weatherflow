"""
Persistent dataset cache manager for WeatherFlow.

Provides a centralized caching system that persists datasets across sessions,
eliminating the need to re-download data every time you log in.

Usage:
    from weatherflow.data.cache import DatasetCache, get_default_cache

    # Get the default cache (uses ~/.weatherflow/datasets/)
    cache = get_default_cache()

    # List cached datasets
    cache.list_datasets()

    # Get path for ERA5 data (creates directory if needed)
    era5_path = cache.get_dataset_path("era5")

    # Check what's cached
    cache.info()
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Default cache location
DEFAULT_CACHE_DIR = Path.home() / ".weatherflow" / "datasets"
CONFIG_FILE = Path.home() / ".weatherflow" / "config.json"
REGISTRY_FILE = "cache_registry.json"


@dataclass
class DatasetInfo:
    """Information about a cached dataset."""

    name: str
    path: str
    size_bytes: int
    created_at: str
    last_accessed: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def size_human(self) -> str:
        """Return human-readable size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


class DatasetCache:
    """
    Manages persistent dataset storage for WeatherFlow.

    Datasets are stored in a persistent directory (default: ~/.weatherflow/datasets/)
    and tracked via a registry file for easy management.

    Environment Variables:
        WEATHERFLOW_CACHE_DIR: Override the default cache directory
        WEATHERFLOW_NO_CACHE: Set to "1" to disable caching (use /tmp)

    Example:
        cache = DatasetCache()

        # Get path for ERA5 data
        era5_dir = cache.get_dataset_path("era5")

        # Register downloaded files
        cache.register_dataset("era5", {
            "years": [2020, 2021],
            "variables": ["temperature", "wind"],
            "levels": [500, 850]
        })

        # List all cached datasets
        for ds in cache.list_datasets():
            print(f"{ds.name}: {ds.size_human()}")
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Override cache directory. If None, uses:
                1. WEATHERFLOW_CACHE_DIR environment variable
                2. Config file setting
                3. Default (~/.weatherflow/datasets/)
        """
        self.cache_dir = Path(self._resolve_cache_dir(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.cache_dir / REGISTRY_FILE
        self._registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _resolve_cache_dir(self, override: Optional[str]) -> str:
        """Resolve the cache directory from various sources."""
        # Check if caching is disabled
        if os.environ.get("WEATHERFLOW_NO_CACHE") == "1":
            import tempfile
            return tempfile.mkdtemp(prefix="weatherflow_")

        # Priority 1: Explicit override
        if override:
            return override

        # Priority 2: Environment variable
        env_dir = os.environ.get("WEATHERFLOW_CACHE_DIR")
        if env_dir:
            return env_dir

        # Priority 3: Config file
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    if "cache_dir" in config:
                        return config["cache_dir"]
            except (json.JSONDecodeError, IOError):
                pass

        # Priority 4: Default
        return str(DEFAULT_CACHE_DIR)

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_registry(self) -> None:
        """Save the cache registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def get_dataset_path(self, dataset_name: str, create: bool = True) -> Path:
        """
        Get the path for a dataset, creating the directory if needed.

        Args:
            dataset_name: Name of the dataset (e.g., "era5", "cmip6")
            create: If True, create the directory if it doesn't exist

        Returns:
            Path to the dataset directory
        """
        path = self.cache_dir / dataset_name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def get_era5_path(
        self,
        years: Optional[Sequence[int]] = None,
        variables: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[int]] = None,
    ) -> Path:
        """
        Get a path for ERA5 data with optional configuration-based subdirectory.

        For simple use cases, returns ~/.weatherflow/datasets/era5/
        For specific configurations, creates a subdirectory based on the config hash.

        Args:
            years: Years of data (optional, for creating config-specific subdirs)
            variables: ERA5 variables (optional)
            levels: Pressure levels (optional)

        Returns:
            Path to the ERA5 dataset directory
        """
        base_path = self.get_dataset_path("era5")

        # If no specific config, return base path
        if not any([years, variables, levels]):
            return base_path

        # Create a config-specific subdirectory
        config_str = json.dumps(
            {
                "years": sorted(years) if years else None,
                "variables": sorted(variables) if variables else None,
                "levels": sorted(levels) if levels else None,
            },
            sort_keys=True,
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        config_path = base_path / f"config_{config_hash}"
        config_path.mkdir(parents=True, exist_ok=True)

        # Store config info
        config_info_path = config_path / "config.json"
        if not config_info_path.exists():
            with open(config_info_path, "w") as f:
                json.dump(
                    {
                        "years": list(years) if years else None,
                        "variables": list(variables) if variables else None,
                        "levels": list(levels) if levels else None,
                        "created_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

        return config_path

    def register_dataset(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a dataset in the cache registry.

        Args:
            name: Dataset name
            metadata: Additional metadata to store
        """
        path = self.get_dataset_path(name, create=False)
        size = self._get_directory_size(path) if path.exists() else 0

        now = datetime.now().isoformat()
        if name not in self._registry:
            self._registry[name] = {
                "created_at": now,
                "last_accessed": now,
                "metadata": metadata or {},
            }
        else:
            self._registry[name]["last_accessed"] = now
            if metadata:
                self._registry[name]["metadata"].update(metadata)

        self._registry[name]["size_bytes"] = size
        self._save_registry()

    def update_access_time(self, name: str) -> None:
        """Update the last accessed time for a dataset."""
        if name in self._registry:
            self._registry[name]["last_accessed"] = datetime.now().isoformat()
            self._save_registry()

    def list_datasets(self) -> List[DatasetInfo]:
        """
        List all cached datasets.

        Returns:
            List of DatasetInfo objects
        """
        datasets = []

        # Scan the cache directory for datasets
        if self.cache_dir.exists():
            for item in self.cache_dir.iterdir():
                if item.is_dir() and item.name != "__pycache__":
                    size = self._get_directory_size(item)
                    registry_info = self._registry.get(item.name, {})

                    datasets.append(
                        DatasetInfo(
                            name=item.name,
                            path=str(item),
                            size_bytes=size,
                            created_at=registry_info.get(
                                "created_at", self._get_dir_creation_time(item)
                            ),
                            last_accessed=registry_info.get(
                                "last_accessed", self._get_dir_creation_time(item)
                            ),
                            metadata=registry_info.get("metadata", {}),
                        )
                    )

        return sorted(datasets, key=lambda x: x.name)

    def info(self, detailed: bool = False) -> str:
        """
        Get information about the cache.

        Args:
            detailed: If True, show detailed information per dataset

        Returns:
            Formatted string with cache information
        """
        lines = [
            "WeatherFlow Dataset Cache",
            "=" * 40,
            f"Location: {self.cache_dir}",
        ]

        datasets = self.list_datasets()
        total_size = sum(ds.size_bytes for ds in datasets)

        lines.append(f"Total datasets: {len(datasets)}")
        lines.append(f"Total size: {self._format_size(total_size)}")
        lines.append("")

        if datasets:
            lines.append("Cached Datasets:")
            lines.append("-" * 40)
            for ds in datasets:
                lines.append(f"  {ds.name}: {ds.size_human()}")
                if detailed:
                    lines.append(f"    Path: {ds.path}")
                    lines.append(f"    Created: {ds.created_at}")
                    lines.append(f"    Last accessed: {ds.last_accessed}")
                    if ds.metadata:
                        lines.append(f"    Metadata: {ds.metadata}")
        else:
            lines.append("No datasets cached yet.")

        return "\n".join(lines)

    def clear(self, dataset_name: Optional[str] = None, confirm: bool = True) -> bool:
        """
        Clear cached data.

        Args:
            dataset_name: If provided, clear only this dataset. Otherwise clear all.
            confirm: If True, require confirmation (always True in code, for CLI safety)

        Returns:
            True if cleared, False if cancelled
        """
        if dataset_name:
            path = self.get_dataset_path(dataset_name, create=False)
            if path.exists():
                shutil.rmtree(path)
                if dataset_name in self._registry:
                    del self._registry[dataset_name]
                    self._save_registry()
                return True
            return False
        else:
            # Clear entire cache
            for item in self.cache_dir.iterdir():
                if item.is_dir() and item.name != "__pycache__":
                    shutil.rmtree(item)
                elif item.is_file() and item.name != REGISTRY_FILE:
                    item.unlink()
            self._registry = {}
            self._save_registry()
            return True

    def has_dataset(self, name: str) -> bool:
        """Check if a dataset exists in the cache."""
        path = self.get_dataset_path(name, create=False)
        return path.exists() and any(path.iterdir())

    def has_era5_year(self, year: int) -> bool:
        """Check if ERA5 data for a specific year is cached."""
        era5_path = self.get_dataset_path("era5", create=False)
        if not era5_path.exists():
            return False

        # Check in base directory and subdirectories
        for nc_file in era5_path.rglob(f"era5_{year}.nc"):
            if nc_file.exists():
                return True
        return False

    def get_cached_era5_years(self) -> List[int]:
        """Get list of years with cached ERA5 data."""
        era5_path = self.get_dataset_path("era5", create=False)
        if not era5_path.exists():
            return []

        years = set()
        for nc_file in era5_path.rglob("era5_*.nc"):
            try:
                year_str = nc_file.stem.replace("era5_", "")
                years.add(int(year_str))
            except ValueError:
                continue

        return sorted(years)

    @staticmethod
    def _get_directory_size(path: Path) -> int:
        """Calculate total size of a directory in bytes."""
        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except (PermissionError, OSError):
            pass
        return total

    @staticmethod
    def _get_dir_creation_time(path: Path) -> str:
        """Get directory creation/modification time."""
        try:
            stat = path.stat()
            # Use mtime as ctime is not reliable on all systems
            return datetime.fromtimestamp(stat.st_mtime).isoformat()
        except OSError:
            return datetime.now().isoformat()

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable size."""
        size = size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


# Global cache instance
_default_cache: Optional[DatasetCache] = None


def get_default_cache() -> DatasetCache:
    """
    Get the default cache instance (singleton).

    Returns:
        The global DatasetCache instance
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = DatasetCache()
    return _default_cache


def get_persistent_era5_path(
    years: Optional[Sequence[int]] = None,
    variables: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[int]] = None,
) -> str:
    """
    Convenience function to get a persistent ERA5 data path.

    This is the recommended way to get a cache path for ERA5 data.

    Args:
        years: Optional list of years
        variables: Optional list of ERA5 variables
        levels: Optional list of pressure levels

    Returns:
        String path to the ERA5 cache directory

    Example:
        # Simple usage - all ERA5 data in one directory
        path = get_persistent_era5_path()

        # Specific configuration
        path = get_persistent_era5_path(
            years=[2020, 2021],
            variables=["temperature"],
            levels=[500, 850]
        )
    """
    cache = get_default_cache()
    return str(cache.get_era5_path(years, variables, levels))


def configure_cache(cache_dir: str) -> None:
    """
    Configure the persistent cache directory.

    This saves the configuration to ~/.weatherflow/config.json

    Args:
        cache_dir: Path to use for the cache directory
    """
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    config["cache_dir"] = cache_dir

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    # Reset the global cache instance
    global _default_cache
    _default_cache = None


def print_cache_info() -> None:
    """Print cache information to stdout."""
    cache = get_default_cache()
    print(cache.info(detailed=True))
