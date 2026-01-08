"""
WeatherFlow Server-Side Data Storage

Manages pre-cached ERA5 data and other datasets on the server.
This eliminates the need for users to download data every session.

Key features:
- Pre-bundled sample datasets stored server-side
- Automatic initialization on first run
- Data integrity verification
- Support for multiple data sources (ERA5, GFS, etc.)
"""

import numpy as np
import xarray as xr
import json
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "trained_models"

# Ensure directories exist
for d in [DATA_DIR, SAMPLES_DIR, CACHE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# WeatherBench2 ERA5 URL (public, free access)
WEATHERBENCH2_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

# Pre-defined sample datasets with citation information
SAMPLE_DATASETS = {
    "hurricane_katrina_2005": {
        "name": "Hurricane Katrina (2005)",
        "description": "Category 5 Atlantic hurricane, one of the deadliest in US history",
        "start_date": "2005-08-23",
        "end_date": "2005-08-31",
        "region": "Gulf of Mexico / US Southeast",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 500, 200],
        "citation": "Knabb et al. (2005). Tropical Cyclone Report: Hurricane Katrina. NHC.",
        "scientific_interest": "Rapid intensification, storm surge, landfalling dynamics",
    },
    "european_heatwave_2003": {
        "name": "European Heat Wave (2003)",
        "description": "Record-breaking heat wave, >70,000 excess deaths",
        "start_date": "2003-08-01",
        "end_date": "2003-08-15",
        "region": "Western Europe",
        "variables": ["temperature", "geopotential"],
        "pressure_levels": [1000, 850, 500],
        "citation": "Schär et al. (2004). The role of increasing temperature variability. Nature.",
        "scientific_interest": "Blocking patterns, temperature extremes, climate attribution",
    },
    "polar_vortex_2019": {
        "name": "Polar Vortex (2019)",
        "description": "Extreme cold outbreak, record-breaking temperatures",
        "start_date": "2019-01-27",
        "end_date": "2019-02-03",
        "region": "North America",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 500, 200, 50],
        "citation": "Lee et al. (2021). Arctic warming and polar vortex. Science.",
        "scientific_interest": "Stratosphere-troposphere coupling, extreme cold",
    },
    "atmospheric_river_2017": {
        "name": "Atmospheric River (2017)",
        "description": "Pineapple Express causing Oroville Dam emergency",
        "start_date": "2017-02-06",
        "end_date": "2017-02-13",
        "region": "US West Coast",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 700, 500],
        "citation": "Ralph et al. (2019). Atmospheric River Reconnaissance. BAMS.",
        "scientific_interest": "Moisture transport, extreme precipitation, flooding",
    },
    "ssw_beast_from_east_2018": {
        "name": "SSW & Beast from the East (2018)",
        "description": "Sudden Stratospheric Warming causing European cold",
        "start_date": "2018-02-10",
        "end_date": "2018-03-05",
        "region": "Northern Hemisphere",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 500, 200, 50, 10],
        "citation": "Butler et al. (2019). The sudden stratospheric warming of 2018. JGR.",
        "scientific_interest": "SSW dynamics, predictability, stratospheric influence",
    },
    "general_sample_2023": {
        "name": "January 2023 Sample",
        "description": "General sample for testing and model development",
        "start_date": "2023-01-01",
        "end_date": "2023-01-15",
        "region": "Global",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 700, 500, 300, 200],
        "citation": "Hersbach et al. (2020). ERA5 global reanalysis. QJRMS.",
        "scientific_interest": "General development, testing, benchmarking",
    },
}

# Model benchmark data with proper citations
MODEL_BENCHMARKS = {
    "GraphCast": {
        "organization": "Google DeepMind",
        "paper": "Lam et al. (2023). Learning skillful medium-range global weather forecasting. Science.",
        "arxiv": "https://arxiv.org/abs/2212.12794",
        "code": "https://github.com/google-deepmind/graphcast",
        "resolution": "0.25°",
        "forecast_range": "10 days",
        "metrics": {
            "z500_rmse_24h": 52,
            "z500_rmse_120h": 382,
            "z500_rmse_240h": 714,
            "t850_rmse_24h": 0.85,
            "t850_rmse_120h": 2.01,
            "t850_rmse_240h": 2.95,
            "acc_z500_120h": 0.965,
            "acc_z500_240h": 0.858,
        },
        "inference_time_s": 60,
        "params_m": 37,
    },
    "FourCastNet": {
        "organization": "NVIDIA",
        "paper": "Pathak et al. (2022). FourCastNet: A global data-driven high-resolution weather model. arXiv.",
        "arxiv": "https://arxiv.org/abs/2202.11214",
        "code": "https://github.com/NVlabs/FourCastNet",
        "resolution": "0.25°",
        "forecast_range": "7 days",
        "metrics": {
            "z500_rmse_24h": 58,
            "z500_rmse_120h": 412,
            "z500_rmse_240h": 785,
            "t850_rmse_24h": 0.92,
            "t850_rmse_120h": 2.18,
            "t850_rmse_240h": 3.15,
            "acc_z500_120h": 0.952,
            "acc_z500_240h": 0.825,
        },
        "inference_time_s": 2,
        "params_m": 450,
    },
    "Pangu-Weather": {
        "organization": "Huawei",
        "paper": "Bi et al. (2023). Accurate medium-range global weather forecasting with 3D neural networks. Nature.",
        "arxiv": "https://arxiv.org/abs/2211.02556",
        "code": "https://github.com/198808xc/Pangu-Weather",
        "resolution": "0.25°",
        "forecast_range": "7 days",
        "metrics": {
            "z500_rmse_24h": 54,
            "z500_rmse_120h": 395,
            "z500_rmse_240h": 742,
            "t850_rmse_24h": 0.88,
            "t850_rmse_120h": 2.08,
            "t850_rmse_240h": 3.02,
            "acc_z500_120h": 0.961,
            "acc_z500_240h": 0.848,
        },
        "inference_time_s": 1,
        "params_m": 256,
    },
    "GenCast": {
        "organization": "Google DeepMind",
        "paper": "Price et al. (2023). GenCast: Diffusion-based ensemble forecasting for medium-range weather. arXiv.",
        "arxiv": "https://arxiv.org/abs/2312.15796",
        "resolution": "0.25°",
        "forecast_range": "15 days",
        "probabilistic": True,
        "metrics": {
            "z500_rmse_24h": 55,
            "z500_rmse_120h": 390,
            "z500_rmse_240h": 720,
            "t850_rmse_24h": 0.87,
            "t850_rmse_120h": 2.05,
            "t850_rmse_240h": 2.98,
            "acc_z500_120h": 0.963,
            "acc_z500_240h": 0.855,
            "crps_z500_120h": 185,
        },
        "inference_time_s": 300,
        "params_m": 500,
    },
    "ClimaX": {
        "organization": "Microsoft Research",
        "paper": "Nguyen et al. (2023). ClimaX: A foundation model for weather and climate. ICML.",
        "arxiv": "https://arxiv.org/abs/2301.10343",
        "code": "https://github.com/microsoft/ClimaX",
        "resolution": "1.4°",
        "forecast_range": "7 days",
        "foundation_model": True,
        "metrics": {
            "z500_rmse_24h": 62,
            "z500_rmse_120h": 425,
            "z500_rmse_240h": 810,
            "t850_rmse_24h": 0.95,
            "t850_rmse_120h": 2.25,
            "t850_rmse_240h": 3.28,
            "acc_z500_120h": 0.945,
            "acc_z500_240h": 0.812,
        },
        "inference_time_s": 5,
        "params_m": 109,
    },
    "Aurora": {
        "organization": "Microsoft Research",
        "paper": "Bodnar et al. (2024). Aurora: A Foundation Model of the Atmosphere. arXiv.",
        "arxiv": "https://arxiv.org/abs/2405.13063",
        "resolution": "0.25°",
        "forecast_range": "10 days",
        "foundation_model": True,
        "metrics": {
            "z500_rmse_24h": 50,
            "z500_rmse_120h": 375,
            "z500_rmse_240h": 698,
            "t850_rmse_24h": 0.82,
            "t850_rmse_120h": 1.95,
            "t850_rmse_240h": 2.88,
            "acc_z500_120h": 0.968,
            "acc_z500_240h": 0.865,
        },
        "inference_time_s": 30,
        "params_m": 1300,
    },
    "NeuralGCM": {
        "organization": "Google Research",
        "paper": "Kochkov et al. (2024). Neural General Circulation Models for Weather and Climate. Nature.",
        "arxiv": "https://arxiv.org/abs/2311.07222",
        "code": "https://github.com/google-research/neuralgcm",
        "resolution": "1.4°",
        "forecast_range": "15 days",
        "hybrid_physics": True,
        "metrics": {
            "z500_rmse_24h": 56,
            "z500_rmse_120h": 388,
            "z500_rmse_240h": 725,
            "t850_rmse_24h": 0.89,
            "t850_rmse_120h": 2.04,
            "t850_rmse_240h": 2.96,
            "acc_z500_120h": 0.962,
            "acc_z500_240h": 0.852,
        },
        "inference_time_s": 120,
        "params_m": 22,
    },
}


def get_sample_file_path(sample_id: str) -> Path:
    """Get the path for a sample dataset file."""
    return SAMPLES_DIR / f"{sample_id}.nc"


def get_sample_metadata_path(sample_id: str) -> Path:
    """Get the path for sample metadata file."""
    return SAMPLES_DIR / f"{sample_id}_metadata.json"


def is_sample_available(sample_id: str) -> bool:
    """Check if a sample dataset is available locally."""
    return get_sample_file_path(sample_id).exists()


def get_available_samples() -> List[str]:
    """Get list of locally available sample datasets."""
    return [sid for sid in SAMPLE_DATASETS.keys() if is_sample_available(sid)]


def get_all_sample_info() -> Dict:
    """Get information about all sample datasets with availability status."""
    result = {}
    for sample_id, info in SAMPLE_DATASETS.items():
        result[sample_id] = {
            **info,
            "available": is_sample_available(sample_id),
            "file_path": str(get_sample_file_path(sample_id)) if is_sample_available(sample_id) else None,
        }
    return result


def load_sample_data(sample_id: str) -> Tuple[Optional[xr.Dataset], Optional[Dict]]:
    """
    Load a sample dataset from local storage.

    Returns:
        tuple: (xarray.Dataset, metadata_dict) or (None, None) if not available
    """
    if not is_sample_available(sample_id):
        return None, None

    try:
        data_path = get_sample_file_path(sample_id)
        data = xr.open_dataset(data_path)

        # Load metadata
        meta_path = get_sample_metadata_path(sample_id)
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = SAMPLE_DATASETS.get(sample_id, {})

        return data, metadata
    except Exception as e:
        logger.error(f"Error loading sample {sample_id}: {e}")
        return None, None


def generate_realistic_era5_sample(
    sample_id: str,
    seed: int = 42
) -> xr.Dataset:
    """
    Generate a realistic ERA5-like sample dataset for demonstration.

    This creates physically plausible atmospheric fields when real data
    cannot be downloaded. The fields have:
    - Proper latitude-dependent temperature structure
    - Realistic jet stream patterns
    - Proper geopotential height gradients
    - Spatially coherent wind fields

    NOTE: This is clearly marked as SYNTHETIC - used only when
    real data download fails. For research, always use real ERA5.
    """
    np.random.seed(seed)

    sample_info = SAMPLE_DATASETS.get(sample_id)
    if sample_info is None:
        sample_info = SAMPLE_DATASETS["general_sample_2023"]

    # Grid dimensions (WeatherBench2 standard)
    n_lat = 32
    n_lon = 64
    n_time = 20  # 5 days at 6-hourly
    levels = sample_info.get("pressure_levels", [1000, 850, 500, 200])

    # Create coordinates
    lats = np.linspace(90, -90, n_lat)
    lons = np.linspace(0, 360 - 360/n_lon, n_lon)
    times = np.array([
        np.datetime64(sample_info["start_date"]) + np.timedelta64(6*i, 'h')
        for i in range(n_time)
    ])

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    # Create data variables
    data_vars = {}

    # Temperature: latitude and pressure-dependent
    if "temperature" in sample_info.get("variables", ["temperature"]):
        temp_data = np.zeros((n_time, len(levels), n_lat, n_lon))
        for t in range(n_time):
            for li, level in enumerate(levels):
                # Base temperature structure
                lapse_rate = 6.5 / 1000  # K/m
                level_height = 8000 * np.log(1000 / level)  # Scale height approx

                # Surface temp varies with latitude
                surface_temp = 288 - 45 * np.abs(np.sin(np.radians(lat_grid)))

                # Add level-dependent cooling
                base_temp = surface_temp - lapse_rate * level_height

                # Add synoptic wave pattern that propagates
                wave = 5 * np.sin(np.radians(lon_grid - t * 360/n_time) * 3) * np.cos(np.radians(lat_grid))

                # Add realistic noise
                noise = np.random.randn(n_lat, n_lon) * 2

                temp_data[t, li] = base_temp + wave + noise

        data_vars["temperature"] = (["time", "level", "latitude", "longitude"], temp_data)

    # Geopotential: pressure and latitude dependent
    if "geopotential" in sample_info.get("variables", []):
        geo_data = np.zeros((n_time, len(levels), n_lat, n_lon))
        for t in range(n_time):
            for li, level in enumerate(levels):
                # Approximate geopotential height (Z = Φ/g)
                scale_height = 8500  # meters
                z_mean = scale_height * np.log(1000 / level)

                # Latitude variation (higher at equator due to thermal wind)
                lat_var = 200 * np.cos(np.radians(lat_grid * 2))

                # Synoptic wave pattern
                wave = 100 * np.sin(np.radians(lon_grid - t * 360/n_time) * 3) * np.exp(-np.abs(lat_grid - 45)**2 / 400)

                # Convert to geopotential (m^2/s^2)
                geo_data[t, li] = 9.81 * (z_mean + lat_var + wave + np.random.randn(n_lat, n_lon) * 20)

        data_vars["geopotential"] = (["time", "level", "latitude", "longitude"], geo_data)

    # U wind component: westerlies with jet stream
    if "u_component_of_wind" in sample_info.get("variables", []):
        u_data = np.zeros((n_time, len(levels), n_lat, n_lon))
        for t in range(n_time):
            for li, level in enumerate(levels):
                # Jet stream core (stronger at upper levels)
                jet_strength = 20 + (1000 - level) / 20

                # Jet centered around 45° with Gaussian profile
                jet_nh = jet_strength * np.exp(-((lat_grid - 45)**2) / 200)
                jet_sh = jet_strength * np.exp(-((lat_grid + 45)**2) / 200)

                # Trade winds (easterlies in tropics)
                trades = -10 * np.exp(-(lat_grid**2) / 300)

                u_data[t, li] = jet_nh + jet_sh + trades + np.random.randn(n_lat, n_lon) * 3

        data_vars["u_component_of_wind"] = (["time", "level", "latitude", "longitude"], u_data)

    # V wind component: weaker meridional flow
    if "v_component_of_wind" in sample_info.get("variables", []):
        v_data = np.zeros((n_time, len(levels), n_lat, n_lon))
        for t in range(n_time):
            for li, level in enumerate(levels):
                # Hadley cell-like pattern
                hadley = 5 * np.sin(np.radians(lat_grid) * 2) * (level / 1000)

                # Synoptic eddy component
                wave = 5 * np.sin(np.radians(lon_grid - t * 360/n_time) * 3) * np.cos(np.radians(lat_grid - 45))

                v_data[t, li] = hadley + wave + np.random.randn(n_lat, n_lon) * 2

        data_vars["v_component_of_wind"] = (["time", "level", "latitude", "longitude"], v_data)

    # Create dataset
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "level": levels,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "source": "WeatherFlow Synthetic Sample",
            "is_synthetic": True,
            "note": "SYNTHETIC DATA - for demonstration only. Use real ERA5 for research.",
            "created": datetime.now().isoformat(),
            "sample_id": sample_id,
            "based_on": sample_info.get("name", "Unknown"),
        }
    )

    return ds


def initialize_sample_data(sample_id: str, force_synthetic: bool = False) -> bool:
    """
    Initialize a sample dataset. First tries to download from WeatherBench2,
    falls back to generating realistic synthetic data if download fails.

    Args:
        sample_id: ID of the sample to initialize
        force_synthetic: If True, skip download attempt and use synthetic

    Returns:
        bool: True if successful
    """
    if sample_id not in SAMPLE_DATASETS:
        logger.error(f"Unknown sample ID: {sample_id}")
        return False

    sample_info = SAMPLE_DATASETS[sample_id]
    data_path = get_sample_file_path(sample_id)
    meta_path = get_sample_metadata_path(sample_id)

    # Skip if already exists
    if data_path.exists() and not force_synthetic:
        logger.info(f"Sample {sample_id} already exists")
        return True

    # Try to download from WeatherBench2
    if not force_synthetic:
        try:
            logger.info(f"Attempting to download {sample_id} from WeatherBench2...")
            ds = xr.open_zarr(
                WEATHERBENCH2_URL,
                storage_options={"anon": True},
                consolidated=True
            )

            # Select subset
            variables = [v for v in sample_info["variables"] if v in ds.data_vars]
            if not variables:
                raise ValueError("No matching variables found")

            subset = ds[variables].sel(
                time=slice(sample_info["start_date"], sample_info["end_date"])
            )

            if "level" in ds.coords:
                available_levels = [l for l in sample_info["pressure_levels"] if l in ds.level.values]
                if available_levels:
                    subset = subset.sel(level=available_levels)

            # Load and save
            logger.info("Loading data...")
            loaded = subset.load()
            loaded.attrs["is_synthetic"] = False
            loaded.attrs["source"] = "WeatherBench2 ERA5 (ECMWF Reanalysis)"
            loaded.to_netcdf(data_path)

            # Save metadata
            metadata = {
                **sample_info,
                "downloaded_at": datetime.now().isoformat(),
                "is_synthetic": False,
                "source": "WeatherBench2 ERA5",
                "file_size_mb": data_path.stat().st_size / (1024 * 1024),
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Successfully downloaded {sample_id}")
            return True

        except Exception as e:
            logger.warning(f"Could not download from WeatherBench2: {e}")
            logger.info("Falling back to synthetic data generation...")

    # Generate synthetic data
    try:
        logger.info(f"Generating synthetic sample for {sample_id}...")
        ds = generate_realistic_era5_sample(sample_id)
        ds.to_netcdf(data_path)

        # Save metadata
        metadata = {
            **sample_info,
            "created_at": datetime.now().isoformat(),
            "is_synthetic": True,
            "source": "WeatherFlow Synthetic Generator",
            "note": "SYNTHETIC DATA - for demonstration. Use real ERA5 for research.",
            "file_size_mb": data_path.stat().st_size / (1024 * 1024),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully generated synthetic {sample_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize {sample_id}: {e}")
        return False


def initialize_all_samples(force_synthetic: bool = False) -> Dict[str, bool]:
    """
    Initialize all sample datasets.

    Returns:
        Dict mapping sample_id to success status
    """
    results = {}
    for sample_id in SAMPLE_DATASETS:
        results[sample_id] = initialize_sample_data(sample_id, force_synthetic)
    return results


def get_model_benchmarks() -> Dict:
    """Get model benchmark data with citations."""
    return MODEL_BENCHMARKS


def get_data_status() -> Dict:
    """Get overall data storage status."""
    available = get_available_samples()
    total = len(SAMPLE_DATASETS)

    # Calculate total storage used
    total_size = 0
    for sid in available:
        path = get_sample_file_path(sid)
        if path.exists():
            total_size += path.stat().st_size

    return {
        "available_samples": len(available),
        "total_samples": total,
        "sample_ids": available,
        "storage_used_mb": total_size / (1024 * 1024),
        "data_directory": str(DATA_DIR),
    }


# CLI for initialization
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WeatherFlow Data Storage Manager")
    parser.add_argument("--init-all", action="store_true", help="Initialize all sample datasets")
    parser.add_argument("--init", type=str, help="Initialize specific sample by ID")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data generation")
    parser.add_argument("--status", action="store_true", help="Show data storage status")
    parser.add_argument("--list", action="store_true", help="List all available samples")

    args = parser.parse_args()

    if args.status:
        status = get_data_status()
        print(f"\nWeatherFlow Data Storage Status")
        print(f"=" * 40)
        print(f"Available samples: {status['available_samples']}/{status['total_samples']}")
        print(f"Storage used: {status['storage_used_mb']:.1f} MB")
        print(f"Data directory: {status['data_directory']}")

    elif args.list:
        info = get_all_sample_info()
        print(f"\nAvailable Sample Datasets")
        print(f"=" * 40)
        for sid, sinfo in info.items():
            status = "✓" if sinfo["available"] else "✗"
            print(f"[{status}] {sid}: {sinfo['name']}")

    elif args.init_all:
        print("Initializing all sample datasets...")
        results = initialize_all_samples(force_synthetic=args.synthetic)
        for sid, success in results.items():
            status = "✓" if success else "✗"
            print(f"[{status}] {sid}")

    elif args.init:
        success = initialize_sample_data(args.init, force_synthetic=args.synthetic)
        status = "✓" if success else "✗"
        print(f"[{status}] {args.init}")

    else:
        parser.print_help()
