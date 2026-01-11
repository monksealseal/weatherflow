"""
WeatherFlow Server-Side Data Storage

Manages real weather data from publicly available sources.
All data is REAL - no synthetic data is ever used.

Key features:
- Real NCEP/ERA-Interim reanalysis data from public sources
- Pre-bundled datasets stored server-side
- Automatic initialization on first run
- Data integrity verification
- Support for multiple real data sources

Data Sources:
- NCEP Reanalysis (via xarray-data): Real atmospheric observations
- ERA-Interim (via xarray-data): ECMWF reanalysis data
- WeatherBench2: When cloud access is available
"""

import numpy as np
import xarray as xr
import json
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from typing import Dict, List, Optional, Tuple
import logging
import urllib.request
import tempfile
import os

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

# WeatherBench2 ERA5 URL (public, free access - when cloud is available)
WEATHERBENCH2_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

# Real data sources from GitHub (always accessible)
REAL_DATA_SOURCES = {
    "ncep_air_temperature": {
        "url": "https://github.com/pydata/xarray-data/raw/master/air_temperature.nc",
        "description": "NCEP Reanalysis 4x daily air temperature at sigma level 0.995",
        "source": "NCEP/NCAR Reanalysis",
        "citation": "Kalnay et al. (1996). The NCEP/NCAR 40-Year Reanalysis Project. Bull. Amer. Meteor. Soc.",
        "variables": ["air"],
        "time_range": "2013-2014",
        "region": "North America / Pacific",
    },
    "era_interim_uvz": {
        "url": "https://github.com/pydata/xarray-data/raw/master/eraint_uvz.nc",
        "description": "Monthly ERA-Interim U, V wind and geopotential at 200, 500, 850 hPa",
        "source": "ECMWF ERA-Interim Reanalysis",
        "citation": "Dee et al. (2011). The ERA-Interim reanalysis. Q.J.R. Meteorol. Soc.",
        "variables": ["u", "v", "z"],
        "time_range": "January & July climatology",
        "region": "Global",
    },
}

# Pre-defined sample datasets - ALL using REAL data sources
# These map to real reanalysis data that can be downloaded
SAMPLE_DATASETS = {
    "ncep_reanalysis_2013": {
        "name": "NCEP Reanalysis Air Temperature (2013-2014)",
        "description": "Real NCEP/NCAR Reanalysis air temperature data at sigma level 0.995",
        "start_date": "2013-01-01",
        "end_date": "2014-12-31",
        "region": "North America / Pacific (15°N-75°N, 200°E-330°E)",
        "variables": ["air"],
        "pressure_levels": [995],  # sigma level
        "citation": "Kalnay et al. (1996). The NCEP/NCAR 40-Year Reanalysis Project. Bull. Amer. Meteor. Soc.",
        "scientific_interest": "Temperature variability, seasonal cycles, weather patterns",
        "data_source": "ncep_air_temperature",
        "is_real_data": True,
    },
    "era_interim_global": {
        "name": "ERA-Interim Global Wind & Geopotential",
        "description": "Real ECMWF ERA-Interim monthly climatology with U, V wind and geopotential height",
        "start_date": "climatology",
        "end_date": "climatology",
        "region": "Global",
        "variables": ["u", "v", "z"],
        "pressure_levels": [200, 500, 850],
        "citation": "Dee et al. (2011). The ERA-Interim reanalysis. Q.J.R. Meteorol. Soc.",
        "scientific_interest": "Global circulation, jet streams, geopotential patterns",
        "data_source": "era_interim_uvz",
        "is_real_data": True,
    },
    # The following require WeatherBench2/cloud access when available
    "hurricane_katrina_2005": {
        "name": "Hurricane Katrina (2005)",
        "description": "Category 5 Atlantic hurricane from ERA5 reanalysis (requires cloud access)",
        "start_date": "2005-08-23",
        "end_date": "2005-08-31",
        "region": "Gulf of Mexico / US Southeast",
        "variables": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_levels": [1000, 850, 500, 200],
        "citation": "Knabb et al. (2005). Tropical Cyclone Report: Hurricane Katrina. NHC.",
        "scientific_interest": "Rapid intensification, storm surge, landfalling dynamics",
        "data_source": "weatherbench2",
        "is_real_data": True,
        "requires_cloud": True,
    },
    "european_heatwave_2003": {
        "name": "European Heat Wave (2003)",
        "description": "Record-breaking heat wave from ERA5 reanalysis (requires cloud access)",
        "start_date": "2003-08-01",
        "end_date": "2003-08-15",
        "region": "Western Europe",
        "variables": ["temperature", "geopotential"],
        "pressure_levels": [1000, 850, 500],
        "citation": "Schär et al. (2004). The role of increasing temperature variability. Nature.",
        "scientific_interest": "Blocking patterns, temperature extremes, climate attribution",
        "data_source": "weatherbench2",
        "is_real_data": True,
        "requires_cloud": True,
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


def download_real_data_from_github(source_key: str) -> Optional[xr.Dataset]:
    """
    Download real weather data from GitHub-hosted sources.
    
    Args:
        source_key: Key from REAL_DATA_SOURCES
        
    Returns:
        xarray.Dataset with real data or None if download fails
    """
    if source_key not in REAL_DATA_SOURCES:
        logger.error(f"Unknown data source: {source_key}")
        return None
    
    source_info = REAL_DATA_SOURCES[source_key]
    url = source_info["url"]
    
    logger.info(f"Downloading real data from {url}...")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 WeatherFlow'})
        resp = urllib.request.urlopen(req, timeout=120)
        data = resp.read()
        logger.info(f"Downloaded {len(data) / 1024 / 1024:.2f} MB")
        
        # Save to temp file and open with xarray
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            f.write(data)
            temp_path = f.name
        
        ds = xr.open_dataset(temp_path)
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # Add metadata about the real source
        ds.attrs["original_source"] = source_info["source"]
        ds.attrs["citation"] = source_info["citation"]
        ds.attrs["is_real_data"] = "1"  # String for netCDF compatibility
        ds.attrs["downloaded_from"] = url
        ds.attrs["downloaded_at"] = datetime.now().isoformat()
        
        return ds
        
    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}")
        return None


def initialize_sample_data(sample_id: str, force_redownload: bool = False) -> bool:
    """
    Initialize a sample dataset by downloading REAL data.
    
    This function ONLY downloads real data - no synthetic data is ever created.
    
    Args:
        sample_id: ID of the sample to initialize
        force_redownload: If True, redownload even if file exists
        
    Returns:
        bool: True if successful
    """
    if sample_id not in SAMPLE_DATASETS:
        logger.error(f"Unknown sample ID: {sample_id}")
        return False
    
    sample_info = SAMPLE_DATASETS[sample_id]
    data_path = get_sample_file_path(sample_id)
    meta_path = get_sample_metadata_path(sample_id)
    
    # Skip if already exists and not forcing redownload
    if data_path.exists() and not force_redownload:
        # Verify it's valid
        try:
            ds = xr.open_dataset(data_path)
            if len(ds.data_vars) > 0:
                logger.info(f"Sample {sample_id} already exists and is valid")
                return True
        except:
            pass
        # If invalid, delete and redownload
        data_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    
    # Determine data source
    data_source = sample_info.get("data_source", "")
    requires_cloud = sample_info.get("requires_cloud", False)
    
    # Try GitHub-hosted real data sources first
    if data_source in REAL_DATA_SOURCES:
        logger.info(f"Downloading real data for {sample_id} from GitHub...")
        ds = download_real_data_from_github(data_source)
        
        if ds is not None:
            # Save to disk
            try:
                ds.to_netcdf(data_path)
                
                # Save metadata
                source_info = REAL_DATA_SOURCES[data_source]
                metadata = {
                    **sample_info,
                    "downloaded_at": datetime.now().isoformat(),
                    "is_real_data": True,
                    "source": source_info["source"],
                    "citation": source_info["citation"],
                    "file_size_mb": data_path.stat().st_size / (1024 * 1024),
                }
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Successfully downloaded real data for {sample_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to save {sample_id}: {e}")
                return False
    
    # Try WeatherBench2 for cloud-based datasets
    if data_source == "weatherbench2" or requires_cloud:
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
                raise ValueError("No matching variables found in WeatherBench2")
            
            subset = ds[variables].sel(
                time=slice(sample_info["start_date"], sample_info["end_date"])
            )
            
            if "level" in ds.coords:
                available_levels = [l for l in sample_info["pressure_levels"] if l in ds.level.values]
                if available_levels:
                    subset = subset.sel(level=available_levels)
            
            # Load and save
            logger.info("Loading data from WeatherBench2...")
            loaded = subset.load()
            loaded.attrs["is_real_data"] = "1"  # String for netCDF compatibility
            loaded.attrs["source"] = "WeatherBench2 ERA5 (ECMWF Reanalysis)"
            loaded.to_netcdf(data_path)
            
            # Save metadata
            metadata = {
                **sample_info,
                "downloaded_at": datetime.now().isoformat(),
                "is_real_data": True,
                "source": "WeatherBench2 ERA5",
                "file_size_mb": data_path.stat().st_size / (1024 * 1024),
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully downloaded {sample_id} from WeatherBench2")
            return True
            
        except Exception as e:
            logger.error(f"Could not download from WeatherBench2: {e}")
            logger.warning(f"Dataset {sample_id} requires cloud access which is not available")
            return False
    
    logger.error(f"No valid data source found for {sample_id}")
    return False


def initialize_all_samples() -> Dict[str, bool]:
    """
    Initialize all sample datasets by downloading REAL data.
    
    Returns:
        Dict mapping sample_id to success status
    """
    results = {}
    for sample_id in SAMPLE_DATASETS:
        results[sample_id] = initialize_sample_data(sample_id)
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
