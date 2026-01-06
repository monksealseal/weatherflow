"""
Data preparation script for WeatherFlow Streamlit App.

Downloads REAL ERA5 data from WeatherBench2 (Google Cloud Storage).
NO SYNTHETIC DATA - only actual atmospheric observations.

WeatherBench2 provides free, public access to processed ERA5 data.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# WeatherBench2 ERA5 URL (public, free access)
WEATHERBENCH2_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"


def download_era5_from_weatherbench2(
    output_dir: str = "streamlit_app/data",
    start_date: str = "2023-01-01",
    end_date: str = "2023-02-28",
    variables: list = None,
    pressure_levels: list = None,
):
    """
    Download REAL ERA5 data from WeatherBench2 Google Cloud Storage.

    This downloads actual ERA5 reanalysis data from ECMWF - NOT synthetic data.

    Args:
        output_dir: Directory to save the data
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        variables: List of variables to download
        pressure_levels: List of pressure levels
    """
    if variables is None:
        variables = ['temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind']

    if pressure_levels is None:
        pressure_levels = [1000, 850, 700, 500, 300, 200]

    print("=" * 70)
    print("WeatherFlow ERA5 Data Download")
    print("=" * 70)
    print("\n🌍 SOURCE: WeatherBench2 / ECMWF ERA5 Reanalysis")
    print("📊 TYPE: REAL atmospheric observations (NOT synthetic)")
    print(f"\n📅 Time period: {start_date} to {end_date}")
    print(f"🔬 Variables: {variables}")
    print(f"📈 Pressure levels: {pressure_levels} hPa")
    print("\n" + "=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Connecting to WeatherBench2...")

    ds = None
    method_used = None

    # Method 1: Direct zarr with anonymous GCS access
    try:
        ds = xr.open_zarr(
            WEATHERBENCH2_URL,
            storage_options={'anon': True},
            consolidated=True
        )
        method_used = "GCS Anonymous"
        print(f"  ✅ Connected via {method_used}")
    except Exception as e1:
        print(f"  ⚠️  Method 1 (GCS) failed: {type(e1).__name__}")

        # Method 2: via gcsfs
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem(token='anon')
            mapper = fs.get_mapper(WEATHERBENCH2_URL)
            ds = xr.open_zarr(mapper, consolidated=True)
            method_used = "gcsfs"
            print(f"  ✅ Connected via {method_used}")
        except Exception as e2:
            print(f"  ⚠️  Method 2 (gcsfs) failed: {type(e2).__name__}")

            # Method 3: HTTP fallback
            try:
                import fsspec
                http_url = WEATHERBENCH2_URL.replace('gs://', 'https://storage.googleapis.com/')
                fs = fsspec.filesystem('http')
                mapper = fs.get_mapper(http_url)
                ds = xr.open_zarr(mapper, consolidated=True)
                method_used = "HTTP"
                print(f"  ✅ Connected via {method_used}")
            except Exception as e3:
                print(f"\n❌ ERROR: Could not connect to WeatherBench2")
                print(f"   Last error: {e3}")
                print("\n   Please check your internet connection and try again.")
                print("   This script requires network access to download REAL ERA5 data.")
                return None

    print(f"\n[2/4] Selecting data subset...")
    print(f"  Available variables: {list(ds.data_vars)[:10]}...")

    # Select subset
    ds_subset = ds[variables].sel(
        time=slice(start_date, end_date),
        level=pressure_levels
    )

    n_times = len(ds_subset.time)
    print(f"  ✅ Selected {n_times} time steps")
    print(f"  ✅ Grid: {ds_subset.latitude.size} × {ds_subset.longitude.size}")

    print(f"\n[3/4] Downloading REAL ERA5 data...")
    print("  This may take a few minutes depending on your connection...")

    ds_loaded = ds_subset.load()

    output_file = output_path / "era5_sample.nc"
    ds_loaded.to_netcdf(output_file)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✅ Saved {file_size_mb:.1f} MB to {output_file}")

    print(f"\n[4/4] Creating metadata...")

    metadata = {
        "source": "WeatherBench2 ERA5 (ECMWF Reanalysis)",
        "is_synthetic": False,
        "url": WEATHERBENCH2_URL,
        "connection_method": method_used,
        "downloaded_at": datetime.now().isoformat(),
        "time_range": {
            "start": str(ds_loaded.time.values[0]),
            "end": str(ds_loaded.time.values[-1]),
            "n_steps": n_times,
            "frequency_hours": 6
        },
        "variables": variables,
        "pressure_levels": pressure_levels,
        "grid": {
            "latitude": int(ds_loaded.latitude.size),
            "longitude": int(ds_loaded.longitude.size),
            "resolution": "64x32"
        },
        "file_size_mb": round(file_size_mb, 2),
        "license": "Copernicus Climate Data Store License",
        "citation": "Hersbach et al. (2020). The ERA5 global reanalysis. Q J R Meteorol Soc."
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Compute normalization statistics
    stats = {}
    for var in variables:
        data = ds_loaded[var].values
        stats[var] = {
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data))
        }

    with open(output_path / "normalization_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  ✅ Metadata and statistics saved")

    print("\n" + "=" * 70)
    print("✅ SUCCESS - Downloaded REAL ERA5 data")
    print("=" * 70)
    print(f"\n📁 Files created:")
    print(f"   • {output_file} ({file_size_mb:.1f} MB)")
    print(f"   • metadata.json")
    print(f"   • normalization_stats.json")
    print(f"\n🌍 This is REAL atmospheric data from ECMWF ERA5 reanalysis.")
    print("   NOT synthetic. NOT simulated. REAL observations.")

    return str(output_file)


def verify_data(data_dir: str = "streamlit_app/data"):
    """Verify the downloaded data is valid and NOT synthetic."""
    data_path = Path(data_dir)
    required_files = ["era5_sample.nc", "metadata.json", "normalization_stats.json"]

    print("\n🔍 Verifying data files...")

    for f in required_files:
        file_path = data_path / f
        if file_path.exists():
            print(f"  ✅ {f} exists")
        else:
            print(f"  ❌ {f} MISSING")
            return False

    # Check metadata
    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)

    if metadata.get("is_synthetic", True):
        print("\n  ❌ ERROR: Data is marked as synthetic!")
        print("     This app requires REAL ERA5 data.")
        return False

    print(f"\n  ✅ Source: {metadata.get('source', 'Unknown')}")
    print(f"  ✅ Synthetic: {metadata.get('is_synthetic', 'Unknown')}")

    # Try loading the NetCDF
    try:
        ds = xr.open_dataset(data_path / "era5_sample.nc")
        print(f"  ✅ NetCDF loads successfully")
        print(f"     Variables: {list(ds.data_vars)}")
        print(f"     Time steps: {len(ds.time)}")
        ds.close()
    except Exception as e:
        print(f"  ❌ Failed to load NetCDF: {e}")
        return False

    print("\n✅ All data files verified - REAL ERA5 data ready!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download REAL ERA5 data from WeatherBench2"
    )
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-02-28", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="streamlit_app/data", help="Output directory")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")

    args = parser.parse_args()

    if args.verify_only:
        verify_data(args.output)
    else:
        result = download_era5_from_weatherbench2(
            output_dir=args.output,
            start_date=args.start,
            end_date=args.end
        )
        if result:
            verify_data(args.output)
