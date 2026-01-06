"""
Data preparation script for WeatherFlow Streamlit App.

Downloads 2 months of ERA5 data from WeatherBench2 (Google Cloud Storage)
and caches it locally for the Streamlit app to use.

Falls back to realistic synthetic data if cloud access fails.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_synthetic_era5_data(
    output_dir: str = "streamlit_app/data",
    start_date: str = "2023-01-01",
    end_date: str = "2023-02-28",
    variables: list = None,
    pressure_levels: list = None,
):
    """
    Create synthetic ERA5-like data when cloud access is unavailable.

    This generates realistic weather patterns based on:
    - Climatological temperature profiles
    - Geostrophic wind relationships
    - Seasonal and diurnal cycles
    - Spatial correlations
    """
    if variables is None:
        variables = ['temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind']

    if pressure_levels is None:
        pressure_levels = [1000, 850, 700, 500, 300, 200]

    print("=" * 60)
    print("WeatherFlow Data Generator (Synthetic)")
    print("=" * 60)
    print("\nGenerating realistic synthetic ERA5-like data...")
    print("(Cloud access unavailable - using physics-based synthetic data)")
    print(f"\nTime period: {start_date} to {end_date}")
    print(f"Variables: {variables}")
    print(f"Pressure levels: {pressure_levels} hPa")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Grid parameters (64x32 resolution)
    n_lat, n_lon = 32, 64
    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(0, 360, n_lon, endpoint=False)

    # Time parameters
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    n_hours = int((end - start).total_seconds() / 3600)
    n_steps = n_hours // 6 + 1  # 6-hourly data
    times = [start + timedelta(hours=6*i) for i in range(n_steps)]

    print(f"\n[1/4] Generating {n_steps} time steps...")

    # Create coordinate arrays
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    lat_rad = np.deg2rad(lat_grid)

    data_vars = {}

    for var in variables:
        print(f"  Generating {var}...")
        var_data = np.zeros((n_steps, len(pressure_levels), n_lat, n_lon), dtype=np.float32)

        for t_idx, t in enumerate(times):
            day_of_year = t.timetuple().tm_yday
            hour = t.hour

            for lev_idx, level in enumerate(pressure_levels):
                if var == 'temperature':
                    # Base temperature decreasing with altitude (lapse rate ~6.5 K/km)
                    altitude_km = {1000: 0, 850: 1.5, 700: 3, 500: 5.5, 300: 9, 200: 12}.get(level, 5)
                    base_temp = 288 - 6.5 * altitude_km

                    # Latitudinal gradient (colder at poles)
                    lat_effect = -40 * np.sin(lat_rad) ** 2

                    # Seasonal cycle
                    seasonal = 15 * np.cos(2 * np.pi * (day_of_year - 172) / 365) * np.sin(lat_rad)

                    # Diurnal cycle (stronger near surface)
                    diurnal = 5 * np.cos(2 * np.pi * (hour - 14) / 24) * (level / 1000) * np.cos(lat_rad)

                    # Synoptic-scale waves
                    wave = 5 * np.sin(3 * np.deg2rad(lon_grid) + 0.1 * t_idx)

                    # Random perturbations
                    noise = 2 * np.random.randn(n_lat, n_lon)

                    var_data[t_idx, lev_idx] = base_temp + lat_effect + seasonal + diurnal + wave + noise

                elif var == 'geopotential':
                    # Hydrostatic approximation
                    R = 287  # J/(kg*K)
                    g = 9.81
                    p0 = 1013.25

                    # Mean temperature for this level
                    altitude_km = {1000: 0, 850: 1.5, 700: 3, 500: 5.5, 300: 9, 200: 12}.get(level, 5)
                    T_mean = 288 - 6.5 * altitude_km

                    # Base geopotential height
                    base_z = (R * T_mean / g) * np.log(p0 / level) * g

                    # Latitude effect
                    lat_effect = 500 * np.cos(lat_rad) * (1000 / level)

                    # Synoptic waves (Rossby waves)
                    wave_amp = 2000 * (1000 / level)
                    wave = wave_amp * np.sin(4 * np.deg2rad(lon_grid) + 0.05 * t_idx + lat_rad)

                    var_data[t_idx, lev_idx] = base_z + lat_effect + wave

                elif var == 'u_component_of_wind':
                    # Geostrophic wind structure
                    f = 2 * 7.292e-5 * np.sin(lat_rad)
                    f = np.where(np.abs(f) < 1e-5, 1e-5 * np.sign(lat_rad + 1e-10), f)

                    # Jet stream structure
                    jet_lat = 45
                    jet_strength = 30 * (1000 / level)
                    u_jet = jet_strength * np.exp(-((lats[:, None] - jet_lat) / 15) ** 2)

                    # Wave perturbations
                    wave = 10 * np.sin(5 * np.deg2rad(lon_grid) + 0.08 * t_idx)

                    var_data[t_idx, lev_idx] = u_jet + wave + 3 * np.random.randn(n_lat, n_lon)

                elif var == 'v_component_of_wind':
                    # Meridional wind
                    wave = 8 * np.cos(4 * np.deg2rad(lon_grid) + 0.06 * t_idx) * np.cos(lat_rad)
                    var_data[t_idx, lev_idx] = wave + 2 * np.random.randn(n_lat, n_lon)

        data_vars[var] = (['time', 'level', 'latitude', 'longitude'], var_data)

    print("\n[2/4] Creating xarray dataset...")

    ds = xr.Dataset(
        data_vars,
        coords={
            'time': times,
            'level': pressure_levels,
            'latitude': lats,
            'longitude': lons
        }
    )

    ds.attrs['source'] = 'WeatherFlow Synthetic Data Generator'
    ds.attrs['description'] = 'Physics-based synthetic ERA5-like data for demonstration'
    ds.attrs['created'] = datetime.now().isoformat()

    print("\n[3/4] Saving to NetCDF...")
    output_file = output_path / "era5_sample.nc"
    ds.to_netcdf(output_file)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved to {output_file} ({file_size_mb:.1f} MB)")

    return ds, output_file, file_size_mb, variables


def download_era5_sample(
    output_dir: str = "streamlit_app/data",
    start_date: str = "2023-01-01",
    end_date: str = "2023-02-28",
    variables: list = None,
    pressure_levels: list = None,
    resolution: str = "64x32"
):
    """
    Download a sample of ERA5 data from WeatherBench2.
    Falls back to synthetic data if cloud access fails.
    """
    if variables is None:
        variables = ['temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind']

    if pressure_levels is None:
        pressure_levels = [1000, 850, 700, 500, 300, 200]

    data_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

    print("=" * 60)
    print("WeatherFlow Data Download")
    print("=" * 60)
    print(f"\nData source: WeatherBench2 ERA5 (Google Cloud Storage)")
    print(f"URL: {data_url}")
    print(f"Time period: {start_date} to {end_date}")
    print(f"Variables: {variables}")
    print(f"Pressure levels: {pressure_levels} hPa")
    print(f"Resolution: {resolution}")
    print("\nThis is FREE, publicly available data - no API key required!")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Connecting to WeatherBench2...")

    ds = None
    use_synthetic = False

    # Try to access cloud data
    try:
        ds = xr.open_zarr(
            data_url,
            storage_options={'anon': True},
            consolidated=True
        )
        print("  ✓ Connected successfully")
    except Exception as e:
        print(f"  ✗ Primary method failed: {type(e).__name__}")
        try:
            print("  Trying gcsfs method...")
            import gcsfs
            fs = gcsfs.GCSFileSystem(token='anon')
            mapper = fs.get_mapper(data_url)
            ds = xr.open_zarr(mapper, consolidated=True)
            print("  ✓ Connected via gcsfs")
        except Exception as e2:
            print(f"  ✗ gcsfs method failed: {type(e2).__name__}")
            print("\n  Falling back to synthetic data generation...")
            use_synthetic = True

    if use_synthetic:
        ds_loaded, output_file, file_size_mb, available_vars = create_synthetic_era5_data(
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            pressure_levels=pressure_levels
        )
        is_synthetic = True
        n_times = len(ds_loaded.time)
    else:
        is_synthetic = False
        print("\n[2/4] Selecting data subset...")

        ds_subset = ds.sel(
            time=slice(start_date, end_date),
            level=pressure_levels
        )

        available_vars = [v for v in variables if v in ds_subset.data_vars]
        if len(available_vars) < len(variables):
            missing = set(variables) - set(available_vars)
            print(f"  Warning: Some variables not found: {missing}")

        ds_subset = ds_subset[available_vars]
        n_times = len(ds_subset.time)

        print(f"  ✓ Selected {n_times} time steps ({n_times * 6} hours of data)")
        print(f"  ✓ Grid size: {ds_subset.latitude.size} x {ds_subset.longitude.size}")

        print("\n[3/4] Downloading and caching data...")
        print("  This may take a few minutes...")

        ds_loaded = ds_subset.load()
        output_file = output_path / "era5_sample.nc"
        ds_loaded.to_netcdf(output_file)

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved to {output_file} ({file_size_mb:.1f} MB)")

    print("\n[4/4] Generating metadata...")

    metadata = {
        "source": "WeatherFlow Synthetic Data" if is_synthetic else "WeatherBench2 ERA5",
        "is_synthetic": is_synthetic,
        "url": "N/A (synthetic)" if is_synthetic else data_url,
        "downloaded_at": datetime.now().isoformat(),
        "time_range": {
            "start": str(ds_loaded.time.values[0]),
            "end": str(ds_loaded.time.values[-1]),
            "n_steps": n_times,
            "frequency_hours": 6
        },
        "variables": available_vars,
        "pressure_levels": pressure_levels,
        "grid": {
            "latitude": int(ds_loaded.latitude.size),
            "longitude": int(ds_loaded.longitude.size),
            "resolution": resolution
        },
        "file_size_mb": round(file_size_mb, 2),
        "license": "Copernicus Climate Data Store License (free for research)",
        "citation": "Hersbach et al. (2020). The ERA5 global reanalysis. Q J R Meteorol Soc."
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  ✓ Metadata saved to {metadata_file}")

    print("\n[Bonus] Computing normalization statistics...")

    stats = {}
    for var in available_vars:
        data = ds_loaded[var].values
        stats[var] = {
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data))
        }

    stats_file = output_path / "normalization_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  ✓ Statistics saved to {stats_file}")

    print("\n" + "=" * 60)
    if is_synthetic:
        print("Synthetic data generation complete!")
        print("(Note: This is physics-based synthetic data for demonstration)")
    else:
        print("Download complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  • {output_file}")
    print(f"  • {metadata_file}")
    print(f"  • {stats_file}")
    print(f"\nYour Streamlit app can now use this data!")

    return str(output_file)


def verify_data(data_dir: str = "streamlit_app/data"):
    """Verify the downloaded data is valid."""
    data_path = Path(data_dir)
    required_files = ["era5_sample.nc", "metadata.json", "normalization_stats.json"]

    print("\nVerifying data files...")

    for f in required_files:
        file_path = data_path / f
        if file_path.exists():
            print(f"  ✓ {f} exists")
        else:
            print(f"  ✗ {f} MISSING")
            return False

    try:
        ds = xr.open_dataset(data_path / "era5_sample.nc")
        print(f"  ✓ NetCDF loads successfully")
        print(f"    Variables: {list(ds.data_vars)}")
        print(f"    Time steps: {len(ds.time)}")
        ds.close()
    except Exception as e:
        print(f"  ✗ Failed to load NetCDF: {e}")
        return False

    print("\n✓ All data files verified!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ERA5 sample data for WeatherFlow")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-02-28", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="streamlit_app/data", help="Output directory")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data generation")

    args = parser.parse_args()

    if args.verify_only:
        verify_data(args.output)
    elif args.synthetic:
        create_synthetic_era5_data(
            output_dir=args.output,
            start_date=args.start,
            end_date=args.end
        )
        verify_data(args.output)
        # Generate metadata for synthetic
        output_path = Path(args.output)
        metadata = {
            "source": "WeatherFlow Synthetic Data",
            "is_synthetic": True,
            "url": "N/A (synthetic)",
            "downloaded_at": datetime.now().isoformat(),
            "time_range": {"start": args.start, "end": args.end, "frequency_hours": 6},
            "variables": ['temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind'],
            "pressure_levels": [1000, 850, 700, 500, 300, 200],
            "grid": {"latitude": 32, "longitude": 64, "resolution": "64x32"},
            "license": "Synthetic data for demonstration",
        }
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    else:
        download_era5_sample(
            output_dir=args.output,
            start_date=args.start,
            end_date=args.end
        )
        verify_data(args.output)
