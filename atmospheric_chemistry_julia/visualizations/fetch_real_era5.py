"""
fetch_real_era5.py

Downloads REAL ERA5 reanalysis data from the public WeatherBench2 Zarr store
using direct HTTP access (no gcsfs/auth needed). Parses zarr metadata and
chunk structure, downloads individual binary chunks, and reassembles into
numpy arrays.

Data: ECMWF ERA5 reanalysis via WeatherBench2 (Rasp et al., 2024)
Grid: 360 × 181 (1° equiangular with poles)
"""

import numpy as np
import json
import urllib.request
import struct
import io
import os
import pickle
import sys

BASE_URL = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_data')
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_url(url, timeout=60):
    """Fetch raw bytes from URL."""
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read()


def load_zarr_metadata():
    """Load consolidated zarr metadata."""
    cache_path = os.path.join(CACHE_DIR, 'zmetadata.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    url = f"{BASE_URL}/.zmetadata"
    print(f"  Fetching zarr metadata...")
    data = fetch_url(url)
    meta = json.loads(data)

    with open(cache_path, 'w') as f:
        json.dump(meta, f)

    return meta


def get_var_info(meta, varname):
    """Extract variable metadata: shape, chunks, dtype, compressor, etc."""
    zarray_key = f"{varname}/.zarray"
    zattrs_key = f"{varname}/.zattrs"

    zarray = meta['metadata'].get(zarray_key, {})
    if isinstance(zarray, str):
        zarray = json.loads(zarray)

    zattrs = meta['metadata'].get(zattrs_key, {})
    if isinstance(zattrs, str):
        zattrs = json.loads(zattrs)

    return {
        'shape': tuple(zarray.get('shape', [])),
        'chunks': tuple(zarray.get('chunks', [])),
        'dtype': zarray.get('dtype', '<f4'),
        'compressor': zarray.get('compressor'),
        'fill_value': zarray.get('fill_value'),
        'order': zarray.get('order', 'C'),
        'dims': zattrs.get('_ARRAY_DIMENSIONS', []),
    }


def download_zarr_chunk(varname, chunk_key):
    """Download a single zarr chunk."""
    url = f"{BASE_URL}/{varname}/{chunk_key}"
    return fetch_url(url)


def decode_chunk(raw_bytes, info):
    """Decode a zarr chunk from raw bytes."""
    compressor = info.get('compressor')
    dtype = np.dtype(info['dtype'])
    chunk_shape = info['chunks']

    if compressor and compressor.get('id') == 'blosc':
        import blosc2
        decompressed = blosc2.decompress(raw_bytes)
        arr = np.frombuffer(decompressed, dtype=dtype)
    elif compressor and compressor.get('id') == 'zlib':
        import zlib
        decompressed = zlib.decompress(raw_bytes)
        arr = np.frombuffer(decompressed, dtype=dtype)
    elif compressor is None:
        arr = np.frombuffer(raw_bytes, dtype=dtype)
    else:
        # Try blosc first as it's the most common
        try:
            import blosc2
            decompressed = blosc2.decompress(raw_bytes)
            arr = np.frombuffer(decompressed, dtype=dtype)
        except Exception:
            arr = np.frombuffer(raw_bytes, dtype=dtype)

    expected_size = int(np.prod(chunk_shape))
    if len(arr) == expected_size:
        arr = arr.reshape(chunk_shape)
    else:
        # Partial chunk at edge
        arr = arr[:expected_size].reshape(chunk_shape) if len(arr) >= expected_size else arr

    return arr


def find_time_index(meta, target_time_str):
    """Find the global time index for a target time."""
    from datetime import datetime
    epoch = datetime(1959, 1, 1)
    target = datetime.fromisoformat(target_time_str.replace('Z', ''))

    # ERA5 6-hourly: compute global time index
    hours_since_epoch = (target - epoch).total_seconds() / 3600
    time_idx = int(hours_since_epoch / 6)

    print(f"  Time {target_time_str}: global_idx={time_idx}")
    return time_idx


def time_chunk_for_var(meta, varname, global_time_idx):
    """Get chunk index and offset within chunk for a specific variable."""
    info = get_var_info(meta, varname)
    time_chunk_size = info['chunks'][0]  # Each variable's own time chunk size
    chunk_idx = global_time_idx // time_chunk_size
    offset = global_time_idx % time_chunk_size
    return chunk_idx, offset


def load_2d_field(meta, varname, global_time_idx):
    """Load a 2D (time, lon, lat) field for one time step."""
    info = get_var_info(meta, varname)
    shape = info['shape']
    chunks = info['chunks']
    ndim = len(shape)

    print(f"    {varname}: shape={shape}, chunks={chunks}, dims={info['dims']}")

    if ndim == 3:
        # (time, lon, lat)
        time_chunk_idx, time_offset = time_chunk_for_var(meta, varname, global_time_idx)
        print(f"      time_chunk={time_chunk_idx}, offset={time_offset}")

        lon_chunks = (shape[1] + chunks[1] - 1) // chunks[1]
        lat_chunks = (shape[2] + chunks[2] - 1) // chunks[2]

        result = np.zeros((shape[1], shape[2]), dtype=np.float32)

        for ic in range(lon_chunks):
            for jc in range(lat_chunks):
                chunk_key = f"{time_chunk_idx}.{ic}.{jc}"
                try:
                    raw = download_zarr_chunk(varname, chunk_key)
                    arr = decode_chunk(raw, info)
                    # Extract our time step
                    if arr.ndim == 3:
                        slice_data = arr[time_offset]
                    else:
                        slice_data = arr

                    i_start = ic * chunks[1]
                    j_start = jc * chunks[2]
                    i_end = min(i_start + slice_data.shape[0], shape[1])
                    j_end = min(j_start + slice_data.shape[-1], shape[2])

                    result[i_start:i_end, j_start:j_end] = slice_data[:i_end-i_start, :j_end-j_start]
                except Exception as e:
                    print(f"      Chunk {chunk_key}: {e}")

        return result

    elif ndim == 2:
        # Static field (lon, lat)
        lon_chunks = (shape[0] + chunks[0] - 1) // chunks[0]
        lat_chunks = (shape[1] + chunks[1] - 1) // chunks[1]

        result = np.zeros(shape, dtype=np.float32)
        for ic in range(lon_chunks):
            for jc in range(lat_chunks):
                chunk_key = f"{ic}.{jc}"
                try:
                    raw = download_zarr_chunk(varname, chunk_key)
                    arr = decode_chunk(raw, info)
                    i_start = ic * chunks[0]
                    j_start = jc * chunks[1]
                    i_end = min(i_start + arr.shape[0], shape[0])
                    j_end = min(j_start + arr.shape[1], shape[1])
                    result[i_start:i_end, j_start:j_end] = arr[:i_end-i_start, :j_end-j_start]
                except Exception as e:
                    print(f"      Chunk {chunk_key}: {e}")
        return result


def load_3d_field(meta, varname, global_time_idx):
    """Load a 3D (time, level, lon, lat) field for one time step."""
    info = get_var_info(meta, varname)
    shape = info['shape']
    chunks = info['chunks']

    time_chunk_idx, time_offset = time_chunk_for_var(meta, varname, global_time_idx)
    print(f"    {varname}: shape={shape}, chunks={chunks}, dims={info['dims']}")
    print(f"      time_chunk={time_chunk_idx}, offset={time_offset}")

    # (time, level, lon, lat)
    nlevels = shape[1]
    nlon = shape[2]
    nlat = shape[3]

    level_chunks_n = (nlevels + chunks[1] - 1) // chunks[1]
    lon_chunks_n = (nlon + chunks[2] - 1) // chunks[2]
    lat_chunks_n = (nlat + chunks[3] - 1) // chunks[3]

    result = np.zeros((nlevels, nlon, nlat), dtype=np.float32)

    for lc in range(level_chunks_n):
        for ic in range(lon_chunks_n):
            for jc in range(lat_chunks_n):
                chunk_key = f"{time_chunk_idx}.{lc}.{ic}.{jc}"
                try:
                    raw = download_zarr_chunk(varname, chunk_key)
                    arr = decode_chunk(raw, info)

                    if arr.ndim == 4:
                        slice_data = arr[time_offset]
                    else:
                        slice_data = arr

                    l_start = lc * chunks[1]
                    i_start = ic * chunks[2]
                    j_start = jc * chunks[3]
                    l_end = min(l_start + slice_data.shape[0], nlevels)
                    i_end = min(i_start + slice_data.shape[1], nlon)
                    j_end = min(j_start + slice_data.shape[2], nlat)

                    result[l_start:l_end, i_start:i_end, j_start:j_end] = \
                        slice_data[:l_end-l_start, :i_end-i_start, :j_end-j_start]
                except Exception as e:
                    print(f"      Chunk {chunk_key}: {e}")

    return result


def load_coordinate(meta, name):
    """Load a 1D coordinate variable."""
    info = get_var_info(meta, name)
    chunk_key = "0"
    raw = download_zarr_chunk(name, chunk_key)
    arr = decode_chunk(raw, info)
    return arr.ravel()


def download_era5_snapshot(target_time='2022-07-15T12:00:00'):
    """Download a full ERA5 snapshot at the target time."""
    cache_file = os.path.join(CACHE_DIR, f'era5_{target_time.replace(":", "").replace("-", "")}.pkl')

    if os.path.exists(cache_file):
        print(f"  Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    meta = load_zarr_metadata()
    time_global = find_time_index(meta, target_time)

    print(f"\n  Downloading coordinates...")
    lon = load_coordinate(meta, 'longitude')
    lat = load_coordinate(meta, 'latitude')

    level_info = get_var_info(meta, 'level')
    n_level_chunks = (level_info['shape'][0] + level_info['chunks'][0] - 1) // level_info['chunks'][0]
    levels = []
    for i in range(n_level_chunks):
        raw = download_zarr_chunk('level', str(i))
        arr = decode_chunk(raw, level_info)
        levels.append(arr.ravel())
    levels = np.concatenate(levels)

    print(f"  Grid: {len(lon)} × {len(lat)}, {len(levels)} levels")
    print(f"  Levels [hPa]: {levels}")

    fields = {
        'longitude': lon,
        'latitude': lat,
        'level': levels,
        'time': target_time,
    }

    # 2D fields
    vars_2d = [
        '2m_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'mean_sea_level_pressure',
        'total_precipitation_6hr',
        'surface_pressure',
        'boundary_layer_height',
        'total_cloud_cover',
        'total_column_water_vapour',
    ]

    print(f"\n  Downloading 2D fields...")
    for var in vars_2d:
        info = get_var_info(meta, var)
        if info['shape']:
            print(f"  Loading {var}...")
            try:
                fields[var] = load_2d_field(meta, var, time_global)
            except Exception as e:
                print(f"    FAILED: {e}")

    # Static fields
    for var in ['land_sea_mask', 'geopotential_at_surface']:
        info = get_var_info(meta, var)
        if info['shape']:
            print(f"  Loading {var} (static)...")
            try:
                fields[var] = load_2d_field(meta, var, 0)
            except Exception as e:
                print(f"    FAILED: {e}")

    # 3D fields
    vars_3d = [
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'specific_humidity',
        'geopotential',
    ]

    print(f"\n  Downloading 3D fields (this may take a few minutes)...")
    for var in vars_3d:
        info = get_var_info(meta, var)
        if info['shape']:
            print(f"  Loading {var}...")
            try:
                fields[var] = load_3d_field(meta, var, time_global)
            except Exception as e:
                print(f"    FAILED: {e}")

    # Cache
    with open(cache_file, 'wb') as f:
        pickle.dump(fields, f)
    print(f"\n  Cached to: {cache_file}")

    return fields


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '2022-07-15T12:00:00'

    print("=" * 65)
    print("  Downloading REAL ERA5 Reanalysis Data")
    print(f"  Target: {target}")
    print(f"  Source: ECMWF ERA5 via WeatherBench2")
    print("=" * 65)

    fields = download_era5_snapshot(target)

    print("\n  Downloaded fields:")
    for k, v in fields.items():
        if isinstance(v, np.ndarray):
            print(f"    {k}: shape={v.shape}, range=[{v.min():.2f}, {v.max():.2f}]")
        else:
            print(f"    {k}: {v}")
