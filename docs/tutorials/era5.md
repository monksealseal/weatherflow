# Working with ERA5 Data

`weatherflow.data.ERA5Dataset` is designed to make ERA5 ingestion painless.
Whether you are streaming from the public WeatherBench2 bucket or loading
pre-downloaded archives, the class handles the mechanics for you. This tutorial
covers the most common scenarios.

## 1. Stream directly from WeatherBench2

The dataset defaults to the WeatherBench2 Google Cloud Storage bucket and tries
several access strategies in order:

1. Anonymous `xr.open_zarr` access.
2. HTTPS mirror via `fsspec.filesystem("http")`.
3. Anonymous `gcsfs` client.
4. Local NetCDF fallback.
5. Local Zarr fallback.

In most environments you can instantiate the loader without any additional
configuration:

```python
from weatherflow.data import ERA5Dataset

dataset = ERA5Dataset(
    variables=["z", "t", "u", "v"],
    pressure_levels=[850, 500, 250],
    time_slice=("2015", "2016"),
    normalize=True,
)
print(len(dataset), dataset.shape)
```

Set `verbose=True` to log each attempted access method and the outcome.

## 2. Use local NetCDF or Zarr archives

If you already have the ERA5 subset on disk, point `data_path` at the archive.
The loader automatically chooses the appropriate backend based on the extension.

```python
dataset = ERA5Dataset(
    data_path="/data/era5/geopotential_temperature.nc",
    variables=["z", "t"],
    pressure_levels=[500],
    time_slice=("2015-01-01", "2015-03-31"),
)
```

For Zarr stores, pass the directory path. WeatherFlow preserves the same
interface regardless of the storage backend.

## 3. Derived diagnostics and normalisation

Enable `add_physics_features=True` to augment the dataset with derived fields
when wind components are present:

- `wind_speed` computed from the \(u\) and \(v\) components.
- `vorticity` estimated via central differences when latitude/longitude grids are
  available.

Normalisation uses the statistics defined in `NORMALIZE_STATS`. You can disable
it for raw values with `normalize=False` or supply custom scaling by wrapping the
returned tensors in your own transform.

## 4. Caching for faster experiments

Set `cache_data=True` to keep retrieved samples in memory. This is useful for
small experiments or unit tests where the dataset fits in RAM. The cache is keyed
by sample index.

```python
dataset = ERA5Dataset(cache_data=True, variables=["z"], pressure_levels=[500])
```

## 5. Explore metadata and coordinates

- `dataset.shape` returns `(variables, levels, lat, lon)` so you can configure the
  model correctly.
- `dataset.get_coords()` returns the latitude/longitude arrays and the pressure
  levels used in the subset.
- The `metadata` entry returned by `dataset[idx]` includes timestamps and the
  requested variable names.

```python
sample = dataset[0]
print(sample["metadata"]["t0"], sample["metadata"]["variables"])
```

## 6. Combine with PyTorch data loaders

Use `create_data_loaders` for convenience:

```python
from weatherflow.data import create_data_loaders

train_loader, val_loader = create_data_loaders(
    variables=["z", "t"],
    pressure_levels=[500],
    train_slice=("2015", "2015"),
    val_slice=("2016", "2016"),
    batch_size=16,
    num_workers=4,
)
```

Adjust `num_workers` to match your storage bandwidth. When reading from local
SSD, a handful of workers keeps the GPU busy; for remote streaming start with 0–2
workers to avoid saturating your connection.

## 7. Troubleshooting tips

- **Authentication** – if anonymous GCS access is blocked, set the environment
  variable `GCSFS_TOKEN` to point at your Google credentials JSON.
- **Proxy support** – configure `HTTPS_PROXY`/`HTTP_PROXY` before instantiating
  the dataset to route traffic through a proxy.
- **Timeouts** – pass `verbose=True` to inspect the failing access method and try
  the next one manually; often switching to a local mirror resolves the issue.

With these techniques you can focus on experimenting with models instead of data
plumbing. Continue with the [Quick Start tutorial](quickstart.md) to see the
loader in action inside a full training script.
