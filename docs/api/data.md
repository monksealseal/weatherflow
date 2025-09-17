# Data Loading API Reference

WeatherFlow ships with dataset classes tailored for atmospheric reanalysis data
and lightweight synthetic experiments. These utilities live in
`weatherflow.data` and integrate seamlessly with PyTorch's `Dataset`/`DataLoader`
API.

## ERA5Dataset

`ERA5Dataset` is the primary entry point for real-world data. It can:

- Stream directly from the public WeatherBench2 Google Cloud Storage bucket with
  multiple fallback strategies (anonymous GCS, HTTPS mirror, local Zarr, NetCDF).
- Select variables by their short names (`"t"`, `"z"`, `"u"`, `"v"`, etc.) and
  map them to the descriptive keys used inside the dataset.
- Slice by pressure level and time using standard Python slices or `(start, end)`
  tuples.
- Optionally normalise the variables using built-in statistics and enrich the
  dataset with derived diagnostics such as wind speed and vorticity.
- Cache samples in memory for rapid prototyping (`cache_data=True`).

Typical usage:

```python
from weatherflow.data import ERA5Dataset

dataset = ERA5Dataset(
    variables=["z", "t"],
    pressure_levels=[500],
    time_slice=("2015", "2015"),
    normalize=True,
    add_physics_features=True,
)

sample = dataset[0]
print(sample["input"].shape, sample["metadata"])
```

::: weatherflow.data.era5.ERA5Dataset
    :members:
    :show-inheritance:

## WeatherDataset

`WeatherDataset` provides a minimal interface for loading local HDF5 archives
containing variables such as geopotential height or temperature. It is useful
for unit tests and small offline experiments.

::: weatherflow.data.datasets.WeatherDataset
    :members:

## Data loader helper

`create_data_loaders` spins up train/validation datasets with consistent
configuration and wraps them in PyTorch `DataLoader` instances. It accepts the
same keyword arguments as `ERA5Dataset` plus `batch_size`/`num_workers`.

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

::: weatherflow.data.era5.create_data_loaders

## Tips

- Inspect `dataset.shape` to discover the `(variables, levels, lat, lon)` layout
  before configuring your model.
- Use `dataset.get_coords()` to retrieve the latitude/longitude grids for
  plotting.
- When working offline, set `data_path` to a local Zarr or NetCDF archive to skip
  network access entirely.
