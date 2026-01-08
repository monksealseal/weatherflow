import numpy as np
import xarray as xr

from weatherflow.data.gaia.regrid import RegridStrategy, TargetGrid, regrid_dataset


def _make_dataset(lat, lon, values):
    return xr.Dataset(
        {
            "precip": (("latitude", "longitude"), values),
        },
        coords={
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_conservative_regrid_preserves_integral():
    lat = np.array([0.0, 1.0, 2.0, 3.0])
    lon = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.ones((4, 4))
    dataset = _make_dataset(lat, lon, values)

    target_lat = np.array([0.5, 2.5])
    target_lon = np.array([0.5, 2.5])
    target = TargetGrid(latitude=target_lat, longitude=target_lon)

    regridded = regrid_dataset(dataset, target, RegridStrategy.CONSERVATIVE)

    def integral(ds):
        weights = np.cos(np.deg2rad(ds["latitude"].values))
        weight_grid = weights[:, None] * np.ones(len(ds["longitude"]))
        return float((ds["precip"].values * weight_grid).sum())

    assert np.isclose(integral(dataset), integral(regridded))


def test_bilinear_regrid_stays_within_bounds():
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    values = np.array([[0.0, 1.0], [2.0, 3.0]])
    dataset = _make_dataset(lat, lon, values)

    target = TargetGrid(latitude=np.array([0.25, 0.75]), longitude=np.array([0.25, 0.75]))
    regridded = regrid_dataset(dataset, target, RegridStrategy.BILINEAR)

    assert regridded["precip"].min() >= values.min() - 1e-6
    assert regridded["precip"].max() <= values.max() + 1e-6
