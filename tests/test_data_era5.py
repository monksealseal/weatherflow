import json

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
from weatherflow.data.era5 import ERA5Dataset


def _build_dummy_era5_dataset(time_steps: int = 2) -> xr.Dataset:
    """Create a minimal in-memory ERA5-like dataset."""
    data = np.ones((time_steps, 1, 2, 3), dtype=np.float32)
    return xr.Dataset(
        {
            "temperature": (
                ("time", "level", "latitude", "longitude"),
                data,
            )
        },
        coords={
            "time": np.array(
                ["2000-01-01", "2001-01-01"], dtype="datetime64[ns]"
            )[:time_steps],
            "level": np.array([500]),
            "latitude": np.linspace(-90, 90, 2),
            "longitude": np.linspace(0, 360, 3, endpoint=False),
        },
    )


def test_era5_dataset_normalizes_and_caches_stats(tmp_path, monkeypatch):
    ds = _build_dummy_era5_dataset()
    call_counter = {"count": 0}

    def fake_open_mfdataset(*args, **kwargs):
        call_counter["count"] += 1
        return ds

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    dataset = ERA5Dataset(
        root_dir=tmp_path,
        years=[2000, 2001],
        variables=["temperature"],
        levels=[500],
    )

    assert call_counter["count"] == 1
    assert len(dataset) == ds.sizes["time"]

    sample = dataset[0]
    assert sample.shape == (1, 1, 2, 3)
    assert torch.allclose(sample, torch.zeros_like(sample), atol=1e-5)

    stats_path = tmp_path / "stats.json"
    assert stats_path.exists()
    stats_content = json.loads(stats_path.read_text())
    assert "mean" in stats_content and "std" in stats_content

    # Re-instantiating should reuse cached statistics without recomputation errors.
    dataset_second = ERA5Dataset(
        root_dir=tmp_path,
        years=[2000, 2001],
        variables=["temperature"],
        levels=[500],
    )
    assert dataset_second.stats["mean"].shape == (1, 1, 1, 1)
