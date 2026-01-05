import numpy as np
import pandas as pd
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
from weatherflow.data.sequence import MultiStepERA5Dataset


def _build_dummy_sequence_dataset(
    monkeypatch, tmp_path, time_steps: int = 6, context: int = 2, pred: int = 2
) -> MultiStepERA5Dataset:
    """Construct an in-memory MultiStepERA5Dataset without disk I/O."""
    # Use datetime coordinates that match the years parameter (2000-2001)
    time_range = pd.date_range("2000-01-01", periods=time_steps, freq="6h")
    ds = xr.Dataset(
        {
            "z": (
                ("time", "level", "latitude", "longitude"),
                np.random.randn(time_steps, 1, 4, 8),
            )
        },
        coords={
            "time": time_range,
            "level": np.array([500]),
            "latitude": np.linspace(-90, 90, 4),
            "longitude": np.linspace(0, 360, 8, endpoint=False),
        },
    )

    call_counter = {"count": 0}

    def fake_open_mfdataset(*args, **kwargs):
        call_counter["count"] += 1
        return ds

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    dataset = MultiStepERA5Dataset(
        root_dir=tmp_path,
        years=[2000, 2001],
        variables=["z"],
        levels=[500],
        context_length=context,
        pred_length=pred,
        stride=1,
    )

    return dataset, call_counter


def test_multistep_dataset_shapes(monkeypatch, tmp_path):
    dataset, call_counter = _build_dummy_sequence_dataset(monkeypatch, tmp_path)
    assert call_counter["count"] == 1
    assert len(dataset) == 3  # (6 - (2+2)) + 1

    sample = dataset[0]
    context = sample["context"]
    target = sample["target"]
    assert context.shape == (2, 1, 1, 4, 8)
    assert target.shape == (2, 1, 1, 4, 8)
    assert sample["metadata"]["context_length"] == 2
    assert sample["metadata"]["pred_length"] == 2
    assert sample["metadata"]["pressure_levels"] == [500]
