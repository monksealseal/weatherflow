import numpy as np
import torch
import xarray as xr

from weatherflow.data.sequence import MultiStepERA5Dataset


def build_dummy_sequence_dataset(
    time_steps: int = 6,
    context: int = 2,
    pred: int = 2,
) -> MultiStepERA5Dataset:
    """Construct an in-memory MultiStepERA5Dataset without I/O."""
    ds = xr.Dataset(
        {
            "z": (
                ("time", "level", "latitude", "longitude"),
                np.random.randn(time_steps, 1, 4, 8),
            )
        },
        coords={
            "time": np.arange(time_steps),
            "level": np.array([500]),
            "latitude": np.linspace(-90, 90, 4),
            "longitude": np.linspace(0, 360, 8, endpoint=False),
        },
    )

    dataset = MultiStepERA5Dataset.__new__(MultiStepERA5Dataset)
    dataset.context_length = context
    dataset.pred_length = pred
    dataset.stride = 1
    dataset.variables = ["z"]
    dataset.pressure_levels = [500]
    dataset.normalize = False
    dataset.normalize_stats = {}
    dataset.cache_data = False
    dataset._cache = None
    dataset.ds = ds
    dataset.times = ds.time
    return dataset


def test_multistep_dataset_shapes():
    dataset = build_dummy_sequence_dataset()
    assert len(dataset) == 3  # (6 - (2+2)) + 1
    sample = dataset[0]
    context = sample["context"]
    target = sample["target"]
    assert context.shape == (2, 1, 1, 4, 8)
    assert target.shape == (2, 1, 1, 4, 8)
    assert sample["metadata"]["context_length"] == 2
    assert sample["metadata"]["pred_length"] == 2
