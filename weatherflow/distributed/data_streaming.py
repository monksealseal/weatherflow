"""Streaming ERA5 dataset utilities for large-scale distributed training."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
import xarray as xr
from braceexpand import braceexpand
from torch.utils.data import IterableDataset, get_worker_info


def _load_stats(stats_path: str) -> Dict[str, torch.Tensor]:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_json = json.load(f)
    return {
        "mean": torch.tensor(stats_json["mean"], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1),
        "std": torch.tensor(stats_json["std"], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1),
    }


@dataclass
class StreamingConfig:
    shard_pattern: str
    sequence_length: int = 2
    shuffle_shards: bool = True
    shard_shuffle_buffer: int = 64
    sample_shuffle_buffer: int = 256
    stats_path: Optional[str] = None
    prefetch: int = 32


class ERA5StreamingDataset(IterableDataset):
    """WebDataset-backed iterable dataset for ERA5 streaming."""

    def __init__(
        self,
        config: StreamingConfig,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        dt_hours: float = 6.0,
    ):
        super().__init__()
        self.config = config
        self.dt_hours = dt_hours
        if world_size is None and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        if rank is None and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        self.world_size = world_size
        self.rank = rank
        self.stats = _load_stats(config.stats_path) if config.stats_path else None

    def _get_shards(self) -> List[str]:
        shards = list(braceexpand(self.config.shard_pattern))
        if not shards:
            raise FileNotFoundError(f"No shards found matching pattern {self.config.shard_pattern}")
        if self.config.shuffle_shards:
            rng = np.random.default_rng()
            rng.shuffle(shards)
        if self.world_size and self.rank is not None:
            shards = shards[self.rank :: self.world_size]
        return shards

    def _normalize(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.stats is None:
            return sample
        mean = self.stats["mean"].to(sample["input"].device)
        std = torch.clamp(self.stats["std"].to(sample["input"].device), min=1e-6)
        sample["input"] = (sample["input"] - mean) / std
        sample["target"] = (sample["target"] - mean) / std
        return sample

    def _wds_pipeline(self, shards: Sequence[str]) -> Iterable[Dict[str, Any]]:
        dataset = wds.WebDataset(shards, resampled=True)
        if self.config.shuffle_shards:
            dataset = dataset.shuffle(self.config.shard_shuffle_buffer)
        dataset = dataset.decode("msgpack").map(lambda sample: sample["msgpack"])
        if self.config.sample_shuffle_buffer > 0:
            dataset = dataset.shuffle(self.config.sample_shuffle_buffer)
        return dataset

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_rank = worker.id if worker else 0
        worker_world_size = worker.num_workers if worker else 1

        shards = self._get_shards()
        shards = shards[worker_rank :: worker_world_size]

        pipeline = self._wds_pipeline(shards)
        loader = wds.WebLoader(pipeline, batch_size=None, num_workers=0)
        if self.config.prefetch:
            loader = loader.prefetch(self.config.prefetch)
        for packed in loader:
            input_arr = packed["input"]
            target_arr = packed["target"]
            timestamps = packed.get("timestamps")
            metadata = packed.get("metadata", {})
            input_tensor = torch.tensor(input_arr, dtype=torch.float32)
            target_tensor = torch.tensor(target_arr, dtype=torch.float32)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
                target_tensor = target_tensor.unsqueeze(0)

            # Build rolling windows
            t_len = input_tensor.shape[0]
            seq_len = self.config.sequence_length
            for start in range(0, max(t_len - seq_len + 1, 1)):
                end = start + seq_len
                window_input = input_tensor[start:end]
                window_target = target_tensor[start:end]
                sample = {
                    "input": window_input,
                    "target": window_target,
                    "timestamps": timestamps[start:end] if isinstance(timestamps, list) else timestamps,
                    "metadata": metadata,
                    "dt": torch.tensor(self.dt_hours, dtype=torch.float32),
                }
                sample = self._normalize(sample)
                yield sample


def prepare_shards(
    zarr_path: str,
    output_pattern: str,
    samples_per_shard: int = 1024,
    keys: Optional[Sequence[str]] = None,
) -> None:
    """Convert ERA5 Zarr archive to WebDataset shards."""
    import msgpack

    ds = xr.open_zarr(zarr_path)
    variables = keys or list(ds.data_vars)
    total = ds.sizes["time"]
    shard_idx = 0
    os.makedirs(os.path.dirname(output_pattern), exist_ok=True)
    buffer: List[bytes] = []

    def _flush(buf: List[bytes], idx: int) -> None:
        if not buf:
            return
        shard_name = output_pattern.format(idx)
        with wds.TarWriter(shard_name) as sink:
            for i, payload in enumerate(buf):
                sink.write({"__key__": f"{i:06d}", "msgpack": payload})

    for start in range(0, total, samples_per_shard):
        end = min(start + samples_per_shard, total)
        chunk = ds.isel(time=slice(start, end))
        for i in range(end - start):
            sample = {var: chunk[var].isel(time=i).values for var in variables}
            payload = msgpack.packb(
                {
                    "input": sample.get("input"),
                    "target": sample.get("target"),
                    "timestamps": sample.get("time"),
                    "metadata": {"source": "era5"},
                }
            )
            buffer.append(payload)
        _flush(buffer, shard_idx)
        shard_idx += 1
        buffer = []
