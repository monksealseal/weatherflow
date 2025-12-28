"""WebDataset-based ERA5 loader with local staging/prefetch."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import webdataset as wds


def _collate(samples):
    xs = [s["x"] for s in samples]
    ys = [s["y"] for s in samples]
    return torch.stack(xs), torch.stack(ys)


def create_webdataset_loader(
    shard_pattern: str,
    batch_size: int = 16,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    resampled: bool = True,
) -> Iterable:
    """
    Create a DataLoader-like iterator over WebDataset shards.

    Args:
        shard_pattern: Pattern like 'data/era5/{0000..0100}.tar'
        batch_size: Batch size.
        num_workers: Number of worker threads for decoding.
        cache_dir: Optional cache directory (uses gcs/http filesystem transparently).
        shuffle: Whether to shuffle samples.
        resampled: Use RepeatedDataset for infinite shards.
    """
    shards = shard_pattern
    if resampled:
        dataset = wds.ResampledShards(shards, deterministic=True)
    else:
        dataset = wds.WebDataset(shards, cache_dir=cache_dir)

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    dataset = dataset.to_tuple("x.pth", "y.pth")
    dataset = dataset.map_tuple(torch.load, torch.load)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batched(batch_size, collation_fn=_collate)
    loader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
    return loader
