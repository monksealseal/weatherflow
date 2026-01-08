"""Sharding utilities for deterministic sampling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import xarray as xr


@dataclass(frozen=True)
class ShardEntry:
    """Metadata for a single shard file."""

    path: str
    start_time: str
    end_time: str
    count: int
    checksum: str


@dataclass(frozen=True)
class ShardManifest:
    """Manifest for reproducible sharded sampling."""

    version: str
    created_at: str
    time_dim: str
    variables: tuple[str, ...]
    shards: tuple[ShardEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "time_dim": self.time_dim,
            "variables": list(self.variables),
            "shards": [entry.__dict__ for entry in self.shards],
        }

    def save(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "shard_manifest.json"
        manifest_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return manifest_path

    @classmethod
    def load(cls, manifest_path: str | Path) -> "ShardManifest":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        shards = tuple(ShardEntry(**entry) for entry in payload["shards"])
        return cls(
            version=payload["version"],
            created_at=payload["created_at"],
            time_dim=payload["time_dim"],
            variables=tuple(payload["variables"]),
            shards=shards,
        )


def write_time_shards(
    dataset: xr.Dataset,
    output_dir: str | Path,
    shard_size: int,
    time_dim: str = "time",
    version: str = "1",
    variables: Iterable[str] | None = None,
) -> ShardManifest:
    """Write time-contiguous shards to disk and return a manifest."""
    if shard_size <= 0:
        raise ValueError("shard_size must be a positive integer.")
    if time_dim not in dataset.dims:
        raise ValueError(f"Dataset is missing the '{time_dim}' dimension.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = dataset.sortby(time_dim)
    time_values = dataset[time_dim].values
    if len(time_values) == 0:
        raise ValueError("Dataset contains no time values.")

    if variables is None:
        variables = list(dataset.data_vars)

    shard_entries: list[ShardEntry] = []
    total = dataset.sizes[time_dim]
    shard_index = 0

    for start in range(0, total, shard_size):
        end = min(start + shard_size, total)
        shard = dataset[tuple(variables)].isel({time_dim: slice(start, end)})
        shard_path = output_path / f"shard_{shard_index:04d}.nc"
        shard.to_netcdf(shard_path)
        checksum = _hash_file(shard_path)

        shard_entries.append(
            ShardEntry(
                path=str(shard_path.name),
                start_time=str(time_values[start]),
                end_time=str(time_values[end - 1]),
                count=end - start,
                checksum=checksum,
            )
        )
        shard_index += 1

    manifest = ShardManifest(
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        time_dim=time_dim,
        variables=tuple(variables),
        shards=tuple(shard_entries),
    )
    manifest.save(output_path)
    return manifest


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
