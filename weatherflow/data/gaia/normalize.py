"""Normalization utilities for deterministic GAIA pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import xarray as xr


@dataclass(frozen=True)
class NormalizationStats:
    """Per-variable normalization statistics computed on training data."""

    version: str
    variables: tuple[str, ...]
    mean: dict[str, float]
    std: dict[str, float]
    computed_at: str

    @classmethod
    def compute(
        cls,
        dataset: xr.Dataset,
        variables: Iterable[str],
        split: str,
        version: str,
    ) -> "NormalizationStats":
        """Compute normalization stats on the training split only."""
        if split != "train":
            raise ValueError("Normalization stats must be computed on the training split.")

        variable_list = tuple(variables)
        if not variable_list:
            raise ValueError("At least one variable must be provided.")

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for name in variable_list:
            data = dataset[name]
            means[name] = float(data.mean().item())
            std = float(data.std().item())
            stds[name] = std if std != 0 else 1e-6

        computed_at = datetime.now(timezone.utc).isoformat()
        return cls(
            version=version,
            variables=variable_list,
            mean=means,
            std=stds,
            computed_at=computed_at,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the stats to a dictionary."""
        payload = asdict(self)
        payload["variables"] = list(self.variables)
        return payload

    def save(self, output_dir: str | Path) -> Path:
        """Save stats to a versioned artifact on disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        artifact_path = output_path / f"normalization_stats_v{self.version}.json"
        artifact_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return artifact_path

    @classmethod
    def load(cls, artifact_path: str | Path) -> "NormalizationStats":
        """Load normalization stats from a versioned artifact."""
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        return cls(
            version=str(payload["version"]),
            variables=tuple(payload["variables"]),
            mean={key: float(value) for key, value in payload["mean"].items()},
            std={key: float(value) for key, value in payload["std"].items()},
            computed_at=str(payload["computed_at"]),
        )

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        """Normalize a dataset using stored statistics."""
        normalized = dataset.copy()
        for name in self.variables:
            mean = self.mean[name]
            std = self.std[name]
            normalized[name] = (normalized[name] - mean) / std
        return normalized

    def denormalize(self, dataset: xr.Dataset) -> xr.Dataset:
        """Invert normalization using stored statistics."""
        restored = dataset.copy()
        for name in self.variables:
            mean = self.mean[name]
            std = self.std[name]
            restored[name] = restored[name] * std + mean
        return restored
