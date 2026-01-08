"""Pipeline utilities for Gaia inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml


@dataclass(frozen=True)
class VariableSchema:
    """Schema definition for a single variable."""

    name: str
    units: str


@dataclass(frozen=True)
class SchemaDefinition:
    """Schema definition for inference inputs."""

    variables: Tuple[VariableSchema, ...]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    @property
    def units(self) -> Tuple[str, ...]:
        return tuple(variable.units for variable in self.variables)


@dataclass(frozen=True)
class InferenceAssets:
    """Loaded assets required for inference."""

    checkpoint: Dict[str, Any]
    normalization_stats: Dict[str, Dict[str, float]]
    schema: SchemaDefinition


class SchemaValidationError(ValueError):
    """Raised when input schema validation fails."""


def load_checkpoint(path: Path, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    """Load a model checkpoint from disk."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location)


def load_normalization_stats(path: Path) -> Dict[str, Dict[str, float]]:
    """Load normalization statistics from JSON, YAML, or torch file."""
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    if stats_path.suffix in {".json"}:
        return json.loads(stats_path.read_text())

    if stats_path.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(stats_path.read_text())

    if stats_path.suffix in {".pt", ".pth"}:
        stats = torch.load(stats_path, map_location="cpu")
        if not isinstance(stats, dict):
            raise ValueError("Normalization stats must be a dictionary.")
        return stats

    raise ValueError(
        "Unsupported normalization stats format. Use JSON, YAML, or torch.")


def load_schema_definition(path: Path) -> SchemaDefinition:
    """Load schema definition from JSON or YAML."""
    schema_path = Path(path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema definition not found: {schema_path}")

    if schema_path.suffix in {".json"}:
        payload = json.loads(schema_path.read_text())
    elif schema_path.suffix in {".yml", ".yaml"}:
        payload = yaml.safe_load(schema_path.read_text())
    else:
        raise ValueError("Schema definition must be JSON or YAML.")

    variables = payload.get("variables")
    if not isinstance(variables, Iterable):
        raise ValueError("Schema definition missing 'variables' list.")

    parsed: List[VariableSchema] = []
    for entry in variables:
        if not isinstance(entry, Mapping):
            raise ValueError("Schema variable entries must be mappings.")
        name = entry.get("name")
        units = entry.get("units")
        if not isinstance(name, str) or not isinstance(units, str):
            raise ValueError("Schema variables must include name and units.")
        parsed.append(VariableSchema(name=name, units=units))

    return SchemaDefinition(variables=tuple(parsed))


def load_inference_assets(
    checkpoint_path: Path,
    normalization_path: Path,
    schema_path: Path,
    map_location: Optional[str] = "cpu",
) -> InferenceAssets:
    """Load checkpoints, normalization stats, and schema definition."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    normalization_stats = load_normalization_stats(normalization_path)
    schema = load_schema_definition(schema_path)
    return InferenceAssets(
        checkpoint=checkpoint,
        normalization_stats=normalization_stats,
        schema=schema,
    )


def _coerce_variables(value: Any) -> Tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(str(item) for item in value)
    return ()


def _coerce_units(value: Any) -> Tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(str(item) for item in value)
    return ()


def validate_schema(
    variables: Sequence[str],
    units: Sequence[str],
    schema: SchemaDefinition,
) -> None:
    """Validate variables and units against the expected schema."""
    expected_vars = schema.names
    expected_units = schema.units

    if tuple(variables) != expected_vars:
        raise SchemaValidationError(
            "Variable mismatch: expected "
            f"{list(expected_vars)}, received {list(variables)}."
        )

    if tuple(units) != expected_units:
        raise SchemaValidationError(
            "Units mismatch: expected "
            f"{list(expected_units)}, received {list(units)}."
        )


def validate_analysis_state(
    state: Mapping[str, Any],
    schema: SchemaDefinition,
    label: str,
) -> None:
    """Validate a single analysis state against schema requirements."""
    variables = _coerce_variables(state.get("variables"))
    units = _coerce_units(state.get("units"))

    if not variables:
        raise SchemaValidationError(
            f"{label} missing required 'variables' list in analysis state."
        )
    if not units:
        raise SchemaValidationError(
            f"{label} missing required 'units' list in analysis state."
        )

    validate_schema(variables, units, schema)


def prepare_inference_inputs(
    analysis_state_t0: Mapping[str, Any],
    analysis_state_t1: Mapping[str, Any],
    schema: SchemaDefinition,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate and return analysis states for inference entry."""
    validate_analysis_state(analysis_state_t0, schema, label="analysis_state_t0")
    validate_analysis_state(analysis_state_t1, schema, label="analysis_state_t1")
    return analysis_state_t0, analysis_state_t1
