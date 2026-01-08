from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class GaiaVariableSchema:
    variables: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_default(cls) -> "GaiaVariableSchema":
        schema_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "gaia"
            / "variables.yaml"
        )
        return cls.from_file(schema_path)

    @classmethod
    def from_file(cls, schema_path: str | Path) -> "GaiaVariableSchema":
        path = Path(schema_path)
        if not path.exists():
            raise FileNotFoundError(f"GAIA schema not found at {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        variables = payload.get("variables")
        if not isinstance(variables, Mapping):
            raise ValueError("GAIA schema must define a 'variables' mapping.")
        return cls(variables=variables)

    def validate(self, metadata: Mapping[str, Any]) -> None:
        if not isinstance(metadata, Mapping):
            raise TypeError("Metadata must be a mapping with a 'variables' section.")
        variables_meta = metadata.get("variables", {})
        if not isinstance(variables_meta, Mapping):
            raise TypeError("Metadata 'variables' entry must be a mapping.")

        expected_vars = set(self.variables)
        provided_vars = set(variables_meta)
        missing = expected_vars - provided_vars
        extra = provided_vars - expected_vars

        errors: list[str] = []
        if missing:
            errors.append(f"Missing variables: {', '.join(sorted(missing))}")
        if extra:
            errors.append(f"Unexpected variables: {', '.join(sorted(extra))}")

        for name in sorted(expected_vars & provided_vars):
            expected_info = self.variables[name]
            actual_info = variables_meta.get(name, {})
            if not isinstance(actual_info, Mapping):
                errors.append(
                    f"Variable '{name}' metadata must be a mapping, got {type(actual_info).__name__}"
                )
                continue

            expected_units = expected_info.get("units")
            if expected_units is not None:
                actual_units = actual_info.get("units")
                if actual_units != expected_units:
                    errors.append(
                        "Units mismatch for variable "
                        f"'{name}': expected '{expected_units}', got '{actual_units}'"
                    )

            expected_levels = expected_info.get("levels")
            if expected_levels is not None:
                actual_levels = actual_info.get("levels")
                if actual_levels is None:
                    errors.append(
                        f"Missing pressure levels for variable '{name}' (expected {expected_levels})"
                    )
                else:
                    expected_set = sorted({int(level) for level in expected_levels})
                    actual_set = sorted({int(level) for level in actual_levels})
                    if expected_set != actual_set:
                        errors.append(
                            "Pressure levels mismatch for variable "
                            f"'{name}': expected {expected_set}, got {actual_set}"
                        )

        if errors:
            message = "Schema validation failed:\n- " + "\n- ".join(errors)
            raise ValueError(message)
