import yaml
import pytest

from weatherflow.data.gaia.schema import GaiaVariableSchema


def _write_schema(tmp_path, payload):
    schema_path = tmp_path / "variables.yaml"
    schema_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return schema_path


def test_schema_reports_missing_variables(tmp_path):
    schema_path = _write_schema(
        tmp_path,
        {
            "variables": {
                "temperature": {"units": "K"},
                "humidity": {"units": "%"},
            }
        },
    )
    schema = GaiaVariableSchema.from_file(schema_path)

    metadata = {"variables": {"temperature": {"units": "K"}}}

    with pytest.raises(ValueError) as excinfo:
        schema.validate(metadata)

    assert "Missing variables: humidity" in str(excinfo.value)


def test_schema_reports_unexpected_units(tmp_path):
    schema_path = _write_schema(
        tmp_path,
        {"variables": {"temperature": {"units": "K"}}},
    )
    schema = GaiaVariableSchema.from_file(schema_path)

    metadata = {"variables": {"temperature": {"units": "C"}}}

    with pytest.raises(ValueError) as excinfo:
        schema.validate(metadata)

    message = str(excinfo.value)
    assert "Units mismatch" in message
    assert "temperature" in message
    assert "expected 'K'" in message


def test_schema_reports_pressure_level_mismatches(tmp_path):
    schema_path = _write_schema(
        tmp_path,
        {"variables": {"wind_u": {"units": "m/s", "levels": [1000, 850]}}},
    )
    schema = GaiaVariableSchema.from_file(schema_path)

    metadata = {"variables": {"wind_u": {"units": "m/s", "levels": [1000]}}}

    with pytest.raises(ValueError) as excinfo:
        schema.validate(metadata)

    message = str(excinfo.value)
    assert "Pressure levels mismatch" in message
    assert "expected [850, 1000]" in message
    assert "got [1000]" in message
