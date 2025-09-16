"""Tests for the WeatherFlow FastAPI server."""
from fastapi.testclient import TestClient

from weatherflow.server.app import create_app

client = TestClient(create_app())


def test_options_endpoint() -> None:
    response = client.get("/api/options")
    assert response.status_code == 200
    payload = response.json()
    assert "variables" in payload
    assert payload["variables"]
    assert "pressureLevels" in payload


def test_experiment_endpoint() -> None:
    payload = {
        "dataset": {
            "variables": ["t", "z"],
            "pressureLevels": [500],
            "gridSize": {"lat": 8, "lon": 16},
            "trainSamples": 12,
            "valSamples": 6
        },
        "model": {
            "hiddenDim": 48,
            "nLayers": 2,
            "useAttention": False,
            "physicsInformed": True
        },
        "training": {
            "epochs": 1,
            "batchSize": 4,
            "learningRate": 0.001,
            "solverMethod": "rk4",
            "timeSteps": 3,
            "lossType": "mse",
            "seed": 7,
            "dynamicsScale": 0.1
        }
    }

    response = client.post("/api/experiments", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["config"]["dataset"]["trainSamples"] == payload["dataset"]["trainSamples"]
    assert data["metrics"]["train"][0]["epoch"] == 1
    assert data["prediction"]["channels"], "Channels should be returned"
    assert data["datasetSummary"]["channelStats"]
    assert data["execution"]["durationSeconds"] > 0
