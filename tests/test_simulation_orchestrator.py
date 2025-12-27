"""Tests for the simulation orchestrator utilities."""
from types import SimpleNamespace

import torch

from weatherflow.simulation import SimulationOrchestrator


def test_resolve_grid_size_defaults_to_tier() -> None:
    orchestrator = SimulationOrchestrator()
    lat, lon, dt = orchestrator.resolve_grid_size(10, 12, "regional")
    assert (lat, lon) == (64, 128)
    assert dt > 0


def test_moisture_and_flux_application() -> None:
    orchestrator = SimulationOrchestrator()
    field = torch.zeros(2, 4, 4)
    moisture_cfg = SimpleNamespace(enable=True, condensation_threshold=0.4)
    flux_cfg = SimpleNamespace(latent_coeff=0.2, sensible_coeff=0.1, drag_coeff=0.05)
    updated = orchestrator.apply_moisture_and_surface_flux(field, moisture_cfg, flux_cfg)
    assert updated.shape == field.shape
    assert torch.isfinite(updated).all()


def test_lod_streamer_tiles_created() -> None:
    orchestrator = SimulationOrchestrator()
    field = torch.randn(3, 6, 6)
    lod_cfg = SimpleNamespace(min_chunk=2, max_chunk=4, overlap=0, max_zoom=1)
    description = orchestrator.stream_level_of_detail(field, lod_cfg)
    assert description["chunkShape"] == [4, 4]
    assert description["tiles"]
