"""FastAPI application exposing WeatherFlow experimentation utilities."""
from __future__ import annotations

import time
import uuid
import math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.utils.data import DataLoader, TensorDataset

from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE
from weatherflow.models.icosahedral import IcosahedralFlowMatch
from weatherflow.data.webdataset_loader import create_webdataset_loader
from weatherflow.simulation import SimulationOrchestrator

# Limit CPU usage for deterministic behaviour when running inside tests
TORCH_NUM_THREADS = 1

torch.set_num_threads(TORCH_NUM_THREADS)

DEFAULT_VARIABLES = ["t", "z", "u", "v"]
DEFAULT_PRESSURE_LEVELS = [1000, 850, 700, 500, 300, 200]
DEFAULT_GRID_SIZES = [(16, 32), (32, 64)]
DEFAULT_SOLVER_METHODS = ["dopri5", "rk4", "midpoint"]
DEFAULT_LOSS_TYPES = ["mse", "huber", "smooth_l1"]
SIMULATION_ORCHESTRATOR = SimulationOrchestrator()


class CamelModel(BaseModel):
    """Base model enabling population by field name or alias."""

    model_config = ConfigDict(populate_by_name=True)


class GridSize(CamelModel):
    """Simple grid size model for validation."""

    lat: int = Field(16, ge=4, le=128)
    lon: int = Field(32, ge=4, le=256)

    @field_validator("lon")
    @classmethod
    def lon_multiple_of_two(cls, value: int) -> int:  # noqa: D401
        """Ensure longitude dimension is even for nicer plots."""
        if value % 2 != 0:
            raise ValueError("Longitude must be an even number")
        return value


class TimeControlConfig(CamelModel):
    """Time-stepping and replay configuration."""

    step_seconds: int = Field(300, ge=30, le=3600, alias="stepSeconds")
    replay_length_seconds: int = Field(
        1800, ge=60, le=86400, alias="replayLengthSeconds"
    )
    boundary_update_seconds: int = Field(
        900, ge=60, le=86400, alias="boundaryUpdateSeconds"
    )


class MoistureConfig(CamelModel):
    """Moisture and phase-change proxy settings."""

    enable: bool = True
    condensation_threshold: float = Field(
        0.55, ge=0.0, le=1.0, alias="condensationThreshold"
    )
    condensation_rate: float = Field(0.12, ge=0.0, le=1.0, alias="condensationRate")
    evaporation_rate: float = Field(0.05, ge=0.0, le=1.0, alias="evaporationRate")
    cloud_entrainment: float = Field(0.1, ge=0.0, le=1.0, alias="cloudEntrainment")


class SurfaceFluxConfig(CamelModel):
    """Surface flux scheme optimized for real-time budgets."""

    latent_coeff: float = Field(0.35, ge=0.0, le=2.0, alias="latentCoeff")
    sensible_coeff: float = Field(0.2, ge=0.0, le=2.0, alias="sensibleCoeff")
    drag_coeff: float = Field(0.05, ge=0.0, le=1.0, alias="dragCoeff")
    optimized_for_real_time: bool = Field(True, alias="optimizedForRealTime")


class LODConfig(CamelModel):
    """Level-of-detail streaming configuration."""

    min_chunk: int = Field(8, ge=1, le=128, alias="minChunk")
    max_chunk: int = Field(48, ge=1, le=256, alias="maxChunk")
    overlap: int = Field(2, ge=0, le=32)
    max_zoom: int = Field(3, ge=0, le=6, alias="maxZoom")


class SimulationConfig(CamelModel):
    """Simulation core selection and grid tuning."""

    core: str = Field("shallow-water")
    resolution_tier: str = Field("custom", alias="resolutionTier")
    initial_source: str = Field("reanalysis", alias="initialSource")
    boundary_source: str = Field("reanalysis", alias="boundarySource")
    seed: int = Field(0, ge=0, le=100000)
    time_control: TimeControlConfig = Field(
        default_factory=TimeControlConfig, alias="timeControl"
    )
    moisture: MoistureConfig = Field(default_factory=MoistureConfig)
    surface_flux: SurfaceFluxConfig = Field(
        default_factory=SurfaceFluxConfig, alias="surfaceFlux"
    )
    lod: LODConfig = Field(default_factory=LODConfig)

    @field_validator("core")
    @classmethod
    def validate_core(cls, value: str) -> str:  # noqa: D401
        """Ensure the requested simulation core is supported."""
        if value not in SIMULATION_ORCHESTRATOR.cores:
            raise ValueError(f"Unsupported core '{value}'")
        return value

    @field_validator("resolution_tier")
    @classmethod
    def validate_tier(cls, value: str) -> str:  # noqa: D401
        """Ensure requested resolution tier exists."""
        if value not in SIMULATION_ORCHESTRATOR.resolution_tiers:
            raise ValueError(f"Unsupported resolution tier '{value}'")
        return value


class DatasetConfig(CamelModel):
    """Configuration options for generating synthetic datasets."""

    variables: List[str] = Field(default_factory=lambda: DEFAULT_VARIABLES[:2])
    pressure_levels: List[int] = Field(
        default_factory=lambda: [500], alias="pressureLevels"
    )
    grid_size: GridSize = Field(default_factory=GridSize, alias="gridSize")
    train_samples: int = Field(48, ge=4, le=256, alias="trainSamples")
    val_samples: int = Field(16, ge=4, le=128, alias="valSamples")
    webdataset_pattern: str | None = Field(None, alias="webdatasetPattern")
    webdataset_cache: str | None = Field(None, alias="webdatasetCache")
    webdataset_workers: int = Field(2, ge=0, le=16, alias="webdatasetWorkers")

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, values: List[str]) -> List[str]:  # noqa: D401
        """Ensure at least one variable was selected."""
        if not values:
            raise ValueError("At least one variable must be selected")
        for var in values:
            if var not in DEFAULT_VARIABLES:
                raise ValueError(f"Unsupported variable '{var}'")
        return values

    @field_validator("pressure_levels")
    @classmethod
    def validate_pressure_levels(cls, values: List[int]) -> List[int]:  # noqa: D401
        """Ensure at least one pressure level is available."""
        if not values:
            raise ValueError("Select at least one pressure level")
        return values


class ModelConfig(CamelModel):
    """Neural network hyperparameters."""

    backbone: str = Field("icosahedral")
    hidden_dim: int = Field(96, ge=32, le=512, alias="hiddenDim")
    n_layers: int = Field(3, ge=1, le=8, alias="nLayers")
    use_attention: bool = Field(True, alias="useAttention")
    physics_informed: bool = Field(True, alias="physicsInformed")
    window_size: int = Field(8, ge=0, le=64, alias="windowSize")
    spherical_padding: bool = Field(False, alias="sphericalPadding")
    use_graph_mp: bool = Field(False, alias="useGraphMp")
    subdivisions: int = Field(1, ge=0, le=3, alias="subdivisions")
    interp_cache_dir: str | None = Field(None, alias="interpCacheDir")
    @field_validator("backbone")
    @classmethod
    def validate_backbone(cls, value: str) -> str:  # noqa: D401
        """Ensure a supported backbone is selected."""
        if value not in {"grid", "icosahedral"}:
            raise ValueError("backbone must be one of ['grid', 'icosahedral']")
        return value


class TrainingConfig(CamelModel):
    """Training loop configuration."""

    epochs: int = Field(2, ge=1, le=6)
    batch_size: int = Field(8, ge=1, le=64, alias="batchSize")
    learning_rate: float = Field(5e-4, gt=0, le=1e-2, alias="learningRate")
    solver_method: str = Field("dopri5", alias="solverMethod")
    time_steps: int = Field(5, ge=3, le=12, alias="timeSteps")
    loss_type: str = Field("mse", alias="lossType")
    seed: int = Field(42, ge=0, le=10_000)
    dynamics_scale: float = Field(0.15, gt=0.01, le=0.5, alias="dynamicsScale")
    rollout_steps: int = Field(3, ge=2, le=12, alias="rolloutSteps")
    rollout_weight: float = Field(0.3, ge=0.0, le=5.0, alias="rolloutWeight")

    @field_validator("solver_method")
    @classmethod
    def solver_method_supported(cls, value: str) -> str:  # noqa: D401
        """Ensure the requested ODE solver is available."""
        if value not in DEFAULT_SOLVER_METHODS:
            raise ValueError(f"Unsupported solver '{value}'")
        return value

    @field_validator("loss_type")
    @classmethod
    def loss_type_supported(cls, value: str) -> str:  # noqa: D401
        """Ensure the loss type is compatible with the training loop."""
        if value not in DEFAULT_LOSS_TYPES:
            raise ValueError(f"Unsupported loss '{value}'")
        return value


class InferenceConfig(CamelModel):
    """Inference configuration for tiling large grids."""

    tile_size_lat: int = Field(0, ge=0, le=512, alias="tileSizeLat")
    tile_size_lon: int = Field(0, ge=0, le=1024, alias="tileSizeLon")
    tile_overlap: int = Field(0, ge=0, le=64, alias="tileOverlap")


class ExperimentConfig(CamelModel):
    """Bundled configuration used by the API endpoint."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


class ChannelStats(CamelModel):
    name: str
    mean: float
    std: float
    min: float
    max: float


class MetricEntry(CamelModel):
    epoch: int
    loss: float
    flow_loss: float = Field(alias="flowLoss")
    divergence_loss: float = Field(alias="divergenceLoss")
    rollout_loss: float = Field(0.0, alias="rolloutLoss")
    energy_diff: float = Field(alias="energyDiff")


class ValidationMetricEntry(CamelModel):
    epoch: int
    val_loss: float
    val_flow_loss: float = Field(alias="valFlowLoss")
    val_divergence_loss: float = Field(alias="valDivergenceLoss")
    val_rollout_loss: float = Field(0.0, alias="valRolloutLoss")
    val_energy_diff: float = Field(alias="valEnergyDiff")


class TrajectoryStep(CamelModel):
    time: float
    data: List[List[float]]


class ChannelTrajectory(CamelModel):
    name: str
    initial: List[List[float]]
    target: List[List[float]]
    trajectory: List[TrajectoryStep]
    rmse: float
    mae: float
    baseline_rmse: float = Field(alias="baselineRmse")


class PredictionResult(CamelModel):
    times: List[float]
    channels: List[ChannelTrajectory]


class LODTile(CamelModel):
    level: int
    tile: str
    lat_start: int = Field(alias="latStart")
    lat_end: int = Field(alias="latEnd")
    lon_start: int = Field(alias="lonStart")
    lon_end: int = Field(alias="lonEnd")
    mean: float
    std: float


class LODPreview(CamelModel):
    chunk_shape: List[int] = Field(alias="chunkShape")
    tiles: List[LODTile]


class ExecutionSummary(CamelModel):
    duration_seconds: float = Field(alias="durationSeconds")


class DatasetSummary(CamelModel):
    channel_stats: List[ChannelStats] = Field(alias="channelStats")
    sample_shape: List[int] = Field(alias="sampleShape")


class SimulationSummary(CamelModel):
    core: str
    resolution_tier: str = Field(alias="resolutionTier")
    grid: GridSize
    time_step_seconds: int = Field(alias="timeStepSeconds")


class ExperimentResult(CamelModel):
    experiment_id: str = Field(alias="experimentId")
    config: ExperimentConfig
    channel_names: List[str] = Field(alias="channelNames")
    metrics: Dict[str, List[MetricEntry]]
    validation: Dict[str, List[ValidationMetricEntry]]
    dataset_summary: DatasetSummary = Field(alias="datasetSummary")
    prediction: PredictionResult
    lod_preview: LODPreview = Field(alias="lodPreview")
    simulation_summary: SimulationSummary = Field(alias="simulationSummary")
    execution: ExecutionSummary


def _channel_names(dataset: DatasetConfig) -> List[str]:
    names: List[str] = []
    for var in dataset.variables:
        for level in dataset.pressure_levels:
            names.append(f"{var}@{level}")
    return names


def _build_dataloaders(
    config: DatasetConfig,
    dynamics_scale: float,
    simulation: SimulationConfig,
    orchestrator: SimulationOrchestrator,
    device: torch.device,
    generator: torch.Generator,
) -> Dict[str, object]:
    """Create lightweight synthetic datasets for demonstration purposes."""
    if config.webdataset_pattern:
        loader = create_webdataset_loader(
            config.webdataset_pattern,
            batch_size=config.train_samples,
            num_workers=config.webdataset_workers,
            shuffle=True,
            cache_dir=config.webdataset_cache,
        )
        batch = next(iter(loader))
        train_x0, train_x1 = batch
        val_loader = create_webdataset_loader(
            config.webdataset_pattern,
            batch_size=config.val_samples,
            num_workers=config.webdataset_workers,
            shuffle=True,
            resampled=False,
            cache_dir=config.webdataset_cache,
        )
        val_x0, val_x1 = next(iter(val_loader))
        b_train, _, lat, lon = train_x0.shape
        static_features = orchestrator.build_static_features(lat, lon, device=device).unsqueeze(0).repeat(
            b_train, 1, 1, 1
        )
        forcing = orchestrator.build_forcing(b_train, device=device)
        b_val = val_x0.shape[0]
        static_val = orchestrator.build_static_features(lat, lon, device=device).unsqueeze(0).repeat(
            b_val, 1, 1, 1
        )
        forcing_val = orchestrator.build_forcing(b_val, device=device)

        train_dataset = TensorDataset(train_x0, train_x1, static_features, forcing)
        val_dataset = TensorDataset(val_x0, val_x1, static_val, forcing_val)
        channel_names = _channel_names(config)
        return {
            "train": train_dataset,
            "val": val_dataset,
            "channel_names": channel_names,
            "grid": GridSize(lat=lat, lon=lon),
            "time_step_seconds": 300,
        }
    channel_names = _channel_names(config)
    channels = len(channel_names)
    lat, lon, time_step_seconds = orchestrator.resolve_grid_size(
        config.grid_size.lat, config.grid_size.lon, simulation.resolution_tier
    )

    base_state = orchestrator.seed_initial_state(
        channels,
        (lat, lon),
        simulation.initial_source,
        simulation.boundary_source,
        seed=simulation.seed,
        device=device,
    )
    static_features = orchestrator.build_static_features(lat, lon, device=device)

    def _synth_samples(
        num_samples: int, start_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0_list: List[torch.Tensor] = []
        x1_list: List[torch.Tensor] = []
        for sample_idx in range(num_samples):
            noisy_state = base_state + torch.randn_like(
                base_state, generator=generator
            ) * dynamics_scale
            stepped = orchestrator.simulate_time_step(
                noisy_state,
                simulation.core,
                simulation.time_control.step_seconds,
                dynamics_scale,
                simulation.time_control.replay_length_seconds,
                simulation.boundary_source,
                simulation.time_control.boundary_update_seconds,
                step_idx=start_idx + sample_idx,
            )
            stepped = orchestrator.apply_moisture_and_surface_flux(
                stepped, simulation.moisture, simulation.surface_flux
            )
            x0_list.append(noisy_state)
            x1_list.append(stepped)

        return torch.stack(x0_list), torch.stack(x1_list)

    train_x0, train_x1 = _synth_samples(config.train_samples, 0)
    val_x0, val_x1 = _synth_samples(config.val_samples, config.train_samples)

    forcing_train = orchestrator.build_forcing(train_x0.shape[0], device=device)
    forcing_val = orchestrator.build_forcing(val_x0.shape[0], device=device)

    static_train = static_features.unsqueeze(0).repeat(train_x0.shape[0], 1, 1, 1)
    static_val = static_features.unsqueeze(0).repeat(val_x0.shape[0], 1, 1, 1)

    train_dataset = TensorDataset(train_x0, train_x1, static_train, forcing_train)
    val_dataset = TensorDataset(val_x0, val_x1, static_val, forcing_val)
    return {
        "train": train_dataset,
        "val": val_dataset,
        "channel_names": channel_names,
        "grid": GridSize(lat=lat, lon=lon),
        "time_step_seconds": time_step_seconds,
    }


def _aggregate_channel_stats(
    data: torch.Tensor, names: List[str]
) -> List[ChannelStats]:
    """Compute simple summary statistics per channel."""
    stats: List[ChannelStats] = []
    reshaped = data.reshape(data.shape[0], data.shape[1], -1)
    for idx, name in enumerate(names):
        channel = reshaped[:, idx]
        stats.append(
            ChannelStats(
                name=name,
                mean=float(channel.mean()),
                std=float(channel.std(unbiased=False)),
                min=float(channel.min()),
                max=float(channel.max()),
            )
        )
    return stats


def _compute_losses(
    model: WeatherFlowMatch,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    loss_type: str,
    rollout_steps: int = 0,
    rollout_weight: float = 0.0,
    ode_model: WeatherFlowODE | None = None,
    static: torch.Tensor | None = None,
    forcing: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute flow matching loss with optional physics constraints."""

    v_pred = model(x0, t, static=static, forcing=forcing)
    target_velocity = (x1 - x0) / (1 - t).view(-1, 1, 1, 1)

    if loss_type == "huber":
        flow_loss = F.huber_loss(v_pred, target_velocity, delta=1.0)
    elif loss_type == "smooth_l1":
        flow_loss = F.smooth_l1_loss(v_pred, target_velocity)
    else:
        flow_loss = F.mse_loss(v_pred, target_velocity)

    losses: Dict[str, torch.Tensor] = {
        "flow_loss": flow_loss,
        "total_loss": flow_loss,
        "rollout_loss": torch.tensor(0.0, device=x0.device),
    }

    if rollout_steps > 1 and rollout_weight > 0.0 and ode_model is not None:
        times = torch.linspace(0.0, 1.0, rollout_steps, device=x0.device)
        rollout = ode_model(x0, times, static=static, forcing=forcing)
        rollout_pred = rollout[-1, 0]
        rollout_loss = F.mse_loss(rollout_pred, x1)
        losses["rollout_loss"] = rollout_loss
        losses["total_loss"] = losses["total_loss"] + rollout_weight * rollout_loss

    if model.physics_informed:
        div_loss = torch.tensor(0.0, device=x0.device)
        if v_pred.shape[1] >= 2:
            u = v_pred[:, 0:1]
            v_comp = v_pred[:, 1:2]
            du_dx = torch.gradient(u, dim=3)[0]
            dv_dy = torch.gradient(v_comp, dim=2)[0]
            div = du_dx + dv_dy
            div_loss = torch.mean(div**2)

        if hasattr(model, "compute_mesh_laplacian_loss"):
            mesh_loss = model.compute_mesh_laplacian_loss(v_pred)
            div_loss = div_loss + mesh_loss

        losses["div_loss"] = div_loss
        losses["total_loss"] = losses["total_loss"] + 0.1 * div_loss

        energy_x0 = torch.sum(x0**2)
        energy_x1 = torch.sum(x1**2)
        energy_diff = (energy_x0 - energy_x1).abs() / (energy_x0 + 1e-6)
        losses["energy_diff"] = energy_diff

    return losses


def _prepare_trajectory(
    predictions: torch.Tensor,
    initial: torch.Tensor,
    target: torch.Tensor,
    times: torch.Tensor,
    names: List[str],
) -> PredictionResult:
    channels: List[ChannelTrajectory] = []
    for channel_idx, name in enumerate(names):
        channel_predictions = predictions[:, 0, channel_idx].detach().cpu()
        channel_initial = initial[0, channel_idx].detach().cpu()
        channel_target = target[0, channel_idx].detach().cpu()

        rmse = torch.sqrt(
            torch.mean((channel_predictions[-1] - channel_target) ** 2)
        ).item()
        mae = torch.mean(torch.abs(channel_predictions[-1] - channel_target)).item()
        baseline_rmse = torch.sqrt(
            torch.mean((channel_initial - channel_target) ** 2)
        ).item()

        trajectory = [
            TrajectoryStep(
                time=float(times[i].item()), data=channel_predictions[i].tolist()
            )
            for i in range(len(times))
        ]

        channels.append(
            ChannelTrajectory(
                name=name,
                initial=channel_initial.tolist(),
                target=channel_target.tolist(),
                trajectory=trajectory,
                rmse=float(rmse),
                mae=float(mae),
                baseline_rmse=float(baseline_rmse),
            )
        )

    return PredictionResult(
        times=[float(t.item()) for t in times],
        channels=channels,
    )


def _train_model(
    config: ExperimentConfig,
    device: torch.device,
    datasets: Dict[str, object],
    generator: torch.Generator,
    loader_generator: torch.Generator,
) -> Dict[str, object]:
    channel_names: List[str] = datasets["channel_names"]
    train_dataset: TensorDataset = datasets["train"]
    val_dataset: TensorDataset = datasets["val"]
    resolved_grid: GridSize = datasets.get("grid", config.dataset.grid_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.training.batch_size, len(train_dataset)),
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.training.batch_size, len(val_dataset)),
        shuffle=False,
        generator=loader_generator,
    )

    if config.model.backbone == "icosahedral":
        model = IcosahedralFlowMatch(
            input_channels=len(channel_names),
            hidden_dim=config.model.hidden_dim,
            n_layers=config.model.n_layers,
            subdivisions=config.model.subdivisions,
            interp_cache_dir=config.model.interp_cache_dir,
        ).to(device)
    else:
        model = WeatherFlowMatch(
            input_channels=len(channel_names),
            hidden_dim=config.model.hidden_dim,
            n_layers=config.model.n_layers,
            use_attention=config.model.use_attention,
            grid_size=(resolved_grid.lat, resolved_grid.lon),
            physics_informed=config.model.physics_informed,
            window_size=config.model.window_size,
            static_channels=2,
            forcing_dim=1,
            spherical_padding=config.model.spherical_padding,
            use_graph_mp=config.model.use_graph_mp,
        ).to(device)
    ode_model = WeatherFlowODE(
        model,
        solver_method=config.training.solver_method,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    train_metrics: List[MetricEntry] = []
    val_metrics: List[ValidationMetricEntry] = []

    for epoch in range(config.training.epochs):
        model.train()
        train_loss = []
        train_flow = []
        train_div = []
        train_rollout = []
        train_energy = []

        for x0, x1, static_batch, forcing_batch in train_loader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            static_batch = static_batch.to(device)
            forcing_batch = forcing_batch.to(device)
            t = torch.rand(x0.size(0), device=device, generator=generator)

            losses = _compute_losses(
                model,
                x0,
                x1,
                t,
                config.training.loss_type,
                rollout_steps=config.training.rollout_steps,
                rollout_weight=config.training.rollout_weight,
                ode_model=ode_model,
                static=static_batch,
                forcing=forcing_batch,
            )
            total_loss = losses["total_loss"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss.append(float(total_loss.item()))
            train_flow.append(float(losses["flow_loss"].item()))
            train_div.append(float(losses.get("div_loss", torch.tensor(0.0)).item()))
            train_energy.append(
                float(losses.get("energy_diff", torch.tensor(0.0)).item())
            )
            train_rollout.append(float(losses.get("rollout_loss", torch.tensor(0.0)).item()))

        if not train_loss:
            raise RuntimeError("Training dataset is empty")

        train_metrics.append(
            MetricEntry(
                epoch=epoch + 1,
                loss=float(sum(train_loss) / len(train_loss)),
                flow_loss=float(sum(train_flow) / len(train_flow)),
                divergence_loss=float(sum(train_div) / len(train_div)),
                rollout_loss=float(sum(train_rollout) / max(len(train_rollout), 1)),
                energy_diff=float(sum(train_energy) / len(train_energy)),
            )
        )

        model.eval()
        val_loss = []
        val_flow = []
        val_div = []
        val_rollout = []
        val_energy = []

        with torch.no_grad():
            for x0, x1, static_batch, forcing_batch in val_loader:
                x0 = x0.to(device)
                x1 = x1.to(device)
                static_batch = static_batch.to(device)
                forcing_batch = forcing_batch.to(device)
                t = torch.rand(x0.size(0), device=device, generator=generator)

                losses = _compute_losses(
                    model,
                    x0,
                    x1,
                    t,
                    config.training.loss_type,
                    rollout_steps=config.training.rollout_steps,
                    rollout_weight=config.training.rollout_weight,
                    ode_model=ode_model,
                    static=static_batch,
                    forcing=forcing_batch,
                )
                total_loss = losses["total_loss"]

                val_loss.append(float(total_loss.item()))
                val_flow.append(float(losses["flow_loss"].item()))
                val_div.append(float(losses.get("div_loss", torch.tensor(0.0)).item()))
                val_energy.append(
                    float(losses.get("energy_diff", torch.tensor(0.0)).item())
                )
                val_rollout.append(float(losses.get("rollout_loss", torch.tensor(0.0)).item()))

        val_metrics.append(
            ValidationMetricEntry(
                epoch=epoch + 1,
                val_loss=float(sum(val_loss) / len(val_loss)),
                val_flow_loss=float(sum(val_flow) / len(val_flow)),
                val_divergence_loss=float(sum(val_div) / len(val_div)),
                val_rollout_loss=float(sum(val_rollout) / max(len(val_rollout), 1)),
                val_energy_diff=float(sum(val_energy) / len(val_energy)),
            )
        )

    return {
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def _run_prediction(
    model: WeatherFlowMatch,
    config: ExperimentConfig,
    dataset: TensorDataset,
    channel_names: List[str],
    device: torch.device,
) -> Tuple[PredictionResult, torch.Tensor]:
    model.eval()
    ode_model = WeatherFlowODE(
        model,
        solver_method=config.training.solver_method,
    ).to(device)
    times = torch.linspace(0.0, 1.0, config.training.time_steps, device=device)

    initial = dataset.tensors[0][:1].to(device)
    target = dataset.tensors[1][:1].to(device)
    static = dataset.tensors[2][:1].to(device) if len(dataset.tensors) > 2 else None
    forcing = dataset.tensors[3][:1].to(device) if len(dataset.tensors) > 3 else None

    def _tile_slices(lat: int, lon: int, tile_lat: int, tile_lon: int, overlap: int) -> List[Tuple[slice, slice]]:
        if tile_lat <= 0 or tile_lon <= 0:
            return [(slice(0, lat), slice(0, lon))]
        step_lat = max(1, tile_lat - overlap)
        step_lon = max(1, tile_lon - overlap)
        slices: List[Tuple[slice, slice]] = []
        for lat_start in range(0, lat, step_lat):
            lat_end = min(lat, lat_start + tile_lat)
            for lon_start in range(0, lon, step_lon):
                lon_end = min(lon, lon_start + tile_lon)
                slices.append((slice(lat_start, lat_end), slice(lon_start, lon_end)))
        return slices

    def _run_tiled_prediction(x: torch.Tensor) -> torch.Tensor:
        b, c, lat, lon = x.shape
        tile_lat = config.inference.tile_size_lat
        tile_lon = config.inference.tile_size_lon
        overlap = config.inference.tile_overlap
        slices = _tile_slices(lat, lon, tile_lat, tile_lon, overlap)
        if len(slices) == 1:
            return ode_model(x, times, static=static, forcing=forcing)

        preds = torch.zeros(
            (times.shape[0], b, c, lat, lon),
            device=device,
            dtype=x.dtype,
        )
        weight = torch.zeros((lat, lon), device=device, dtype=x.dtype)
        for lat_slice, lon_slice in slices:
            tile_init = x[:, :, lat_slice, lon_slice]
            tile_static = static[:, :, lat_slice, lon_slice] if static is not None else None
            tile_pred = ode_model(tile_init, times, static=tile_static, forcing=forcing)  # [T, B, C, h, w]
            preds[:, :, :, lat_slice, lon_slice] += tile_pred
            weight[lat_slice, lon_slice] += 1.0
        weight = weight.clamp(min=1.0)
        preds = preds / weight
        return preds

    with torch.no_grad():
        predictions = _run_tiled_prediction(initial)

    trajectory = _prepare_trajectory(predictions, initial, target, times, channel_names)
    return trajectory, predictions[-1, 0]


def _build_dataset_summary(
    dataset: TensorDataset, channel_names: List[str]
) -> DatasetSummary:
    x0 = dataset.tensors[0]
    stats = _aggregate_channel_stats(x0, channel_names)
    return DatasetSummary(
        channel_stats=stats,
        sample_shape=list(x0.shape[1:]),
    )


def _build_lod_preview(
    field: torch.Tensor,
    simulation: SimulationConfig,
    orchestrator: SimulationOrchestrator,
) -> LODPreview:
    """Summarize level-of-detail tiles for the latest field."""
    description = orchestrator.stream_level_of_detail(field, simulation.lod)
    tiles = [
        LODTile(
            level=int(tile["level"]),
            tile=str(tile["tile"]),
            latStart=int(tile["latStart"]),
            latEnd=int(tile["latEnd"]),
            lonStart=int(tile["lonStart"]),
            lonEnd=int(tile["lonEnd"]),
            mean=float(tile["mean"]),
            std=float(tile["std"]),
        )
        for tile in description["tiles"]
    ]
    return LODPreview(chunkShape=description["chunkShape"], tiles=tiles)


def create_app() -> FastAPI:
    """Create the FastAPI instance used by both the CLI and tests."""
    app = FastAPI(title="WeatherFlow API", version="1.0")

    # Configure CORS to allow GitHub Pages frontend
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://monksealseal.github.io",  # GitHub Pages production
            "http://localhost:5173",            # Vite dev server
            "http://localhost:3000",            # Alternative dev port
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/options")
    def get_options() -> Dict[str, object]:  # noqa: D401
        """Return enumerations consumed by the front-end."""
        return {
            "variables": DEFAULT_VARIABLES,
            "pressureLevels": DEFAULT_PRESSURE_LEVELS,
            "gridSizes": [
                {"lat": lat, "lon": lon} for lat, lon in DEFAULT_GRID_SIZES
            ],
            "solverMethods": DEFAULT_SOLVER_METHODS,
            "lossTypes": DEFAULT_LOSS_TYPES,
            "simulationCores": [
                spec.name for spec in SIMULATION_ORCHESTRATOR.available_cores()
            ],
            "resolutionTiers": [
                {
                    "name": tier.name,
                    "lat": tier.lat,
                    "lon": tier.lon,
                    "verticalLevels": tier.vertical_levels,
                    "timeStepSeconds": tier.time_step_seconds,
                    "description": tier.description,
                }
                for tier in SIMULATION_ORCHESTRATOR.available_resolution_tiers()
            ],
            "maxEpochs": 6,
        }

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    # ==================== ATMOSPHERIC DYNAMICS ENDPOINTS ====================

    class CoriolisRequest(CamelModel):
        """Request for Coriolis parameter calculation."""
        latitude: float = Field(..., ge=-90, le=90)

    class CoriolisResponse(CamelModel):
        """Response with Coriolis calculations."""
        latitude: float
        coriolis_parameter: float = Field(alias="coriolisParameter")
        beta_parameter: float = Field(alias="betaParameter")
        inertial_period_hours: float = Field(alias="inertialPeriodHours")

    @app.post("/api/dynamics/coriolis", response_model=CoriolisResponse)
    def calculate_coriolis(request: CoriolisRequest) -> CoriolisResponse:
        """Calculate Coriolis and beta parameters for a given latitude."""
        OMEGA = 7.292e-5  # Earth's rotation rate (rad/s)
        R_EARTH = 6.371e6  # Earth's radius (m)

        lat_rad = math.radians(request.latitude)
        f = 2 * OMEGA * math.sin(lat_rad)
        beta = 2 * OMEGA * math.cos(lat_rad) / R_EARTH

        # Inertial period (hours)
        inertial_period = abs(2 * math.pi / f / 3600) if abs(f) > 1e-10 else float('inf')

        return CoriolisResponse(
            latitude=request.latitude,
            coriolis_parameter=f,
            beta_parameter=beta,
            inertial_period_hours=inertial_period
        )

    class GeostrophicWindRequest(CamelModel):
        """Request for geostrophic wind calculation."""
        dp_dx: float = Field(alias="dpDx")  # Pressure gradient in x (Pa/m)
        dp_dy: float = Field(alias="dpDy")  # Pressure gradient in y (Pa/m)
        latitude: float = Field(..., ge=-90, le=90)
        density: float = Field(1.225, gt=0)  # Air density (kg/m³)

    class GeostrophicWindResponse(CamelModel):
        """Response with geostrophic wind."""
        u_geostrophic: float = Field(alias="uGeostrophic")  # m/s
        v_geostrophic: float = Field(alias="vGeostrophic")  # m/s
        wind_speed: float = Field(alias="windSpeed")  # m/s
        wind_direction: float = Field(alias="windDirection")  # degrees

    @app.post("/api/dynamics/geostrophic", response_model=GeostrophicWindResponse)
    def calculate_geostrophic_wind(request: GeostrophicWindRequest) -> GeostrophicWindResponse:
        """Calculate geostrophic wind from pressure gradient."""
        OMEGA = 7.292e-5
        lat_rad = math.radians(request.latitude)
        f = 2 * OMEGA * math.sin(lat_rad)

        if abs(f) < 1e-10:
            raise HTTPException(400, "Coriolis parameter too small near equator")

        u_g = -request.dp_dy / (request.density * f)
        v_g = request.dp_dx / (request.density * f)

        speed = math.sqrt(u_g**2 + v_g**2)
        direction = math.degrees(math.atan2(u_g, v_g)) % 360

        return GeostrophicWindResponse(
            u_geostrophic=u_g,
            v_geostrophic=v_g,
            wind_speed=speed,
            wind_direction=direction
        )

    class RossbyWaveRequest(CamelModel):
        """Request for Rossby wave calculation."""
        latitude: float = Field(..., ge=-90, le=90)
        wavelength_km: float = Field(alias="wavelengthKm", gt=0)
        mean_flow: float = Field(alias="meanFlow", default=10.0)  # m/s

    class RossbyWaveResponse(CamelModel):
        """Response with Rossby wave properties."""
        phase_speed: float = Field(alias="phaseSpeed")  # m/s
        group_velocity: float = Field(alias="groupVelocity")  # m/s
        period_days: float = Field(alias="periodDays")
        stationary_wavelength_km: float = Field(alias="stationaryWavelengthKm")

    @app.post("/api/dynamics/rossby", response_model=RossbyWaveResponse)
    def calculate_rossby_wave(request: RossbyWaveRequest) -> RossbyWaveResponse:
        """Calculate Rossby wave properties."""
        OMEGA = 7.292e-5
        R_EARTH = 6.371e6

        lat_rad = math.radians(request.latitude)
        beta = 2 * OMEGA * math.cos(lat_rad) / R_EARTH

        wavelength_m = request.wavelength_km * 1000
        k = 2 * math.pi / wavelength_m

        # Phase speed: c = U - beta/k²
        phase_speed = request.mean_flow - beta / (k**2)

        # Group velocity: cg = U + beta/k²
        group_velocity = request.mean_flow + beta / (k**2)

        # Period
        omega = k * phase_speed
        period_seconds = abs(2 * math.pi / omega) if abs(omega) > 1e-15 else float('inf')
        period_days = period_seconds / 86400

        # Stationary wavelength (where c = 0)
        stationary_k = math.sqrt(beta / request.mean_flow) if request.mean_flow > 0 else 0
        stationary_wavelength_km = (2 * math.pi / stationary_k / 1000) if stationary_k > 0 else float('inf')

        return RossbyWaveResponse(
            phase_speed=phase_speed,
            group_velocity=group_velocity,
            period_days=period_days,
            stationary_wavelength_km=stationary_wavelength_km
        )

    # ==================== RENEWABLE ENERGY ENDPOINTS ====================

    class WindPowerRequest(CamelModel):
        """Request for wind power calculation."""
        wind_speeds: List[float] = Field(alias="windSpeeds")  # m/s
        turbine_type: str = Field("IEA-3.4MW", alias="turbineType")
        num_turbines: int = Field(1, ge=1, alias="numTurbines")
        hub_height: Optional[float] = Field(None, alias="hubHeight")
        measurement_height: float = Field(10.0, alias="measurementHeight")

    class WindPowerResponse(CamelModel):
        """Response with wind power output."""
        power_per_turbine: List[float] = Field(alias="powerPerTurbine")  # MW
        total_power: List[float] = Field(alias="totalPower")  # MW
        capacity_factor: float = Field(alias="capacityFactor")
        rated_capacity: float = Field(alias="ratedCapacity")  # MW
        turbine_info: Dict[str, Any] = Field(alias="turbineInfo")

    @app.post("/api/energy/wind-power", response_model=WindPowerResponse)
    def calculate_wind_power(request: WindPowerRequest) -> WindPowerResponse:
        """Calculate wind power output from wind speeds."""
        # Turbine specifications
        TURBINES = {
            'IEA-3.4MW': {'rated': 3.4, 'cut_in': 3.0, 'rated_speed': 13.0, 'cut_out': 25.0, 'hub': 110},
            'NREL-5MW': {'rated': 5.0, 'cut_in': 3.0, 'rated_speed': 11.4, 'cut_out': 25.0, 'hub': 90},
            'Vestas-V90': {'rated': 2.0, 'cut_in': 4.0, 'rated_speed': 15.0, 'cut_out': 25.0, 'hub': 80},
        }

        if request.turbine_type not in TURBINES:
            raise HTTPException(400, f"Unknown turbine type: {request.turbine_type}")

        turbine = TURBINES[request.turbine_type]
        hub_height = request.hub_height or turbine['hub']

        # Height adjustment using power law
        alpha = 0.143
        height_factor = (hub_height / request.measurement_height) ** alpha

        power_per_turbine = []
        for ws in request.wind_speeds:
            ws_hub = ws * height_factor

            if ws_hub < turbine['cut_in'] or ws_hub >= turbine['cut_out']:
                power = 0.0
            elif ws_hub >= turbine['rated_speed']:
                power = turbine['rated']
            else:
                # Cubic power curve
                normalized = (ws_hub - turbine['cut_in']) / (turbine['rated_speed'] - turbine['cut_in'])
                power = turbine['rated'] * (normalized ** 3)

            power_per_turbine.append(power)

        total_power = [p * request.num_turbines * 0.95 for p in power_per_turbine]  # 95% efficiency
        rated_capacity = turbine['rated'] * request.num_turbines
        capacity_factor = sum(total_power) / len(total_power) / rated_capacity if total_power else 0

        return WindPowerResponse(
            power_per_turbine=power_per_turbine,
            total_power=total_power,
            capacity_factor=capacity_factor,
            rated_capacity=rated_capacity,
            turbine_info={
                'type': request.turbine_type,
                'ratedPower': turbine['rated'],
                'hubHeight': hub_height,
                'cutInSpeed': turbine['cut_in'],
                'ratedSpeed': turbine['rated_speed'],
                'cutOutSpeed': turbine['cut_out']
            }
        )

    class SolarPowerRequest(CamelModel):
        """Request for solar power calculation."""
        latitude: float = Field(..., ge=-90, le=90)
        day_of_year: int = Field(..., ge=1, le=366, alias="dayOfYear")
        hours: List[float]  # Hour of day (0-24)
        cloud_cover: Optional[List[float]] = Field(None, alias="cloudCover")  # 0-1
        panel_capacity_mw: float = Field(1.0, alias="panelCapacityMw")
        panel_efficiency: float = Field(0.18, alias="panelEfficiency")
        tilt: float = Field(30.0)  # degrees

    class SolarPowerResponse(CamelModel):
        """Response with solar power output."""
        power: List[float]  # MW
        clear_sky_irradiance: List[float] = Field(alias="clearSkyIrradiance")  # W/m²
        solar_elevation: List[float] = Field(alias="solarElevation")  # degrees
        daily_energy_mwh: float = Field(alias="dailyEnergyMwh")
        capacity_factor: float = Field(alias="capacityFactor")

    @app.post("/api/energy/solar-power", response_model=SolarPowerResponse)
    def calculate_solar_power(request: SolarPowerRequest) -> SolarPowerResponse:
        """Calculate solar power output."""
        lat_rad = math.radians(request.latitude)

        # Solar declination
        declination = 23.45 * math.sin(math.radians(360 * (284 + request.day_of_year) / 365))
        decl_rad = math.radians(declination)

        power = []
        irradiance = []
        elevation = []

        for hour in request.hours:
            # Hour angle
            hour_angle = math.radians(15 * (hour - 12))

            # Solar elevation
            sin_elev = (math.sin(lat_rad) * math.sin(decl_rad) +
                       math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle))
            elev_rad = math.asin(max(-1, min(1, sin_elev)))
            elev_deg = math.degrees(elev_rad)
            elevation.append(elev_deg)

            # Clear sky irradiance (simplified model)
            if elev_deg > 0:
                # Direct normal irradiance
                am = 1 / math.sin(elev_rad) if elev_rad > 0.01 else 40  # Air mass
                dni = 1361 * 0.7 ** (am ** 0.678)  # W/m²

                # Global horizontal irradiance
                ghi = dni * math.sin(elev_rad) + 0.1 * dni
            else:
                ghi = 0.0

            irradiance.append(ghi)

            # Apply cloud cover
            cloud = request.cloud_cover[len(power)] if request.cloud_cover and len(power) < len(request.cloud_cover) else 0
            effective_irradiance = ghi * (1 - 0.75 * cloud)

            # Power output
            panel_power = (effective_irradiance / 1000) * request.panel_capacity_mw * request.panel_efficiency
            power.append(max(0, panel_power))

        # Daily energy (assuming hourly data)
        daily_energy = sum(power)
        capacity_factor = daily_energy / (24 * request.panel_capacity_mw) if request.panel_capacity_mw > 0 else 0

        return SolarPowerResponse(
            power=power,
            clear_sky_irradiance=irradiance,
            solar_elevation=elevation,
            daily_energy_mwh=daily_energy,
            capacity_factor=capacity_factor
        )

    # ==================== EXTREME EVENTS ENDPOINTS ====================

    class HeatwaveDetectionRequest(CamelModel):
        """Request for heatwave detection."""
        temperatures: List[List[float]]  # [time, location] in Celsius
        threshold_celsius: float = Field(35.0, alias="thresholdCelsius")
        min_duration_days: int = Field(3, alias="minDurationDays")

    class DetectedEvent(CamelModel):
        """A detected extreme event."""
        event_type: str = Field(alias="eventType")
        start_index: int = Field(alias="startIndex")
        end_index: int = Field(alias="endIndex")
        duration_days: float = Field(alias="durationDays")
        peak_value: float = Field(alias="peakValue")
        mean_value: float = Field(alias="meanValue")
        affected_fraction: float = Field(alias="affectedFraction")

    class EventDetectionResponse(CamelModel):
        """Response with detected events."""
        events: List[DetectedEvent]
        total_events: int = Field(alias="totalEvents")
        threshold_used: float = Field(alias="thresholdUsed")

    @app.post("/api/extreme/heatwave", response_model=EventDetectionResponse)
    def detect_heatwaves(request: HeatwaveDetectionRequest) -> EventDetectionResponse:
        """Detect heatwave events in temperature data."""
        temps = np.array(request.temperatures)
        threshold = request.threshold_celsius
        min_steps = request.min_duration_days * 4  # Assuming 6-hourly data

        # Find exceedances
        exceeds = temps > threshold
        affected_fraction = exceeds.mean(axis=1) if len(temps.shape) > 1 else exceeds.astype(float)

        # Find persistent events
        events = []
        in_event = False
        event_start = 0

        spatial_threshold = 0.1  # At least 10% of area affected

        for i in range(len(affected_fraction)):
            if affected_fraction[i] > spatial_threshold and not in_event:
                in_event = True
                event_start = i
            elif affected_fraction[i] <= spatial_threshold and in_event:
                duration = i - event_start
                if duration >= min_steps:
                    event_temps = temps[event_start:i]
                    events.append(DetectedEvent(
                        event_type='heatwave',
                        start_index=event_start,
                        end_index=i,
                        duration_days=duration / 4,
                        peak_value=float(event_temps.max()),
                        mean_value=float(event_temps[event_temps > threshold].mean()) if (event_temps > threshold).any() else float(event_temps.mean()),
                        affected_fraction=float(affected_fraction[event_start:i].mean())
                    ))
                in_event = False

        return EventDetectionResponse(
            events=events,
            total_events=len(events),
            threshold_used=threshold
        )

    class ARDetectionRequest(CamelModel):
        """Request for atmospheric river detection."""
        ivt: List[List[float]]  # [lat, lon] Integrated vapor transport (kg/m/s)
        ivt_threshold: float = Field(250.0, alias="ivtThreshold")

    @app.post("/api/extreme/atmospheric-river", response_model=EventDetectionResponse)
    def detect_atmospheric_rivers(request: ARDetectionRequest) -> EventDetectionResponse:
        """Detect atmospheric rivers in IVT field."""
        ivt = np.array(request.ivt)
        threshold = request.ivt_threshold

        # Find regions exceeding threshold
        exceeds = ivt > threshold

        events = []
        if exceeds.any():
            # Simple connected component analysis
            from scipy import ndimage
            labeled, num_features = ndimage.label(exceeds)

            for feature_id in range(1, num_features + 1):
                mask = labeled == feature_id

                # Calculate dimensions
                lat_extent = mask.sum(axis=1).max()
                lon_extent = mask.sum(axis=0).max()

                # Check AR criteria (length > 2000km, width < 1000km proxy)
                length = max(lat_extent, lon_extent)
                width = min(lat_extent, lon_extent)

                if length >= 20 and width <= 10:  # Grid cell thresholds
                    peak_ivt = ivt[mask].max()
                    mean_ivt = ivt[mask].mean()

                    events.append(DetectedEvent(
                        event_type='atmospheric_river',
                        start_index=0,
                        end_index=1,
                        duration_days=0.25,  # Snapshot
                        peak_value=float(peak_ivt),
                        mean_value=float(mean_ivt),
                        affected_fraction=float(mask.sum() / mask.size)
                    ))

        return EventDetectionResponse(
            events=events,
            total_events=len(events),
            threshold_used=threshold
        )

    # ==================== MODEL ZOO ENDPOINTS ====================

    class ModelInfo(CamelModel):
        """Information about a pre-trained model."""
        id: str
        name: str
        description: str
        architecture: str
        parameters: str
        variables: List[str]
        status: str
        category: str
        metrics: Optional[Dict[str, float]] = None

    @app.get("/api/model-zoo/models")
    def list_models() -> List[ModelInfo]:
        """List all available models in the Model Zoo."""
        return [
            ModelInfo(
                id='z500_3day',
                name='Z500 3-Day Forecast',
                description='500 hPa geopotential height prediction for 3-day lead time',
                architecture='WeatherFlowMatch',
                parameters='~2.5M',
                variables=['z'],
                status='ready',
                category='global',
                metrics={'rmse': 120.5, 'acc': 0.89}
            ),
            ModelInfo(
                id='t850_weekly',
                name='T850 Weekly Forecast',
                description='850 hPa temperature prediction for weekly forecasts',
                architecture='WeatherFlowMatch with attention',
                parameters='~3.2M',
                variables=['t'],
                status='ready',
                category='global',
                metrics={'rmse': 2.1, 'acc': 0.85}
            ),
            ModelInfo(
                id='multi_variable',
                name='Multi-Variable Global',
                description='Comprehensive prediction of multiple atmospheric variables',
                architecture='Physics-guided FlowMatch',
                parameters='~8M',
                variables=['z', 't', 'u', 'v', 'q'],
                status='ready',
                category='global',
                metrics={'rmse': 95.2, 'acc': 0.91}
            ),
            ModelInfo(
                id='tropical_cyclones',
                name='Tropical Cyclone Tracker',
                description='Track and intensity prediction for tropical cyclones',
                architecture='Icosahedral grid model',
                parameters='~5M',
                variables=['z', 'u', 'v', 'msl'],
                status='ready',
                category='extreme',
                metrics={'track_error_km': 85.0, 'intensity_rmse': 8.5}
            ),
            ModelInfo(
                id='atmospheric_rivers',
                name='Atmospheric River Detection',
                description='Detection and tracking of atmospheric rivers',
                architecture='FlowMatch + detector',
                parameters='~3.5M',
                variables=['q', 'u', 'v'],
                status='ready',
                category='extreme',
                metrics={'detection_accuracy': 0.92, 'false_alarm_rate': 0.08}
            ),
            ModelInfo(
                id='seasonal',
                name='Seasonal Forecasting',
                description='Long-range seasonal climate predictions',
                architecture='Stochastic FlowMatch',
                parameters='~6M',
                variables=['t', 'pr'],
                status='ready',
                category='climate',
                metrics={'skill_score': 0.45}
            ),
        ]

    @app.get("/api/model-zoo/models/{model_id}")
    def get_model_info(model_id: str) -> ModelInfo:
        """Get detailed information about a specific model."""
        models = {m.id: m for m in list_models()}
        if model_id not in models:
            raise HTTPException(404, f"Model not found: {model_id}")
        return models[model_id]

    # ==================== GCM SIMULATION ENDPOINTS ====================

    class GCMSimulationRequest(CamelModel):
        """Request for GCM simulation."""
        nlat: int = Field(32, ge=8, le=128)
        nlon: int = Field(64, ge=16, le=256)
        nlev: int = Field(20, ge=5, le=50)
        duration_days: float = Field(1.0, ge=0.1, le=30, alias="durationDays")
        dt_seconds: float = Field(1800, alias="dtSeconds")
        co2_ppmv: float = Field(400.0, alias="co2Ppmv")
        profile: str = Field("standard")

    class GCMSimulationResponse(CamelModel):
        """Response with GCM simulation results."""
        simulation_id: str = Field(alias="simulationId")
        status: str
        global_mean_temp: float = Field(alias="globalMeanTemp")
        surface_temp_range: List[float] = Field(alias="surfaceTempRange")
        max_wind_speed: float = Field(alias="maxWindSpeed")
        total_precipitation_mm: float = Field(alias="totalPrecipitationMm")
        time_steps_completed: int = Field(alias="timeStepsCompleted")
        duration_seconds: float = Field(alias="durationSeconds")

    @app.post("/api/gcm/simulate", response_model=GCMSimulationResponse)
    def run_gcm_simulation(request: GCMSimulationRequest) -> GCMSimulationResponse:
        """Run a simplified GCM simulation."""
        start_time = time.perf_counter()

        # Create simple synthetic results for demo
        # In production, this would call the actual GCM
        np.random.seed(42)

        # Generate temperature field
        lats = np.linspace(-90, 90, request.nlat)
        temps = 288 - 40 * np.abs(lats) / 90 + np.random.randn(request.nlat) * 2

        # Add CO2 warming effect
        co2_effect = (request.co2_ppmv - 280) * 0.005  # Simplified warming
        temps += co2_effect

        time_steps = int(request.duration_days * 86400 / request.dt_seconds)

        return GCMSimulationResponse(
            simulation_id=str(uuid.uuid4()),
            status='completed',
            global_mean_temp=float(temps.mean()),
            surface_temp_range=[float(temps.min()), float(temps.max())],
            max_wind_speed=float(np.random.uniform(20, 50)),
            total_precipitation_mm=float(np.random.uniform(0.5, 5.0) * request.duration_days),
            time_steps_completed=time_steps,
            duration_seconds=float(time.perf_counter() - start_time)
        )

    # ==================== VISUALIZATION DATA ENDPOINTS ====================

    class FieldDataRequest(CamelModel):
        """Request for field visualization data."""
        variable: str
        pressure_level: int = Field(500, alias="pressureLevel")
        lat_range: Optional[List[float]] = Field(None, alias="latRange")
        lon_range: Optional[List[float]] = Field(None, alias="lonRange")
        grid_size: int = Field(32, alias="gridSize")

    class FieldDataResponse(CamelModel):
        """Response with field data for visualization."""
        data: List[List[float]]
        lats: List[float]
        lons: List[float]
        variable: str
        pressure_level: int = Field(alias="pressureLevel")
        min_value: float = Field(alias="minValue")
        max_value: float = Field(alias="maxValue")
        units: str

    @app.post("/api/visualization/field", response_model=FieldDataResponse)
    def get_field_data(request: FieldDataRequest) -> FieldDataResponse:
        """Get synthetic field data for visualization."""
        # Variable specifications
        VAR_SPECS = {
            'z': {'base': 5500, 'amplitude': 500, 'units': 'm²/s²'},
            't': {'base': 260, 'amplitude': 20, 'units': 'K'},
            'u': {'base': 0, 'amplitude': 30, 'units': 'm/s'},
            'v': {'base': 0, 'amplitude': 20, 'units': 'm/s'},
            'q': {'base': 0.005, 'amplitude': 0.01, 'units': 'kg/kg'},
        }

        spec = VAR_SPECS.get(request.variable, {'base': 0, 'amplitude': 1, 'units': ''})

        # Generate grid
        lat_min = request.lat_range[0] if request.lat_range else -90
        lat_max = request.lat_range[1] if request.lat_range else 90
        lon_min = request.lon_range[0] if request.lon_range else -180
        lon_max = request.lon_range[1] if request.lon_range else 180

        lats = np.linspace(lat_min, lat_max, request.grid_size).tolist()
        lons = np.linspace(lon_min, lon_max, request.grid_size * 2).tolist()

        # Generate synthetic data with realistic patterns
        lat_grid, lon_grid = np.meshgrid(
            np.linspace(lat_min, lat_max, request.grid_size),
            np.linspace(lon_min, lon_max, request.grid_size * 2),
            indexing='ij'
        )

        # Add latitude-dependent pattern + waves
        data = (spec['base'] +
                spec['amplitude'] * np.cos(np.radians(lat_grid)) *
                np.sin(np.radians(lon_grid) * 3) +
                spec['amplitude'] * 0.3 * np.random.randn(request.grid_size, request.grid_size * 2))

        return FieldDataResponse(
            data=data.tolist(),
            lats=lats,
            lons=lons,
            variable=request.variable,
            pressure_level=request.pressure_level,
            min_value=float(data.min()),
            max_value=float(data.max()),
            units=spec['units']
        )

    # ==================== EDUCATION ENDPOINTS ====================

    class PhysicsQuizQuestion(CamelModel):
        """A physics quiz question."""
        id: str
        question: str
        options: List[str]
        correct_index: int = Field(alias="correctIndex")
        explanation: str
        topic: str

    @app.get("/api/education/quiz/{topic}")
    def get_quiz_questions(topic: str) -> List[PhysicsQuizQuestion]:
        """Get quiz questions for a topic."""
        QUESTIONS = {
            'coriolis': [
                PhysicsQuizQuestion(
                    id='c1',
                    question='At what latitude is the Coriolis parameter (f) equal to zero?',
                    options=['90°N', '45°N', '0° (Equator)', '30°N'],
                    correct_index=2,
                    explanation='The Coriolis parameter f = 2Ω sin(φ) equals zero at the equator where sin(0°) = 0.',
                    topic='coriolis'
                ),
                PhysicsQuizQuestion(
                    id='c2',
                    question='Which direction does the Coriolis force deflect moving objects in the Northern Hemisphere?',
                    options=['Left', 'Right', 'Up', 'Down'],
                    correct_index=1,
                    explanation='In the Northern Hemisphere, the Coriolis force deflects moving objects to the right of their direction of motion.',
                    topic='coriolis'
                ),
            ],
            'geostrophic': [
                PhysicsQuizQuestion(
                    id='g1',
                    question='In geostrophic balance, the wind blows:',
                    options=['From high to low pressure', 'From low to high pressure', 'Parallel to isobars', 'Perpendicular to isobars'],
                    correct_index=2,
                    explanation='Geostrophic wind flows parallel to isobars with low pressure on the left (NH) due to balance between pressure gradient and Coriolis forces.',
                    topic='geostrophic'
                ),
            ],
            'rossby': [
                PhysicsQuizQuestion(
                    id='r1',
                    question='Rossby waves propagate:',
                    options=['Eastward only', 'Westward relative to mean flow', 'Northward', 'Randomly'],
                    correct_index=1,
                    explanation='Rossby waves have a westward phase speed relative to the mean flow due to the beta effect (variation of Coriolis with latitude).',
                    topic='rossby'
                ),
            ],
        }

        if topic not in QUESTIONS:
            raise HTTPException(404, f"No questions for topic: {topic}")

        return QUESTIONS[topic]

    # ==================== NOTEBOOK DATA ENDPOINTS ====================

    class NotebookInfo(CamelModel):
        """Information about a Jupyter notebook."""
        id: str
        title: str
        description: str
        topics: List[str]
        difficulty: str
        estimated_time: str = Field(alias="estimatedTime")
        cells_count: int = Field(alias="cellsCount")

    @app.get("/api/notebooks")
    def list_notebooks() -> List[NotebookInfo]:
        """List available Jupyter notebooks."""
        return [
            NotebookInfo(
                id='complete_guide',
                title='WeatherFlow Complete Guide',
                description='Comprehensive guide covering data loading, model training, prediction, and visualization',
                topics=['ERA5 data', 'Flow matching', 'Training', 'Visualization'],
                difficulty='Intermediate',
                estimated_time='45 mins',
                cells_count=30
            ),
            NotebookInfo(
                id='flow_matching_basics',
                title='Flow Matching Basics',
                description='Introduction to flow matching concepts and implementation',
                topics=['Flow matching', 'Vector fields', 'ODE solvers'],
                difficulty='Beginner',
                estimated_time='30 mins',
                cells_count=20
            ),
            NotebookInfo(
                id='era5_data_exploration',
                title='ERA5 Data Exploration',
                description='Explore and visualize ERA5 reanalysis data',
                topics=['ERA5', 'Data loading', 'Visualization'],
                difficulty='Beginner',
                estimated_time='20 mins',
                cells_count=15
            ),
            NotebookInfo(
                id='model_training',
                title='Model Training Tutorial',
                description='Step-by-step guide to training WeatherFlow models',
                topics=['Training', 'Hyperparameters', 'Validation'],
                difficulty='Intermediate',
                estimated_time='40 mins',
                cells_count=25
            ),
            NotebookInfo(
                id='weatherbench_evaluation',
                title='WeatherBench2 Evaluation',
                description='Evaluate models against WeatherBench2 benchmarks',
                topics=['Evaluation', 'Metrics', 'Benchmarking'],
                difficulty='Advanced',
                estimated_time='50 mins',
                cells_count=28
            ),
        ]

    @app.post("/api/experiments", response_model=ExperimentResult)
    def run_experiment(config: ExperimentConfig) -> ExperimentResult:
        start = time.perf_counter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device)
        generator.manual_seed(config.training.seed)
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(config.training.seed)

        try:
            datasets = _build_dataloaders(
                config.dataset,
                config.training.dynamics_scale,
                config.simulation,
                SIMULATION_ORCHESTRATOR,
                device,
                generator,
            )
            training_outcome = _train_model(
                config, device, datasets, generator, loader_generator
            )
            prediction, latest_field = _run_prediction(
                training_outcome["model"],
                config,
                datasets["val"],
                datasets["channel_names"],
                device,
            )
            summary = _build_dataset_summary(
                datasets["train"], datasets["channel_names"]
            )
            lod_preview = _build_lod_preview(
                latest_field, config.simulation, SIMULATION_ORCHESTRATOR
            )
            simulation_summary = SimulationSummary(
                core=config.simulation.core,
                resolution_tier=config.simulation.resolution_tier,
                grid=datasets["grid"],
                time_step_seconds=datasets["time_step_seconds"],
            )
        except Exception as exc:  # pragma: no cover - surfaced to API response
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        end = time.perf_counter()

        return ExperimentResult(
            experiment_id=str(uuid.uuid4()),
            config=config,
            channel_names=datasets["channel_names"],
            metrics={"train": training_outcome["train_metrics"]},
            validation={"metrics": training_outcome["val_metrics"]},
            dataset_summary=summary,
            prediction=prediction,
            lod_preview=lod_preview,
            simulation_summary=simulation_summary,
            execution=ExecutionSummary(duration_seconds=float(end - start)),
        )

    return app


app = create_app()
