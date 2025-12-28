"""FastAPI application exposing WeatherFlow experimentation utilities."""
from __future__ import annotations

import time
import uuid
from typing import Dict, List, Tuple

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

    backbone: str = Field("grid")
    hidden_dim: int = Field(96, ge=32, le=512, alias="hiddenDim")
    n_layers: int = Field(3, ge=1, le=8, alias="nLayers")
    use_attention: bool = Field(True, alias="useAttention")
    physics_informed: bool = Field(True, alias="physicsInformed")
    window_size: int = Field(8, ge=0, le=64, alias="windowSize")
    spherical_padding: bool = Field(False, alias="sphericalPadding")
    use_graph_mp: bool = Field(False, alias="useGraphMp")
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


class InferenceConfig(CamelModel):
    """Inference configuration for tiling large grids."""

    tile_size_lat: int = Field(0, ge=0, le=512, alias="tileSizeLat")
    tile_size_lon: int = Field(0, ge=0, le=1024, alias="tileSizeLon")
    tile_overlap: int = Field(0, ge=0, le=64, alias="tileOverlap")

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
            num_workers=2,
            shuffle=True,
        )
        batch = next(iter(loader))
        train_x0, train_x1 = batch
        val_loader = create_webdataset_loader(
            config.webdataset_pattern,
            batch_size=config.val_samples,
            num_workers=2,
            shuffle=True,
            resampled=False,
        )
        val_x0, val_x1 = next(iter(val_loader))
        # Dummy static/forcing placeholders (zeroed) matching shapes
        b_train, _, lat, lon = train_x0.shape
        static_features = torch.zeros(b_train, 2, lat, lon, device=device)
        forcing = torch.zeros(b_train, 1, device=device)
        b_val = val_x0.shape[0]
        static_val = torch.zeros(b_val, 2, lat, lon, device=device)
        forcing_val = torch.zeros(b_val, 1, device=device)

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
        if v_pred.shape[1] >= 2:
            u = v_pred[:, 0:1]
            v_comp = v_pred[:, 1:2]
            du_dx = torch.gradient(u, dim=3)[0]
            dv_dy = torch.gradient(v_comp, dim=2)[0]
            div = du_dx + dv_dy
            div_loss = torch.mean(div**2)
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
