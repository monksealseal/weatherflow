"""FastAPI application exposing WeatherFlow experimentation utilities."""
from __future__ import annotations

import time
import uuid
from typing import Dict, List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from torch.utils.data import DataLoader, TensorDataset

from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE

# Limit CPU usage for deterministic behaviour when running inside tests
TORCH_NUM_THREADS = 1

torch.set_num_threads(TORCH_NUM_THREADS)

DEFAULT_VARIABLES = ["t", "z", "u", "v"]
DEFAULT_PRESSURE_LEVELS = [1000, 850, 700, 500, 300, 200]
DEFAULT_GRID_SIZES = [(16, 32), (32, 64)]
DEFAULT_SOLVER_METHODS = ["dopri5", "rk4", "midpoint"]
DEFAULT_LOSS_TYPES = ["mse", "huber", "smooth_l1"]


class CamelModel(BaseModel):
    """Base model enabling population by field name or alias."""

    class Config:
        allow_population_by_field_name = True


class GridSize(CamelModel):
    """Simple grid size model for validation."""

    lat: int = Field(16, ge=4, le=128)
    lon: int = Field(32, ge=4, le=256)

    @validator("lon")
    def lon_multiple_of_two(cls, value: int) -> int:  # noqa: D401
        """Ensure longitude dimension is even for nicer plots."""
        if value % 2 != 0:
            raise ValueError("Longitude must be an even number")
        return value


class DatasetConfig(CamelModel):
    """Configuration options for generating synthetic datasets."""

    variables: List[str] = Field(default_factory=lambda: DEFAULT_VARIABLES[:2])
    pressure_levels: List[int] = Field(default_factory=lambda: [500], alias="pressureLevels")
    grid_size: GridSize = Field(default_factory=GridSize, alias="gridSize")
    train_samples: int = Field(48, ge=4, le=256, alias="trainSamples")
    val_samples: int = Field(16, ge=4, le=128, alias="valSamples")

    @validator("variables")
    def validate_variables(cls, values: List[str]) -> List[str]:  # noqa: D401
        """Ensure at least one variable was selected."""
        if not values:
            raise ValueError("At least one variable must be selected")
        for var in values:
            if var not in DEFAULT_VARIABLES:
                raise ValueError(f"Unsupported variable '{var}'")
        return values

    @validator("pressure_levels")
    def validate_pressure_levels(cls, values: List[int]) -> List[int]:  # noqa: D401
        """Ensure at least one pressure level is available."""
        if not values:
            raise ValueError("Select at least one pressure level")
        return values


class ModelConfig(CamelModel):
    """Neural network hyperparameters."""

    hidden_dim: int = Field(96, ge=32, le=512, alias="hiddenDim")
    n_layers: int = Field(3, ge=1, le=8, alias="nLayers")
    use_attention: bool = Field(True, alias="useAttention")
    physics_informed: bool = Field(True, alias="physicsInformed")


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

    @validator("solver_method")
    def solver_method_supported(cls, value: str) -> str:  # noqa: D401
        """Ensure the requested ODE solver is available."""
        if value not in DEFAULT_SOLVER_METHODS:
            raise ValueError(f"Unsupported solver '{value}'")
        return value

    @validator("loss_type")
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
    energy_diff: float = Field(alias="energyDiff")


class ValidationMetricEntry(CamelModel):
    epoch: int
    val_loss: float
    val_flow_loss: float = Field(alias="valFlowLoss")
    val_divergence_loss: float = Field(alias="valDivergenceLoss")
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


class ExecutionSummary(CamelModel):
    duration_seconds: float = Field(alias="durationSeconds")


class DatasetSummary(CamelModel):
    channel_stats: List[ChannelStats] = Field(alias="channelStats")
    sample_shape: List[int] = Field(alias="sampleShape")


class ExperimentResult(CamelModel):
    experiment_id: str = Field(alias="experimentId")
    config: ExperimentConfig
    channel_names: List[str] = Field(alias="channelNames")
    metrics: Dict[str, List[MetricEntry]]
    validation: Dict[str, List[ValidationMetricEntry]]
    dataset_summary: DatasetSummary = Field(alias="datasetSummary")
    prediction: PredictionResult
    execution: ExecutionSummary


def _channel_names(dataset: DatasetConfig) -> List[str]:
    names: List[str] = []
    for var in dataset.variables:
        for level in dataset.pressure_levels:
            names.append(f"{var}@{level}")
    return names


def _build_dataloaders(config: DatasetConfig, dynamics_scale: float) -> Dict[str, object]:
    """Create lightweight synthetic datasets for demonstration purposes."""
    channel_names = _channel_names(config)
    channels = len(channel_names)
    lat = config.grid_size.lat
    lon = config.grid_size.lon

    def _synth_samples(num_samples: int) -> torch.Tensor:
        base = torch.randn(num_samples, channels, lat, lon)
        return base

    train_x0 = _synth_samples(config.train_samples)
    train_x1 = train_x0 + dynamics_scale * torch.randn_like(train_x0)

    val_x0 = _synth_samples(config.val_samples)
    val_x1 = val_x0 + dynamics_scale * torch.randn_like(val_x0)

    train_dataset = TensorDataset(train_x0, train_x1)
    val_dataset = TensorDataset(val_x0, val_x1)
    return {
        "train": train_dataset,
        "val": val_dataset,
        "channel_names": channel_names,
    }


def _aggregate_channel_stats(data: torch.Tensor, names: List[str]) -> List[ChannelStats]:
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
) -> Dict[str, torch.Tensor]:
    """Compute flow matching loss with optional physics constraints."""

    v_pred = model(x0, t)
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
    }

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

        rmse = torch.sqrt(torch.mean((channel_predictions[-1] - channel_target) ** 2)).item()
        mae = torch.mean(torch.abs(channel_predictions[-1] - channel_target)).item()
        baseline_rmse = torch.sqrt(torch.mean((channel_initial - channel_target) ** 2)).item()

        trajectory = [
            TrajectoryStep(time=float(times[i].item()), data=channel_predictions[i].tolist())
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
) -> Dict[str, object]:
    channel_names: List[str] = datasets["channel_names"]
    train_dataset: TensorDataset = datasets["train"]
    val_dataset: TensorDataset = datasets["val"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.training.batch_size, len(train_dataset)),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.training.batch_size, len(val_dataset)),
        shuffle=False,
    )

    model = WeatherFlowMatch(
        input_channels=len(channel_names),
        hidden_dim=config.model.hidden_dim,
        n_layers=config.model.n_layers,
        use_attention=config.model.use_attention,
        grid_size=(config.dataset.grid_size.lat, config.dataset.grid_size.lon),
        physics_informed=config.model.physics_informed,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    train_metrics: List[MetricEntry] = []
    val_metrics: List[ValidationMetricEntry] = []

    for epoch in range(config.training.epochs):
        model.train()
        train_loss = []
        train_flow = []
        train_div = []
        train_energy = []

        for x0, x1 in train_loader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            t = torch.rand(x0.size(0), device=device)

            losses = _compute_losses(model, x0, x1, t, config.training.loss_type)
            total_loss = losses["total_loss"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss.append(float(total_loss.item()))
            train_flow.append(float(losses["flow_loss"].item()))
            train_div.append(float(losses.get("div_loss", torch.tensor(0.0)).item()))
            train_energy.append(float(losses.get("energy_diff", torch.tensor(0.0)).item()))

        if not train_loss:
            raise RuntimeError("Training dataset is empty")

        train_metrics.append(
            MetricEntry(
                epoch=epoch + 1,
                loss=float(sum(train_loss) / len(train_loss)),
                flow_loss=float(sum(train_flow) / len(train_flow)),
                divergence_loss=float(sum(train_div) / len(train_div)),
                energy_diff=float(sum(train_energy) / len(train_energy)),
            )
        )

        model.eval()
        val_loss = []
        val_flow = []
        val_div = []
        val_energy = []

        with torch.no_grad():
            for x0, x1 in val_loader:
                x0 = x0.to(device)
                x1 = x1.to(device)
                t = torch.rand(x0.size(0), device=device)

                losses = _compute_losses(model, x0, x1, t, config.training.loss_type)
                total_loss = losses["total_loss"]

                val_loss.append(float(total_loss.item()))
                val_flow.append(float(losses["flow_loss"].item()))
                val_div.append(float(losses.get("div_loss", torch.tensor(0.0)).item()))
                val_energy.append(float(losses.get("energy_diff", torch.tensor(0.0)).item()))

        val_metrics.append(
            ValidationMetricEntry(
                epoch=epoch + 1,
                val_loss=float(sum(val_loss) / len(val_loss)),
                val_flow_loss=float(sum(val_flow) / len(val_flow)),
                val_divergence_loss=float(sum(val_div) / len(val_div)),
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
) -> PredictionResult:
    model.eval()
    ode_model = WeatherFlowODE(
        model,
        solver_method=config.training.solver_method,
    ).to(device)
    times = torch.linspace(0.0, 1.0, config.training.time_steps, device=device)

    initial, target = dataset.tensors[0][:1].to(device), dataset.tensors[1][:1].to(device)

    with torch.no_grad():
        predictions = ode_model(initial, times)

    return _prepare_trajectory(predictions, initial, target, times, channel_names)


def _build_dataset_summary(dataset: TensorDataset, channel_names: List[str]) -> DatasetSummary:
    x0 = dataset.tensors[0]
    stats = _aggregate_channel_stats(x0, channel_names)
    return DatasetSummary(
        channel_stats=stats,
        sample_shape=list(x0.shape[1:]),
    )


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
            "maxEpochs": 6,
        }

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/experiments", response_model=ExperimentResult)
    def run_experiment(config: ExperimentConfig) -> ExperimentResult:
        start = time.perf_counter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            torch.manual_seed(config.training.seed)
            datasets = _build_dataloaders(config.dataset, config.training.dynamics_scale)
            training_outcome = _train_model(config, device, datasets)
            prediction = _run_prediction(
                training_outcome["model"],
                config,
                datasets["val"],
                datasets["channel_names"],
                device,
            )
            summary = _build_dataset_summary(datasets["train"], datasets["channel_names"])
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
            execution=ExecutionSummary(duration_seconds=float(end - start)),
        )

    return app


app = create_app()
