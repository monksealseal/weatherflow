import torch

from weatherflow.training.gaia import (
    RolloutCurriculum,
    calibration_step,
    finetune_step,
    pretrain_step,
)
from weatherflow.training.gaia.calibrate import CalibrationBatch
from weatherflow.training.gaia.finetune import FinetuneBatch
from weatherflow.training.gaia.pretrain import PretrainBatch


class ToyPretrainModel(torch.nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(features, features)
        self.order_head = torch.nn.Linear(features, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)

    def temporal_ordering_logits(self, sequence: torch.Tensor) -> torch.Tensor:
        pooled = sequence.mean(dim=1)
        return self.order_head(pooled)


class ToyAutoregressiveModel(torch.nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(features, features)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        last_step = context[:, -1]
        return self.projection(last_step)


class ToyEnsembleModel(torch.nn.Module):
    def __init__(self, features: int, members: int, horizon: int) -> None:
        super().__init__()
        self.members = members
        self.horizon = horizon
        self.projection = torch.nn.Linear(features, features)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        last_step = context[:, -1]
        base = self.projection(last_step)
        base = base[:, None, :].repeat(1, self.horizon, 1)
        noise = torch.randn(self.members, *base.shape, device=base.device) * 0.1
        return base.unsqueeze(0).repeat(self.members, 1, 1, 1) + noise


def test_pretrain_step_runs_backward() -> None:
    torch.manual_seed(0)
    model = ToyPretrainModel(features=4)
    inputs = torch.randn(8, 5, 4)
    loss, metrics = pretrain_step(model, PretrainBatch(inputs=inputs), mask_ratio=0.2)
    loss.backward()
    assert "total_loss" in metrics


def test_finetune_step_runs_backward() -> None:
    torch.manual_seed(1)
    model = ToyAutoregressiveModel(features=3)
    context = torch.randn(6, 4, 3)
    targets = torch.randn(6, 3, 3)
    curriculum = RolloutCurriculum(start_horizon=1, end_horizon=3, total_steps=10)
    loss, metrics = finetune_step(
        model,
        FinetuneBatch(context=context, targets=targets),
        curriculum,
        global_step=5,
    )
    loss.backward()
    assert metrics["horizon"] >= 1


def test_calibration_step_includes_spectral_term() -> None:
    torch.manual_seed(2)
    model = ToyEnsembleModel(features=2, members=4, horizon=2)
    context = torch.randn(5, 3, 2)
    targets = torch.randn(5, 2, 2)
    loss, metrics = calibration_step(
        model,
        CalibrationBatch(context=context, targets=targets),
        spectral_weight=0.5,
    )
    loss.backward()
    assert "spectral_crps" in metrics
