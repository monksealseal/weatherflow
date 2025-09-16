"""Integration tests for the :mod:`weatherflow.training.flow_trainer` module."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from weatherflow.models.base import BaseWeatherModel
from weatherflow.training.flow_trainer import FlowTrainer, compute_flow_loss


class _SyntheticWeatherDataset(Dataset):
    """Generate consecutive synthetic weather frames."""

    def __init__(self, size: int = 12, channels: int = 3, grid_shape: tuple[int, int] = (8, 16)) -> None:
        super().__init__()
        self.size = size
        self.channels = channels
        self.grid_shape = grid_shape

        generator = torch.Generator().manual_seed(0)
        self.data = torch.randn(size, channels, *grid_shape, generator=generator)

    def __len__(self) -> int:  # type: ignore[override]
        return self.size - 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.data[index], self.data[index + 1]


class _TinyWeatherModel(BaseWeatherModel):
    """Minimal :class:`BaseWeatherModel` implementation for testing."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.encoder(x)
        time_embed = t.view(-1, 1, 1, 1).expand_as(features[:, :1])
        features_with_time = torch.cat([features, time_embed], dim=1)
        return self.decoder(features_with_time)

    def mass_conservation_constraint(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.shape[1] < 2:
            return torch.tensor(0.0, device=x.device)

        du_dx = torch.gradient(x[:, 0], dim=2)[0]
        dv_dy = torch.gradient(x[:, 1], dim=1)[0]
        divergence = du_dx + dv_dy
        return torch.mean(divergence.square())

    def energy_conservation_constraint(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        energy = torch.sum(x.square(), dim=1)
        return torch.var(energy)


def test_flow_trainer_end_to_end(tmp_path: Path) -> None:
    dataset = _SyntheticWeatherDataset()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = _TinyWeatherModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = FlowTrainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        use_amp=False,
        checkpoint_dir=str(tmp_path),
        physics_regularization=True,
        physics_lambda=0.1,
    )

    train_metrics = trainer.train_epoch(train_loader)
    assert set(train_metrics) == {"loss", "flow_loss", "physics_loss"}

    val_metrics = trainer.validate(val_loader)
    assert set(val_metrics) == {"val_loss", "val_flow_loss", "val_physics_loss"}

    checkpoint_name = "trainer.pt"
    trainer.save_checkpoint(checkpoint_name)
    checkpoint_path = tmp_path / checkpoint_name
    assert checkpoint_path.exists()

    trainer.load_checkpoint(checkpoint_name)

    x0, x1 = next(iter(train_loader))
    t = torch.rand(x0.size(0))
    v_t = model(x0, t)
    loss = compute_flow_loss(v_t, x0, x1, t)
    assert loss.item() >= 0
