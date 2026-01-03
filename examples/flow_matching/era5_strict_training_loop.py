"""Production-ready flow matching loop that maps pure noise to real ERA5 data.

This script:
* Streams normalized ERA5 tensors from NetCDF files using
  ``weatherflow.data.era5.ERA5Dataset``.
* Samples Gaussian noise as the source distribution and learns a vector field
  that transports the noise toward the real ERA5 state.
* Runs a lightweight validation pass every few epochs to track progress on real
  data.
* Saves checkpoints to disk so you can resume or load the trained weights
  elsewhere.

Usage (Colab-friendly):
    python examples/flow_matching/era5_strict_training_loop.py \\
        --data-root /content/drive/MyDrive/era5 \\
        --checkpoint-dir /content/drive/MyDrive/weatherflow_ckpts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn, optim

from weatherflow.data.era5 import create_data_loaders


@dataclass
class TrainConfig:
    """Configuration for the ERA5 flow-matching run."""

    data_root: Path
    checkpoint_dir: Path
    train_years: Sequence[int]
    val_years: Sequence[int]
    variables: Sequence[str]
    levels: Sequence[int]
    batch_size: int = 4
    num_workers: int = 0
    learning_rate: float = 1e-4
    num_epochs: int = 10
    val_every: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleVectorField(nn.Module):
    """Minimal CNN vector field that conditions on continuous time."""

    def __init__(self, input_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:  # type: ignore[override]
        time_emb = self.time_mlp(t.view(-1, 1)).unsqueeze(-1).unsqueeze(-1)
        return self.net(x_t + time_emb)


def _flatten_batch(real_batch: torch.Tensor) -> torch.Tensor:
    """Collapse variable and level dims into a channel axis."""
    batch_size, n_vars, n_levels, lat, lon = real_batch.shape
    return real_batch.view(batch_size, n_vars * n_levels, lat, lon)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    epoch_loss = 0.0

    for real_batch in loader:
        x_1 = _flatten_batch(real_batch.to(device))
        batch_size = x_1.shape[0]

        x_0 = torch.randn_like(x_1, device=device)
        t = torch.rand(batch_size, device=device)
        t_view = t.view(-1, 1, 1, 1)

        x_t = (1 - t_view) * x_0 + t_view * x_1
        target_vector = x_1 - x_0

        predicted_vector = model(x_t, t)
        loss = torch.mean((predicted_vector - target_vector) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def validate_epoch(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> float:
    model.eval()
    total_loss = 0.0

    for real_batch in loader:
        x_1 = _flatten_batch(real_batch.to(device))
        batch_size = x_1.shape[0]

        x_0 = torch.randn_like(x_1, device=device)
        t = torch.rand(batch_size, device=device)
        t_view = t.view(-1, 1, 1, 1)

        x_t = (1 - t_view) * x_0 + t_view * x_1
        target_vector = x_1 - x_0

        predicted_vector = model(x_t, t)
        loss = torch.mean((predicted_vector - target_vector) ** 2)
        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(model: nn.Module, checkpoint_dir: Path, epoch: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"weatherflow_epoch_{epoch}.pt"
    torch.save(model.state_dict(), path)
    return path


def run_training(cfg: TrainConfig) -> None:
    train_loader, val_loader = create_data_loaders(
        root_dir=str(cfg.data_root),
        train_years=cfg.train_years,
        val_years=cfg.val_years,
        variables=cfg.variables,
        levels=cfg.levels,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        download=False,
    )

    input_channels = len(cfg.variables) * len(cfg.levels)
    model = SimpleVectorField(input_channels).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(f"🔥 Training on REAL ERA5 data from: {cfg.data_root}")
    print(f"   Device: {cfg.device}")
    print(f"   Batches per epoch: {len(train_loader)}")

    for epoch in range(cfg.num_epochs):
        train_loss = train_epoch(model, train_loader, cfg.device, optimizer)

        log_message = (
            f"✅ Epoch {epoch + 1}/{cfg.num_epochs} | Train Loss: {train_loss:.6f}"
        )
        if (epoch + 1) % cfg.val_every == 0:
            val_loss = validate_epoch(model, val_loader, cfg.device)
            log_message += f" | Val Loss: {val_loss:.6f}"
        print(log_message)

        ckpt_path = save_checkpoint(model, cfg.checkpoint_dir, epoch + 1)
        print(f"   💾 Model saved to: {ckpt_path}")


def parse_years(years: Iterable[str]) -> Sequence[int]:
    return [int(year) for year in years]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Strict flow matching on real ERA5 data."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder containing ERA5 NetCDF files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Where to store model checkpoints.",
    )
    parser.add_argument(
        "--train-years",
        nargs="+",
        default=["2018", "2019"],
        help="Training years (e.g., 2018 2019).",
    )
    parser.add_argument(
        "--val-years",
        nargs="+",
        default=["2020"],
        help="Validation years (e.g., 2020).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["u_component_of_wind", "v_component_of_wind"],
        help="ERA5 variable names.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["850", "500"],
        help="Pressure levels (hPa).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=5)

    args = parser.parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        train_years=parse_years(args.train_years),
        val_years=parse_years(args.val_years),
        variables=args.variables,
        levels=parse_years(args.levels),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        val_every=args.val_every,
    )

    run_training(config)
