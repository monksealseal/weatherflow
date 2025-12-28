import torch
from torch.utils.data import DataLoader, TensorDataset

from weatherflow.training.flow_trainer import FlowTrainer


class DummyModel(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Simple learnable parameter to avoid empty param list issues
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, t, style=None):
        # Return a simple scaled version to keep shapes consistent
        return (x * 0.0 + t.view(-1, 1, 1, 1)) * self.scale


def build_loader(batch: int = 2, channels: int = 2):
    x0 = torch.randn(batch, channels, 4, 4)
    x1 = torch.randn(batch, channels, 4, 4)
    ds = TensorDataset(x0, x1)
    return DataLoader(ds, batch_size=batch)


def test_flow_trainer_runs_train_and_val():
    loader = build_loader()
    model = DummyModel(channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = FlowTrainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        use_amp=False,
        grad_clip=1.0,
        ema_decay=0.9,
        seed=42,
        noise_std=(0.0, 0.05),
    )

    train_metrics = trainer.train_epoch(loader)
    val_metrics = trainer.validate(loader)

    assert "loss" in train_metrics
    assert "val_loss" in val_metrics
    assert "val_rmse" in val_metrics
    assert "val_mae" in val_metrics
