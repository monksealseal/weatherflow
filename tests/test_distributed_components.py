import os

import torch
import torch.distributed as dist
import pytest

from weatherflow.distributed.loss_trajectory import TrajectoryFlowLoss, TrajectoryLossConfig
from weatherflow.distributed.model_large import WeatherFlowFoundation, WeatherFlowFoundationConfig
from weatherflow.distributed.trainer_distributed import DistributedFlowTrainer, TrainerConfig


def setup_module(module):
    os.environ["WANDB_MODE"] = "offline"


def teardown_module(module):
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def test_foundation_model_forward_shapes():
    cfg = WeatherFlowFoundationConfig(
        input_channels=4,
        levels=1,
        embed_dim=64,
        depth=2,
        num_heads=4,
        patch_size=2,
        static_channels=0,
        forcing_dim=0,
        use_grad_checkpointing=False,
        sphere_regularization=False,
    )
    model = WeatherFlowFoundation(cfg)
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    out = model(x, t)
    assert out.shape == x.shape


def test_trajectory_flow_loss_outputs():
    cfg = WeatherFlowFoundationConfig(
        input_channels=4,
        levels=1,
        embed_dim=32,
        depth=1,
        num_heads=4,
        patch_size=2,
        static_channels=0,
        forcing_dim=0,
        use_grad_checkpointing=False,
        sphere_regularization=False,
    )
    model = WeatherFlowFoundation(cfg)
    traj = torch.randn(1, 3, 4, 4, 4)
    dt = torch.tensor(1.0)
    loss_fn = TrajectoryFlowLoss(TrajectoryLossConfig(max_pairs=2))
    loss_dict = loss_fn(traj, dt=dt, model=model)
    assert "total_loss" in loss_dict and loss_dict["total_loss"].dim() == 0
    assert loss_dict["flow_loss"] >= 0
    assert loss_dict["div_loss"] >= 0


def test_fsdp_checkpoint_roundtrip(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("FSDP checkpoint test requires CUDA availability.")
    if not dist.is_available():
        return
    if not dist.is_initialized():
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    # Silence wandb in tests
    import wandb

    monkeypatch.setattr(wandb, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)

    cfg = WeatherFlowFoundationConfig(
        input_channels=4,
        levels=1,
        embed_dim=32,
        depth=1,
        num_heads=4,
        patch_size=2,
        static_channels=0,
        forcing_dim=0,
        use_grad_checkpointing=False,
        sphere_regularization=False,
    )
    model = WeatherFlowFoundation(cfg)
    dummy_batch = {
        "input": torch.randn(1, 4, 4, 4),
        "target": torch.randn(1, 4, 4, 4),
        "dt": torch.tensor(1.0),
    }
    trainer_cfg = TrainerConfig(
        lr=1e-3,
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=1,
        max_steps=1,
        use_compile=False,
        use_bf16=False,
        validation_interval=10,
    )
    trainer = DistributedFlowTrainer(
        model=model,
        train_loader=[dummy_batch],
        val_loader=None,
        cfg=trainer_cfg,
        loss_fn=TrajectoryFlowLoss(TrajectoryLossConfig(max_pairs=1)),
    )
    trainer.step = 5
    ckpt_path = tmp_path / "test.pt"
    trainer.save_checkpoint(str(ckpt_path))
    assert ckpt_path.exists()
    trainer.load_checkpoint(str(ckpt_path))
    assert trainer.step == 5
