"""Launch script for distributed WeatherFlow foundation model training."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import yaml

from .data_streaming import ERA5StreamingDataset, StreamingConfig
from .loss_trajectory import TrajectoryFlowLoss
from .model_large import WeatherFlowFoundation, WeatherFlowFoundationConfig
from .trainer_distributed import DistributedFlowTrainer, TrainerConfig


def _init_distributed() -> None:
    if dist.is_initialized():
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict[str, Any]) -> tuple:
    data_cfg = cfg["data"]
    streaming_cfg = StreamingConfig(
        shard_pattern=data_cfg["shard_pattern"],
        sequence_length=data_cfg.get("sequence_length", 2),
        stats_path=data_cfg.get("stats_path"),
        shard_shuffle_buffer=data_cfg.get("shard_shuffle_buffer", 64),
        sample_shuffle_buffer=data_cfg.get("sample_shuffle_buffer", 256),
        prefetch=data_cfg.get("prefetch", 32),
    )
    train_dataset = ERA5StreamingDataset(streaming_cfg, dt_hours=data_cfg.get("dt_hours", 6.0))
    val_dataset = None
    if data_cfg.get("val_shard_pattern"):
        val_cfg = StreamingConfig(
            shard_pattern=data_cfg["val_shard_pattern"],
            sequence_length=data_cfg.get("sequence_length", 2),
            stats_path=data_cfg.get("stats_path"),
            shard_shuffle_buffer=data_cfg.get("shard_shuffle_buffer", 64),
            sample_shuffle_buffer=data_cfg.get("sample_shuffle_buffer", 256),
            prefetch=data_cfg.get("prefetch", 32),
        )
        val_dataset = ERA5StreamingDataset(val_cfg, dt_hours=data_cfg.get("dt_hours", 6.0))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 1),
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=data_cfg.get("batch_size", 1),
            num_workers=data_cfg.get("num_workers", 2),
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _init_distributed()
    rank = dist.get_rank()

    model_cfg = WeatherFlowFoundationConfig(**cfg["model"])
    model = WeatherFlowFoundation(model_cfg)
    train_loader, val_loader = build_dataloaders(cfg)

    trainer_cfg = TrainerConfig(**cfg["trainer"])
    loss_fn = TrajectoryFlowLoss()
    trainer = DistributedFlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=trainer_cfg,
        loss_fn=loss_fn,
    )
    try:
        trainer.train()
    except Exception:
        if rank == 0:
            print("Encountered exception, shutting down...", file=sys.stderr)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
