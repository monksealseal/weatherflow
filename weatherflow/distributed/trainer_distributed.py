"""Distributed FSDP trainer for WeatherFlow foundation models."""
from __future__ import annotations

import os
import signal
import math
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import FullStateDictConfig, LocalStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb

from .loss_trajectory import TrajectoryFlowLoss
from .model_large import TransformerBlock


@dataclass
class TrainerConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    clip_grad: float = 1.0
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    keep_last: int = 5
    use_compile: bool = True
    use_bf16: bool = True
    validation_interval: int = 1000


def _get_mixed_precision(enable: bool) -> Optional[MixedPrecision]:
    if not enable:
        return None
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )


def _create_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainerConfig) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return float(step) / float(max(1, cfg.warmup_steps))
        progress = (step - cfg.warmup_steps) / float(max(1, cfg.max_steps - cfg.warmup_steps))
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    return LambdaLR(optimizer, lr_lambda)


@contextmanager
def fsdp_state_dict_type(model: FSDP, rank0_only: bool = True):
    cfg = FullStateDictConfig(rank0_only=rank0_only, offload_to_cpu=True)
    local_cfg = LocalStateDictConfig()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg, local_cfg):
        yield


class DistributedFlowTrainer:
    """End-to-end distributed trainer built on PyTorch FSDP."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Iterable,
        val_loader: Optional[Iterable],
        cfg: TrainerConfig,
        loss_fn: Optional[TrajectoryFlowLoss] = None,
        compile_mode: str = "reduce-overhead",
    ):
        if not dist.is_initialized():
            raise RuntimeError("DistributedFlowTrainer requires torch.distributed to be initialized.")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.loss_fn = loss_fn or TrajectoryFlowLoss()
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        auto_wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
        mixed_precision = _get_mixed_precision(cfg.use_bf16)
        self.model = FSDP(
            model.to(self.device),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap,
        )
        if cfg.use_compile:
            self.model = torch.compile(self.model, mode=compile_mode)

        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = _create_scheduler(self.optimizer, cfg)
        self.step = 0
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._stop = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        if self.rank == 0:
            wandb.init(project="weatherflow-foundation", config=cfg.__dict__)

    def _handle_signal(self, signum: int, frame: object) -> None:
        self._stop = True

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_states = batch["input"].to(self.device)
        targets = batch["target"].to(input_states.device)
        dt = batch.get("dt", torch.tensor(1.0, device=input_states.device)).to(input_states.device)
        static = batch.get("static")
        forcing = batch.get("forcing")

        loss_out = self.loss_fn(
            input_states,
            dt=dt,
            model=self.model,
            targets=targets,
            static=static.to(input_states.device) if static is not None else None,
            forcing=forcing.to(input_states.device) if forcing is not None else None,
        )
        return loss_out

    def train(self) -> None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        for batch in self.train_loader:
            if self._stop:
                break
            self.model.train()
            with torch.cuda.amp.autocast(enabled=self.cfg.use_bf16 and torch.cuda.is_available(), dtype=torch.bfloat16 if self.cfg.use_bf16 else torch.float32):
                loss_dict = self._forward(batch)
                loss = loss_dict["total_loss"]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.clip_grad is not None:
                FSDP.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            if self.rank == 0:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()}, "lr": self.scheduler.get_last_lr()[0], "step": self.step})

            if self.step % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint(f"step-{self.step}.pt")
                self._cleanup_old_checkpoints()

            if self.val_loader is not None and self.step % self.cfg.validation_interval == 0:
                self.validate()
            if self.step >= self.cfg.max_steps:
                break
        dist.barrier()

    def validate(self) -> None:
        if self.val_loader is None:
            return
        self.model.eval()
        total_loss = torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
        count = torch.tensor(0, device=total_loss.device)
        with torch.no_grad():
            for batch in self.val_loader:
                loss_dict = self._forward(batch)
                total_loss += loss_dict["total_loss"]
                count += 1
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if self.rank == 0 and count > 0:
            wandb.log({"val_loss": (total_loss / count).item(), "step": self.step})
        self.model.train()

    def save_checkpoint(self, name: str) -> None:
        path = self.checkpoint_dir / name
        state = {
            "step": self.step,
            "scheduler": self.scheduler.state_dict(),
        }
        with fsdp_state_dict_type(self.model, rank0_only=True):
            state["model"] = self.model.state_dict()
            state["optimizer"] = self.optimizer.state_dict()
        if self.rank == 0:
            torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        map_location = {"cuda:%d" % 0: "cuda:%d" % torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=map_location)
        with fsdp_state_dict_type(self.model, rank0_only=False):
            self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint.get("step", 0)

    def _cleanup_old_checkpoints(self) -> None:
        if self.rank != 0:
            return
        ckpts = sorted(self.checkpoint_dir.glob("step-*.pt"), key=os.path.getmtime)
        if len(ckpts) <= self.cfg.keep_last:
            return
        for ckpt in ckpts[:-self.cfg.keep_last]:
            ckpt.unlink(missing_ok=True)


__all__ = ["DistributedFlowTrainer", "TrainerConfig"]
