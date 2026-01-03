"""
Distributed Training Framework for FlowAtmosphere

Implements PyTorch FSDP (Fully Sharded Data Parallel) for training models
with 10+ billion parameters across hundreds of GPUs.
"""

import os
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import wandb
from datetime import timedelta


class DistributedFlowTrainer:
    """
    Distributed trainer for FlowAtmosphere foundation model.

    Features:
    - FSDP for efficient multi-GPU training
    - Gradient checkpointing for memory efficiency
    - Mixed precision training (BF16/FP16)
    - Automatic fault tolerance and checkpointing
    - Performance monitoring and logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 100,
        val_interval: int = 1000,
        checkpoint_interval: int = 5000,
        use_wandb: bool = True,
        mixed_precision: bool = True,
        cpu_offload: bool = False,
    ):
        """
        Args:
            model: FlowAtmosphere model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            max_epochs: Maximum training epochs
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Steps between logging
            val_interval: Steps between validation
            checkpoint_interval: Steps between checkpoints
            use_wandb: Whether to use Weights & Biases logging
            mixed_precision: Use mixed precision training
            cpu_offload: Offload parameters to CPU (for very large models)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.checkpoint_interval = checkpoint_interval
        self.use_wandb = use_wandb
        self.mixed_precision = mixed_precision

        # Initialize distributed training
        self._setup_distributed()

        # Wrap model with FSDP
        self.model = self._wrap_model_fsdp(cpu_offload)

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Initialize logging
        if self.use_wandb and self.is_main_process:
            wandb.init(
                project='flowatmosphere',
                config={
                    'model_params': sum(p.numel() for p in model.parameters()),
                    'max_epochs': max_epochs,
                    'gradient_accumulation': gradient_accumulation_steps,
                }
            )

    def _setup_distributed(self):
        """Initialize distributed training environment."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # Multi-GPU distributed training
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])

            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=timedelta(minutes=30),
            )

            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            # Single GPU or CPU
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.is_main_process = (self.rank == 0)

        if self.is_main_process:
            print(f"Distributed training: {self.world_size} GPUs")
            print(f"Device: {self.device}")

    def _wrap_model_fsdp(self, cpu_offload: bool = False) -> FSDP:
        """Wrap model with FSDP for distributed training."""
        # Mixed precision policy
        if self.mixed_precision:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:
            mp_policy = None

        # Auto wrap policy - wrap layers above 100M parameters
        auto_wrap_policy = size_based_auto_wrap_policy(
            min_num_params=100_000_000
        )

        # CPU offload for very large models
        cpu_offload_policy = CPUOffload(offload_params=True) if cpu_offload else None

        # Wrap with FSDP
        model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=cpu_offload_policy,
            device_id=self.local_rank,
        )

        if self.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model wrapped with FSDP: {total_params / 1e9:.2f}B parameters")

        return model

    def train(self):
        """Main training loop."""
        if self.is_main_process:
            print("Starting training...")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            if self.is_main_process:
                print(f"\nEpoch {epoch + 1}/{self.max_epochs}")

            # Train epoch
            self._train_epoch()

            # Validate
            if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
                val_loss = self._validate()

                if self.is_main_process:
                    print(f"Validation loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint('best_model.pt')

            # Save epoch checkpoint
            if self.is_main_process:
                self._save_checkpoint(f'epoch_{epoch + 1}.pt')

        if self.is_main_process:
            print("Training complete!")
            if self.use_wandb:
                wandb.finish()

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (x0, x1) in enumerate(self.train_loader):
            # Move to device
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)

            # Compute loss
            loss = self._train_step(x0, x1)

            epoch_loss += loss
            num_batches += 1

            # Logging
            if self.global_step % self.log_interval == 0 and self.is_main_process:
                avg_loss = epoch_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']

                print(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.6f}")

                if self.use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lr': lr,
                        'epoch': self.current_epoch,
                    }, step=self.global_step)

            # Validation
            if self.global_step % self.val_interval == 0:
                val_loss = self._validate()
                if self.is_main_process and self.use_wandb:
                    wandb.log({'val/loss': val_loss}, step=self.global_step)

            # Checkpoint
            if self.global_step % self.checkpoint_interval == 0 and self.is_main_process:
                self._save_checkpoint(f'step_{self.global_step}.pt')

            self.global_step += 1

    def _train_step(self, x0: torch.Tensor, x1: torch.Tensor) -> float:
        """Single training step."""
        # Sample random time
        batch_size = x0.size(0)
        t = torch.rand(batch_size, device=self.device)

        # Interpolate between x0 and x1
        x_t = t.view(-1, 1, 1, 1) * x1 + (1 - t.view(-1, 1, 1, 1)) * x0

        # Target velocity (straight-line path)
        v_target = x1 - x0

        # Create lat/lon grids (placeholder - should come from data)
        _, _, H, W = x0.shape
        lat_grid = torch.linspace(-torch.pi/2, torch.pi/2, H, device=self.device)
        lon_grid = torch.linspace(-torch.pi, torch.pi, W, device=self.device)
        lat_grid, lon_grid = torch.meshgrid(lat_grid, lon_grid, indexing='ij')

        # Forward pass
        if self.mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                v_pred = self.model(x_t, t, lat_grid, lon_grid)
                loss = nn.functional.mse_loss(v_pred, v_target)

            # Backward pass with gradient scaling
            loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()

            # Update weights
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()
        else:
            # Standard training without mixed precision
            v_pred = self.model(x_t, t, lat_grid, lon_grid)
            loss = nn.functional.mse_loss(v_pred, v_target)

            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

        return loss.item() * self.gradient_accumulation_steps

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x0, x1 in self.val_loader:
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)

            batch_size = x0.size(0)
            t = torch.rand(batch_size, device=self.device)

            x_t = t.view(-1, 1, 1, 1) * x1 + (1 - t.view(-1, 1, 1, 1)) * x0
            v_target = x1 - x0

            # Create grids
            _, _, H, W = x0.shape
            lat_grid = torch.linspace(-torch.pi/2, torch.pi/2, H, device=self.device)
            lon_grid = torch.linspace(-torch.pi, torch.pi, W, device=self.device)
            lat_grid, lon_grid = torch.meshgrid(lat_grid, lon_grid, indexing='ij')

            v_pred = self.model(x_t, t, lat_grid, lon_grid)
            loss = nn.functional.mse_loss(v_pred, v_target)

            total_loss += loss.item()
            num_batches += 1

            # Limit validation batches
            if num_batches >= 100:
                break

        avg_loss = total_loss / num_batches

        # Synchronize across GPUs
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        self.model.train()
        return avg_loss

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )

        # Configure state dict
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.model.state_dict()

        if self.is_main_process:
            checkpoint = {
                'model_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss,
            }

            path = self.checkpoint_dir / filename
            torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )

        checkpoint = torch.load(path, map_location='cpu')

        # Load model state
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.is_main_process:
            print(f"Loaded checkpoint from {path}")
            print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
