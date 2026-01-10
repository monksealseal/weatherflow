"""
Video Diffusion Trainer for Atmospheric Sequence Prediction

Specialized training infrastructure for fine-tuning video diffusion models
(like Stable Video Diffusion) on atmospheric data sequences.

Worldsphere uses this for:
- Predicting sequences of 25 frames from atmospheric data
- Multi-variable prediction (e.g., satellite + wind at each timestep)
- Leveraging pre-trained diffusion models for atmospheric understanding

Features:
- Sequence prediction from initial frame(s)
- Multi-variable output (multiple images per timestep)
- DDPM and DDIM training/sampling
- Classifier-free guidance support
- RMSE tracking for experiment comparison
- Hyperparameter logging
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class VideoDiffusionConfig:
    """Configuration for video diffusion training."""

    # Model architecture
    in_channels: int = 3  # Input channels per frame
    out_channels: int = 3  # Output channels per frame
    model_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    num_heads: int = 8

    # Sequence configuration
    num_frames: int = 25  # Number of frames to predict
    conditioning_frames: int = 1  # Number of conditioning frames
    frame_size: Tuple[int, int] = (256, 256)
    variables_per_frame: int = 1  # For multi-variable prediction

    # Diffusion parameters
    num_timesteps: int = 1000
    beta_schedule: str = "linear"  # "linear" or "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    # Training options
    use_amp: bool = True
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    use_ema: bool = True

    # Classifier-free guidance
    use_cfg: bool = True
    cfg_scale: float = 7.5
    uncond_prob: float = 0.1  # Probability of dropping conditioning

    # Sampling
    num_inference_steps: int = 50  # DDIM steps
    eta: float = 0.0  # DDIM eta (0 = deterministic)

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_every: int = 10

    # Experiment tracking
    experiment_name: str = "video_diffusion_experiment"
    variable_names: List[str] = field(default_factory=lambda: ["satellite_brightness"])
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = asdict(self)
        # Convert tuples to lists for JSON serialization
        result["channel_mult"] = list(result["channel_mult"])
        result["attention_resolutions"] = list(result["attention_resolutions"])
        result["frame_size"] = list(result["frame_size"])
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoDiffusionConfig":
        """Create config from dictionary."""
        # Convert lists back to tuples
        if "channel_mult" in data:
            data["channel_mult"] = tuple(data["channel_mult"])
        if "attention_resolutions" in data:
            data["attention_resolutions"] = tuple(data["attention_resolutions"])
        if "frame_size" in data:
            data["frame_size"] = tuple(data["frame_size"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class VideoDiffusionMetrics:
    """Metrics from video diffusion training."""

    epoch: int = 0
    step: int = 0

    # Losses
    loss: float = 0.0
    mse_loss: float = 0.0
    vlb_loss: float = 0.0

    # Per-frame metrics
    frame_rmse: List[float] = field(default_factory=list)
    frame_mae: List[float] = field(default_factory=list)

    # Sequence metrics
    sequence_rmse: float = 0.0
    sequence_mae: float = 0.0
    temporal_consistency: float = 0.0

    # Variable-specific metrics
    variable_rmse: Dict[str, float] = field(default_factory=dict)

    # Training info
    learning_rate: float = 0.0
    gpu_memory_mb: float = 0.0
    samples_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class NoiseScheduler:
    """
    Noise scheduler for diffusion training and sampling.

    Supports linear and cosine beta schedules.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: str = "cuda",
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).to(device)

        # Posterior variance
        self.posterior_variance = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        ).to(device)

    def add_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x at timestep t."""
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)

        noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        return noisy, noise

    def get_velocity(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Get velocity target for v-prediction."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x


class VideoDiffusionTrainer:
    """
    Training infrastructure for video diffusion models.

    Supports:
    - Fine-tuning pre-trained video diffusion models
    - Training from scratch for atmospheric sequences
    - Multi-variable sequence prediction
    - Comprehensive RMSE tracking

    Example:
        >>> model = VideoDiffusionModel(...)
        >>> config = VideoDiffusionConfig(
        ...     num_frames=25,
        ...     variable_names=["brightness_temp", "wind_speed"],
        ...     epochs=100
        ... )
        >>> trainer = VideoDiffusionTrainer(model, config)
        >>> history = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[VideoDiffusionConfig] = None,
    ):
        """
        Initialize video diffusion trainer.

        Args:
            model: Video diffusion model
            config: Training configuration
        """
        if config is None:
            config = VideoDiffusionConfig()

        self.config = config
        self.device = config.device
        self.model = model.to(self.device)

        # Create noise scheduler
        self.scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.beta_schedule,
            device=self.device,
        )

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate / 100,
        )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # EMA model
        if config.use_ema:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.history: List[VideoDiffusionMetrics] = []
        self.best_rmse = float("inf")

        # Checkpoint directory
        if config.checkpoint_dir:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Experiment log
        self.experiment_log = {
            "config": config.to_dict(),
            "start_time": datetime.now().isoformat(),
            "history": [],
        }

    def _create_ema_model(self) -> nn.Module:
        """Create EMA copy of model."""
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def _update_ema(self) -> None:
        """Update EMA weights."""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.mul_(self.config.ema_decay).add_(
                    param.data, alpha=1 - self.config.ema_decay
                )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dictionary containing:
                - "frames": (batch, num_frames, channels, height, width)
                - "condition": Optional conditioning frames

        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()

        frames = batch["frames"].to(self.device)
        condition = batch.get("condition")
        if condition is not None:
            condition = condition.to(self.device)

        batch_size = frames.shape[0]

        # Sample random timesteps
        t = torch.randint(
            0, self.config.num_timesteps, (batch_size,), device=self.device
        )

        # Add noise
        noise = torch.randn_like(frames)
        noisy_frames, _ = self.scheduler.add_noise(frames, t, noise)

        # Classifier-free guidance: randomly drop conditioning
        if self.config.use_cfg and condition is not None:
            drop_mask = torch.rand(batch_size, device=self.device) < self.config.uncond_prob
            condition = torch.where(
                drop_mask.view(-1, 1, 1, 1, 1),
                torch.zeros_like(condition),
                condition,
            )

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            # Model predicts noise
            noise_pred = self.model(noisy_frames, t, condition)

            # MSE loss
            loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
            if self.config.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.optimizer.step()

        # Update EMA
        self._update_ema()

        self.global_step += 1

        return {"loss": loss.item(), "mse_loss": loss.item()}

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> VideoDiffusionMetrics:
        """Train one epoch."""
        metrics = VideoDiffusionMetrics(epoch=self.current_epoch)
        num_batches = len(train_loader)

        total_loss = 0.0
        start_time = datetime.now()
        num_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            step_metrics = self.train_step(batch)
            total_loss += step_metrics["loss"]
            num_samples += batch["frames"].shape[0]

            pbar.set_postfix({"loss": f"{step_metrics['loss']:.4f}"})

        # Compute metrics
        elapsed = (datetime.now() - start_time).total_seconds()
        metrics.loss = total_loss / num_batches
        metrics.mse_loss = total_loss / num_batches
        metrics.learning_rate = self.optimizer.param_groups[0]["lr"]
        metrics.samples_per_second = num_samples / elapsed

        if torch.cuda.is_available():
            metrics.gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        num_samples: int = 4,
    ) -> VideoDiffusionMetrics:
        """
        Validate the model by generating sequences and comparing to ground truth.

        Args:
            val_loader: Validation data loader
            num_samples: Number of samples to generate for validation

        Returns:
            Validation metrics
        """
        model = self.ema_model if self.ema_model else self.model
        model.eval()

        metrics = VideoDiffusionMetrics(epoch=self.current_epoch)
        total_rmse = 0.0
        total_mae = 0.0
        frame_rmses = []

        samples_validated = 0

        for batch in val_loader:
            if samples_validated >= num_samples:
                break

            frames = batch["frames"].to(self.device)
            condition = batch.get("condition")
            if condition is not None:
                condition = condition.to(self.device)

            # Generate sequences
            generated = self.sample(
                condition=condition,
                batch_size=frames.shape[0],
            )

            # Compute metrics
            rmse = torch.sqrt(F.mse_loss(generated, frames)).item()
            mae = F.l1_loss(generated, frames).item()

            total_rmse += rmse
            total_mae += mae

            # Per-frame RMSE
            for f in range(frames.shape[1]):
                frame_rmse = torch.sqrt(
                    F.mse_loss(generated[:, f], frames[:, f])
                ).item()
                if len(frame_rmses) <= f:
                    frame_rmses.append([])
                frame_rmses[f].append(frame_rmse)

            samples_validated += frames.shape[0]

        # Average metrics
        num_batches = max(1, samples_validated // val_loader.batch_size)
        metrics.sequence_rmse = total_rmse / num_batches
        metrics.sequence_mae = total_mae / num_batches
        metrics.rmse = metrics.sequence_rmse

        # Per-frame metrics
        metrics.frame_rmse = [np.mean(fr) for fr in frame_rmses] if frame_rmses else []
        metrics.frame_mae = []

        return metrics

    @torch.no_grad()
    def sample(
        self,
        condition: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample sequences using DDIM.

        Args:
            condition: Optional conditioning frames
            batch_size: Number of sequences to generate
            num_inference_steps: Number of DDIM steps

        Returns:
            Generated sequences (batch, num_frames, channels, height, width)
        """
        model = self.ema_model if self.ema_model else self.model
        model.eval()

        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        # Initialize with noise
        shape = (
            batch_size,
            self.config.num_frames,
            self.config.out_channels,
            self.config.frame_size[0],
            self.config.frame_size[1],
        )
        x = torch.randn(shape, device=self.device)

        # DDIM sampling steps
        step_ratio = self.config.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.config.num_timesteps, step_ratio))[::-1]

        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise
            noise_pred = model(x, t_batch, condition)

            # Classifier-free guidance
            if self.config.use_cfg and condition is not None:
                noise_pred_uncond = model(x, t_batch, None)
                noise_pred = noise_pred_uncond + self.config.cfg_scale * (
                    noise_pred - noise_pred_uncond
                )

            # DDIM step
            alpha_bar_t = self.scheduler.alphas_cumprod[t]
            alpha_bar_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else torch.tensor(1.0, device=self.device)
            )

            pred_x0 = (
                x - torch.sqrt(1 - alpha_bar_t) * noise_pred
            ) / torch.sqrt(alpha_bar_t)

            # Clip prediction
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * noise_pred

            # DDIM step
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt

            # Add noise for stochastic sampling
            if self.config.eta > 0 and i < len(timesteps) - 1:
                sigma = self.config.eta * torch.sqrt(
                    (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
                )
                noise = torch.randn_like(x)
                x = x + sigma * noise

        return x

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        progress_callback: Optional[Callable[[int, VideoDiffusionMetrics], None]] = None,
    ) -> List[VideoDiffusionMetrics]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            progress_callback: Optional callback(epoch, metrics)

        Returns:
            List of metrics for each epoch
        """
        logger.info(f"Starting video diffusion training: {self.config.experiment_name}")
        logger.info(f"Num frames: {self.config.num_frames}")
        logger.info(f"Device: {self.device}")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                train_metrics.sequence_rmse = val_metrics.sequence_rmse
                train_metrics.rmse = val_metrics.rmse
                train_metrics.frame_rmse = val_metrics.frame_rmse

            # Update LR scheduler
            self.lr_scheduler.step()

            # Store history
            self.history.append(train_metrics)
            self.experiment_log["history"].append(train_metrics.to_dict())

            # Update best RMSE
            if train_metrics.rmse < self.best_rmse:
                self.best_rmse = train_metrics.rmse
                if self.checkpoint_dir:
                    self.save_checkpoint("best_model.pt")

            # Save periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Progress callback
            if progress_callback:
                progress_callback(epoch, train_metrics)

            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Loss: {train_metrics.loss:.4f} | "
                f"RMSE: {train_metrics.rmse:.4f} | "
                f"LR: {train_metrics.learning_rate:.2e}"
            )

        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint("final_model.pt")
            self._save_experiment_log()

        logger.info(f"Training complete. Best RMSE: {self.best_rmse:.4f}")

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "config": self.config.to_dict(),
            "best_rmse": self.best_rmse,
            "history": [m.to_dict() for m in self.history],
        }

        if self.ema_model:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not set")

        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_rmse = checkpoint.get("best_rmse", float("inf"))

        if self.ema_model and "ema_model_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def _save_experiment_log(self) -> None:
        """Save experiment log to JSON."""
        if not self.checkpoint_dir:
            return

        self.experiment_log["end_time"] = datetime.now().isoformat()
        self.experiment_log["best_rmse"] = self.best_rmse

        log_path = self.checkpoint_dir / "experiment_log.json"
        with open(log_path, "w") as f:
            json.dump(self.experiment_log, f, indent=2)

        logger.info(f"Experiment log saved: {log_path}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        if not self.history:
            return {}

        return {
            "experiment_name": self.config.experiment_name,
            "num_frames": self.config.num_frames,
            "epochs_completed": len(self.history),
            "best_rmse": self.best_rmse,
            "final_loss": self.history[-1].loss if self.history else 0,
            "final_rmse": self.history[-1].rmse if self.history else 0,
            "variables": self.config.variable_names,
            "hyperparameters": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_timesteps": self.config.num_timesteps,
                "num_inference_steps": self.config.num_inference_steps,
                "cfg_scale": self.config.cfg_scale,
            },
        }
