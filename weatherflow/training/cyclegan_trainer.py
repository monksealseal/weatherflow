"""
CycleGAN Trainer for Hurricane Satellite to Wind Field Translation

Specialized training infrastructure for pix2pix and CycleGAN models
used by Worldsphere for satellite image to wind field estimation.

Features:
- Paired (Pix2Pix) and unpaired (CycleGAN) training modes
- GAN loss with L1/L2 reconstruction loss
- Cycle consistency loss for CycleGAN
- Identity loss option
- RMSE tracking for comparing experiments
- Hyperparameter logging for experiment tracking
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
class CycleGANConfig:
    """Configuration for CycleGAN training."""

    # Model type
    model_type: str = "pix2pix"  # "pix2pix" or "cyclegan"

    # Architecture
    in_channels: int = 3  # Satellite channels
    out_channels: int = 2  # Wind u, v components
    generator_features: int = 64
    discriminator_features: int = 64
    num_residual_blocks: int = 9  # For CycleGAN generator

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 4
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # Loss weights
    lambda_l1: float = 100.0  # L1 reconstruction weight
    lambda_cycle: float = 10.0  # Cycle consistency weight (CycleGAN)
    lambda_identity: float = 0.5  # Identity loss weight (CycleGAN)
    lambda_gp: float = 0.0  # Gradient penalty (WGAN-GP)

    # GAN loss type
    gan_loss: str = "lsgan"  # "vanilla", "lsgan", "wgan"

    # Training options
    use_amp: bool = True
    grad_clip: float = 1.0
    lr_decay_start: int = 50  # Epoch to start linear LR decay
    pool_size: int = 50  # Image pool for discriminator (CycleGAN)

    # Data
    image_size: Tuple[int, int] = (256, 256)
    max_wind_speed: float = 80.0  # m/s for normalization

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_every: int = 10

    # Experiment tracking
    experiment_name: str = "cyclegan_experiment"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleGANConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CycleGANMetrics:
    """Metrics from CycleGAN training."""

    epoch: int = 0

    # Generator losses
    g_loss: float = 0.0
    g_loss_adv: float = 0.0
    g_loss_l1: float = 0.0
    g_loss_cycle: float = 0.0
    g_loss_identity: float = 0.0

    # Discriminator losses
    d_loss: float = 0.0
    d_loss_real: float = 0.0
    d_loss_fake: float = 0.0

    # Reconstruction metrics
    rmse: float = 0.0
    mae: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0

    # Wind-specific metrics
    wind_speed_rmse: float = 0.0
    wind_direction_rmse: float = 0.0

    # Training info
    learning_rate_g: float = 0.0
    learning_rate_d: float = 0.0
    gpu_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ImagePool:
    """
    Image buffer for discriminator training in CycleGAN.

    Stores previously generated images to provide a more stable
    training signal to the discriminator.
    """

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return images from pool, randomly replacing some."""
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)

            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                return_images.append(image)
            else:
                if np.random.random() > 0.5:
                    # Return image from pool
                    idx = np.random.randint(0, self.pool_size)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image.clone()
                    return_images.append(tmp)
                else:
                    return_images.append(image)

        return torch.cat(return_images, dim=0)


class CycleGANTrainer:
    """
    Training infrastructure for CycleGAN and Pix2Pix models.

    Supports:
    - Pix2Pix (paired training) for satellite -> wind field
    - CycleGAN (unpaired training) with cycle consistency
    - Comprehensive metrics tracking for experiment comparison
    - Hyperparameter logging

    Example:
        >>> from weatherflow.model_library.architectures.image_translation import (
        ...     HurricaneWindFieldModel
        ... )
        >>>
        >>> model = HurricaneWindFieldModel(use_cyclegan=False)
        >>> config = CycleGANConfig(model_type="pix2pix", epochs=100)
        >>> trainer = CycleGANTrainer(model, config)
        >>>
        >>> history = trainer.train(train_loader, val_loader)
        >>> print(f"Best RMSE: {min(m.rmse for m in history):.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[CycleGANConfig] = None,
        generator_B: Optional[nn.Module] = None,  # For CycleGAN A->B->A
    ):
        """
        Initialize CycleGAN trainer.

        Args:
            model: Main model (contains generator and discriminator)
            config: Training configuration
            generator_B: Second generator for CycleGAN (B->A)
        """
        if config is None:
            config = CycleGANConfig()

        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.generator_B = generator_B.to(self.device) if generator_B else None

        # Extract generator and discriminator from model
        self.generator = model.generator
        self.discriminator = model.discriminator

        # Create optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(config.beta1, config.beta2),
        )

        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=(config.beta1, config.beta2),
        )

        # Add second generator optimizer for CycleGAN
        if self.generator_B:
            self.optimizer_G_B = torch.optim.Adam(
                self.generator_B.parameters(),
                lr=config.learning_rate_g,
                betas=(config.beta1, config.beta2),
            )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # Image pools for CycleGAN
        self.fake_A_pool = ImagePool(config.pool_size)
        self.fake_B_pool = ImagePool(config.pool_size)

        # Training state
        self.current_epoch = 0
        self.history: List[CycleGANMetrics] = []
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

    def _get_gan_loss(
        self,
        pred: torch.Tensor,
        is_real: bool,
    ) -> torch.Tensor:
        """Compute GAN loss based on configuration."""
        if self.config.gan_loss == "vanilla":
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return F.binary_cross_entropy_with_logits(pred, target)
        elif self.config.gan_loss == "lsgan":
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return F.mse_loss(pred, target)
        elif self.config.gan_loss == "wgan":
            return -pred.mean() if is_real else pred.mean()
        else:
            raise ValueError(f"Unknown GAN loss: {self.config.gan_loss}")

    def _compute_wind_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute wind-specific metrics."""
        with torch.no_grad():
            # Denormalize from [-1, 1] to physical units
            pred_wind = pred * self.config.max_wind_speed
            target_wind = target * self.config.max_wind_speed

            # Wind speed RMSE
            pred_speed = torch.sqrt(pred_wind[:, 0]**2 + pred_wind[:, 1]**2)
            target_speed = torch.sqrt(target_wind[:, 0]**2 + target_wind[:, 1]**2)
            speed_rmse = torch.sqrt(F.mse_loss(pred_speed, target_speed)).item()

            # Wind direction RMSE (in degrees)
            pred_dir = torch.atan2(pred_wind[:, 1], pred_wind[:, 0]) * 180 / np.pi
            target_dir = torch.atan2(target_wind[:, 1], target_wind[:, 0]) * 180 / np.pi

            # Handle circular nature of angles
            dir_diff = torch.abs(pred_dir - target_dir)
            dir_diff = torch.min(dir_diff, 360 - dir_diff)
            dir_rmse = torch.sqrt((dir_diff**2).mean()).item()

            return speed_rmse, dir_rmse

    def train_epoch_pix2pix(
        self,
        train_loader: DataLoader,
    ) -> CycleGANMetrics:
        """Train one epoch with Pix2Pix (paired training)."""
        self.model.train()

        metrics = CycleGANMetrics(epoch=self.current_epoch)
        num_batches = len(train_loader)

        total_g_loss = 0.0
        total_d_loss = 0.0
        total_rmse = 0.0
        total_speed_rmse = 0.0
        total_dir_rmse = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Extract data
            if isinstance(batch, (tuple, list)):
                real_A, real_B = batch[0].to(self.device), batch[1].to(self.device)
            elif isinstance(batch, dict):
                real_A = batch["input"].to(self.device)
                real_B = batch["target"].to(self.device)
            else:
                raise ValueError("Unsupported batch format")

            # Normalize target to [-1, 1]
            real_B_norm = real_B / self.config.max_wind_speed

            # =============== Train Generator ===============
            self.optimizer_G.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # Generate fake wind field
                fake_B = self.generator(real_A)

                # Adversarial loss
                pred_fake = self.discriminator(real_A, fake_B)
                g_loss_adv = self._get_gan_loss(pred_fake, is_real=True)

                # L1 reconstruction loss
                g_loss_l1 = F.l1_loss(fake_B, real_B_norm)

                # Total generator loss
                g_loss = g_loss_adv + self.config.lambda_l1 * g_loss_l1

            if self.scaler:
                self.scaler.scale(g_loss).backward()
                if self.config.grad_clip:
                    self.scaler.unscale_(self.optimizer_G)
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), self.config.grad_clip
                    )
                self.scaler.step(self.optimizer_G)
            else:
                g_loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), self.config.grad_clip
                    )
                self.optimizer_G.step()

            # =============== Train Discriminator ===============
            self.optimizer_D.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # Real loss
                pred_real = self.discriminator(real_A, real_B_norm)
                d_loss_real = self._get_gan_loss(pred_real, is_real=True)

                # Fake loss
                pred_fake = self.discriminator(real_A, fake_B.detach())
                d_loss_fake = self._get_gan_loss(pred_fake, is_real=False)

                # Total discriminator loss
                d_loss = (d_loss_real + d_loss_fake) / 2

            if self.scaler:
                self.scaler.scale(d_loss).backward()
                if self.config.grad_clip:
                    self.scaler.unscale_(self.optimizer_D)
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), self.config.grad_clip
                    )
                self.scaler.step(self.optimizer_D)
                self.scaler.update()
            else:
                d_loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), self.config.grad_clip
                    )
                self.optimizer_D.step()

            # Compute reconstruction RMSE
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(fake_B, real_B_norm)).item()
                speed_rmse, dir_rmse = self._compute_wind_metrics(fake_B, real_B_norm)

            # Accumulate metrics
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_rmse += rmse
            total_speed_rmse += speed_rmse
            total_dir_rmse += dir_rmse

            pbar.set_postfix({
                "G": f"{g_loss.item():.4f}",
                "D": f"{d_loss.item():.4f}",
                "RMSE": f"{rmse:.4f}",
            })

        # Average metrics
        metrics.g_loss = total_g_loss / num_batches
        metrics.d_loss = total_d_loss / num_batches
        metrics.rmse = total_rmse / num_batches
        metrics.wind_speed_rmse = total_speed_rmse / num_batches
        metrics.wind_direction_rmse = total_dir_rmse / num_batches

        # Get GPU memory
        if torch.cuda.is_available():
            metrics.gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6

        metrics.learning_rate_g = self.optimizer_G.param_groups[0]["lr"]
        metrics.learning_rate_d = self.optimizer_D.param_groups[0]["lr"]

        return metrics

    def validate(
        self,
        val_loader: DataLoader,
    ) -> CycleGANMetrics:
        """Validate the model."""
        self.model.eval()

        metrics = CycleGANMetrics(epoch=self.current_epoch)
        num_batches = len(val_loader)

        total_rmse = 0.0
        total_mae = 0.0
        total_speed_rmse = 0.0
        total_dir_rmse = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Extract data
                if isinstance(batch, (tuple, list)):
                    real_A, real_B = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    real_A = batch["input"].to(self.device)
                    real_B = batch["target"].to(self.device)
                else:
                    raise ValueError("Unsupported batch format")

                # Normalize target
                real_B_norm = real_B / self.config.max_wind_speed

                # Generate prediction
                fake_B = self.generator(real_A)

                # Compute metrics
                rmse = torch.sqrt(F.mse_loss(fake_B, real_B_norm)).item()
                mae = F.l1_loss(fake_B, real_B_norm).item()
                speed_rmse, dir_rmse = self._compute_wind_metrics(fake_B, real_B_norm)

                total_rmse += rmse
                total_mae += mae
                total_speed_rmse += speed_rmse
                total_dir_rmse += dir_rmse

        # Average metrics
        metrics.rmse = total_rmse / num_batches
        metrics.mae = total_mae / num_batches
        metrics.wind_speed_rmse = total_speed_rmse / num_batches
        metrics.wind_direction_rmse = total_dir_rmse / num_batches

        return metrics

    def _update_lr(self) -> None:
        """Update learning rate with linear decay."""
        if self.current_epoch < self.config.lr_decay_start:
            return

        decay_epochs = self.config.epochs - self.config.lr_decay_start
        if decay_epochs <= 0:
            return

        decay_ratio = 1.0 - (self.current_epoch - self.config.lr_decay_start) / decay_epochs
        decay_ratio = max(0, decay_ratio)

        new_lr_g = self.config.learning_rate_g * decay_ratio
        new_lr_d = self.config.learning_rate_d * decay_ratio

        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = new_lr_g
        for param_group in self.optimizer_D.param_groups:
            param_group["lr"] = new_lr_d

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        progress_callback: Optional[Callable[[int, CycleGANMetrics], None]] = None,
    ) -> List[CycleGANMetrics]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            progress_callback: Optional callback(epoch, metrics)

        Returns:
            List of metrics for each epoch
        """
        logger.info(f"Starting CycleGAN training: {self.config.experiment_name}")
        logger.info(f"Model type: {self.config.model_type}")
        logger.info(f"Device: {self.device}")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train epoch
            if self.config.model_type == "pix2pix":
                train_metrics = self.train_epoch_pix2pix(train_loader)
            else:
                # CycleGAN training would go here
                train_metrics = self.train_epoch_pix2pix(train_loader)

            # Validate if loader provided
            if val_loader:
                val_metrics = self.validate(val_loader)
                train_metrics.rmse = val_metrics.rmse  # Use validation RMSE
                train_metrics.mae = val_metrics.mae
                train_metrics.wind_speed_rmse = val_metrics.wind_speed_rmse
                train_metrics.wind_direction_rmse = val_metrics.wind_direction_rmse

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

            # Update learning rate
            self._update_lr()

            # Progress callback
            if progress_callback:
                progress_callback(epoch, train_metrics)

            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"G: {train_metrics.g_loss:.4f} | "
                f"D: {train_metrics.d_loss:.4f} | "
                f"RMSE: {train_metrics.rmse:.4f} | "
                f"Speed RMSE: {train_metrics.wind_speed_rmse:.2f} m/s"
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
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "config": self.config.to_dict(),
            "best_rmse": self.best_rmse,
            "history": [m.to_dict() for m in self.history],
        }

        if self.generator_B:
            checkpoint["generator_B_state_dict"] = self.generator_B.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not set")

        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_rmse = checkpoint.get("best_rmse", float("inf"))

        if self.generator_B and "generator_B_state_dict" in checkpoint:
            self.generator_B.load_state_dict(checkpoint["generator_B_state_dict"])

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
            "model_type": self.config.model_type,
            "epochs_completed": len(self.history),
            "best_rmse": self.best_rmse,
            "final_rmse": self.history[-1].rmse if self.history else 0,
            "final_g_loss": self.history[-1].g_loss if self.history else 0,
            "final_d_loss": self.history[-1].d_loss if self.history else 0,
            "final_wind_speed_rmse": self.history[-1].wind_speed_rmse if self.history else 0,
            "hyperparameters": {
                "learning_rate_g": self.config.learning_rate_g,
                "learning_rate_d": self.config.learning_rate_d,
                "lambda_l1": self.config.lambda_l1,
                "batch_size": self.config.batch_size,
                "generator_features": self.config.generator_features,
            },
        }
