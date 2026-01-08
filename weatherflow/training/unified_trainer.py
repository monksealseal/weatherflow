"""
Unified Training Framework for Weather AI Models

Supports training across all model architectures:
- Flow Matching (WeatherFlowMatch, IcosahedralFlowMatch)
- Vision Transformers (FourCastNet, ClimaX)
- Graph Neural Networks (GraphCast)
- 3D Transformers (Pangu-Weather)
- Diffusion Models (GenCast)
- Image Translation (Pix2Pix, CycleGAN)
- GAIA and Foundation Models

Features:
- Mini training runs for quick experiments
- Physics-informed losses
- Experiment tracking
- Checkpoint management
- Multiple loss types and schedulers
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for unified training."""

    # Model settings
    model_name: str = "custom"
    model_category: str = "flow_matching"  # flow_matching, transformer, gnn, diffusion, gan

    # Training settings
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_epochs: int = 1

    # Optimizer settings
    optimizer: str = "adamw"  # adam, adamw, sgd, lamb
    scheduler: str = "cosine"  # cosine, linear, step, none

    # Loss settings
    loss_type: str = "mse"  # mse, huber, smooth_l1, mae

    # Physics settings
    use_physics_loss: bool = False
    physics_weight: float = 0.1
    physics_losses: List[str] = field(default_factory=lambda: ["divergence", "geostrophic"])

    # Flow matching specific
    flow_weighting: str = "time"  # time, none

    # Diffusion specific
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"  # linear, cosine

    # GAN specific
    gan_mode: str = "vanilla"  # vanilla, lsgan, wgan
    lambda_l1: float = 100.0

    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    num_workers: int = 4

    # Checkpoint settings
    checkpoint_dir: Optional[str] = None
    save_every: int = 5
    keep_last_n: int = 3

    # Logging
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "weatherflow-experiments"

    # Mini training
    mini_mode: bool = False
    mini_samples: int = 100
    mini_val_samples: int = 20

    # Experiment
    experiment_name: str = "experiment"
    experiment_tags: List[str] = field(default_factory=list)
    seed: int = 42


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_mse: float = 0.0
    val_mse: float = 0.0
    train_mae: float = 0.0
    val_mae: float = 0.0
    physics_loss: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    throughput: float = 0.0  # samples per second
    gpu_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LossRegistry:
    """Registry for different loss functions."""

    _losses: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn: Callable):
            cls._losses[name] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._losses:
            raise ValueError(f"Loss '{name}' not found. Available: {list(cls._losses.keys())}")
        return cls._losses[name]

    @classmethod
    def list_losses(cls) -> List[str]:
        return list(cls._losses.keys())


# Register standard losses
@LossRegistry.register("mse")
def mse_loss(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return F.mse_loss(pred, target)


@LossRegistry.register("mae")
def mae_loss(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return F.l1_loss(pred, target)


@LossRegistry.register("huber")
def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0, **kwargs) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)


@LossRegistry.register("smooth_l1")
def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


@LossRegistry.register("flow_matching")
def flow_matching_loss(
    v_pred: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    weighting: str = "time",
    **kwargs
) -> torch.Tensor:
    """Rectified flow matching loss."""
    target_velocity = x1 - x0
    diff = v_pred - target_velocity

    if weighting == "time":
        weight = (t * (1 - t)).clamp(min=1e-3).view(-1, 1, 1, 1)
        return (diff.pow(2) * weight).mean()
    return diff.pow(2).mean()


@LossRegistry.register("diffusion")
def diffusion_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """Diffusion denoising loss."""
    return F.mse_loss(noise_pred, noise)


@LossRegistry.register("gan_generator")
def gan_generator_loss(
    fake_pred: torch.Tensor,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    mode: str = "vanilla",
    lambda_l1: float = 100.0,
    **kwargs
) -> torch.Tensor:
    """GAN generator loss with L1 reconstruction."""
    if mode == "vanilla":
        adversarial_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )
    elif mode == "lsgan":
        adversarial_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    elif mode == "wgan":
        adversarial_loss = -fake_pred.mean()
    else:
        adversarial_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )

    l1_loss = F.l1_loss(fake_images, real_images)
    return adversarial_loss + lambda_l1 * l1_loss


@LossRegistry.register("gan_discriminator")
def gan_discriminator_loss(
    real_pred: torch.Tensor,
    fake_pred: torch.Tensor,
    mode: str = "vanilla",
    **kwargs
) -> torch.Tensor:
    """GAN discriminator loss."""
    if mode == "vanilla":
        real_loss = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )
    elif mode == "lsgan":
        real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    elif mode == "wgan":
        real_loss = -real_pred.mean()
        fake_loss = fake_pred.mean()
    else:
        real_loss = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )

    return (real_loss + fake_loss) * 0.5


class PhysicsLossModule(nn.Module):
    """Physics-informed loss calculations for weather models."""

    # Earth constants
    EARTH_RADIUS = 6.371e6  # meters
    EARTH_OMEGA = 7.2921e-5  # rad/s
    GRAVITY = 9.81  # m/s^2

    def __init__(
        self,
        losses: List[str] = ["divergence"],
        lat_range: Tuple[float, float] = (-90, 90),
        grid_size: Tuple[int, int] = (721, 1440),
    ):
        super().__init__()
        self.losses = losses
        self.lat_range = lat_range
        self.grid_size = grid_size

        # Precompute latitude-dependent Coriolis parameter
        lat = torch.linspace(lat_range[0], lat_range[1], grid_size[0])
        self.register_buffer("f", 2 * self.EARTH_OMEGA * torch.sin(torch.deg2rad(lat)))

        # Grid spacing
        dlat = (lat_range[1] - lat_range[0]) / grid_size[0]
        dlon = 360.0 / grid_size[1]
        self.register_buffer("dlat", torch.tensor(dlat * np.pi / 180 * self.EARTH_RADIUS))
        self.register_buffer("dlon", torch.tensor(dlon * np.pi / 180))

    def divergence_loss(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Mass continuity: du/dx + dv/dy ≈ 0 for incompressible flow."""
        # Finite difference gradients
        du_dx = (u[..., :, 2:] - u[..., :, :-2]) / (2 * self.dlon * self.EARTH_RADIUS)
        dv_dy = (v[..., 2:, :] - v[..., :-2, :]) / (2 * self.dlat)

        # Trim to match dimensions
        du_dx = du_dx[..., 1:-1, :]
        dv_dy = dv_dy[..., :, 1:-1]

        div = du_dx + dv_dy
        return div.pow(2).mean()

    def geostrophic_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Geostrophic balance: f*u ≈ -g*dz/dy, f*v ≈ g*dz/dx."""
        # Compute geopotential gradients
        dz_dx = (z[..., :, 2:] - z[..., :, :-2]) / (2 * self.dlon * self.EARTH_RADIUS)
        dz_dy = (z[..., 2:, :] - z[..., :-2, :]) / (2 * self.dlat)

        # Compute geostrophic winds
        f = self.f.view(1, 1, -1, 1)
        u_geo = -self.GRAVITY * dz_dy / (f[..., 1:-1, :] + 1e-8)
        v_geo = self.GRAVITY * dz_dx / (f[..., :, 1:-1] + 1e-8)

        # Compare with actual winds
        u_actual = u[..., 1:-1, 1:-1]
        v_actual = v[..., 1:-1, 1:-1]

        loss_u = F.mse_loss(u_actual, u_geo[..., 1:-1])
        loss_v = F.mse_loss(v_actual, v_geo[..., 1:-1, :])

        return loss_u + loss_v

    def energy_spectrum_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Encourage correct energy spectra (k^-3 for 2D turbulence)."""
        # 2D FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        # Power spectrum
        pred_power = (pred_fft.real.pow(2) + pred_fft.imag.pow(2)).mean(dim=(0, 1))
        target_power = (target_fft.real.pow(2) + target_fft.imag.pow(2)).mean(dim=(0, 1))

        # Log-space comparison
        pred_log = torch.log(pred_power + 1e-8)
        target_log = torch.log(target_power + 1e-8)

        return F.mse_loss(pred_log, target_log)

    def pv_conservation_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Potential vorticity conservation loss."""
        # Relative vorticity
        dv_dx = (v[..., :, 2:] - v[..., :, :-2]) / (2 * self.dlon * self.EARTH_RADIUS)
        du_dy = (u[..., 2:, :] - u[..., :-2, :]) / (2 * self.dlat)

        zeta = dv_dx[..., 1:-1, :] - du_dy[..., :, 1:-1]

        # Absolute vorticity
        f = self.f.view(1, 1, -1, 1)
        eta = zeta + f[..., 1:-1, 1:-1]

        # Potential temperature gradient (simplified)
        dtheta_dz = theta[..., 1:-1, 1:-1]  # Placeholder for vertical gradient

        # PV = eta * dtheta/dz
        pv = eta * (dtheta_dz + 1e-8)

        # PV should be approximately conserved (minimize variance in time)
        return pv.var()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        variables: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        """Compute combined physics loss.

        Args:
            pred: Model prediction [B, C, H, W]
            target: Target values [B, C, H, W]
            variables: Dict mapping variable names to channel indices
        """
        total_loss = torch.tensor(0.0, device=pred.device)

        # Default variable mapping for ERA5-like data
        if variables is None:
            variables = {"u": 0, "v": 1, "z": 2, "t": 3}

        for loss_name in self.losses:
            if loss_name == "divergence" and "u" in variables and "v" in variables:
                u = pred[:, variables["u"]]
                v = pred[:, variables["v"]]
                total_loss = total_loss + self.divergence_loss(u, v)

            elif loss_name == "geostrophic" and all(k in variables for k in ["u", "v", "z"]):
                u = pred[:, variables["u"]]
                v = pred[:, variables["v"]]
                z = pred[:, variables["z"]]
                total_loss = total_loss + self.geostrophic_loss(u, v, z)

            elif loss_name == "energy_spectrum":
                total_loss = total_loss + self.energy_spectrum_loss(pred, target)

            elif loss_name == "pv" and all(k in variables for k in ["u", "v", "t"]):
                u = pred[:, variables["u"]]
                v = pred[:, variables["v"]]
                theta = pred[:, variables["t"]]
                total_loss = total_loss + self.pv_conservation_loss(u, v, theta)

        return total_loss


class UnifiedTrainer:
    """
    Unified trainer for all weather AI model architectures.

    Supports:
    - Flow matching models (velocity field prediction)
    - Deterministic models (direct prediction)
    - Diffusion models (noise prediction)
    - GAN models (generator + discriminator)
    - Physics-informed training
    - Mini training runs for quick experiments
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        discriminator: Optional[nn.Module] = None,
    ):
        """
        Initialize unified trainer.

        Args:
            model: The model to train (generator for GANs)
            config: Training configuration
            discriminator: Optional discriminator for GAN training
        """
        self.model = model.to(config.device)
        self.config = config
        self.discriminator = discriminator
        if discriminator is not None:
            self.discriminator = discriminator.to(config.device)

        # Set random seed
        self._set_seed(config.seed)

        # Setup optimizer(s)
        self.optimizer = self._create_optimizer(self.model)
        self.optimizer_d = None
        if self.discriminator is not None:
            self.optimizer_d = self._create_optimizer(self.discriminator)

        # Setup scheduler
        self.scheduler = self._create_scheduler(self.optimizer)
        self.scheduler_d = None
        if self.optimizer_d is not None:
            self.scheduler_d = self._create_scheduler(self.optimizer_d)

        # Setup AMP
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and "cuda" in config.device else None

        # Setup physics loss
        self.physics_loss_module = None
        if config.use_physics_loss:
            self.physics_loss_module = PhysicsLossModule(losses=config.physics_losses)
            self.physics_loss_module = self.physics_loss_module.to(config.device)

        # Setup checkpoint directory
        if config.checkpoint_dir:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history: List[TrainingMetrics] = []

        # Initialize wandb if enabled
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.experiment_name,
                    config=asdict(config),
                    tags=config.experiment_tags,
                )
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                config.use_wandb = False

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        config = self.config
        params = model.parameters()

        if config.optimizer == "adam":
            return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "sgd":
            return torch.optim.SGD(params, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        elif config.optimizer == "lamb":
            try:
                from torch_optimizer import Lamb
                return Lamb(params, lr=config.learning_rate, weight_decay=config.weight_decay)
            except ImportError:
                logger.warning("torch_optimizer not installed, falling back to AdamW")
                return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create learning rate scheduler based on config."""
        config = self.config
        total_steps = config.epochs
        warmup_steps = config.warmup_epochs

        if config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
            )
        elif config.scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_steps
            )
        elif config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            return None

    def _compute_loss(
        self,
        batch: Union[Tuple, Dict],
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss based on model category.

        Returns:
            Tuple of (loss_tensor, metrics_dict)
        """
        config = self.config
        device = config.device

        # Parse batch based on format
        if isinstance(batch, dict):
            x0 = batch["input"].to(device)
            x1 = batch["target"].to(device)
            style = batch.get("style")
            if style is not None:
                style = style.to(device)
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                x0, x1 = batch
                x0, x1 = x0.to(device), x1.to(device)
                style = None
            else:
                x0 = batch[0].to(device)
                x1 = batch[1].to(device) if len(batch) > 1 else x0
                style = None
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")

        metrics = {}

        # Flow matching training
        if config.model_category == "flow_matching":
            # Sample time
            t = torch.rand(x0.size(0), device=device)
            t_broadcast = t.view(-1, 1, 1, 1)

            # Interpolate
            x_t = torch.lerp(x0, x1, t_broadcast)

            # Predict velocity
            if hasattr(self.model, "supports_style_conditioning") and self.model.supports_style_conditioning:
                v_pred = self.model(x_t, t, style=style)
            else:
                v_pred = self.model(x_t, t)

            # Flow matching loss
            loss = LossRegistry.get("flow_matching")(
                v_pred, x0, x1, t, weighting=config.flow_weighting
            )

            # Compute prediction for metrics (at t=1)
            pred = x0 + v_pred  # Simplified, should use ODE solver
            metrics["flow_loss"] = loss.item()

        # Diffusion training
        elif config.model_category == "diffusion":
            # Sample timestep
            t = torch.randint(0, config.diffusion_steps, (x0.size(0),), device=device)

            # Add noise
            noise = torch.randn_like(x1)
            alpha = self._get_alpha(t).view(-1, 1, 1, 1)
            x_noisy = torch.sqrt(alpha) * x1 + torch.sqrt(1 - alpha) * noise

            # Predict noise
            noise_pred = self.model(x_noisy, t)

            # Diffusion loss
            loss = LossRegistry.get("diffusion")(noise_pred, noise, t)

            pred = x1  # For metrics
            metrics["diffusion_loss"] = loss.item()

        # GAN training (generator step)
        elif config.model_category == "gan":
            # Generator forward
            fake = self.model(x0)

            if self.discriminator is not None:
                # Discriminator prediction on fake
                fake_pred = self.discriminator(torch.cat([x0, fake], dim=1))

                # Generator loss
                loss = LossRegistry.get("gan_generator")(
                    fake_pred, x1, fake,
                    mode=config.gan_mode,
                    lambda_l1=config.lambda_l1
                )
            else:
                # No discriminator - just L1 loss
                loss = F.l1_loss(fake, x1)

            pred = fake
            metrics["gan_g_loss"] = loss.item()

        # Standard deterministic model
        else:
            # Forward pass
            pred = self.model(x0)

            # Standard loss
            loss_fn = LossRegistry.get(config.loss_type)
            loss = loss_fn(pred, x1)

        # Add physics loss if enabled
        if self.physics_loss_module is not None and is_training:
            physics_loss = self.physics_loss_module(pred, x1)
            loss = loss + config.physics_weight * physics_loss
            metrics["physics_loss"] = physics_loss.item()

        # Compute additional metrics
        with torch.no_grad():
            metrics["mse"] = F.mse_loss(pred, x1).item()
            metrics["mae"] = F.l1_loss(pred, x1).item()

        return loss, metrics

    def _get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha values for diffusion at timestep t."""
        # Linear schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.config.diffusion_steps, device=t.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod[t]

    def _train_discriminator_step(self, batch) -> Dict[str, float]:
        """Train discriminator for one step (GAN only)."""
        if self.discriminator is None:
            return {}

        config = self.config
        device = config.device

        # Parse batch
        if isinstance(batch, dict):
            x0 = batch["input"].to(device)
            x1 = batch["target"].to(device)
        else:
            x0, x1 = batch
            x0, x1 = x0.to(device), x1.to(device)

        # Generate fake images
        with torch.no_grad():
            fake = self.model(x0)

        # Real and fake predictions
        real_pred = self.discriminator(torch.cat([x0, x1], dim=1))
        fake_pred = self.discriminator(torch.cat([x0, fake.detach()], dim=1))

        # Discriminator loss
        loss = LossRegistry.get("gan_discriminator")(
            real_pred, fake_pred, mode=config.gan_mode
        )

        # Backward
        self.optimizer_d.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_d)
        else:
            loss.backward()
            self.optimizer_d.step()

        return {"gan_d_loss": loss.item()}

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()

        config = self.config
        total_loss = 0.0
        all_metrics: Dict[str, List[float]] = {}
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Train discriminator first (for GANs)
            if config.model_category == "gan":
                d_metrics = self._train_discriminator_step(batch)
                for k, v in d_metrics.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                loss, metrics = self._compute_loss(batch, is_training=True)

            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                self.optimizer.step()

            # Update AMP scaler
            if self.scaler is not None:
                self.scaler.update()

            # Track metrics
            total_loss += loss.item()
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

            num_batches += 1

            # Mini mode - early stop
            if config.mini_mode and batch_idx + 1 >= config.mini_samples // config.batch_size:
                break

        # Step scheduler
        if self.scheduler is not None and self.current_epoch >= config.warmup_epochs:
            self.scheduler.step()
        if self.scheduler_d is not None and self.current_epoch >= config.warmup_epochs:
            self.scheduler_d.step()

        # Compute averages
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        avg_metrics["loss"] = total_loss / max(num_batches, 1)
        avg_metrics["epoch_time"] = time.time() - start_time
        avg_metrics["throughput"] = (num_batches * config.batch_size) / avg_metrics["epoch_time"]
        avg_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return avg_metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        config = self.config
        total_loss = 0.0
        all_metrics: Dict[str, List[float]] = {}
        num_batches = 0

        for batch_idx, batch in enumerate(val_loader):
            loss, metrics = self._compute_loss(batch, is_training=False)

            total_loss += loss.item()
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

            num_batches += 1

            # Mini mode - early stop
            if config.mini_mode and batch_idx + 1 >= config.mini_val_samples // config.batch_size:
                break

        avg_metrics = {f"val_{k}": np.mean(v) for k, v in all_metrics.items()}
        avg_metrics["val_loss"] = total_loss / max(num_batches, 1)

        return avg_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        progress_callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> List[TrainingMetrics]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            progress_callback: Optional callback(epoch, metrics) for progress updates

        Returns:
            List of TrainingMetrics for each epoch
        """
        config = self.config

        logger.info(f"Starting training: {config.experiment_name}")
        logger.info(f"Model: {config.model_name}, Category: {config.model_category}")
        logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
        logger.info(f"Mini mode: {config.mini_mode}")

        for epoch in range(config.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics.get("loss", 0),
                val_loss=val_metrics.get("val_loss", 0),
                train_mse=train_metrics.get("mse", 0),
                val_mse=val_metrics.get("val_mse", 0),
                train_mae=train_metrics.get("mae", 0),
                val_mae=val_metrics.get("val_mae", 0),
                physics_loss=train_metrics.get("physics_loss", 0),
                learning_rate=train_metrics.get("learning_rate", 0),
                epoch_time=train_metrics.get("epoch_time", 0),
                throughput=train_metrics.get("throughput", 0),
                gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            )
            self.history.append(metrics)

            # Log progress
            if epoch % config.log_every == 0 or epoch == config.epochs - 1:
                logger.info(
                    f"Epoch {epoch}/{config.epochs} - "
                    f"Train Loss: {metrics.train_loss:.4f}, "
                    f"Val Loss: {metrics.val_loss:.4f}, "
                    f"LR: {metrics.learning_rate:.2e}"
                )

            # Wandb logging
            if config.use_wandb:
                try:
                    import wandb
                    wandb.log(metrics.to_dict())
                except Exception as e:
                    logger.warning(f"Wandb logging failed: {e}")

            # Progress callback
            if progress_callback is not None:
                progress_callback(epoch, all_metrics)

            # Save checkpoint
            if self.checkpoint_dir is not None:
                if epoch % config.save_every == 0 or epoch == config.epochs - 1:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

                # Save best model
                if val_metrics.get("val_loss", float("inf")) < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")

        # Save final model
        if self.checkpoint_dir is not None:
            self.save_checkpoint("final_model.pt")
            self._save_history()

        # Close wandb
        if config.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        return self.history

    def save_checkpoint(self, filename: str) -> Path:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("Checkpoint directory not set")

        path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "history": [m.to_dict() for m in self.history],
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.discriminator is not None:
            checkpoint["discriminator_state_dict"] = self.discriminator.state_dict()
            checkpoint["optimizer_d_state_dict"] = self.optimizer_d.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        return path

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.discriminator is not None and "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

        logger.info(f"Checkpoint loaded: {path}")

    def _save_history(self) -> None:
        """Save training history to JSON."""
        if self.checkpoint_dir is None:
            return

        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "config": asdict(self.config),
                "history": [m.to_dict() for m in self.history],
            }, f, indent=2)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training run."""
        if not self.history:
            return {}

        final = self.history[-1]
        best_epoch = min(range(len(self.history)), key=lambda i: self.history[i].val_loss or float("inf"))

        return {
            "experiment_name": self.config.experiment_name,
            "model_name": self.config.model_name,
            "model_category": self.config.model_category,
            "epochs_trained": len(self.history),
            "final_train_loss": final.train_loss,
            "final_val_loss": final.val_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": best_epoch,
            "total_time_seconds": sum(m.epoch_time for m in self.history),
            "avg_throughput": np.mean([m.throughput for m in self.history]),
            "max_gpu_memory_mb": max(m.gpu_memory_mb for m in self.history),
        }


def quick_train(
    model: nn.Module,
    train_data: Union[Dataset, DataLoader],
    val_data: Optional[Union[Dataset, DataLoader]] = None,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    model_category: str = "transformer",
    mini_mode: bool = True,
    **kwargs,
) -> Tuple[nn.Module, List[TrainingMetrics]]:
    """
    Quick training helper for rapid experimentation.

    Args:
        model: Model to train
        train_data: Training dataset or dataloader
        val_data: Optional validation dataset or dataloader
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_category: Model category (flow_matching, transformer, diffusion, gan)
        mini_mode: Enable mini training mode for quick experiments
        **kwargs: Additional TrainingConfig parameters

    Returns:
        Tuple of (trained_model, training_history)

    Example:
        >>> model = MyModel()
        >>> train_ds = ERA5Dataset(...)
        >>> model, history = quick_train(model, train_ds, epochs=3, mini_mode=True)
    """
    # Create dataloaders if needed
    if isinstance(train_data, Dataset):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        train_loader = train_data

    if val_data is not None:
        if isinstance(val_data, Dataset):
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            val_loader = val_data
    else:
        val_loader = None

    # Create config
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_category=model_category,
        mini_mode=mini_mode,
        **kwargs,
    )

    # Train
    trainer = UnifiedTrainer(model, config)
    history = trainer.train(train_loader, val_loader)

    return model, history
