import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple, Callable

# Configure logging
logger = logging.getLogger(__name__)

from .metrics import energy_ratio, mae, persistence_rmse, rmse
from .utils import set_global_seed

def compute_flow_loss(
    v_t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    alpha: float = 0.2,
    min_velocity: float = 5.0,
    loss_type: str = 'mse'
) -> torch.Tensor:
    """Compute flow matching loss between predicted and target velocities."""
    # Compute target velocity as straight-line path
    # Avoid division by zero at t=1 with small epsilon
    target_velocity = (x1 - x0) / (1.0 - t.view(-1, 1, 1, 1) + 1e-8)
    
    # Apply minimum velocity threshold for numerical stability
    velocity_norm = torch.norm(target_velocity.reshape(target_velocity.shape[0], -1), 
                              dim=1, keepdim=True)
    velocity_norm = velocity_norm.reshape(-1, 1, 1, 1)
    scale_factor = torch.where(
        velocity_norm < min_velocity,
        min_velocity / (velocity_norm + 1e-8),
        torch.ones_like(velocity_norm)
    )
    target_velocity = target_velocity * scale_factor
    
    # Compute loss based on specified type
    if loss_type == 'huber':
        return F.huber_loss(v_t, target_velocity, delta=1.0)
    elif loss_type == 'smooth_l1':
        return F.smooth_l1_loss(v_t, target_velocity)
    else:  # Default to MSE
        return F.mse_loss(v_t, target_velocity)


class FlowTrainer:
    """Training infrastructure for flow matching models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_amp: bool = True,
        use_wandb: bool = False,
        checkpoint_dir: Optional[str] = None,
        scheduler: Optional[object] = None,
        physics_regularization: bool = False,
        physics_lambda: float = 0.1,
        loss_type: str = 'mse',
        grad_clip: Optional[float] = 1.0,
        ema_decay: Optional[float] = None,
        seed: Optional[int] = None,
        noise_std: Optional[Tuple[float, float]] = None,
    ):
        """Initialize the trainer."""
        if seed is not None:
            set_global_seed(seed, deterministic=False)

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        # Only use AMP if CUDA is available and requested
        self.use_amp = use_amp and torch.cuda.is_available() and 'cuda' in device
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.use_wandb = use_wandb
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.scheduler = scheduler
        self.physics_regularization = physics_regularization
        self.physics_lambda = physics_lambda
        self.loss_type = loss_type
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.noise_std = noise_std
        
        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training metrics
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self._ema_params: Optional[List[torch.Tensor]] = None
        if self.ema_decay is not None:
            self._init_ema()

    def _init_ema(self) -> None:
        """Initialize exponential moving average shadow weights."""
        self._ema_params = [p.detach().clone() for p in self.model.parameters()]

    def _update_ema(self) -> None:
        """Update EMA weights with current parameters."""
        if self._ema_params is None or self.ema_decay is None:
            return
        with torch.no_grad():
            for shadow, param in zip(self._ema_params, self.model.parameters()):
                shadow.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

    def _ema_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return a state dict built from EMA parameters for eval."""
        if self._ema_params is None:
            return None
        return {k: v.clone() for k, v in zip(self.model.state_dict().keys(), self._ema_params)}

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        physics_loss = 0.0
        flow_loss = 0.0
        num_batches = len(train_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract data
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x0, x1 = batch
                x0, x1 = x0.to(self.device), x1.to(self.device)
                style = None
            elif isinstance(batch, dict):
                x0 = batch['input'].to(self.device)
                x1 = batch['target'].to(self.device)
                style = batch.get('style')
                if style is not None:
                    style = style.to(self.device)
            else:
                raise ValueError("Unsupported batch format")

            # Optional stochastic interpolant noise
            if self.noise_std is not None:
                sigma_min, sigma_max = self.noise_std
                sigma = torch.rand(x0.size(0), device=self.device) * (sigma_max - sigma_min) + sigma_min
                noise = torch.randn_like(x0)
                x0_noisy = x0 + sigma.view(-1, 1, 1, 1) * noise
                x1_noisy = x1 + sigma.view(-1, 1, 1, 1) * noise
            else:
                x0_noisy, x1_noisy = x0, x1
            
            # Sample time points
            t = torch.rand(x0.size(0), device=self.device)
            
            # Forward pass with AMP if enabled (use torch.autocast for device-agnostic support)
            device_type = getattr(self.device, "type", None)
            with torch.autocast(
                device_type="cuda" if device_type == "cuda" else "cpu",
                enabled=self.use_amp and device_type in ("cuda", "cpu"),
            ):
                # Compute model prediction
                if getattr(self.model, 'supports_style_conditioning', False):
                    v_t = self.model(x0_noisy, t, style=style)
                else:
                    v_t = self.model(x0_noisy, t)
                
                # Compute flow matching loss
                batch_flow_loss = compute_flow_loss(
                    v_t, x0_noisy, x1_noisy, t, 
                    loss_type=self.loss_type
                )
                
                # Add physics regularization if enabled
                batch_physics_loss = torch.tensor(0.0, device=self.device)
                if self.physics_regularization and hasattr(self.model, 'compute_physics_loss'):
                    batch_physics_loss = self.model.compute_physics_loss(v_t, x0)
                    batch_loss = batch_flow_loss + self.physics_lambda * batch_physics_loss
                    physics_loss += batch_physics_loss.item()
                else:
                    batch_loss = batch_flow_loss
                
                # Track losses
                flow_loss += batch_flow_loss.item()
                total_loss += batch_loss.item()
            
            # Backward pass with AMP if enabled
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(batch_loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch_loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # EMA update
            self._update_ema()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': batch_loss.item(),
                'flow_loss': batch_flow_loss.item()
            })
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({
                    'batch_loss': batch_loss.item(),
                    'batch_flow_loss': batch_flow_loss.item(),
                    'batch_physics_loss': batch_physics_loss.item() if self.physics_regularization else 0.0
                })
        
        # Update learning rate scheduler if provided
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_flow_loss = flow_loss / num_batches
        avg_physics_loss = physics_loss / num_batches if self.physics_regularization else 0.0
        
        # Return metrics
        metrics = {
            'loss': avg_loss,
            'flow_loss': avg_flow_loss,
            'physics_loss': avg_physics_loss
        }
        
        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        physics_loss = 0.0
        flow_loss = 0.0
        num_batches = len(val_loader)
        rmse_total = 0.0
        mae_total = 0.0
        energy_total = 0.0
        persistence_total = 0.0
        
        eval_state = None
        # Optionally evaluate with EMA weights
        if self._ema_params is not None:
            eval_state = self.model.state_dict()
            self.model.load_state_dict(self._ema_state_dict())  # type: ignore

        with torch.no_grad():
            for batch in val_loader:
                # Extract data
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    x0, x1 = batch
                    x0, x1 = x0.to(self.device), x1.to(self.device)
                    style = None
                elif isinstance(batch, dict):
                    x0 = batch['input'].to(self.device)
                    x1 = batch['target'].to(self.device)
                    style = batch.get('style')
                    if style is not None:
                        style = style.to(self.device)
                else:
                    raise ValueError("Unsupported batch format")

                if self.noise_std is not None:
                    sigma_min, sigma_max = self.noise_std
                    sigma = torch.rand(x0.size(0), device=self.device) * (sigma_max - sigma_min) + sigma_min
                    noise = torch.randn_like(x0)
                    x0_noisy = x0 + sigma.view(-1, 1, 1, 1) * noise
                    x1_noisy = x1 + sigma.view(-1, 1, 1, 1) * noise
                else:
                    x0_noisy, x1_noisy = x0, x1
                
                # Sample time points
                t = torch.rand(x0.size(0), device=self.device)
                
                # Compute model prediction
                if getattr(self.model, 'supports_style_conditioning', False):
                    v_t = self.model(x0_noisy, t, style=style)
                else:
                    v_t = self.model(x0_noisy, t)
                
                # Compute flow matching loss
                batch_flow_loss = compute_flow_loss(
                    v_t, x0_noisy, x1_noisy, t, 
                    loss_type=self.loss_type
                )
                
                # Add physics regularization if enabled
                if self.physics_regularization and hasattr(self.model, 'compute_physics_loss'):
                    batch_physics_loss = self.model.compute_physics_loss(v_t, x0)
                    batch_loss = batch_flow_loss + self.physics_lambda * batch_physics_loss
                    physics_loss += batch_physics_loss.item()
                else:
                    batch_loss = batch_flow_loss
                
                # Track losses
                flow_loss += batch_flow_loss.item()
                total_loss += batch_loss.item()

                # Metrics
                rmse_total += rmse(v_t, x1).item()
                mae_total += mae(v_t, x1).item()
                energy_total += energy_ratio(v_t, x1).item()
                persistence_total += persistence_rmse(x0, x1).item()

        # Restore original weights if EMA was used
        if eval_state is not None:
            self.model.load_state_dict(eval_state)
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_flow_loss = flow_loss / num_batches
        avg_physics_loss = physics_loss / num_batches if self.physics_regularization else 0.0
        avg_rmse = rmse_total / num_batches
        avg_mae = mae_total / num_batches
        avg_energy = energy_total / num_batches
        avg_persistence_rmse = persistence_total / num_batches
        
        # Return metrics
        metrics = {
            'val_loss': avg_loss,
            'val_flow_loss': avg_flow_loss,
            'val_physics_loss': avg_physics_loss,
            'val_rmse': avg_rmse,
            'val_mae': avg_mae,
            'val_energy_ratio': avg_energy,
            'val_persistence_rmse': avg_persistence_rmse,
        }
        
        return metrics
        
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            logger.warning("Checkpoint directory not set, skipping checkpoint save")
            return
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss
        }
        
        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not set")
        
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load other training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load scheduler state if available
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
