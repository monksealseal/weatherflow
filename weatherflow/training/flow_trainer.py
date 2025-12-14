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
        loss_type: str = 'mse'
    ):
        """Initialize the trainer."""
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and 'cuda' in device
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        self.use_wandb = use_wandb
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.scheduler = scheduler
        self.physics_regularization = physics_regularization
        self.physics_lambda = physics_lambda
        self.loss_type = loss_type
        
        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training metrics
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
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
            
            # Sample time points
            t = torch.rand(x0.size(0), device=self.device)
            
            # Forward pass with AMP if enabled
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Compute model prediction
                if getattr(self.model, 'supports_style_conditioning', False):
                    v_t = self.model(x0, t, style=style)
                else:
                    v_t = self.model(x0, t)
                
                # Compute flow matching loss
                batch_flow_loss = compute_flow_loss(
                    v_t, x0, x1, t, 
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
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch_loss.backward()
                self.optimizer.step()
            
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
                
                # Sample time points
                t = torch.rand(x0.size(0), device=self.device)
                
                # Compute model prediction
                if getattr(self.model, 'supports_style_conditioning', False):
                    v_t = self.model(x0, t, style=style)
                else:
                    v_t = self.model(x0, t)
                
                # Compute flow matching loss
                batch_flow_loss = compute_flow_loss(
                    v_t, x0, x1, t, 
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
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_flow_loss = flow_loss / num_batches
        avg_physics_loss = physics_loss / num_batches if self.physics_regularization else 0.0
        
        # Return metrics
        metrics = {
            'val_loss': avg_loss,
            'val_flow_loss': avg_flow_loss,
            'val_physics_loss': avg_physics_loss
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
