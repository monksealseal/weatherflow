
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Callable
import wandb
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

class WeatherTrainer:
    """Advanced training infrastructure from our successful experiments."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        experiment_dir: Optional[str] = None,
        use_amp: bool = True,
        use_wandb: bool = False,
        project_name: str = 'weatherflow'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and 'cuda' in device
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup experiment tracking
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name)
            wandb.watch(model)
        
        # Setup experiment directory
        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.experiment_dir = None
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'physics_losses': []
        }
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        physics_weight: float = 0.1,
        grad_clip: Optional[float] = 1.0
    ) -> Dict[str, float]:
        """Train for one epoch with physics-informed loss."""
        self.model.train()
        total_loss = 0
        physics_loss = 0
        n_batches = len(train_loader)
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass with automatic mixed precision
                with autocast(enabled=self.use_amp):
                    pred = self.model(x)
                    mse_loss = nn.functional.mse_loss(pred, y)
                    phys_loss = self.model.compute_physics_loss(pred, y)
                    loss = mse_loss + physics_weight * phys_loss
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                
                # Update metrics
                total_loss += mse_loss.item()
                physics_loss += phys_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'phys_loss': f"{phys_loss.item():.4f}"
                })
        
        return {
            'loss': total_loss / n_batches,
            'physics_loss': physics_loss / n_batches
        }
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Validate model with detailed metrics."""
        self.model.eval()
        total_loss = 0
        physics_loss = 0
        n_batches = len(val_loader)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    pred = self.model(x)
                    mse_loss = nn.functional.mse_loss(pred, y)
                    phys_loss = self.model.compute_physics_loss(pred, y)
                
                total_loss += mse_loss.item()
                physics_loss += phys_loss.item()
                
                # Store predictions for analysis
                predictions.append(pred.cpu())
                targets.append(y.cpu())
        
        # Compute additional metrics
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
        mae = torch.mean(torch.abs(predictions - targets))
        
        return {
            'val_loss': total_loss / n_batches,
            'physics_loss': physics_loss / n_batches,
            'rmse': rmse.item(),
            'mae': mae.item()
        }
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        physics_weight: float = 0.1,
        grad_clip: Optional[float] = 1.0,
        patience: int = 10,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """Full training loop with comprehensive monitoring."""
        
        # Early stopping setup
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(
                train_loader,
                physics_weight=physics_weight,
                grad_clip=grad_clip
            )
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['physics_losses'].append(train_metrics['physics_loss'])
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **train_metrics,
                    **val_metrics,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                patience_counter = 0
                
                if save_best and self.experiment_dir:
                    self.save_checkpoint(
                        self.experiment_dir / 'best_model.pt',
                        epoch,
                        val_metrics
                    )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print epoch summary
            print(f"
Epoch {epoch}:")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Physics Loss: {train_metrics['physics_loss']:.4f}")
            print(f"RMSE: {val_metrics['rmse']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return self.history
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint with metadata."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['metrics']['val_loss']
        self.best_epoch = checkpoint['epoch']
