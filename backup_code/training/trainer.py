
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from ..utils import WeatherVisualizer
from ..physics import AtmosphericPhysics
from tqdm.auto import tqdm
import wandb
import numpy as np
from pathlib import Path

class WeatherModelTrainer:
    """Advanced training utilities from our successful experiments."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_wandb: bool = False,
        experiment_dir: Optional[str] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.use_wandb = use_wandb
        self.visualizer = WeatherVisualizer()
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        self.physics = AtmosphericPhysics()
        
        if self.experiment_dir:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, dataloader, physics_weight: float = 0.1):
        """Train for one epoch with physics-informed loss."""
        self.model.train()
        total_loss = 0
        physics_loss = 0
        batch_metrics = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            self.optimizer.zero_grad()
            
            # Get data
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            pred = self.model(x)
            
            # Calculate losses
            mse_loss = self.loss_fn(pred, y)
            phys_loss = self.model.compute_physics_loss(pred, y)
            loss = mse_loss + physics_weight * phys_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            metrics = {
                'batch_loss': loss.item(),
                'mse_loss': mse_loss.item(),
                'physics_loss': phys_loss.item()
            }
            batch_metrics.append(metrics)
            
            total_loss += mse_loss.item()
            physics_loss += phys_loss.item()
            
            # Log batch metrics
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch/loss': metrics['batch_loss'],
                    'batch/mse': metrics['mse_loss'],
                    'batch/physics': metrics['physics_loss']
                })
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': total_loss / len(dataloader),
            'physics_loss': physics_loss / len(dataloader),
            'batch_metrics': batch_metrics
        }
        
        return epoch_metrics
    
    def validate(self, dataloader):
        """Validate model performance with detailed metrics."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                
                total_loss += loss.item()
                predictions.append(pred.cpu())
                targets.append(y.cpu())
        
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        
        # Compute detailed metrics
        metrics = {
            'val_loss': total_loss / len(dataloader),
            'rmse': torch.sqrt(torch.mean((predictions - targets) ** 2)).item(),
            'mae': torch.mean(torch.abs(predictions - targets)).item()
        }
        
        return metrics, predictions, targets
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        callbacks: List = None,
        physics_weight: float = 0.1,
        save_checkpoints: bool = True
    ):
        """Full training loop with comprehensive monitoring."""
        history = {
            'train_loss': [], 'val_loss': [], 
            'physics_loss': [], 'rmse': [], 'mae': []
        }
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, physics_weight)
            
            # Validation
            val_metrics, predictions, targets = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['physics_loss'].append(train_metrics['physics_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['rmse'].append(val_metrics['rmse'])
            history['mae'].append(val_metrics['mae'])
            
            # Logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/physics_loss': train_metrics['physics_loss'],
                    'val/loss': val_metrics['val_loss'],
                    'val/rmse': val_metrics['rmse'],
                    'val/mae': val_metrics['mae']
                })
            
            # Visualization
            if epoch % 5 == 0:
                self._create_validation_plots(predictions, targets, epoch)
            
            # Save checkpoint
            if save_checkpoints and self.experiment_dir:
                self._save_checkpoint(epoch, val_metrics)
            
            # Callbacks
            if callbacks:
                stop_training = False
                for callback in callbacks:
                    if callback(val_metrics, self.model):
                        print(f"Early stopping at epoch {epoch}")
                        stop_training = True
                if stop_training:
                    break
            
            # Print progress
            self._print_epoch_summary(epoch, train_metrics, val_metrics)
        
        return history
    
    def _create_validation_plots(self, predictions, targets, epoch):
        """Create and save validation visualizations."""
        # Prediction comparison
        fig, _ = self.visualizer.plot_comparison(
            targets[0], predictions[0],
            title=f"Epoch {epoch} Predictions"
        )
        
        if self.use_wandb:
            wandb.log({"predictions": wandb.Image(fig)})
        
        if self.experiment_dir:
            fig.savefig(
                self.experiment_dir / f"pred_comparison_epoch_{epoch}.png"
            )
        
        plt.close(fig)
    
    def _save_checkpoint(self, epoch, metrics):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if not hasattr(self, 'best_val_loss') or metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_model_path = self.experiment_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
    
    def _print_epoch_summary(self, epoch, train_metrics, val_metrics):
        """Print detailed epoch summary."""
        print(f"
Epoch {epoch} Summary:")
        print("-" * 50)
        print(f"Training Loss: {train_metrics['loss']:.4f}")
        print(f"Physics Loss: {train_metrics['physics_loss']:.4f}")
        print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
        print(f"RMSE: {val_metrics['rmse']:.4f}")
        print(f"MAE: {val_metrics['mae']:.4f}")
        print("-" * 50)
