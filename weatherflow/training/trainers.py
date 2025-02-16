import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from pathlib import Path

class WeatherTrainer:
    """Base trainer for weather prediction models"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        
    def train(self, train_loader, val_loader):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                loss = self._training_step(batch)
                train_loss += loss.item()
            
            # Validation
            val_loss = self._validate(val_loader)
            
            # Logging
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss
            }
            
            if self.config.use_wandb:
                wandb.log(metrics)
            
            # Model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"best_model_epoch_{epoch}.pt")
                
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    def _training_step(self, batch):
        """Single training step"""
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_loss(batch)
                val_loss += loss.item()
                
        return val_loss / len(val_loader)
    
    def _compute_loss(self, batch):
        """Compute loss for a batch"""
        raise NotImplementedError("Implement in subclass")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        save_path = Path(self.config.save_dir) / filename
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, save_path)
        
class PhysicsGuidedTrainer(WeatherTrainer):
    """Trainer for PhysicsGuidedAttention model"""
    def _compute_loss(self, batch):
        x, y = batch
        x = {k: v.to(self.device) for k, v in x.items()}
        y = {k: v.to(self.device) for k, v in y.items()}
        
        pred = self.model(x)
        mse_loss = torch.nn.functional.mse_loss(pred, y)
        physics_loss = self._compute_physics_loss(pred, y)
        
        return mse_loss + self.config.physics_weight * physics_loss
    
    def _compute_physics_loss(self, pred, target):
        """Compute physics-based loss terms"""
        energy_loss = self._energy_conservation(pred, target)
        return energy_loss

class StochasticFlowTrainer(WeatherTrainer):
    """Trainer for StochasticFlowModel"""
    def _compute_loss(self, batch):
        x, y = batch
        t = torch.rand(x.size(0), device=self.device)
        
        pred = self.model(x, t)
        return torch.nn.functional.mse_loss(pred, y)
