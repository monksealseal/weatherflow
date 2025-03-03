
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb
import os
from pathlib import Path
from typing import Optional, Dict

def compute_flow_loss(v_t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, 
                     t: torch.Tensor, alpha: float = 0.2, min_velocity: float = 5.0) -> torch.Tensor:
    """Compute flow matching loss."""
    target_velocity = (x1 - x0) / (1 - t + 1e-8)
    return F.mse_loss(v_t, target_velocity)

class FlowTrainer:
    """Training infrastructure for flow matching models."""
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_amp: bool = True,
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and 'cuda' in device
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        self.use_wandb = use_wandb
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x0, x1) in enumerate(tqdm(train_loader, desc="Training")):
            x0, x1 = x0.to(self.device), x1.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                t = torch.rand(x0.size(0), device=self.device)
                v_t = self.model(x0, t)
                loss = compute_flow_loss(v_t, x0, x1, t)
            
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.use_wandb:
                wandb.log({"batch_loss": loss.item()})
        
        return {"loss": total_loss / len(train_loader)}


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..path import ProbPath, GaussianProbPath

class FlowMatchTrainer:
    """Flow matching trainer implementing Algorithm 3 (p.27).
    
    Trains a flow model using the conditional flow matching objective.
    """
    
    def __init__(
        self,
        model: nn.Module,
        path: ProbPath,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        use_amp: bool = False
    ):
        self.model = model
        self.path = path
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    def train_step(self, z: torch.Tensor) -> dict:
        """Perform a single training step as in Algorithm 3.
        
        Args:
            z: Data sample
            
        Returns:
            Dictionary of metrics
        """
        z = z.to(self.device)
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Sample time uniformly as in Algorithm 3, line 3
            t = torch.rand(z.size(0), device=self.device)
            
            # Sample noise and compute x as in Algorithm 3, line 4-5
            if isinstance(self.path, GaussianProbPath):
                x = self.path.sample_path(z, t)
                
                # Compute target vector field
                u_target = self.path.get_conditional_vector_field(x, z, t)
            else:
                # General case for any probability path
                x = self.path.sample_path(z, t)
                u_target = self.path.get_conditional_vector_field(x, z, t)
            
            # Compute model prediction
            u_pred = self.model(x, t)
            
            # Compute loss as in Algorithm 3, line 6
            loss = F.mse_loss(u_pred, u_target)
        
        # Update model parameters as in Algorithm 3, line 7
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def train_epoch(self, data_loader: DataLoader) -> dict:
        """Train for one epoch.
        
        Args:
            data_loader: Data loader providing samples
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0
        
        for batch in data_loader:
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
        
        return {"loss": total_loss / len(data_loader)}
