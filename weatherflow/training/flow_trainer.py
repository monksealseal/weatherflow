
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
