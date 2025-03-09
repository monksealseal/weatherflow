
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Import our improved modules
from weatherflow.models.base import BaseWeatherModel
from weatherflow.training.flow_trainer import FlowTrainer, compute_flow_loss

# Create a simple synthetic dataset
class SyntheticWeatherDataset(Dataset):
    def __init__(self, size=100, grid_shape=(16, 32), channels=3):
        self.size = size
        self.grid_shape = grid_shape
        self.channels = channels
        
        # Generate random data
        np.random.seed(42)  # For reproducibility
        self.data = torch.randn(size, channels, *grid_shape)
        
    def __len__(self):
        return self.size - 1  # -1 because we need pairs
        
    def __getitem__(self, idx):
        # Return consecutive frames as input/target
        return self.data[idx], self.data[idx + 1]

# Create a simple model that implements BaseWeatherModel
class SimpleWeatherModel(BaseWeatherModel):
    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        
        # Simple CNN architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),  # +1 for time
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t):
        """Forward pass with time conditioning."""
        # Encode input
        features = self.encoder(x)
        
        # Add time information
        t = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        features_with_time = torch.cat([features, t], dim=1)
        
        # Decode to get velocity field
        velocity = self.decoder(features_with_time)
        
        return velocity
    
    def mass_conservation_constraint(self, x):
        """Implement mass conservation constraint."""
        # Simple divergence calculation for velocity field
        if x.shape[1] >= 2:
            # Assume first two channels are u,v components
            u = x[:, 0]
            v = x[:, 1]
            
            # Compute spatial derivatives
            du_dx = torch.gradient(u, dim=2)[0]
            dv_dy = torch.gradient(v, dim=1)[0]
            
            # Divergence
            div = du_dx + dv_dy
            
            return torch.mean(div**2)
        else:
            # Fallback for single-channel data
            return torch.tensor(0.0, device=x.device)
    
    def energy_conservation_constraint(self, x):
        """Implement energy conservation constraint."""
        # Simple energy calculation (sum of squared values)
        return torch.mean(torch.var(torch.sum(x**2, dim=1)))

# Test function
def test_flow_trainer():
    print("Testing FlowTrainer...")
    
    # Create dataset and data loaders
    train_dataset = SyntheticWeatherDataset(size=100)
    val_dataset = SyntheticWeatherDataset(size=20)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model and optimizer
    model = SimpleWeatherModel(in_channels=3, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create checkpoint directory
    checkpoint_dir = Path("test_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create trainer
    trainer = FlowTrainer(
        model=model,
        optimizer=optimizer,
        device="cpu",  # Use CPU for testing
        use_amp=False,  # Disable AMP for testing
        checkpoint_dir=str(checkpoint_dir),
        physics_regularization=True,
        physics_lambda=0.1
    )
    
    # Train for a few epochs
    print("Training for 2 epochs...")
    num_epochs = 2
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    # Test loading a checkpoint
    print("Testing checkpoint loading...")
    trainer.load_checkpoint("checkpoint_epoch_2.pt")
    
    # Test compute_flow_loss function
    print("Testing compute_flow_loss function...")
    batch = next(iter(train_loader))
    x0, x1 = batch
    t = torch.rand(x0.size(0))
    v_t = model(x0, t)
    
    # Test different loss types
    mse_loss = compute_flow_loss(v_t, x0, x1, t, loss_type='mse')
    huber_loss = compute_flow_loss(v_t, x0, x1, t, loss_type='huber')
    smooth_l1_loss = compute_flow_loss(v_t, x0, x1, t, loss_type='smooth_l1')
    
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"Huber Loss: {huber_loss.item():.4f}")
    print(f"Smooth L1 Loss: {smooth_l1_loss.item():.4f}")
    
    # Clean up
    for file in checkpoint_dir.glob("*.pt"):
        file.unlink()
    checkpoint_dir.rmdir()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_flow_trainer()
