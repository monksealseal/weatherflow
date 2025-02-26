#!/usr/bin/env python3
"""
ERA5 Flow Matching Example Script

This script demonstrates how to use the WeatherFlow library to:
1. Load ERA5 data
2. Train a flow matching model
3. Generate weather predictions
4. Visualize and evaluate results

Author: monksealseal
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm

# Import WeatherFlow components
from weatherflow.data import ERA5Dataset, create_data_loaders
from weatherflow.models import WeatherFlowMatch, WeatherFlowODE
from weatherflow.utils import WeatherVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ERA5 Flow Matching Example')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to ERA5 data')
    parser.add_argument('--variables', type=str, nargs='+', default=['z', 't'],
                        help='Variables to use')
    parser.add_argument('--pressure-levels', type=int, nargs='+', default=[500],
                        help='Pressure levels to use')
    parser.add_argument('--train-years', type=str, nargs='+', default=['2015'],
                        help='Years for training data')
    parser.add_argument('--val-years', type=str, nargs='+', default=['2016'],
                        help='Years for validation data')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for model')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of layers in model')
    parser.add_argument('--use-attention', action='store_true',
                        help='Use attention in model')
    parser.add_argument('--physics-informed', action='store_true',
                        help='Use physics constraints')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--save-model', action='store_true',
                        help='Save model checkpoint')
    parser.add_argument('--save-results', action='store_true',
                        help='Save results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set experiment name
    if args.exp_name is None:
        args.exp_name = f'flow_match_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    return args

def setup_output_dir(args):
    """Create output directory structure."""
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dirs = {
        'models': output_dir / 'models',
        'plots': output_dir / 'plots',
        'predictions': output_dir / 'predictions'
    }
    
    for d in dirs.values():
        d.mkdir(exist_ok=True)
        
    # Save arguments
    with open(output_dir / 'args.txt', 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')
    
    return dirs

def load_data(args):
    """Load ERA5 data."""
    logger.info("Loading data...")
    
    # Format time slices
    train_slice = (f"{args.train_years[0]}", f"{args.train_years[-1]}-12-31")
    val_slice = (f"{args.val_years[0]}", f"{args.val_years[-1]}-12-31")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        variables=args.variables,
        pressure_levels=args.pressure_levels,
        data_path=args.data_path,
        train_slice=train_slice,
        val_slice=val_slice,
        batch_size=args.batch_size,
        num_workers=4,
        normalize=True
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    return train_loader, val_loader

def setup_model(args, input_channels):
    """Setup flow matching model."""
    logger.info("Setting up model...")
    
    # Create model
    model = WeatherFlowMatch(
        input_channels=input_channels,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        use_attention=args.use_attention,
        physics_informed=args.physics_informed
    )
    
    # Move to device
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    return model, optimizer, scheduler

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        x0, x1 = batch['input'].to(device), batch['target'].to(device)
        
        # Generate random timesteps
        t = torch.rand(x0.size(0), device=device)
        
        # Compute loss
        losses = model.compute_flow_loss(x0, x1, t)
        loss = losses['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}"
        })
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Get data
            x0, x1 = batch['input'].to(device), batch['target'].to(device)
            
            # Generate random timesteps
            t = torch.rand(x0.size(0), device=device)
            
            # Compute loss
            losses = model.compute_flow_loss(x0, x1, t)
            loss = losses['total_loss']
            
            # Update metrics
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train(model, train_loader, val_loader, optimizer, scheduler, args, output_dirs):
    """Train the model."""
    logger.info("Starting training...")
    
    # Setup for tracking metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, args.device)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                    f"Train loss: {train_loss:.4f}, "
                    f"Val loss: {val_loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, output_dirs['models'] / 'best_model.pt')
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dirs['plots'] / 'training_curves.png')
    
    logger.info("Training completed!")
    return train_losses, val_losses

def generate_predictions(model, val_loader, args, output_dirs):
    """Generate predictions using trained model."""
    logger.info("Generating predictions...")
    
    # Create ODE model for prediction
    ode_model = WeatherFlowODE(
        flow_model=model,
        solver_method='dopri5',
        rtol=1e-4,
        atol=1e-4
    )
    
    # Create visualizer
    visualizer = WeatherVisualizer(save_dir=str(output_dirs['plots']))
    
    # Get first batch
    batch = next(iter(val_loader))
    x0, x1 = batch['input'].to(args.device), batch['target'].to(args.device)
    metadata = batch['metadata']
    
    # Generate predictions for different lead times
    lead_times = torch.linspace(0, 1, 5)
    
    # Generate predictions
    with torch.no_grad():
        predictions = ode_model(x0, lead_times)
    
    # Create comparison plots
    for i, lt in enumerate(lead_times[1:]):  # Skip t=0
        lt_idx = i + 1
        
        # Select first sample
        pred = predictions[lt_idx, 0].cpu()
        true = x0[0].cpu() if lt_idx == 0 else x1[0].cpu()
        
        # Plot comparison
        for var_idx, var_name in enumerate(args.variables):
            title = f"Prediction vs Truth (Lead time: {lt.item():.2f})"
            fig, _ = visualizer.plot_comparison(
                true_data={var_name: true[var_idx]},
                pred_data={var_name: pred[var_idx]},
                var_name=var_name,
                title=title
            )
            
            # Save figure
            fig.savefig(output_dirs['plots'] / f'comparison_{var_name}_t{lt_idx}.png')
            plt.close(fig)
    
    # Create animation for one variable
    var_idx = 0  # First variable
    var_name = args.variables[var_idx]
    
    anim = visualizer.create_prediction_animation(
        predictions[:, 0, var_idx].cpu(),
        var_name=var_name,
        title=f"{var_name} Prediction",
        save_path=str(output_dirs['plots'] / f'animation_{var_name}.gif')
    )
    
    logger.info("Predictions generated!")
    return predictions

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directories
    output_dirs = setup_output_dir(args)
    
    # Load data
    train_loader, val_loader = load_data(args)
    
    # Get input shape
    sample_batch = next(iter(train_loader))
    input_channels = sample_batch['input'].shape[1]
    
    # Setup model
    model, optimizer, scheduler = setup_model(args, input_channels)
    
    # Print model summary
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    train_losses, val_losses = train(model, train_loader, val_loader, optimizer, scheduler, args, output_dirs)
    
    # Generate predictions
    predictions = generate_predictions(model, val_loader, args, output_dirs)
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main()