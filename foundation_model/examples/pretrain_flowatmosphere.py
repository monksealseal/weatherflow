"""
Example script for pre-training FlowAtmosphere foundation model.

This script demonstrates how to:
1. Set up the massive data pipeline
2. Initialize the FlowFormer model
3. Configure multi-objective pre-training
4. Train with distributed FSDP
"""

import torch
import torch.distributed as dist
from pathlib import Path

from foundation_model.models.flow_atmosphere import FlowAtmosphere
from foundation_model.data.massive_pipeline import MassiveDataPipeline, CurriculumDataScheduler
from foundation_model.training.distributed_trainer import DistributedFlowTrainer
from foundation_model.objectives.pretraining import MultiObjectivePretraining, CurriculumPretraining


def main():
    # Configuration
    config = {
        # Model architecture
        'input_channels': 128,  # Multi-variable, multi-level
        'd_model': 1024,  # 10B parameters with depth
        'num_layers': 24,
        'num_heads': 16,
        'd_ff': 4096,
        'dropout': 0.1,
        'window_size': 16,

        # Training
        'batch_size': 8,  # Per GPU
        'gradient_accumulation_steps': 4,
        'max_epochs': 100,
        'learning_rate': 1e-4,
        'max_grad_norm': 1.0,

        # Data
        'data_sources': [
            '/data/era5_zarr',
            '/data/cmip6_zarr',
        ],
        'num_workers': 8,

        # Checkpointing
        'checkpoint_dir': './checkpoints/flowatm-10b',
    }

    print("=" * 80)
    print("FlowAtmosphere Pre-training")
    print("=" * 80)

    # 1. Initialize data pipeline
    print("\n1. Initializing massive data pipeline...")
    data_pipeline = MassiveDataPipeline(
        data_sources=config['data_sources'],
        cache_dir='/tmp/flowatm_cache',
        num_workers=config['num_workers'],
    )

    # Create data loaders
    train_loader = data_pipeline.create_training_dataset(
        split='train',
        batch_size=config['batch_size'],
        shuffle=True,
    )

    val_loader = data_pipeline.create_training_dataset(
        split='val',
        batch_size=config['batch_size'],
        shuffle=False,
    )

    print(f"✓ Data pipeline initialized")

    # 2. Initialize FlowAtmosphere model
    print("\n2. Initializing FlowAtmosphere model...")
    model = FlowAtmosphere(
        input_channels=config['input_channels'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        window_size=config['window_size'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {total_params / 1e9:.2f}B parameters")

    # 3. Set up multi-objective pre-training
    print("\n3. Setting up multi-objective pre-training...")
    pretraining = MultiObjectivePretraining(
        model=model,
        mask_ratio=0.15,
        temporal_window=8,
    )

    # Curriculum learning
    curriculum = CurriculumPretraining(
        pretraining_module=pretraining,
        num_stages=5,
    )

    print(f"✓ Pre-training objectives configured")
    print(f"  - Flow Matching (base)")
    print(f"  - Masked Variable Modeling")
    print(f"  - Temporal Jigsaw Puzzle")
    print(f"  - Climate Invariance Learning")

    # 4. Initialize optimizer and scheduler
    print("\n4. Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Cosine annealing with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['max_epochs'],
        eta_min=1e-6,
    )

    print(f"✓ Optimizer: AdamW with lr={config['learning_rate']}")

    # 5. Initialize distributed trainer
    print("\n5. Initializing distributed trainer with FSDP...")
    trainer = DistributedFlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=config['max_epochs'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        max_grad_norm=config['max_grad_norm'],
        checkpoint_dir=config['checkpoint_dir'],
        use_wandb=True,
        mixed_precision=True,
    )

    print(f"✓ Distributed trainer initialized")

    # 6. Start pre-training
    print("\n6. Starting pre-training...")
    print("=" * 80)

    trainer.train()

    print("\n" + "=" * 80)
    print("Pre-training complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
