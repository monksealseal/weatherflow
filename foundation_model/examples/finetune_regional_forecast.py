"""
Example script for fine-tuning FlowAtmosphere for regional forecasting using LoRA.

This demonstrates parameter-efficient adaptation of the 10B parameter model
to a specific region (e.g., Europe) with minimal compute.
"""

import torch
from pathlib import Path

from foundation_model.models.flow_atmosphere import FlowAtmosphere
from foundation_model.adaptation.peft import (
    LoRAConfig,
    PEFTEngine,
    TaskSpecificAdapter,
)


def main():
    print("=" * 80)
    print("FlowAtmosphere Regional Fine-Tuning with LoRA")
    print("=" * 80)

    # 1. Load pre-trained FlowAtmosphere model
    print("\n1. Loading pre-trained FlowAtmosphere-10B model...")

    model = FlowAtmosphere(
        input_channels=128,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
        pretrained_path='./checkpoints/flowatm-10b/best_model.pt',
    )

    print(f"✓ Loaded pre-trained model")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e9:.2f}B")

    # 2. Create regional forecast adapter
    print("\n2. Creating LoRA adapter for Europe forecasting...")

    adapter_engine = TaskSpecificAdapter(model)

    # Create adapter for European 10-day forecast
    adapter = adapter_engine.create_forecast_adapter(
        region='europe',
        lead_time_hours=240,  # 10 days
    )

    trainable_params = sum(p.numel() for p in adapter.lora_modules.parameters())
    print(f"✓ Created LoRA adapter")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"  Percentage of base model: {100 * trainable_params / total_params:.4f}%")

    # 3. Prepare regional data
    print("\n3. Loading European regional data...")

    # In practice, load regional ERA5 data
    # For demonstration, use dummy data
    from torch.utils.data import TensorDataset, DataLoader

    # Dummy data: 100 samples, 128 channels, 256x256 spatial
    train_x0 = torch.randn(100, 128, 256, 256)
    train_x1 = torch.randn(100, 128, 256, 256)
    train_dataset = TensorDataset(train_x0, train_x1)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_x0 = torch.randn(20, 128, 256, 256)
    val_x1 = torch.randn(20, 128, 256, 256)
    val_dataset = TensorDataset(val_x0, val_x1)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"✓ Data loaded")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # 4. Fine-tune adapter
    print("\n4. Fine-tuning LoRA adapter...")

    adapter_engine.peft_engine.fine_tune_adapter(
        adapter_name='forecast_europe_240h',
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    print(f"✓ Fine-tuning complete")

    # 5. Save adapter
    print("\n5. Saving adapter...")

    output_path = './adapters/forecast_europe_240h.pt'
    Path('./adapters').mkdir(exist_ok=True)

    adapter_engine.peft_engine.save_adapter(
        'forecast_europe_240h',
        output_path
    )

    print(f"✓ Adapter saved to {output_path}")
    print(f"  File size: ~{trainable_params * 4 / 1e6:.1f} MB (vs {total_params * 4 / 1e9:.1f} GB for full model)")

    # 6. Demonstrate inference
    print("\n6. Running inference with adapted model...")

    # Sample forecast
    test_state = torch.randn(1, 128, 256, 256)
    if torch.cuda.is_available():
        test_state = test_state.cuda()
        adapter = adapter.cuda()

    lat_grid = torch.linspace(-torch.pi/2, torch.pi/2, 256)
    lon_grid = torch.linspace(-torch.pi, torch.pi, 256)
    lat_grid, lon_grid = torch.meshgrid(lat_grid, lon_grid, indexing='ij')

    if torch.cuda.is_available():
        lat_grid = lat_grid.cuda()
        lon_grid = lon_grid.cuda()

    with torch.no_grad():
        forecast = adapter.base_model.forecast(
            initial_state=test_state,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            lead_times=[24, 48, 72, 96, 120, 144, 168, 192, 216, 240],
            num_ensemble=4,
        )

    print(f"✓ Generated 10-day ensemble forecast")
    print(f"  Output shape: {forecast.shape}")

    print("\n" + "=" * 80)
    print("Fine-tuning complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Evaluate on regional test set")
    print("  - Compare with baseline models")
    print("  - Deploy for operational forecasting")


if __name__ == '__main__':
    main()
