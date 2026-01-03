"""
Parameter-Efficient Fine-Tuning (PEFT) for FlowAtmosphere

Implements LoRA (Low-Rank Adaptation) and other PEFT methods to enable
efficient adaptation of the massive foundation model to specific tasks
with minimal compute and parameters.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    r: int = 8  # Rank of low-rank adaptation
    alpha: float = 16.0  # Scaling factor
    dropout: float = 0.1
    target_modules: List[str] = None  # Module names to apply LoRA


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.

    Adds trainable low-rank decomposition matrices to frozen pretrained weights.
    Instead of training W, we train: W + (B @ A) where A and B are low-rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

        # Scaling
        self.scaling = alpha / r

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, original_weight: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x: Input tensor
            original_weight: Frozen pretrained weight

        Returns:
            Adapted output
        """
        # Original forward
        original_out = F.linear(x, original_weight)

        # LoRA adaptation
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling

        return original_out + lora_out


class LoRAAdapter(nn.Module):
    """
    LoRA adapter that wraps around FlowAtmosphere model.

    Freezes the base model and adds trainable low-rank adaptations
    to specified modules.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: LoRAConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Add LoRA layers
        self.lora_layers: Dict[str, LoRALayer] = {}
        self._inject_lora_layers()

        # Register as nn.ModuleDict for proper parameter tracking
        self.lora_modules = nn.ModuleDict(self.lora_layers)

    def _inject_lora_layers(self):
        """Inject LoRA layers into target modules."""
        target_modules = self.config.target_modules or [
            'attn_proj_q', 'attn_proj_k', 'attn_proj_v', 'attn_proj_o',
            'ffn.fc1', 'ffn.fc2',
        ]

        for name, module in self.base_model.named_modules():
            # Check if this module should have LoRA
            should_adapt = any(target in name for target in target_modules)

            if should_adapt and isinstance(module, nn.Linear):
                # Create LoRA layer
                lora = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=self.config.r,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout,
                )

                self.lora_layers[name] = lora
                print(f"Added LoRA to {name}: {module.in_features} -> {module.out_features}, rank={self.config.r}")

    def forward(self, *args, **kwargs):
        """Forward pass with LoRA adaptations."""
        # Store original forward methods
        original_forwards = {}

        # Replace linear layers with LoRA-adapted versions
        for name, lora_layer in self.lora_layers.items():
            # Get the actual module
            module = self._get_module_by_name(name)

            if module is not None and isinstance(module, nn.Linear):
                # Save original forward
                original_forwards[name] = module.forward

                # Create adapted forward function
                original_weight = module.weight

                def make_lora_forward(lora, orig_weight):
                    def lora_forward(x):
                        return lora(x, orig_weight)
                    return lora_forward

                # Replace forward
                module.forward = make_lora_forward(lora_layer, original_weight)

        # Run base model with LoRA adaptations
        output = self.base_model(*args, **kwargs)

        # Restore original forward methods
        for name, original_forward in original_forwards.items():
            module = self._get_module_by_name(name)
            if module is not None:
                module.forward = original_forward

        return output

    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get module by dotted name."""
        parts = name.split('.')
        module = self.base_model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None

        return module

    def save_adapter(self, path: str):
        """Save only LoRA parameters (very small)."""
        lora_state = {
            name: layer.state_dict()
            for name, layer in self.lora_layers.items()
        }

        torch.save({
            'lora_state_dict': lora_state,
            'config': self.config,
        }, path)

        print(f"Saved LoRA adapter to {path}")

    def load_adapter(self, path: str):
        """Load LoRA parameters."""
        checkpoint = torch.load(path, map_location='cpu')

        for name, state_dict in checkpoint['lora_state_dict'].items():
            if name in self.lora_layers:
                self.lora_layers[name].load_state_dict(state_dict)

        print(f"Loaded LoRA adapter from {path}")


class PEFTEngine:
    """
    Engine for managing parameter-efficient fine-tuning.

    Handles:
    - Adapter injection
    - Task-specific fine-tuning
    - Adapter merging and swapping
    - Multi-task adapter management
    """

    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.current_adapter: Optional[str] = None

    def create_adapter(
        self,
        name: str,
        config: Optional[LoRAConfig] = None,
    ) -> LoRAAdapter:
        """
        Create a new task-specific adapter.

        Args:
            name: Adapter name (e.g., 'forecast_europe', 'downscale_california')
            config: LoRA configuration

        Returns:
            LoRA adapter instance
        """
        if config is None:
            config = LoRAConfig()

        adapter = LoRAAdapter(self.base_model, config)
        self.adapters[name] = adapter

        print(f"Created adapter '{name}' with {self._count_adapter_params(adapter)} trainable parameters")

        return adapter

    def fine_tune_adapter(
        self,
        adapter_name: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
    ):
        """
        Fine-tune a specific adapter.

        Args:
            adapter_name: Name of adapter to fine-tune
            train_loader: Training data
            val_loader: Validation data
            num_epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for adapter parameters
            device: Device to use
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")

        adapter = self.adapters[adapter_name]
        adapter.to(device)

        # Optimizer only for LoRA parameters
        optimizer = torch.optim.AdamW(
            adapter.lora_modules.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        best_val_loss = float('inf')

        print(f"Fine-tuning adapter '{adapter_name}'...")

        for epoch in range(num_epochs):
            # Training
            adapter.train()
            train_loss = 0.0

            for batch_idx, (x0, x1) in enumerate(train_loader):
                x0, x1 = x0.to(device), x1.to(device)

                optimizer.zero_grad()

                # Forward pass
                # Assume adapter wraps forecast method
                # Simplified - actual implementation depends on task
                loss = self._compute_task_loss(adapter, x0, x1)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            adapter.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x0, x1 in val_loader:
                    x0, x1 = x0.to(device), x1.to(device)
                    loss = self._compute_task_loss(adapter, x0, x1)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Save best adapter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_adapter(adapter_name, f'best_{adapter_name}.pt')

    def _compute_task_loss(
        self,
        adapter: LoRAAdapter,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute task-specific loss."""
        # Simplified - should be customized per task
        # For now, use flow matching loss

        B = x0.size(0)
        t = torch.rand(B, device=x0.device)

        x_t = t.view(-1, 1, 1, 1) * x1 + (1 - t.view(-1, 1, 1, 1)) * x0
        v_target = x1 - x0

        # Create grids (placeholder)
        _, _, H, W = x0.shape
        lat_grid = torch.linspace(-torch.pi/2, torch.pi/2, H, device=x0.device)
        lon_grid = torch.linspace(-torch.pi, torch.pi, W, device=x0.device)
        lat_grid, lon_grid = torch.meshgrid(lat_grid, lon_grid, indexing='ij')

        # Forward with adapter
        v_pred = adapter(x_t, t, lat_grid, lon_grid)

        loss = F.mse_loss(v_pred, v_target)
        return loss

    def switch_adapter(self, adapter_name: str):
        """Switch to a different adapter."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")

        self.current_adapter = adapter_name
        print(f"Switched to adapter '{adapter_name}'")

    def merge_adapters(
        self,
        adapter_names: List[str],
        weights: Optional[List[float]] = None,
        merged_name: str = 'merged',
    ):
        """
        Merge multiple adapters into a single adapter.

        Useful for multi-task models.

        Args:
            adapter_names: Names of adapters to merge
            weights: Weights for each adapter (default: equal)
            merged_name: Name for merged adapter
        """
        if weights is None:
            weights = [1.0 / len(adapter_names)] * len(adapter_names)

        # Create new adapter
        config = LoRAConfig()
        merged_adapter = self.create_adapter(merged_name, config)

        # Merge LoRA parameters
        for lora_name in merged_adapter.lora_layers.keys():
            merged_A = torch.zeros_like(merged_adapter.lora_layers[lora_name].lora_A)
            merged_B = torch.zeros_like(merged_adapter.lora_layers[lora_name].lora_B)

            for adapter_name, weight in zip(adapter_names, weights):
                if adapter_name in self.adapters:
                    adapter = self.adapters[adapter_name]
                    if lora_name in adapter.lora_layers:
                        merged_A += weight * adapter.lora_layers[lora_name].lora_A.data
                        merged_B += weight * adapter.lora_layers[lora_name].lora_B.data

            merged_adapter.lora_layers[lora_name].lora_A.data = merged_A
            merged_adapter.lora_layers[lora_name].lora_B.data = merged_B

        print(f"Merged {len(adapter_names)} adapters into '{merged_name}'")

        return merged_adapter

    def save_adapter(self, adapter_name: str, path: str):
        """Save specific adapter."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")

        self.adapters[adapter_name].save_adapter(path)

    def load_adapter(self, adapter_name: str, path: str, config: Optional[LoRAConfig] = None):
        """Load adapter from file."""
        if adapter_name not in self.adapters:
            # Create adapter first
            self.create_adapter(adapter_name, config)

        self.adapters[adapter_name].load_adapter(path)

    def _count_adapter_params(self, adapter: LoRAAdapter) -> int:
        """Count trainable parameters in adapter."""
        return sum(p.numel() for p in adapter.lora_modules.parameters() if p.requires_grad)

    def get_adapter_stats(self) -> Dict[str, Dict]:
        """Get statistics for all adapters."""
        stats = {}

        for name, adapter in self.adapters.items():
            stats[name] = {
                'num_params': self._count_adapter_params(adapter),
                'num_lora_layers': len(adapter.lora_layers),
                'rank': adapter.config.r,
            }

        return stats


class TaskSpecificAdapter:
    """
    High-level interface for creating task-specific adaptations.

    Provides pre-configured adapters for common atmospheric tasks.
    """

    def __init__(self, base_model: nn.Module):
        self.peft_engine = PEFTEngine(base_model)

    def create_forecast_adapter(
        self,
        region: str,
        lead_time_hours: int = 240,
    ) -> LoRAAdapter:
        """Create adapter for regional forecasting."""
        config = LoRAConfig(
            r=16,  # Higher rank for forecasting
            alpha=32.0,
        )

        name = f'forecast_{region}_{lead_time_hours}h'
        return self.peft_engine.create_adapter(name, config)

    def create_downscale_adapter(
        self,
        region: str,
        resolution_factor: int = 4,
    ) -> LoRAAdapter:
        """Create adapter for downscaling."""
        config = LoRAConfig(
            r=8,
            alpha=16.0,
        )

        name = f'downscale_{region}_{resolution_factor}x'
        return self.peft_engine.create_adapter(name, config)

    def create_extreme_event_adapter(
        self,
        event_type: str,  # 'hurricane', 'heatwave', 'drought', etc.
    ) -> LoRAAdapter:
        """Create adapter for extreme event prediction."""
        config = LoRAConfig(
            r=12,
            alpha=24.0,
        )

        name = f'extreme_{event_type}'
        return self.peft_engine.create_adapter(name, config)

    def create_s2s_adapter(
        self,
        lead_weeks: int = 6,
    ) -> LoRAAdapter:
        """Create adapter for sub-seasonal to seasonal prediction."""
        config = LoRAConfig(
            r=20,  # Higher rank for long-term prediction
            alpha=40.0,
        )

        name = f's2s_{lead_weeks}weeks'
        return self.peft_engine.create_adapter(name, config)
