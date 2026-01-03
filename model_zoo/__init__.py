"""
WeatherFlow Model Zoo

Pre-trained models for weather prediction and atmospheric analysis.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch

MODEL_ZOO_DIR = Path(__file__).parent


class ModelMetadata:
    """Container for model metadata from model cards."""

    def __init__(self, metadata_dict: Dict[str, Any]):
        self.data = metadata_dict

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def summary(self) -> str:
        """Generate a human-readable summary of the model."""
        lines = [
            f"Model: {self.data['name']} ({self.data['model_id']})",
            f"Description: {self.data['description']}",
            f"",
            f"Architecture:",
            f"  Type: {self.data['architecture']['model_type']}",
            f"  Parameters: {self.data['architecture']['parameter_count']:,}",
            f"  Hidden Dim: {self.data['architecture']['hidden_dim']}",
            f"  Layers: {self.data['architecture']['n_layers']}",
            f"",
            f"Training Data:",
            f"  Variables: {', '.join(self.data['training_data']['variables'])}",
            f"  Pressure Levels: {self.data['training_data']['pressure_levels']}",
            f"  Resolution: {self.data['training_data']['spatial_resolution']}",
            f"",
            f"Performance (Test Set):",
        ]

        # Add performance metrics
        for lead_time, metrics in self.data['performance_metrics']['lead_times'].items():
            lines.append(f"  {lead_time}:")
            lines.append(f"    ACC: {metrics.get('acc', 'N/A')}")
            lines.append(f"    RMSE: {metrics.get('rmse', 'N/A')}")

        lines.extend([
            f"",
            f"File: {self.data['file_info']['checkpoint_file']}",
            f"Size: {self.data['file_info']['file_size_mb']:.1f} MB",
        ])

        return "\n".join(lines)


def list_models(category: Optional[str] = None) -> List[str]:
    """
    List all available pre-trained models.

    Args:
        category: Optional category filter (e.g., 'global_forecasting', 'extreme_events')

    Returns:
        List of model IDs
    """
    models = []

    # Search for model cards in the zoo directory
    search_dirs = [MODEL_ZOO_DIR]
    if category:
        category_dir = MODEL_ZOO_DIR / category
        if category_dir.exists():
            search_dirs = [category_dir]

    for search_dir in search_dirs:
        for model_card in search_dir.rglob("model_card.json"):
            with open(model_card) as f:
                data = json.load(f)
                models.append(data['model_id'])

    return sorted(models)


def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get information about a specific model.

    Args:
        model_id: Unique model identifier

    Returns:
        Dictionary containing model information
    """
    metadata = _load_model_metadata(model_id)

    return {
        'model_id': metadata['model_id'],
        'name': metadata['name'],
        'description': metadata['description'],
        'variables': metadata['training_data']['variables'],
        'pressure_levels': metadata['training_data']['pressure_levels'],
        'metrics': metadata['performance_metrics'],
        'file_size_mb': metadata['file_info']['file_size_mb'],
    }


def _load_model_metadata(model_id: str) -> Dict[str, Any]:
    """Load model metadata from model card."""
    # Search for the model card
    for model_card_path in MODEL_ZOO_DIR.rglob("model_card.json"):
        with open(model_card_path) as f:
            data = json.load(f)
            if data['model_id'] == model_id:
                return data

    raise ValueError(f"Model '{model_id}' not found in the model zoo")


def _find_model_checkpoint(model_id: str, metadata: Dict[str, Any]) -> Path:
    """Find the checkpoint file for a model."""
    checkpoint_name = metadata['file_info']['checkpoint_file']

    # Search in the same directory as the model card
    for model_card_path in MODEL_ZOO_DIR.rglob("model_card.json"):
        with open(model_card_path) as f:
            data = json.load(f)
            if data['model_id'] == model_id:
                checkpoint_path = model_card_path.parent / checkpoint_name
                if checkpoint_path.exists():
                    return checkpoint_path
                else:
                    raise FileNotFoundError(
                        f"Checkpoint file '{checkpoint_name}' not found. "
                        f"You may need to download it using: "
                        f"python model_zoo/download_model.py {model_id}"
                    )

    raise ValueError(f"Model '{model_id}' not found in the model zoo")


def load_model(
    model_id: str,
    device: Optional[str] = None,
    return_metadata: bool = True
) -> Tuple[torch.nn.Module, Optional[ModelMetadata]]:
    """
    Load a pre-trained model from the Model Zoo.

    Args:
        model_id: Unique model identifier (e.g., 'wf_z500_3day_v1')
        device: Device to load the model on ('cuda', 'cpu', or None for auto)
        return_metadata: Whether to return the model metadata

    Returns:
        Tuple of (model, metadata) if return_metadata=True, else just model

    Example:
        >>> model, metadata = load_model('wf_z500_3day_v1')
        >>> print(metadata.summary())
        >>> predictions = model(input_data)
    """
    # Load metadata
    metadata_dict = _load_model_metadata(model_id)
    metadata = ModelMetadata(metadata_dict)

    # Find checkpoint
    checkpoint_path = _find_model_checkpoint(model_id, metadata_dict)

    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Import the model class
    from weatherflow import models

    model_type = metadata['architecture']['model_type']
    model_class = getattr(models, model_type)

    # Instantiate model with architecture parameters
    arch = metadata['architecture']
    model_kwargs = {
        'input_channels': arch['input_channels'],
        'hidden_dim': arch['hidden_dim'],
        'n_layers': arch['n_layers'],
    }

    # Add optional parameters
    if arch.get('use_attention'):
        model_kwargs['use_attention'] = True
    if arch.get('physics_informed'):
        model_kwargs['physics_informed'] = True
    if arch.get('grid_size'):
        model_kwargs['grid_size'] = tuple(arch['grid_size'])

    model = model_class(**model_kwargs)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    if return_metadata:
        return model, metadata
    else:
        return model


def download_model(model_id: str, force: bool = False) -> Path:
    """
    Download a model that is hosted externally.

    Args:
        model_id: Unique model identifier
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded checkpoint
    """
    metadata = _load_model_metadata(model_id)

    # Check if model is already downloaded
    try:
        checkpoint_path = _find_model_checkpoint(model_id, metadata)
        if not force:
            print(f"Model '{model_id}' is already downloaded at {checkpoint_path}")
            return checkpoint_path
    except FileNotFoundError:
        pass

    # Download logic would go here
    # For now, just provide instructions
    raise NotImplementedError(
        f"Model '{model_id}' requires external download. "
        f"Please run: python model_zoo/download_model.py {model_id}"
    )


__all__ = [
    'list_models',
    'get_model_info',
    'load_model',
    'download_model',
    'ModelMetadata',
]
