"""
WeatherFlow Checkpoint Management Utilities

This module provides utilities for managing trained model checkpoints
across the Streamlit app. It handles saving, loading, and listing checkpoints.
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Checkpoint directory (consistent with data_storage.py)
BASE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = BASE_DIR / "data" / "trained_models"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0) / (1024**3)
        info["cuda_memory_cached"] = torch.cuda.memory_reserved(0) / (1024**3)
    
    return info


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    config: Dict,
    model_name: str = "weatherflow",
    extra_info: Optional[Dict] = None,
) -> Path:
    """
    Save a model checkpoint to disk.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer (optional)
        epoch: Current epoch number
        train_loss: Training loss at this epoch
        val_loss: Validation loss at this epoch (optional)
        config: Model configuration dictionary
        model_name: Name prefix for the checkpoint file
        extra_info: Additional information to save
        
    Returns:
        Path to the saved checkpoint
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_epoch{epoch:04d}_{timestamp}.pt"
    filepath = CHECKPOINTS_DIR / filename
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if extra_info is not None:
        checkpoint["extra_info"] = extra_info
    
    torch.save(checkpoint, filepath)
    
    # Also save a metadata JSON for easy browsing
    metadata_path = filepath.with_suffix(".json")
    metadata = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "file_size_mb": filepath.stat().st_size / (1024 * 1024),
    }
    if extra_info:
        metadata["extra_info"] = extra_info
        
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Checkpoint saved: {filepath}")
    return filepath


def load_checkpoint(
    filepath: Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load a checkpoint from disk.
    
    Args:
        filepath: Path to the checkpoint file
        model: Optional model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load the checkpoint to
        
    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        logger.info(f"Model weights loaded from {filepath}")
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Optimizer state loaded from {filepath}")
    
    return checkpoint


def list_checkpoints() -> List[Dict]:
    """
    List all available checkpoints with metadata.
    
    Returns:
        List of dictionaries with checkpoint information
    """
    checkpoints = []
    
    for pt_file in sorted(CHECKPOINTS_DIR.glob("*.pt"), reverse=True):
        info = {
            "filename": pt_file.name,
            "filepath": str(pt_file),
            "file_size_mb": pt_file.stat().st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(pt_file.stat().st_mtime).isoformat(),
        }
        
        # Try to load metadata JSON if it exists
        json_path = pt_file.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path) as f:
                    metadata = json.load(f)
                info.update(metadata)
            except Exception:
                pass
        else:
            # Try to extract basic info from the checkpoint
            try:
                checkpoint = torch.load(pt_file, map_location="cpu")
                info["epoch"] = checkpoint.get("epoch", "?")
                info["train_loss"] = checkpoint.get("train_loss", "?")
                info["val_loss"] = checkpoint.get("val_loss", "?")
                info["config"] = checkpoint.get("config", {})
            except Exception:
                pass
        
        checkpoints.append(info)
    
    return checkpoints


def get_best_checkpoint() -> Optional[Dict]:
    """
    Get the best checkpoint based on validation loss.
    
    Returns:
        Checkpoint info dictionary or None if no checkpoints exist
    """
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        return None
    
    # Filter to those with valid val_loss
    valid_checkpoints = [
        c for c in checkpoints 
        if isinstance(c.get("val_loss"), (int, float))
    ]
    
    if not valid_checkpoints:
        # Fall back to most recent
        return checkpoints[0] if checkpoints else None
    
    # Return checkpoint with lowest val_loss
    return min(valid_checkpoints, key=lambda c: c["val_loss"])


def get_latest_checkpoint() -> Optional[Dict]:
    """
    Get the most recently modified checkpoint.
    
    Returns:
        Checkpoint info dictionary or None if no checkpoints exist
    """
    checkpoints = list_checkpoints()
    return checkpoints[0] if checkpoints else None


def has_trained_model() -> bool:
    """Check if any trained model checkpoints exist."""
    return len(list(CHECKPOINTS_DIR.glob("*.pt"))) > 0


def delete_checkpoint(filepath: Path) -> bool:
    """
    Delete a checkpoint and its metadata.
    
    Args:
        filepath: Path to the checkpoint file
        
    Returns:
        True if successful
    """
    try:
        filepath = Path(filepath)
        if filepath.exists():
            filepath.unlink()
        
        json_path = filepath.with_suffix(".json")
        if json_path.exists():
            json_path.unlink()
        
        logger.info(f"Checkpoint deleted: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete checkpoint: {e}")
        return False


def cleanup_old_checkpoints(keep_best: int = 3, keep_latest: int = 2) -> int:
    """
    Clean up old checkpoints, keeping only the best and most recent ones.
    
    Args:
        keep_best: Number of best checkpoints to keep
        keep_latest: Number of latest checkpoints to keep
        
    Returns:
        Number of checkpoints deleted
    """
    checkpoints = list_checkpoints()
    
    if len(checkpoints) <= keep_best + keep_latest:
        return 0
    
    # Identify checkpoints to keep
    keep_set = set()
    
    # Keep the best ones (by val_loss)
    valid_checkpoints = [
        c for c in checkpoints 
        if isinstance(c.get("val_loss"), (int, float))
    ]
    valid_checkpoints.sort(key=lambda c: c["val_loss"])
    for c in valid_checkpoints[:keep_best]:
        keep_set.add(c["filename"])
    
    # Keep the latest ones
    for c in checkpoints[:keep_latest]:
        keep_set.add(c["filename"])
    
    # Delete the rest
    deleted = 0
    for c in checkpoints:
        if c["filename"] not in keep_set:
            if delete_checkpoint(Path(c["filepath"])):
                deleted += 1
    
    return deleted


def create_model_from_config(config: Dict) -> torch.nn.Module:
    """
    Create a WeatherFlowMatch model from a configuration dictionary.
    
    Args:
        config: Model configuration
        
    Returns:
        WeatherFlowMatch model instance
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from weatherflow.models.flow_matching import WeatherFlowMatch
    
    model = WeatherFlowMatch(
        input_channels=config.get("input_channels", 4),
        hidden_dim=config.get("hidden_dim", 128),
        n_layers=config.get("n_layers", 4),
        use_attention=config.get("use_attention", True),
        grid_size=tuple(config.get("grid_size", (32, 64))),
        physics_informed=config.get("physics_informed", True),
        window_size=config.get("window_size", 8),
    )
    
    return model


def load_model_for_inference(checkpoint_path: Optional[Path] = None) -> Tuple[Optional[torch.nn.Module], Optional[Dict]]:
    """
    Load a model ready for inference.
    
    Args:
        checkpoint_path: Path to checkpoint, or None to use best/latest
        
    Returns:
        Tuple of (model, config) or (None, None) if no checkpoint available
    """
    if checkpoint_path is None:
        # Try to get best checkpoint
        checkpoint_info = get_best_checkpoint()
        if checkpoint_info is None:
            checkpoint_info = get_latest_checkpoint()
        
        if checkpoint_info is None:
            return None, None
        
        checkpoint_path = Path(checkpoint_info["filepath"])
    
    if not checkpoint_path.exists():
        return None, None
    
    try:
        device = get_device()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        config = checkpoint.get("config", {})
        model = create_model_from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        return model, config
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None
