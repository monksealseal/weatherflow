"""
Worldsphere Model Registry

Central registry for all Worldsphere AI models including:
- CycleGAN/Pix2Pix models for image-to-image translation
- Video diffusion models for sequence prediction
- Hybrid models combining multiple approaches

Features:
- Model versioning
- Metadata tracking
- Performance history
- Model comparison
"""

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models supported by Worldsphere."""

    CYCLEGAN = "cyclegan"
    PIX2PIX = "pix2pix"
    VIDEO_DIFFUSION = "video_diffusion"
    STABLE_VIDEO_DIFFUSION = "stable_video_diffusion"
    HYBRID = "hybrid"


class ModelStatus(Enum):
    """Status of a model in the registry."""

    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    # Identity
    model_id: str = ""
    name: str = ""
    version: str = "1.0.0"
    model_type: ModelType = ModelType.PIX2PIX
    status: ModelStatus = ModelStatus.TRAINING

    # Description
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = "Worldsphere"

    # Architecture
    architecture: str = ""
    input_variables: List[str] = field(default_factory=list)
    output_variables: List[str] = field(default_factory=list)
    input_shape: Tuple[int, ...] = (3, 256, 256)
    output_shape: Tuple[int, ...] = (2, 256, 256)

    # For sequence models
    num_frames: int = 1
    conditioning_frames: int = 0

    # Training info
    training_dataset: str = ""
    training_samples: int = 0
    training_epochs: int = 0
    training_time_hours: float = 0.0

    # Performance metrics
    best_rmse: float = float("inf")
    best_mae: float = float("inf")
    validation_rmse: float = float("inf")
    test_rmse: float = float("inf")

    # Hurricane-specific metrics
    wind_speed_rmse: float = float("inf")
    wind_direction_rmse: float = float("inf")

    # Hyperparameters used
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    trained_at: str = ""

    # Paths
    checkpoint_path: str = ""
    config_path: str = ""
    experiment_log_path: str = ""

    # Lineage
    parent_model_id: Optional[str] = None
    base_model: Optional[str] = None  # e.g., "stable-video-diffusion-img2vid"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["model_type"] = self.model_type.value
        result["status"] = self.status.value
        result["input_shape"] = list(self.input_shape)
        result["output_shape"] = list(self.output_shape)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        data = data.copy()
        if "model_type" in data:
            data["model_type"] = ModelType(data["model_type"])
        if "status" in data:
            data["status"] = ModelStatus(data["status"])
        if "input_shape" in data:
            data["input_shape"] = tuple(data["input_shape"])
        if "output_shape" in data:
            data["output_shape"] = tuple(data["output_shape"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class WorldsphereModelRegistry:
    """
    Central registry for Worldsphere AI models.

    Provides:
    - Model registration and versioning
    - Performance tracking
    - Model comparison
    - Export/import functionality

    Example:
        >>> registry = WorldsphereModelRegistry("./models")
        >>> registry.register_model(model, metadata)
        >>> best_model = registry.get_best_model(ModelType.CYCLEGAN, metric="rmse")
    """

    def __init__(self, base_dir: Union[str, Path] = "./worldsphere_models"):
        """
        Initialize the model registry.

        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.base_dir / "models"
        self.metadata_dir = self.base_dir / "metadata"
        self.exports_dir = self.base_dir / "exports"

        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)

        # Load existing registry
        self.models: Dict[str, ModelMetadata] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load existing model metadata."""
        registry_file = self.metadata_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, "r") as f:
                    data = json.load(f)
                for model_id, meta_dict in data.items():
                    self.models[model_id] = ModelMetadata.from_dict(meta_dict)
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.metadata_dir / "registry.json"
        data = {model_id: meta.to_dict() for model_id, meta in self.models.items()}
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_model_id(self, name: str, model_type: ModelType) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_type.value}_{name}_{timestamp}"

    def register_model(
        self,
        model: "nn.Module",  # type: ignore[name-defined]
        metadata: ModelMetadata,
        save_weights: bool = True,
    ) -> str:
        """
        Register a model in the registry.

        Args:
            model: PyTorch model
            metadata: Model metadata
            save_weights: Whether to save model weights

        Returns:
            Model ID
        """
        # Generate ID if not provided
        if not metadata.model_id:
            metadata.model_id = self._generate_model_id(
                metadata.name or "model", metadata.model_type
            )

        # Set timestamps
        now = datetime.now().isoformat()
        if not metadata.created_at:
            metadata.created_at = now
        metadata.updated_at = now

        # Save model weights
        if save_weights:
            model_dir = self.models_dir / metadata.model_id
            model_dir.mkdir(exist_ok=True)

            weights_path = model_dir / "model.pt"
            torch.save(model.state_dict(), weights_path)
            metadata.checkpoint_path = str(weights_path)

            # Save config
            config_path = model_dir / "metadata.json"
            with open(config_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
            metadata.config_path = str(config_path)

        # Add to registry
        self.models[metadata.model_id] = metadata
        self._save_registry()

        logger.info(f"Registered model: {metadata.model_id}")
        return metadata.model_id

    def update_model(
        self,
        model_id: str,
        updates: Dict[str, Any],
        model: Optional[Any] = None,
    ) -> None:
        """
        Update model metadata.

        Args:
            model_id: Model ID
            updates: Dictionary of fields to update
            model: Optionally update model weights
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")

        metadata = self.models[model_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        metadata.updated_at = datetime.now().isoformat()

        # Update weights if provided
        if model is not None and metadata.checkpoint_path:
            torch.save(model.state_dict(), metadata.checkpoint_path)

        # Save updated metadata
        if metadata.config_path:
            with open(metadata.config_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

        self._save_registry()
        logger.info(f"Updated model: {model_id}")

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)

    def load_model_weights(
        self,
        model_id: str,
        model: Any,
        device: str = "cuda",
    ) -> Any:
        """
        Load model weights from registry.

        Args:
            model_id: Model ID
            model: Model instance to load weights into
            device: Device to load to

        Returns:
            Model with loaded weights
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")

        metadata = self.models[model_id]
        if not metadata.checkpoint_path or not Path(metadata.checkpoint_path).exists():
            raise ValueError(f"Checkpoint not found for model: {model_id}")

        state_dict = torch.load(metadata.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        logger.info(f"Loaded weights for model: {model_id}")
        return model

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "updated_at",
        ascending: bool = False,
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering.

        Args:
            model_type: Filter by model type
            status: Filter by status
            tags: Filter by tags (any match)
            sort_by: Field to sort by
            ascending: Sort order

        Returns:
            List of matching models
        """
        models = list(self.models.values())

        # Apply filters
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        if status:
            models = [m for m in models if m.status == status]
        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]

        # Sort
        models.sort(
            key=lambda m: getattr(m, sort_by, ""),
            reverse=not ascending,
        )

        return models

    def get_best_model(
        self,
        model_type: Optional[ModelType] = None,
        metric: str = "best_rmse",
        status: Optional[ModelStatus] = None,
    ) -> Optional[ModelMetadata]:
        """
        Get the best model by a metric.

        Args:
            model_type: Filter by model type
            metric: Metric to optimize (lower is better for rmse/mae)
            status: Filter by status

        Returns:
            Best model metadata or None
        """
        models = self.list_models(model_type=model_type, status=status)

        if not models:
            return None

        # For RMSE/MAE, lower is better
        return min(models, key=lambda m: getattr(m, metric, float("inf")))

    def compare_models(
        self,
        model_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple models.

        Args:
            model_ids: List of model IDs to compare
            metrics: Metrics to compare (default: rmse, mae, wind_speed_rmse)

        Returns:
            Comparison dictionary
        """
        if metrics is None:
            metrics = ["best_rmse", "best_mae", "wind_speed_rmse", "wind_direction_rmse"]

        models = [self.models.get(mid) for mid in model_ids if mid in self.models]

        if not models:
            return {}

        comparison = {
            "models": [
                {
                    "model_id": m.model_id,
                    "name": m.name,
                    "model_type": m.model_type.value,
                    "status": m.status.value,
                }
                for m in models
            ],
            "metrics": {},
            "best_by_metric": {},
        }

        for metric in metrics:
            values = {m.model_id: getattr(m, metric, float("inf")) for m in models}
            comparison["metrics"][metric] = values

            # Find best model for this metric
            best_id = min(values.keys(), key=lambda k: values[k])
            comparison["best_by_metric"][metric] = best_id

        return comparison

    def delete_model(self, model_id: str, delete_files: bool = True) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model ID
            delete_files: Also delete model files

        Returns:
            True if deleted successfully
        """
        if model_id not in self.models:
            return False

        metadata = self.models[model_id]

        # Delete files
        if delete_files:
            model_dir = self.models_dir / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)

        # Remove from registry
        del self.models[model_id]
        self._save_registry()

        logger.info(f"Deleted model: {model_id}")
        return True

    def export_model(
        self,
        model_id: str,
        export_path: Optional[Path] = None,
        include_training_logs: bool = True,
    ) -> Path:
        """
        Export a model for sharing or deployment.

        Args:
            model_id: Model ID
            export_path: Path to export to
            include_training_logs: Include training logs in export

        Returns:
            Path to exported model
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")

        metadata = self.models[model_id]
        model_dir = self.models_dir / model_id

        if export_path is None:
            export_path = self.exports_dir / f"{model_id}_export"

        export_path.mkdir(parents=True, exist_ok=True)

        # Copy model files
        if model_dir.exists():
            for file in model_dir.iterdir():
                if include_training_logs or not file.name.endswith("_log.json"):
                    shutil.copy(file, export_path / file.name)

        # Create export manifest
        manifest = {
            "model_id": model_id,
            "exported_at": datetime.now().isoformat(),
            "metadata": metadata.to_dict(),
        }

        with open(export_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Exported model to: {export_path}")
        return export_path

    def import_model(
        self,
        import_path: Union[str, Path],
        new_name: Optional[str] = None,
    ) -> str:
        """
        Import a model from export.

        Args:
            import_path: Path to exported model
            new_name: Optional new name for imported model

        Returns:
            New model ID
        """
        import_path = Path(import_path)
        manifest_path = import_path / "manifest.json"

        if not manifest_path.exists():
            raise ValueError(f"Invalid export: manifest.json not found in {import_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        metadata = ModelMetadata.from_dict(manifest["metadata"])

        # Generate new ID
        if new_name:
            metadata.name = new_name
        metadata.model_id = self._generate_model_id(
            metadata.name or "imported", metadata.model_type
        )

        # Copy files
        new_model_dir = self.models_dir / metadata.model_id
        new_model_dir.mkdir(exist_ok=True)

        for file in import_path.iterdir():
            if file.name != "manifest.json":
                shutil.copy(file, new_model_dir / file.name)

        # Update paths
        metadata.checkpoint_path = str(new_model_dir / "model.pt")
        metadata.config_path = str(new_model_dir / "metadata.json")

        # Save updated metadata
        with open(metadata.config_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Register
        self.models[metadata.model_id] = metadata
        self._save_registry()

        logger.info(f"Imported model: {metadata.model_id}")
        return metadata.model_id

    def get_performance_history(
        self,
        model_type: Optional[ModelType] = None,
        metric: str = "best_rmse",
    ) -> List[Dict[str, Any]]:
        """
        Get performance history for models over time.

        Args:
            model_type: Filter by model type
            metric: Metric to track

        Returns:
            List of performance records sorted by date
        """
        models = self.list_models(model_type=model_type, sort_by="trained_at")

        history = []
        for m in models:
            if m.trained_at:
                history.append({
                    "model_id": m.model_id,
                    "name": m.name,
                    "date": m.trained_at,
                    metric: getattr(m, metric, None),
                    "hyperparameters": m.hyperparameters,
                })

        return history

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        type_counts = {}
        status_counts = {}

        for m in self.models.values():
            type_counts[m.model_type.value] = type_counts.get(m.model_type.value, 0) + 1
            status_counts[m.status.value] = status_counts.get(m.status.value, 0) + 1

        best_cyclegan = self.get_best_model(ModelType.CYCLEGAN)
        best_pix2pix = self.get_best_model(ModelType.PIX2PIX)
        best_diffusion = self.get_best_model(ModelType.VIDEO_DIFFUSION)

        return {
            "total_models": len(self.models),
            "by_type": type_counts,
            "by_status": status_counts,
            "best_cyclegan_rmse": best_cyclegan.best_rmse if best_cyclegan else None,
            "best_pix2pix_rmse": best_pix2pix.best_rmse if best_pix2pix else None,
            "best_diffusion_rmse": best_diffusion.best_rmse if best_diffusion else None,
        }


# Global registry instance
_registry: Optional[WorldsphereModelRegistry] = None


def get_registry(base_dir: Optional[str] = None) -> WorldsphereModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None or base_dir is not None:
        _registry = WorldsphereModelRegistry(base_dir or "./worldsphere_models")
    return _registry
