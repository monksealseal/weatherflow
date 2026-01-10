"""
Worldsphere Data Manager

Centralized data management for Worldsphere AI models:
- Dataset registration and versioning
- Preprocessing pipeline management
- Data loading utilities
- Sequence data handling for diffusion models

Features:
- Unified data access for CycleGAN and diffusion models
- Preprocessing script management
- Data versioning
- Quality checks
"""

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Dataset = object
    DataLoader = None

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    # Identity
    dataset_id: str = ""
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Data type
    data_type: str = "paired"  # "paired", "unpaired", "sequence"
    source_type: str = "satellite"  # "satellite", "reanalysis", "synthetic"

    # Dimensions
    num_samples: int = 0
    image_size: Tuple[int, int] = (256, 256)
    input_channels: int = 3
    output_channels: int = 2

    # For sequence data
    sequence_length: int = 1
    variables_per_frame: int = 1

    # Variables
    input_variables: List[str] = field(default_factory=list)
    output_variables: List[str] = field(default_factory=list)

    # Split info
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0

    # Quality info
    quality_score: float = 0.0
    completeness: float = 1.0
    has_missing_values: bool = False

    # Paths
    data_path: str = ""
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""
    preprocessing_script: str = ""

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["image_size"] = list(self.image_size)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        """Create from dictionary."""
        data = data.copy()
        if "image_size" in data:
            data["image_size"] = tuple(data["image_size"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PreprocessingPipeline:
    """Preprocessing pipeline configuration."""

    # Identity
    pipeline_id: str = ""
    name: str = ""
    description: str = ""

    # Steps
    steps: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration
    normalize: bool = True
    normalize_method: str = "minmax"  # "minmax", "zscore", "custom"
    resize: bool = True
    target_size: Tuple[int, int] = (256, 256)

    # Augmentation
    augment: bool = True
    augmentation_config: Dict[str, Any] = field(default_factory=dict)

    # For wind data
    max_wind_speed: float = 80.0
    wind_normalization: str = "physical"  # "physical", "statistical"

    # For satellite data
    satellite_channels: List[str] = field(default_factory=list)
    brightness_scaling: bool = True

    # For sequence data
    frame_stride: int = 1
    temporal_augmentation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["target_size"] = list(self.target_size)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingPipeline":
        """Create from dictionary."""
        data = data.copy()
        if "target_size" in data:
            data["target_size"] = tuple(data["target_size"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class WorldsphereDataset(Dataset):
    """
    PyTorch Dataset for Worldsphere data.

    Supports:
    - Paired data (satellite -> wind field)
    - Unpaired data (for CycleGAN)
    - Sequence data (for diffusion models)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        data_type: str = "paired",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        sequence_length: int = 1,
        return_dict: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to data directory or file
            data_type: Type of data (paired, unpaired, sequence)
            transform: Transform for input data
            target_transform: Transform for target data
            sequence_length: Length of sequences for sequence data
            return_dict: Return dictionary instead of tuple
        """
        self.data_path = Path(data_path)
        self.data_type = data_type
        self.transform = transform
        self.target_transform = target_transform
        self.sequence_length = sequence_length
        self.return_dict = return_dict

        # Load data index
        self.samples = self._load_index()

    def _load_index(self) -> List[Dict[str, Any]]:
        """Load data index."""
        index_path = self.data_path / "index.json"

        if index_path.exists():
            with open(index_path, "r") as f:
                return json.load(f)

        # Auto-generate index from directory structure
        samples = []

        if self.data_type == "paired":
            input_dir = self.data_path / "input"
            target_dir = self.data_path / "target"

            if input_dir.exists() and target_dir.exists():
                for input_file in sorted(input_dir.glob("*.npy")):
                    target_file = target_dir / input_file.name
                    if target_file.exists():
                        samples.append({
                            "input": str(input_file),
                            "target": str(target_file),
                        })

        elif self.data_type == "sequence":
            sequences_dir = self.data_path / "sequences"

            if sequences_dir.exists():
                for seq_dir in sorted(sequences_dir.iterdir()):
                    if seq_dir.is_dir():
                        frames = sorted(seq_dir.glob("*.npy"))
                        if len(frames) >= self.sequence_length:
                            samples.append({
                                "frames": [str(f) for f in frames],
                            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        sample = self.samples[idx]

        if self.data_type == "paired":
            input_data = np.load(sample["input"])
            target_data = np.load(sample["target"])

            input_tensor = torch.from_numpy(input_data).float()
            target_tensor = torch.from_numpy(target_data).float()

            if self.transform:
                input_tensor = self.transform(input_tensor)
            if self.target_transform:
                target_tensor = self.target_transform(target_tensor)

            if self.return_dict:
                return {"input": input_tensor, "target": target_tensor}
            return input_tensor, target_tensor

        elif self.data_type == "sequence":
            frames = []
            for frame_path in sample["frames"][:self.sequence_length]:
                frame = np.load(frame_path)
                frames.append(torch.from_numpy(frame).float())

            frames_tensor = torch.stack(frames)

            if self.transform:
                frames_tensor = self.transform(frames_tensor)

            # First frame(s) as condition
            condition = frames_tensor[:1]  # First frame
            target = frames_tensor

            if self.return_dict:
                return {"frames": target, "condition": condition}
            return target, condition

        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")


class WorldsphereDataManager:
    """
    Central data manager for Worldsphere.

    Manages:
    - Dataset registration
    - Preprocessing pipelines
    - Data loading
    - Quality checks
    """

    def __init__(self, base_dir: Union[str, Path] = "./worldsphere_data"):
        """
        Initialize data manager.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.datasets_dir = self.base_dir / "datasets"
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.pipelines_dir = self.base_dir / "pipelines"

        self.datasets_dir.mkdir(exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.pipelines_dir.mkdir(exist_ok=True)

        # Load registries
        self.datasets: Dict[str, DatasetConfig] = {}
        self.pipelines: Dict[str, PreprocessingPipeline] = {}
        self._load_registries()

    def _load_registries(self) -> None:
        """Load dataset and pipeline registries."""
        # Load datasets
        datasets_file = self.base_dir / "datasets_registry.json"
        if datasets_file.exists():
            try:
                with open(datasets_file, "r") as f:
                    data = json.load(f)
                for ds_id, ds_data in data.items():
                    self.datasets[ds_id] = DatasetConfig.from_dict(ds_data)
            except Exception as e:
                logger.warning(f"Failed to load datasets registry: {e}")

        # Load pipelines
        pipelines_file = self.base_dir / "pipelines_registry.json"
        if pipelines_file.exists():
            try:
                with open(pipelines_file, "r") as f:
                    data = json.load(f)
                for pl_id, pl_data in data.items():
                    self.pipelines[pl_id] = PreprocessingPipeline.from_dict(pl_data)
            except Exception as e:
                logger.warning(f"Failed to load pipelines registry: {e}")

    def _save_registries(self) -> None:
        """Save registries to disk."""
        # Save datasets
        datasets_file = self.base_dir / "datasets_registry.json"
        with open(datasets_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.datasets.items()}, f, indent=2)

        # Save pipelines
        pipelines_file = self.base_dir / "pipelines_registry.json"
        with open(pipelines_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.pipelines.items()}, f, indent=2)

    def register_dataset(
        self,
        name: str,
        data_path: Union[str, Path],
        config: Optional[DatasetConfig] = None,
        compute_statistics: bool = True,
    ) -> str:
        """
        Register a new dataset.

        Args:
            name: Dataset name
            data_path: Path to data
            config: Optional dataset configuration
            compute_statistics: Whether to compute data statistics

        Returns:
            Dataset ID
        """
        if config is None:
            config = DatasetConfig()

        # Generate ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.dataset_id = f"{name}_{timestamp}"
        config.name = name
        config.data_path = str(data_path)

        # Set timestamps
        now = datetime.now().isoformat()
        config.created_at = now
        config.updated_at = now

        # Copy data to managed location
        data_path = Path(data_path)
        target_dir = self.datasets_dir / config.dataset_id

        if data_path.is_dir():
            shutil.copytree(data_path, target_dir)
        else:
            target_dir.mkdir(exist_ok=True)
            shutil.copy(data_path, target_dir)

        config.data_path = str(target_dir)

        # Compute statistics
        if compute_statistics:
            config.statistics = self._compute_statistics(target_dir, config.data_type)
            config.num_samples = config.statistics.get("num_samples", 0)

        # Register
        self.datasets[config.dataset_id] = config
        self._save_registries()

        logger.info(f"Registered dataset: {config.dataset_id}")
        return config.dataset_id

    def _compute_statistics(
        self,
        data_path: Path,
        data_type: str = "paired",
    ) -> Dict[str, Any]:
        """Compute data statistics."""
        stats = {"num_samples": 0}

        if data_type == "paired":
            input_dir = data_path / "input"
            if input_dir.exists():
                files = list(input_dir.glob("*.npy"))
                stats["num_samples"] = len(files)

                if files:
                    sample = np.load(files[0])
                    stats["shape"] = list(sample.shape)
                    stats["dtype"] = str(sample.dtype)

                    # Compute statistics from sample of files
                    sample_files = files[:min(100, len(files))]
                    all_data = [np.load(f) for f in sample_files]
                    stacked = np.stack(all_data)

                    stats["mean"] = float(stacked.mean())
                    stats["std"] = float(stacked.std())
                    stats["min"] = float(stacked.min())
                    stats["max"] = float(stacked.max())

        elif data_type == "sequence":
            sequences_dir = data_path / "sequences"
            if sequences_dir.exists():
                stats["num_samples"] = len(list(sequences_dir.iterdir()))

        return stats

    def register_pipeline(
        self,
        name: str,
        config: Optional[PreprocessingPipeline] = None,
    ) -> str:
        """
        Register a preprocessing pipeline.

        Args:
            name: Pipeline name
            config: Pipeline configuration

        Returns:
            Pipeline ID
        """
        if config is None:
            config = PreprocessingPipeline()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.pipeline_id = f"{name}_{timestamp}"
        config.name = name

        self.pipelines[config.pipeline_id] = config
        self._save_registries()

        logger.info(f"Registered pipeline: {config.pipeline_id}")
        return config.pipeline_id

    def get_dataset(self, dataset_id: str) -> Optional[DatasetConfig]:
        """Get dataset configuration."""
        return self.datasets.get(dataset_id)

    def get_pipeline(self, pipeline_id: str) -> Optional[PreprocessingPipeline]:
        """Get pipeline configuration."""
        return self.pipelines.get(pipeline_id)

    def list_datasets(
        self,
        data_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[DatasetConfig]:
        """List registered datasets."""
        datasets = list(self.datasets.values())

        if data_type:
            datasets = [d for d in datasets if d.data_type == data_type]
        if tags:
            datasets = [d for d in datasets if any(t in d.tags for t in tags)]

        return datasets

    def list_pipelines(self) -> List[PreprocessingPipeline]:
        """List registered pipelines."""
        return list(self.pipelines.values())

    def create_dataloader(
        self,
        dataset_id: str,
        split: str = "train",
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pipeline_id: Optional[str] = None,
    ) -> DataLoader:
        """
        Create a DataLoader for a dataset.

        Args:
            dataset_id: Dataset ID
            split: Data split (train, val, test)
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of data loading workers
            pipeline_id: Optional preprocessing pipeline

        Returns:
            PyTorch DataLoader
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_id}")

        config = self.datasets[dataset_id]

        # Get data path for split
        data_path = Path(config.data_path)
        if split == "train" and config.train_path:
            data_path = Path(config.train_path)
        elif split == "val" and config.val_path:
            data_path = Path(config.val_path)
        elif split == "test" and config.test_path:
            data_path = Path(config.test_path)

        # Create transform from pipeline
        transform = None
        if pipeline_id and pipeline_id in self.pipelines:
            pipeline = self.pipelines[pipeline_id]
            transform = self._create_transform(pipeline)

        # Create dataset
        dataset = WorldsphereDataset(
            data_path=data_path,
            data_type=config.data_type,
            transform=transform,
            sequence_length=config.sequence_length,
        )

        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def _create_transform(
        self,
        pipeline: PreprocessingPipeline,
    ) -> Callable:
        """Create transform function from pipeline."""

        def transform(x: torch.Tensor) -> torch.Tensor:
            # Resize
            if pipeline.resize:
                h, w = pipeline.target_size
                if x.dim() == 3:  # C, H, W
                    x = torch.nn.functional.interpolate(
                        x.unsqueeze(0), size=(h, w), mode="bilinear"
                    ).squeeze(0)
                elif x.dim() == 4:  # T, C, H, W
                    x = torch.nn.functional.interpolate(
                        x, size=(h, w), mode="bilinear"
                    )

            # Normalize
            if pipeline.normalize:
                if pipeline.normalize_method == "minmax":
                    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                    x = x * 2 - 1  # Scale to [-1, 1]
                elif pipeline.normalize_method == "zscore":
                    x = (x - x.mean()) / (x.std() + 1e-8)

            return x

        return transform

    def delete_dataset(self, dataset_id: str, delete_files: bool = True) -> bool:
        """Delete a dataset."""
        if dataset_id not in self.datasets:
            return False

        config = self.datasets[dataset_id]

        if delete_files and config.data_path:
            data_path = Path(config.data_path)
            if data_path.exists():
                shutil.rmtree(data_path)

        del self.datasets[dataset_id]
        self._save_registries()

        logger.info(f"Deleted dataset: {dataset_id}")
        return True

    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""
        if pipeline_id not in self.pipelines:
            return False

        del self.pipelines[pipeline_id]
        self._save_registries()

        logger.info(f"Deleted pipeline: {pipeline_id}")
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get data manager summary."""
        datasets = list(self.datasets.values())

        total_samples = sum(d.num_samples for d in datasets)
        type_counts = {}
        for d in datasets:
            type_counts[d.data_type] = type_counts.get(d.data_type, 0) + 1

        return {
            "total_datasets": len(datasets),
            "total_samples": total_samples,
            "by_type": type_counts,
            "total_pipelines": len(self.pipelines),
            "storage_path": str(self.base_dir),
        }

    def create_sample_dataset(
        self,
        name: str = "sample_hurricane",
        num_samples: int = 100,
        image_size: Tuple[int, int] = (256, 256),
        data_type: str = "paired",
    ) -> str:
        """
        Create a sample dataset for testing.

        Args:
            name: Dataset name
            num_samples: Number of samples to generate
            image_size: Image dimensions
            data_type: Type of data

        Returns:
            Dataset ID
        """
        sample_dir = self.datasets_dir / f"{name}_sample"
        sample_dir.mkdir(exist_ok=True)

        if data_type == "paired":
            input_dir = sample_dir / "input"
            target_dir = sample_dir / "target"
            input_dir.mkdir(exist_ok=True)
            target_dir.mkdir(exist_ok=True)

            for i in range(num_samples):
                # Generate synthetic satellite image (3 channels)
                satellite = np.random.randn(3, *image_size).astype(np.float32)
                # Generate synthetic wind field (2 channels: u, v)
                wind = np.random.randn(2, *image_size).astype(np.float32) * 40

                np.save(input_dir / f"sample_{i:05d}.npy", satellite)
                np.save(target_dir / f"sample_{i:05d}.npy", wind)

        elif data_type == "sequence":
            sequences_dir = sample_dir / "sequences"
            sequences_dir.mkdir(exist_ok=True)

            for i in range(num_samples):
                seq_dir = sequences_dir / f"seq_{i:05d}"
                seq_dir.mkdir(exist_ok=True)

                for t in range(25):
                    frame = np.random.randn(3, *image_size).astype(np.float32)
                    np.save(seq_dir / f"frame_{t:03d}.npy", frame)

        config = DatasetConfig(
            name=name,
            data_type=data_type,
            num_samples=num_samples,
            image_size=image_size,
            input_channels=3,
            output_channels=2 if data_type == "paired" else 3,
            input_variables=["satellite_ch1", "satellite_ch2", "satellite_ch3"],
            output_variables=["u_wind", "v_wind"] if data_type == "paired" else ["brightness"],
        )

        return self.register_dataset(name, sample_dir, config, compute_statistics=True)
