from .flow_trainer import FlowTrainer, compute_flow_loss
from .metrics import energy_ratio, mae, persistence_rmse, rmse
from .utils import set_global_seed
from .cyclegan_trainer import CycleGANTrainer, CycleGANConfig, CycleGANMetrics
from .video_diffusion_trainer import (
    VideoDiffusionTrainer,
    VideoDiffusionConfig,
    VideoDiffusionMetrics,
)

__all__ = [
    # Flow matching
    'FlowTrainer',
    'compute_flow_loss',
    # CycleGAN
    'CycleGANTrainer',
    'CycleGANConfig',
    'CycleGANMetrics',
    # Video Diffusion
    'VideoDiffusionTrainer',
    'VideoDiffusionConfig',
    'VideoDiffusionMetrics',
    # Metrics
    'rmse',
    'mae',
    'energy_ratio',
    'persistence_rmse',
    'set_global_seed',
]

# FLUX fine-tuning (optional heavy dependency)
try:
    from .flux_finetune_trainer import (
        FluxFineTuneTrainer,
        FluxFineTuneConfig,
    )
    __all__.extend(['FluxFineTuneTrainer', 'FluxFineTuneConfig'])
except ImportError:
    pass
