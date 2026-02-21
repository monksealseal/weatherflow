"""
FLUX.1-dev LoRA Fine-Tuning Trainer (library integration)

Thin wrapper that re-exports the trainer and config from the examples module
so they can be imported via ``from weatherflow.training import FluxFineTuneTrainer``.
"""

import sys
from pathlib import Path

# Ensure the examples directory is importable
_examples_root = Path(__file__).resolve().parent.parent.parent / "examples" / "flux_img2img"
if str(_examples_root.parent) not in sys.path:
    sys.path.insert(0, str(_examples_root.parent))

from flux_img2img.finetune_flux_img2img import (  # noqa: E402
    FluxFineTuneConfig,
    FluxImg2ImgDataset,
    FluxImg2ImgTrainer as FluxFineTuneTrainer,
)

__all__ = [
    "FluxFineTuneConfig",
    "FluxFineTuneTrainer",
    "FluxImg2ImgDataset",
]
