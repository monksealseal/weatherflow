"""
FlowAtmosphere - A Next-Generation Weather Foundation Model

A transformer-based foundation model for atmospheric science built on WeatherFlow's
flow matching core. Enables zero/few-shot learning for diverse downstream tasks
beyond simple forecasting.
"""

from .models.flow_former import FlowFormer, HierarchicalSphericalTransformer
from .models.flow_atmosphere import FlowAtmosphere
from .training.distributed_trainer import DistributedFlowTrainer
from .objectives.pretraining import MultiObjectivePretraining
from .adaptation.peft import LoRAAdapter, PEFTEngine
from .data.massive_pipeline import MassiveDataPipeline

__version__ = "0.1.0"

__all__ = [
    "FlowFormer",
    "HierarchicalSphericalTransformer",
    "FlowAtmosphere",
    "DistributedFlowTrainer",
    "MultiObjectivePretraining",
    "LoRAAdapter",
    "PEFTEngine",
    "MassiveDataPipeline",
]
