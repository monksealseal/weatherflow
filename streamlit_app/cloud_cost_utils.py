"""
Cloud Cost Estimation Utilities

Provides cost estimates for training on Google Cloud Platform (GCP).
Helps users understand costs BEFORE launching training jobs.
"""

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GPUConfig:
    """GPU configuration and pricing."""
    name: str
    memory_gb: int
    hourly_cost: float  # USD per hour
    tflops: float  # Training TFLOPS
    recommended_batch_size: int


# GCP GPU pricing (approximate, as of late 2024)
GCP_GPU_CONFIGS = {
    "T4": GPUConfig("NVIDIA T4", 16, 0.35, 8.1, 32),
    "A100-40GB": GPUConfig("NVIDIA A100 40GB", 40, 3.67, 156, 128),
    "A100-80GB": GPUConfig("NVIDIA A100 80GB", 80, 4.89, 156, 256),
    "V100": GPUConfig("NVIDIA V100", 16, 2.48, 14, 64),
    "L4": GPUConfig("NVIDIA L4", 24, 0.73, 30, 64),
    "H100": GPUConfig("NVIDIA H100", 80, 10.80, 378, 512),
}

# CPU pricing
GCP_CPU_HOURLY = 0.03  # per vCPU
GCP_MEMORY_HOURLY = 0.004  # per GB


def estimate_training_time(
    model_params: int,
    dataset_size: int,
    batch_size: int,
    epochs: int,
    gpu_tflops: float,
) -> float:
    """
    Estimate training time in hours.

    Args:
        model_params: Number of model parameters
        dataset_size: Number of training samples
        batch_size: Training batch size
        epochs: Number of training epochs
        gpu_tflops: GPU compute power in TFLOPS

    Returns:
        Estimated hours of training time
    """
    # Rough estimate: 6 FLOPs per parameter per sample for forward+backward
    flops_per_sample = model_params * 6
    total_samples = dataset_size * epochs
    batches = total_samples / batch_size

    # Include communication overhead (~20%)
    effective_tflops = gpu_tflops * 0.8

    total_flops = flops_per_sample * total_samples
    seconds = total_flops / (effective_tflops * 1e12)

    # Add overhead for data loading, checkpointing, etc (~30%)
    seconds *= 1.3

    return seconds / 3600


def estimate_memory_requirements(
    model_params: int,
    batch_size: int,
    input_shape: Tuple[int, ...],
) -> float:
    """
    Estimate GPU memory requirements in GB.

    Args:
        model_params: Number of model parameters
        batch_size: Training batch size
        input_shape: Shape of input tensor (C, H, W)

    Returns:
        Estimated GPU memory in GB
    """
    # Model weights (fp32 = 4 bytes)
    weights_gb = model_params * 4 / (1024**3)

    # Optimizer states (Adam has 2 states per param)
    optimizer_gb = weights_gb * 2

    # Gradients
    gradients_gb = weights_gb

    # Activations (rough estimate)
    c, h, w = input_shape
    activation_elements = batch_size * c * h * w * 10  # Rough multiplier
    activations_gb = activation_elements * 4 / (1024**3)

    # Total with 20% buffer
    total = (weights_gb + optimizer_gb + gradients_gb + activations_gb) * 1.2

    return total


def recommend_gpu(memory_required_gb: float) -> str:
    """Recommend appropriate GPU based on memory requirements."""
    for name, config in sorted(GCP_GPU_CONFIGS.items(), key=lambda x: x[1].hourly_cost):
        if config.memory_gb >= memory_required_gb * 1.2:  # 20% headroom
            return name
    return "H100"  # Largest available


def estimate_training_cost(
    model_params: int,
    dataset_size: int,
    batch_size: int,
    epochs: int,
    gpu_type: str,
    num_gpus: int = 1,
) -> Dict:
    """
    Estimate total training cost on GCP.

    Args:
        model_params: Number of model parameters
        dataset_size: Number of training samples
        batch_size: Training batch size per GPU
        epochs: Number of training epochs
        gpu_type: GPU type (e.g., "A100-40GB")
        num_gpus: Number of GPUs

    Returns:
        Dictionary with cost breakdown
    """
    gpu_config = GCP_GPU_CONFIGS.get(gpu_type, GCP_GPU_CONFIGS["T4"])

    # Adjust batch size for multi-GPU
    effective_batch = batch_size * num_gpus

    # Estimate training time
    training_hours = estimate_training_time(
        model_params,
        dataset_size,
        effective_batch,
        epochs,
        gpu_config.tflops * num_gpus
    )

    # GPU cost
    gpu_cost = training_hours * gpu_config.hourly_cost * num_gpus

    # CPU/memory cost (rough estimate: 8 vCPUs, 32GB per GPU)
    cpu_cost = training_hours * (8 * GCP_CPU_HOURLY * num_gpus)
    memory_cost = training_hours * (32 * GCP_MEMORY_HOURLY * num_gpus)

    # Storage cost (minimal for short runs)
    storage_cost = training_hours * 0.05  # ~$0.05/hour for SSD

    # Total
    total_cost = gpu_cost + cpu_cost + memory_cost + storage_cost

    return {
        "gpu_type": gpu_config.name,
        "num_gpus": num_gpus,
        "estimated_hours": training_hours,
        "gpu_cost": gpu_cost,
        "cpu_cost": cpu_cost,
        "memory_cost": memory_cost,
        "storage_cost": storage_cost,
        "total_cost": total_cost,
        "cost_per_epoch": total_cost / max(epochs, 1),
        "recommended_batch_size": gpu_config.recommended_batch_size,
    }


def format_cost_estimate(estimate: Dict) -> str:
    """Format cost estimate as human-readable string."""
    return f"""
**Estimated Training Cost (GCP)**

| Item | Cost |
|------|------|
| GPU ({estimate['gpu_type']} x{estimate['num_gpus']}) | ${estimate['gpu_cost']:.2f} |
| CPU | ${estimate['cpu_cost']:.2f} |
| Memory | ${estimate['memory_cost']:.2f} |
| Storage | ${estimate['storage_cost']:.2f} |
| **Total** | **${estimate['total_cost']:.2f}** |

**Time Estimate:** {estimate['estimated_hours']:.1f} hours
**Cost per Epoch:** ${estimate['cost_per_epoch']:.2f}
"""


def get_multi_node_config(
    model_params: int,
    target_hours: float = 24,
    epochs: int = 100,
    dataset_size: int = 10000,
) -> Dict:
    """
    Calculate optimal multi-node configuration to meet target training time.

    Args:
        model_params: Number of model parameters
        target_hours: Target training time in hours
        epochs: Number of training epochs
        dataset_size: Number of training samples

    Returns:
        Recommended configuration
    """
    best_config = None
    best_cost = float('inf')

    for gpu_type, gpu_config in GCP_GPU_CONFIGS.items():
        for num_gpus in [1, 2, 4, 8]:
            estimate = estimate_training_cost(
                model_params,
                dataset_size,
                gpu_config.recommended_batch_size,
                epochs,
                gpu_type,
                num_gpus
            )

            if estimate['estimated_hours'] <= target_hours and estimate['total_cost'] < best_cost:
                best_cost = estimate['total_cost']
                best_config = {
                    'gpu_type': gpu_type,
                    'num_gpus': num_gpus,
                    **estimate
                }

    if best_config is None:
        # If no config meets target, return fastest
        best_config = estimate_training_cost(
            model_params, dataset_size, 256, epochs, "H100", 8
        )
        best_config['gpu_type'] = "H100"
        best_config['num_gpus'] = 8
        best_config['note'] = "Target time may not be achievable with current GPUs"

    return best_config


# Quick reference table
QUICK_COST_REFERENCE = """
## Quick Cost Reference

| Training Job | Est. Time | Est. Cost |
|-------------|-----------|-----------|
| Small demo (1M params, 1000 samples, 10 epochs) | ~0.5 hours | ~$0.20 |
| Medium (10M params, 10K samples, 50 epochs) | ~4 hours | ~$15 |
| Large (100M params, 100K samples, 100 epochs) | ~48 hours | ~$400 |
| Production (1B params, 1M samples, 50 epochs) | ~1 week | ~$5,000+ |

*Estimates based on T4 GPU. A100 costs more but trains faster.*
"""
