import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return torch.mean(torch.abs(pred - target))


def energy_ratio(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ratio of L2 energy between prediction and target."""
    pred_energy = torch.sum(pred**2)
    target_energy = torch.sum(target**2).clamp(min=1e-8)
    return pred_energy / target_energy


def persistence_rmse(baseline: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE of a persistence (x0) baseline against the target."""
    return rmse(baseline, target)
