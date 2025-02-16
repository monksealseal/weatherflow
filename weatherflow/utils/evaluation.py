
import torch
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
import matplotlib.pyplot as plt
from ..physics import AtmosphericPhysics

class WeatherMetrics:
    """Comprehensive weather-specific evaluation metrics from our experiments."""
    
    def __init__(self):
        self.physics = AtmosphericPhysics()
    
    def calculate_all_metrics(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        pressure_levels: Optional[np.ndarray] = None,
        lat_grid: Optional[np.ndarray] = None,
        lon_grid: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive set of evaluation metrics."""
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        metrics = {}
        
        # Basic error metrics
        metrics.update(self.calculate_error_metrics(predictions, targets))
        
        # Pattern correlation
        metrics.update(self.calculate_pattern_correlation(predictions, targets))
        
        # Scale-dependent metrics
        metrics.update(self.calculate_scale_dependent_scores(predictions, targets))
        
        # Physics-based metrics
        if all(x is not None for x in [pressure_levels, lat_grid, lon_grid]):
            metrics.update(self.calculate_physics_metrics(
                predictions, targets, pressure_levels, lat_grid, lon_grid
            ))
        
        return metrics
    
    def calculate_error_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic error metrics."""
        # Mean metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Normalized metrics
        nrmse = rmse / np.std(targets)
        
        # Bias and variance
        bias = np.mean(predictions - targets)
        error_variance = np.var(predictions - targets)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'nrmse': nrmse,
            'bias': bias,
            'error_variance': error_variance
        }
    
    def calculate_pattern_correlation(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        spatial_dims: Tuple[int, int] = (-2, -1)
    ) -> Dict[str, float]:
        """Calculate pattern correlation metrics."""
        # Reshape for spatial correlations
        pred_flat = predictions.reshape(*predictions.shape[:-2], -1)
        target_flat = targets.reshape(*targets.shape[:-2], -1)
        
        # Pattern correlation
        pattern_corr = np.mean([
            stats.pearsonr(p, t)[0]
            for p, t in zip(pred_flat, target_flat)
        ])
        
        # Anomaly correlation coefficient (ACC)
        pred_anom = predictions - np.mean(predictions, axis=spatial_dims, keepdims=True)
        target_anom = targets - np.mean(targets, axis=spatial_dims, keepdims=True)
        
        acc = np.mean([
            stats.pearsonr(p.flatten(), t.flatten())[0]
            for p, t in zip(pred_anom, target_anom)
        ])
        
        return {
            'pattern_correlation': pattern_corr,
            'acc': acc
        }
    
    def calculate_scale_dependent_scores(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        scales: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, float]:
        """Calculate error metrics at different spatial scales."""
        metrics = {}
        
        for scale in scales:
            if scale > 1:
                # Downsample both predictions and targets
                pred_coarse = self._downsample(predictions, scale)
                target_coarse = self._downsample(targets, scale)
                
                # Calculate metrics at this scale
                mse = np.mean((pred_coarse - target_coarse) ** 2)
                metrics[f'mse_scale_{scale}'] = mse
                
                # Calculate variance ratio at this scale
                pred_var = np.var(pred_coarse)
                target_var = np.var(target_coarse)
                metrics[f'variance_ratio_scale_{scale}'] = pred_var / target_var
        
        return metrics
    
    def calculate_physics_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        pressure_levels: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray
    ) -> Dict[str, float]:
        """Calculate physics-based evaluation metrics."""
        metrics = {}
        
        # Static stability error
        if predictions.shape[1] > 1:  # Multiple pressure levels
            pred_stability = self.physics.static_stability(
                predictions[..., 0], pressure_levels
            )
            target_stability = self.physics.static_stability(
                targets[..., 0], pressure_levels
            )
            metrics['stability_mse'] = np.mean(
                (pred_stability - target_stability) ** 2
            )
        
        # Thermal wind balance error
        if predictions.shape[-1] >= 3:  # Has u, v, T components
            pred_thermal = self.physics.thermal_wind_balance(
                predictions, pressure_levels, lat_grid
            )
            target_thermal = self.physics.thermal_wind_balance(
                targets, pressure_levels, lat_grid
            )
            metrics['thermal_wind_mse'] = np.mean(
                (pred_thermal - target_thermal) ** 2
            )
        
        # Conservation metrics
        conservation_metrics = self.calculate_conservation_metrics(
            predictions, targets
        )
        metrics.update(conservation_metrics)
        
        return metrics
    
    def calculate_conservation_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate conservation law violations."""
        metrics = {}
        
        # Mass conservation (if we have wind components)
        if predictions.shape[-1] >= 2:
            div_pred = np.gradient(predictions[..., 0], axis=-1) +                       np.gradient(predictions[..., 1], axis=-2)
            div_target = np.gradient(targets[..., 0], axis=-1) +                         np.gradient(targets[..., 1], axis=-2)
            
            metrics['mass_conservation_error'] = np.mean(
                np.abs(div_pred) - np.abs(div_target)
            )
        
        # Energy conservation
        energy_pred = np.sum(predictions ** 2, axis=(-3, -2, -1))
        energy_target = np.sum(targets ** 2, axis=(-3, -2, -1))
        metrics['energy_conservation_error'] = np.mean(
            np.abs(energy_pred - energy_target)
        )
        
        return metrics
    
    def _downsample(
        self,
        x: np.ndarray,
        scale: int
    ) -> np.ndarray:
        """Downsample tensor by averaging over blocks."""
        if scale == 1:
            return x
        
        # Ensure dimensions are divisible by scale
        *leading, h, w = x.shape
        h = (h // scale) * scale
        w = (w // scale) * scale
        x = x[..., :h, :w]
        
        # Reshape and mean
        shape = (*leading, h // scale, scale, w // scale, scale)
        return x.reshape(shape).mean(axis=(-1, -3))

class WeatherEvaluator:
    """Comprehensive evaluation tools for weather predictions."""
    
    def __init__(self):
        self.metrics = WeatherMetrics()
    
    def evaluate_prediction(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        pressure_levels: Optional[np.ndarray] = None,
        lat_grid: Optional[np.ndarray] = None,
        lon_grid: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model predictions comprehensively."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        return self.metrics.calculate_all_metrics(
            predictions,
            targets,
            pressure_levels,
            lat_grid,
            lon_grid
        )
    
    def evaluate_ensemble(
        self,
        predictions: np.ndarray,  # Shape: [n_models, ...]
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate ensemble predictions."""
        results = {}
        
        # Individual model metrics
        individual_metrics = []
        for i, pred in enumerate(predictions):
            metrics = self.metrics.calculate_all_metrics(pred, targets)
            individual_metrics.append(metrics)
        results['individual'] = individual_metrics
        
        # Ensemble mean metrics
        if weights is None:
            weights = np.ones(len(predictions)) / len(predictions)
        ensemble_mean = np.average(predictions, axis=0, weights=weights)
        results['ensemble'] = self.metrics.calculate_all_metrics(
            ensemble_mean, targets
        )
        
        # Ensemble spread metrics
        ensemble_std = np.std(predictions, axis=0)
        spread_skill = np.corrcoef(
            ensemble_std.flatten(),
            np.abs(ensemble_mean - targets).flatten()
        )[0, 1]
        results['spread_skill'] = spread_skill
        
        return results
