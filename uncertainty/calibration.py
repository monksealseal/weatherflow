
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.stats import norm
import seaborn as sns

class CalibrationAnalyzer:
    """Advanced calibration analysis from our experiments."""
    
    def analyze_calibration(
        self,
        predictions: torch.Tensor,  # Shape: [n_models, n_samples, ...]
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Analyze prediction calibration with multiple metrics."""
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Compute z-scores
        z_scores = (targets - mean_pred) / (std_pred + 1e-8)
        
        # Expected vs actual coverage
        coverages = []
        expected_coverages = np.linspace(0.1, 0.9, 9)
        
        for coverage in expected_coverages:
            threshold = norm.ppf((1 + coverage) / 2)
            actual_coverage = (torch.abs(z_scores) <= threshold).float().mean()
            coverages.append(actual_coverage.item())
        
        # Compute calibration metrics
        calibration_error = np.mean((np.array(coverages) - expected_coverages) ** 2)
        
        # Reliability diagram data
        hist, bin_edges = np.histogram(z_scores.numpy(), bins=n_bins, range=(-3, 3))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return {
            'calibration_error': calibration_error,
            'expected_coverages': expected_coverages,
            'actual_coverages': coverages,
            'z_score_hist': hist,
            'z_score_bins': bin_centers
        }
    
    def plot_calibration_diagnostics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create comprehensive calibration diagnostic plots."""
        metrics = self.analyze_calibration(predictions, targets)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Coverage plot
        axs[0, 0].plot(
            metrics['expected_coverages'],
            metrics['actual_coverages'],
            'b-', label='Actual'
        )
        axs[0, 0].plot([0, 1], [0, 1], 'k--', label='Ideal')
        axs[0, 0].set_xlabel('Expected Coverage')
        axs[0, 0].set_ylabel('Actual Coverage')
        axs[0, 0].set_title('Prediction Coverage Analysis')
        axs[0, 0].legend()
        
        # Z-score distribution
        sns.histplot(
            data=predictions.std(dim=0).numpy().flatten(),
            ax=axs[0, 1],
            bins=30
        )
        axs[0, 1].set_xlabel('Prediction Uncertainty (Std)')
        axs[0, 1].set_title('Uncertainty Distribution')
        
        # Reliability diagram
        axs[1, 0].bar(
            metrics['z_score_bins'],
            metrics['z_score_hist'] / metrics['z_score_hist'].sum(),
            width=0.2
        )
        x = np.linspace(-3, 3, 100)
        axs[1, 0].plot(x, norm.pdf(x) * 0.2, 'r-', label='Standard Normal')
        axs[1, 0].set_xlabel('Z-Score')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].set_title('Reliability Diagram')
        axs[1, 0].legend()
        
        # Error vs Uncertainty
        error = torch.abs(predictions.mean(dim=0) - targets)
        uncertainty = predictions.std(dim=0)
        axs[1, 1].scatter(
            uncertainty.numpy().flatten(),
            error.numpy().flatten(),
            alpha=0.1
        )
        axs[1, 1].set_xlabel('Predicted Uncertainty')
        axs[1, 1].set_ylabel('Absolute Error')
        axs[1, 1].set_title('Error vs Uncertainty')
        
        plt.tight_layout()
        return fig, axs
