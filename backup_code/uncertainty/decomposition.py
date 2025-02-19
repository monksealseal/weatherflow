
import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

class UncertaintyDecomposition:
    """Uncertainty decomposition methods from our experiments."""
    
    def decompose_uncertainty(
        self,
        predictions: torch.Tensor,  # Shape: [n_models, n_samples, ...]
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Decompose prediction uncertainty into components."""
        # Aleatoric uncertainty (data uncertainty)
        aleatoric = predictions.var(dim=0)  # Variance across models
        
        # Epistemic uncertainty (model uncertainty)
        mean_pred = predictions.mean(dim=0)
        epistemic = torch.mean((predictions - mean_pred.unsqueeze(0))**2, dim=0)
        
        # Total uncertainty
        total = aleatoric + epistemic
        
        # Uncertainty ratio
        uncertainty_ratio = epistemic / (total + 1e-8)
        
        return {
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'total': total,
            'uncertainty_ratio': uncertainty_ratio
        }
    
    def plot_uncertainty_decomposition(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        variable_idx: int = 0
    ):
        """Visualize uncertainty decomposition."""
        components = self.decompose_uncertainty(predictions, targets)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total uncertainty
        im0 = axs[0, 0].imshow(
            components['total'][variable_idx].numpy(),
            cmap='viridis'
        )
        axs[0, 0].set_title('Total Uncertainty')
        plt.colorbar(im0, ax=axs[0, 0])
        
        # Aleatoric uncertainty
        im1 = axs[0, 1].imshow(
            components['aleatoric'][variable_idx].numpy(),
            cmap='viridis'
        )
        axs[0, 1].set_title('Aleatoric Uncertainty')
        plt.colorbar(im1, ax=axs[0, 1])
        
        # Epistemic uncertainty
        im2 = axs[1, 0].imshow(
            components['epistemic'][variable_idx].numpy(),
            cmap='viridis'
        )
        axs[1, 0].set_title('Epistemic Uncertainty')
        plt.colorbar(im2, ax=axs[1, 0])
        
        # Uncertainty ratio
        im3 = axs[1, 1].imshow(
            components['uncertainty_ratio'][variable_idx].numpy(),
            cmap='RdYlBu'
        )
        axs[1, 1].set_title('Epistemic/Total Ratio')
        plt.colorbar(im3, ax=axs[1, 1])
        
        plt.tight_layout()
        return fig, axs
