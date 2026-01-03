
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple

class FlowVisualizer:
    """Visualization tools for flow matching models."""
    
    def __init__(self, update_freq: int = 100):
        self.update_freq = update_freq
    
    def visualize_flow(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
        n_steps: int = 10,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """Visualize flow evolution.

        Args:
            model: Flow matching model
            x0: Initial state
            x1: Target state
            n_steps: Number of steps to visualize
            save_path: Path to save figure (if None, figure is not saved)
            show: Whether to display the plot (default: False for headless compatibility)
        """
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            times = torch.linspace(0, 1, n_steps, device=device)
            states = [x0.cpu()]

            for t in times[1:]:
                t_batch = t.expand(x0.size(0))
                v_t = model(states[-1].to(device), t_batch)
                next_state = states[-1].to(device) + v_t * (1/n_steps)
                states.append(next_state.cpu())

        # Plot evolution
        fig, axes = plt.subplots(2, n_steps//2, figsize=(20, 8))
        axes = axes.flatten()

        for i, state in enumerate(states):
            axes[i].imshow(state[0, 0].numpy(), cmap='RdBu_r')
            axes[i].set_title(f't = {times[i]:.2f}')
            axes[i].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
