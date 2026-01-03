"""
Multi-Objective Pre-Training for FlowAtmosphere

Implements novel pre-training objectives that teach fundamental atmospheric physics:
1. Masked Variable Modeling
2. Temporal Jigsaw Puzzle
3. Climate Invariance Learning
4. Flow Matching (base objective)
"""

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiObjectivePretraining(nn.Module):
    """
    Multi-task pre-training objectives for learning atmospheric physics.

    Combines multiple self-supervised objectives to learn rich, transferable
    representations of atmospheric dynamics.
    """

    def __init__(
        self,
        model: nn.Module,
        mask_ratio: float = 0.15,
        temporal_window: int = 8,
        climate_invariance_weight: float = 0.1,
        masked_modeling_weight: float = 0.5,
        temporal_jigsaw_weight: float = 0.3,
        flow_matching_weight: float = 1.0,
    ):
        """
        Args:
            model: FlowFormer or FlowAtmosphere model
            mask_ratio: Fraction of variables to mask
            temporal_window: Window size for temporal jigsaw
            climate_invariance_weight: Weight for climate invariance loss
            masked_modeling_weight: Weight for masked modeling loss
            temporal_jigsaw_weight: Weight for temporal jigsaw loss
            flow_matching_weight: Weight for flow matching loss
        """
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        self.temporal_window = temporal_window

        # Loss weights
        self.weights = {
            'flow_matching': flow_matching_weight,
            'masked_modeling': masked_modeling_weight,
            'temporal_jigsaw': temporal_jigsaw_weight,
            'climate_invariance': climate_invariance_weight,
        }

        # Prediction heads for auxiliary tasks
        d_model = model.d_model if hasattr(model, 'd_model') else 512

        # Masked variable prediction head
        self.mask_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128),  # Predict masked variables
        )

        # Temporal order prediction head
        self.order_predictor = nn.Sequential(
            nn.Linear(d_model * temporal_window, 512),
            nn.GELU(),
            nn.Linear(512, np.math.factorial(temporal_window)),  # All permutations
        )

        # Climate signal encoder
        self.climate_encoder = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
        )

    def forward(
        self,
        x_sequence: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all pre-training objectives.

        Args:
            x_sequence: Sequence of atmospheric states, shape [B, T, C, H, W]
            lat_grid: Latitude coordinates
            lon_grid: Longitude coordinates

        Returns:
            Dictionary of losses
        """
        losses = {}

        # 1. Flow Matching Loss (base objective)
        flow_loss = self.flow_matching_loss(x_sequence, lat_grid, lon_grid)
        losses['flow_matching'] = flow_loss

        # 2. Masked Variable Modeling
        masked_loss = self.masked_variable_modeling(x_sequence, lat_grid, lon_grid)
        losses['masked_modeling'] = masked_loss

        # 3. Temporal Jigsaw Puzzle
        temporal_loss = self.temporal_jigsaw(x_sequence, lat_grid, lon_grid)
        losses['temporal_jigsaw'] = temporal_loss

        # 4. Climate Invariance
        climate_loss = self.climate_invariance_loss(x_sequence, lat_grid, lon_grid)
        losses['climate_invariance'] = climate_loss

        # Compute weighted total loss
        total_loss = sum(
            self.weights[key] * value
            for key, value in losses.items()
            if key in self.weights
        )
        losses['total'] = total_loss

        return losses

    def flow_matching_loss(
        self,
        x_sequence: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard flow matching loss.

        Learn to predict velocity field between consecutive states.
        """
        B, T, C, H, W = x_sequence.shape

        # Sample pairs (x0, x1) from sequence
        idx = torch.randint(0, T - 1, (B,))
        x0 = x_sequence[torch.arange(B), idx]
        x1 = x_sequence[torch.arange(B), idx + 1]

        # Sample time
        t = torch.rand(B, device=x_sequence.device)

        # Interpolate
        x_t = t.view(-1, 1, 1, 1) * x1 + (1 - t.view(-1, 1, 1, 1)) * x0

        # Target velocity
        v_target = x1 - x0

        # Predict velocity
        if hasattr(self.model, 'flow_model'):
            # FlowAtmosphere
            v_pred = self.model.flow_model(x_t, t, lat_grid, lon_grid)
        else:
            # FlowFormer
            v_pred = self.model(x_t, t, lat_grid, lon_grid)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return loss

    def masked_variable_modeling(
        self,
        x_sequence: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Masked Variable Modeling (like BERT for weather).

        Randomly mask atmospheric variables and train model to reconstruct them.
        This forces the model to learn relationships between variables.
        """
        B, T, C, H, W = x_sequence.shape

        # Select random timestep
        idx = torch.randint(0, T, (B,))
        x = x_sequence[torch.arange(B), idx]

        # Create mask (mask entire variables, not individual pixels)
        num_masked = int(C * self.mask_ratio)
        mask_indices = torch.randperm(C)[:num_masked]

        # Create masked input
        x_masked = x.clone()
        x_masked[:, mask_indices] = 0.0  # Zero out masked variables

        # Add mask token embedding (learnable)
        # For simplicity, use zeros here
        mask_token = torch.zeros_like(x[:, :1])
        x_masked[:, mask_indices] = mask_token.expand(-1, num_masked, -1, -1)

        # Encode with model
        t = torch.zeros(B, device=x.device)

        if hasattr(self.model, 'flow_model'):
            features = self.model.flow_model.encoder(x_masked, lat_grid, lon_grid)
        else:
            features = self.model.encoder(x_masked, lat_grid, lon_grid)

        # features shape: [B, H, W, d_model]

        # Predict masked variables
        # Average over spatial dimensions for simplicity
        features_pooled = features.mean(dim=[1, 2])  # [B, d_model]

        # Predict masked values
        predicted = self.mask_predictor(features_pooled)  # [B, num_channels]

        # Ground truth (pooled)
        target = x.mean(dim=[-2, -1])  # [B, C]

        # Loss only on masked variables
        loss = F.mse_loss(predicted[:, mask_indices], target[:, mask_indices])

        return loss

    def temporal_jigsaw(
        self,
        x_sequence: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Temporal Jigsaw Puzzle.

        Shuffle sequence of time steps and train model to predict correct order.
        This forces model to learn temporal dynamics.
        """
        B, T, C, H, W = x_sequence.shape

        if T < self.temporal_window:
            # Not enough timesteps
            return torch.tensor(0.0, device=x_sequence.device)

        # Extract temporal window
        start_idx = torch.randint(0, T - self.temporal_window + 1, (B,))

        windows = []
        for i in range(B):
            window = x_sequence[i, start_idx[i]:start_idx[i] + self.temporal_window]
            windows.append(window)

        windows = torch.stack(windows, dim=0)  # [B, temporal_window, C, H, W]

        # Generate random permutation
        perms = []
        perm_indices = []
        for i in range(B):
            perm = torch.randperm(self.temporal_window)
            perms.append(perm)

            # Convert permutation to index (for classification)
            # Simplified: use hash of permutation as class
            perm_idx = sum([p.item() * (10 ** i) for i, p in enumerate(perm)])
            perm_indices.append(perm_idx % np.math.factorial(self.temporal_window))

        perm_indices = torch.tensor(perm_indices, device=x_sequence.device)

        # Apply permutation to windows
        shuffled_windows = []
        for i in range(B):
            shuffled = windows[i, perms[i]]
            shuffled_windows.append(shuffled)

        shuffled_windows = torch.stack(shuffled_windows, dim=0)

        # Encode each timestep
        features_list = []
        for t_idx in range(self.temporal_window):
            x_t = shuffled_windows[:, t_idx]
            t = torch.zeros(B, device=x_sequence.device)

            if hasattr(self.model, 'flow_model'):
                feats = self.model.flow_model.encoder(x_t, lat_grid, lon_grid)
            else:
                feats = self.model.encoder(x_t, lat_grid, lon_grid)

            # Pool spatial dimensions
            feats_pooled = feats.mean(dim=[1, 2])  # [B, d_model]
            features_list.append(feats_pooled)

        # Concatenate temporal features
        temporal_features = torch.cat(features_list, dim=-1)  # [B, d_model * T]

        # Predict permutation
        logits = self.order_predictor(temporal_features)  # [B, num_permutations]

        # Cross-entropy loss
        # Note: This is simplified - full implementation needs proper permutation indexing
        # For now, use a simplified loss
        loss = F.cross_entropy(logits, perm_indices)

        return loss

    def climate_invariance_loss(
        self,
        x_sequence: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Climate Invariance Loss.

        Force model to learn representations that are invariant to year-to-year
        weather noise but sensitive to long-term climate signals.

        This is done by contrasting:
        - States from same climate regime (should have similar representations)
        - States from different climate regimes (should have different representations)
        """
        B, T, C, H, W = x_sequence.shape

        # Assume sequence represents same climate regime (e.g., same decade)
        # Sample two different timesteps from same sequence (positive pair)
        idx1 = torch.randint(0, T, (B,))
        idx2 = torch.randint(0, T, (B,))

        x1 = x_sequence[torch.arange(B), idx1]
        x2 = x_sequence[torch.arange(B), idx2]

        # Encode both states
        t = torch.zeros(B, device=x_sequence.device)

        if hasattr(self.model, 'flow_model'):
            feat1 = self.model.flow_model.encoder(x1, lat_grid, lon_grid)
            feat2 = self.model.flow_model.encoder(x2, lat_grid, lon_grid)
        else:
            feat1 = self.model.encoder(x1, lat_grid, lon_grid)
            feat2 = self.model.encoder(x2, lat_grid, lon_grid)

        # Extract climate signal (low-frequency component)
        feat1 = feat1.permute(0, 3, 1, 2)  # [B, d_model, H, W]
        feat2 = feat2.permute(0, 3, 1, 2)

        climate1 = self.climate_encoder(feat1)  # [B, 128]
        climate2 = self.climate_encoder(feat2)

        # Contrastive loss - pull positive pairs together
        # For simplicity, use cosine similarity
        similarity = F.cosine_similarity(climate1, climate2, dim=-1)

        # Maximize similarity (minimize negative similarity)
        loss = -similarity.mean()

        return loss


class CurriculumPretraining:
    """
    Curriculum learning wrapper for pre-training.

    Progressively increases task difficulty:
    1. Start with low-resolution data, simple variables
    2. Gradually increase resolution
    3. Add more variables and complexity
    4. Introduce extreme events and rare phenomena
    """

    def __init__(
        self,
        pretraining_module: MultiObjectivePretraining,
        num_stages: int = 5,
    ):
        self.pretraining = pretraining_module
        self.num_stages = num_stages
        self.current_stage = 0

        # Define curriculum stages
        self.stages = self._define_stages()

    def _define_stages(self) -> List[Dict]:
        """Define curriculum stages with increasing difficulty."""
        stages = []

        # Stage 0: Low resolution, basic variables (u, v, t, z)
        stages.append({
            'resolution': (32, 64),
            'num_variables': 4,
            'phenomena': ['large_scale_circulation'],
            'loss_weights': {
                'flow_matching': 1.0,
                'masked_modeling': 0.1,
                'temporal_jigsaw': 0.1,
                'climate_invariance': 0.05,
            }
        })

        # Stage 1: Medium resolution, add humidity
        stages.append({
            'resolution': (64, 128),
            'num_variables': 8,
            'phenomena': ['large_scale_circulation', 'fronts'],
            'loss_weights': {
                'flow_matching': 1.0,
                'masked_modeling': 0.3,
                'temporal_jigsaw': 0.2,
                'climate_invariance': 0.1,
            }
        })

        # Stage 2: High resolution, add more levels
        stages.append({
            'resolution': (128, 256),
            'num_variables': 16,
            'phenomena': ['large_scale_circulation', 'fronts', 'convection'],
            'loss_weights': {
                'flow_matching': 1.0,
                'masked_modeling': 0.5,
                'temporal_jigsaw': 0.3,
                'climate_invariance': 0.1,
            }
        })

        # Stage 3: Very high resolution, full complexity
        stages.append({
            'resolution': (256, 512),
            'num_variables': 32,
            'phenomena': ['all'],
            'loss_weights': {
                'flow_matching': 1.0,
                'masked_modeling': 0.5,
                'temporal_jigsaw': 0.3,
                'climate_invariance': 0.1,
            }
        })

        # Stage 4: Full resolution, extreme events
        stages.append({
            'resolution': (720, 1440),
            'num_variables': 64,
            'phenomena': ['all', 'extreme_events'],
            'loss_weights': {
                'flow_matching': 1.0,
                'masked_modeling': 0.5,
                'temporal_jigsaw': 0.3,
                'climate_invariance': 0.1,
            }
        })

        return stages

    def get_current_config(self) -> Dict:
        """Get configuration for current stage."""
        return self.stages[self.current_stage]

    def advance_stage(self):
        """Move to next curriculum stage."""
        if self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            config = self.stages[self.current_stage]

            # Update loss weights
            self.pretraining.weights = config['loss_weights']

            print(f"Advanced to curriculum stage {self.current_stage + 1}/{self.num_stages}")
            print(f"Resolution: {config['resolution']}")
            print(f"Variables: {config['num_variables']}")
