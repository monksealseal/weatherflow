"""
FlowAtmosphere: Unified Foundation Model for Atmospheric Science

A single, massive pre-trained model that can perform diverse atmospheric tasks
through a natural language or programmatic interface.
"""

from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
from .flow_former import FlowFormer, HierarchicalSphericalTransformer


class FlowAtmosphere(nn.Module):
    """
    FlowAtmosphere: A GPT-like foundation model for atmospheric science.

    This model provides a unified interface for diverse tasks:
    - Weather forecasting
    - Climate downscaling
    - Teleconnection analysis
    - Extreme event attribution
    - Sub-seasonal to seasonal prediction
    - And more...

    The model dynamically adapts its computational graph based on the task.
    """

    def __init__(
        self,
        input_channels: int = 128,  # Multi-variable, multi-level inputs
        d_model: int = 1024,  # 10B parameters with depth
        num_layers: int = 24,
        num_heads: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.1,
        window_size: int = 16,
        max_lead_time: int = 240,  # Hours
        num_pressure_levels: int = 13,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_lead_time = max_lead_time
        self.num_pressure_levels = num_pressure_levels

        # Core flow matching model
        self.flow_model = FlowFormer(
            input_channels=input_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            window_size=window_size,
        )

        # Task-specific heads (initialized but can be adapted)
        self.task_heads = nn.ModuleDict({
            'forecast': self._build_forecast_head(),
            'downscale': self._build_downscale_head(),
            'teleconnection': self._build_teleconnection_head(),
            'attribution': self._build_attribution_head(),
            's2s_prediction': self._build_s2s_head(),
        })

        # Natural language interface (text encoder)
        self.text_encoder = self._build_text_encoder()

        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def _build_forecast_head(self) -> nn.Module:
        """Build head for weather forecasting task."""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.input_channels),
        )

    def _build_downscale_head(self) -> nn.Module:
        """Build head for spatial downscaling."""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.ConvTranspose2d(self.d_model, self.d_model // 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(self.d_model // 2, self.input_channels, kernel_size=4, stride=2, padding=1),
        )

    def _build_teleconnection_head(self) -> nn.Module:
        """Build head for teleconnection pattern analysis."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.d_model, 512),
            nn.GELU(),
            nn.Linear(512, 128),  # Teleconnection indices
        )

    def _build_attribution_head(self) -> nn.Module:
        """Build head for extreme event attribution."""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),  # Attribution score
            nn.Sigmoid(),
        )

    def _build_s2s_head(self) -> nn.Module:
        """Build head for sub-seasonal to seasonal prediction."""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.input_channels * self.max_lead_time // 24),
        )

    def _build_text_encoder(self) -> nn.Module:
        """Build text encoder for natural language queries."""
        # Simplified - in production use pretrained language model
        return nn.Sequential(
            nn.Embedding(50000, 512),  # Vocab size, embed dim
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=8),
                num_layers=6,
            ),
            nn.Linear(512, self.d_model),
        )

    def forecast(
        self,
        initial_state: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        lead_times: List[int],
        num_ensemble: int = 1,
    ) -> torch.Tensor:
        """
        Generate weather forecast.

        Args:
            initial_state: Initial weather state, shape [B, C, H, W]
            lat_grid: Latitude coordinates, shape [H, W]
            lon_grid: Longitude coordinates, shape [H, W]
            lead_times: Forecast lead times in hours
            num_ensemble: Number of ensemble members

        Returns:
            Forecasts, shape [B, num_ensemble, len(lead_times), C, H, W]
        """
        B, C, H, W = initial_state.shape
        device = initial_state.device

        forecasts = []

        for ensemble_idx in range(num_ensemble):
            # Add noise for ensemble perturbation
            if ensemble_idx > 0:
                noise = torch.randn_like(initial_state) * 0.01
                current_state = initial_state + noise
            else:
                current_state = initial_state

            ensemble_forecast = [current_state]

            # Integrate forward in time using flow matching
            for lead_time in lead_times:
                # Normalize time to [0, 1]
                t = torch.tensor([lead_time / self.max_lead_time], device=device)
                t = t.expand(B)

                # Compute velocity field
                v = self.flow_model(current_state, t, lat_grid, lon_grid)

                # Euler step (can use more sophisticated ODE solver)
                dt = 1.0 / self.max_lead_time
                current_state = current_state + v * dt

                ensemble_forecast.append(current_state)

            forecasts.append(torch.stack(ensemble_forecast[1:], dim=1))

        return torch.stack(forecasts, dim=1)

    def downscale(
        self,
        coarse_state: torch.Tensor,
        lat_grid_coarse: torch.Tensor,
        lon_grid_coarse: torch.Tensor,
        target_resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Downscale coarse resolution data to higher resolution.

        Args:
            coarse_state: Coarse weather state, shape [B, C, H_c, W_c]
            lat_grid_coarse: Coarse latitude grid
            lon_grid_coarse: Coarse longitude grid
            target_resolution: (H_target, W_target)

        Returns:
            Downscaled state, shape [B, C, H_target, W_target]
        """
        # Encode coarse state
        t = torch.zeros(coarse_state.size(0), device=coarse_state.device)
        features = self.flow_model.encoder(
            coarse_state, lat_grid_coarse, lon_grid_coarse
        )

        # Apply downscaling head
        # Note: This is simplified - full implementation needs proper upsampling
        B, H, W, D = features.shape
        features = features.permute(0, 3, 1, 2)  # [B, D, H, W]

        # Upsample to target resolution
        downscaled = nn.functional.interpolate(
            features,
            size=target_resolution,
            mode='bilinear',
            align_corners=False,
        )

        # Project to output channels
        downscaled = self.task_heads['downscale'](downscaled.permute(0, 2, 3, 1))

        return downscaled

    def analyze_teleconnection(
        self,
        state_sequence: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze teleconnection patterns (e.g., ENSO, NAO, PDO).

        Args:
            state_sequence: Weather states over time, shape [B, T, C, H, W]
            lat_grid: Latitude coordinates
            lon_grid: Longitude coordinates

        Returns:
            Dictionary of teleconnection indices
        """
        B, T, C, H, W = state_sequence.shape

        # Encode each time step
        indices = []
        for t_idx in range(T):
            state = state_sequence[:, t_idx]
            t = torch.ones(B, device=state.device) * (t_idx / T)

            features = self.flow_model.encoder(state, lat_grid, lon_grid)
            features = features.permute(0, 3, 1, 2)  # [B, D, H, W]

            # Apply teleconnection head
            tc_indices = self.task_heads['teleconnection'](features)
            indices.append(tc_indices)

        indices = torch.stack(indices, dim=1)  # [B, T, num_indices]

        # Parse into named teleconnection patterns
        return {
            'ENSO': indices[:, :, 0],
            'NAO': indices[:, :, 1],
            'PDO': indices[:, :, 2],
            'SAM': indices[:, :, 3],
            # ... more patterns
        }

    def attribute_event(
        self,
        event_state: torch.Tensor,
        baseline_state: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attribute extreme event to climate change.

        Args:
            event_state: Observed extreme event state
            baseline_state: Baseline (pre-industrial or counterfactual) state
            lat_grid: Latitude coordinates
            lon_grid: Longitude coordinates

        Returns:
            Attribution score (0-1), shape [B, H, W]
        """
        t = torch.zeros(event_state.size(0), device=event_state.device)

        # Encode both states
        event_features = self.flow_model.encoder(event_state, lat_grid, lon_grid)
        baseline_features = self.flow_model.encoder(baseline_state, lat_grid, lon_grid)

        # Compute difference
        diff_features = event_features - baseline_features

        # Apply attribution head
        attribution = self.task_heads['attribution'](diff_features)
        attribution = attribution.squeeze(-1)  # [B, H, W]

        return attribution

    def s2s_forecast(
        self,
        initial_state: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        lead_weeks: int = 6,
    ) -> torch.Tensor:
        """
        Sub-seasonal to seasonal (S2S) forecast.

        Args:
            initial_state: Initial state
            lat_grid: Latitude coordinates
            lon_grid: Longitude coordinates
            lead_weeks: Number of weeks to forecast

        Returns:
            Weekly averaged forecasts, shape [B, lead_weeks, C, H, W]
        """
        # Use flow matching with longer integration
        forecasts = []

        for week in range(lead_weeks):
            t = torch.ones(initial_state.size(0), device=initial_state.device) * \
                ((week + 1) / lead_weeks)

            # Multi-step integration (simplified)
            state = initial_state
            for _ in range(7):  # Daily steps
                v = self.flow_model(state, t, lat_grid, lon_grid)
                state = state + v * (1.0 / (lead_weeks * 7))

            forecasts.append(state)

        return torch.stack(forecasts, dim=1)

    def answer_query(
        self,
        query: str,
        context_state: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Answer natural language query about atmospheric state.

        Example queries:
        - "Show me the evolution of the polar vortex strength over the last week"
        - "Will there be a heatwave in Europe next month?"
        - "How is ENSO affecting precipitation in California?"

        Args:
            query: Natural language question
            context_state: Current atmospheric state
            lat_grid: Latitude coordinates
            lon_grid: Longitude coordinates

        Returns:
            Dictionary with 'answer' (text) and 'visualization' (tensor)
        """
        # Tokenize query (simplified)
        # In production, use proper tokenizer
        query_tokens = self._tokenize_query(query)
        query_tokens = query_tokens.to(context_state.device)

        # Encode query
        query_embedding = self.text_encoder(query_tokens)  # [seq_len, d_model]
        query_embedding = query_embedding.mean(dim=0)  # Pool to [d_model]

        # Encode atmospheric state
        t = torch.zeros(context_state.size(0), device=context_state.device)
        state_features = self.flow_model.encoder(context_state, lat_grid, lon_grid)
        state_features = state_features.mean(dim=[0, 1, 2])  # Pool to [d_model]

        # Combine query and state
        combined = query_embedding + state_features

        # Generate answer (simplified - full implementation would use language model)
        # This is a placeholder for demonstration
        answer_text = self._generate_answer(combined, query)

        # Generate visualization
        visualization = self._generate_visualization(combined, context_state)

        return {
            'answer': answer_text,
            'visualization': visualization,
        }

    def _tokenize_query(self, query: str) -> torch.Tensor:
        """Tokenize natural language query."""
        # Simplified tokenization - use proper tokenizer in production
        tokens = [hash(word) % 50000 for word in query.lower().split()]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    def _generate_answer(self, features: torch.Tensor, query: str) -> str:
        """Generate textual answer to query."""
        # Placeholder - full implementation would use language model decoder
        return f"Analysis for query: '{query}' based on atmospheric state features."

    def _generate_visualization(
        self,
        features: torch.Tensor,
        context_state: torch.Tensor
    ) -> torch.Tensor:
        """Generate visualization tensor."""
        # Return relevant atmospheric field
        return context_state[:, 0:1]  # Return first channel as example

    def load_pretrained(self, path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded pretrained model from {path}")

    def save_checkpoint(self, path: str, optimizer_state: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'input_channels': self.input_channels,
                'd_model': self.d_model,
                'num_layers': self.num_layers,
                'max_lead_time': self.max_lead_time,
                'num_pressure_levels': self.num_pressure_levels,
            }
        }
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
