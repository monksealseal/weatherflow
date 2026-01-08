"""
GenCast Model Architecture

Implementation based on:
    "GenCast: Diffusion-based ensemble forecasting for medium-range weather"
    Price et al., 2023
    Google DeepMind

Key innovations:
    - Denoising diffusion for probabilistic weather prediction
    - Conditional generation on initial conditions
    - Ensemble generation through multiple samples
    - Score-based sampling for uncertainty quantification
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from weatherflow.model_library.architectures.base import (
    ProbabilisticWeatherModel,
    TimeEmbedding,
    LeadTimeEmbedding,
    ConvBlock,
    ResidualBlock,
)
from weatherflow.model_library.registry import (
    ModelCategory,
    ModelInfo,
    ModelScale,
    register_model,
)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """Self-attention block for the diffusion UNet."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, 3, self.num_heads, channels // self.num_heads, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention
        scale = (channels // self.num_heads) ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(batch, channels, height, width)
        out = self.proj(out)

        return out + residual


class ResnetBlock(nn.Module):
    """ResNet block with time embedding for diffusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 8,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(time_emb)[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class DownBlock(nn.Module):
    """Downsampling block for UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        has_attn: bool = False,
        downsample: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        for i in range(num_layers):
            self.blocks.append(
                ResnetBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    time_emb_dim,
                )
            )
            if has_attn:
                self.attns.append(AttentionBlock(out_channels))
            else:
                self.attns.append(nn.Identity())

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        outputs = []
        for block, attn in zip(self.blocks, self.attns):
            x = block(x, time_emb)
            x = attn(x)
            outputs.append(x)
        x = self.downsample(x)
        return x, outputs


class UpBlock(nn.Module):
    """Upsampling block for UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        has_attn: bool = False,
        upsample: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        for i in range(num_layers):
            self.blocks.append(
                ResnetBlock(
                    in_channels + out_channels if i == 0 else out_channels,
                    out_channels,
                    time_emb_dim,
                )
            )
            if has_attn:
                self.attns.append(AttentionBlock(out_channels))
            else:
                self.attns.append(nn.Identity())

        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1) if upsample else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        for block, attn in zip(self.blocks, self.attns):
            x = block(x, time_emb)
            x = attn(x)
        x = self.upsample(x)
        return x


class DiffusionUNet(nn.Module):
    """
    UNet architecture for diffusion-based weather prediction.

    Takes noisy weather state and diffusion timestep,
    predicts the noise to be removed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,  # Conditioning (initial state)
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16, 8),
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial conv (noisy state + conditioning)
        self.conv_in = nn.Conv2d(in_channels + cond_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        current_res = 64  # Assume starting resolution

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            has_attn = current_res in attn_resolutions
            downsample = i < len(channel_mults) - 1

            self.down_blocks.append(
                DownBlock(
                    channels, out_ch, time_emb_dim,
                    num_layers=num_res_blocks,
                    has_attn=has_attn,
                    downsample=downsample,
                )
            )
            channels = out_ch
            if downsample:
                current_res //= 2

        # Middle
        self.mid_block1 = ResnetBlock(channels, channels, time_emb_dim)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResnetBlock(channels, channels, time_emb_dim)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            has_attn = current_res in attn_resolutions
            upsample = i < len(channel_mults) - 1

            self.up_blocks.append(
                UpBlock(
                    channels, out_ch, time_emb_dim,
                    num_layers=num_res_blocks,
                    has_attn=has_attn,
                    upsample=upsample,
                )
            )
            channels = out_ch
            if upsample:
                current_res *= 2

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy state (batch, channels, h, w)
            t: Diffusion timestep (batch,)
            cond: Conditioning (initial state) (batch, cond_channels, h, w)

        Returns:
            Predicted noise (batch, channels, h, w)
        """
        # Time embedding
        time_emb = self.time_emb(t)

        # Concatenate input with conditioning
        x = torch.cat([x, cond], dim=1)
        x = self.conv_in(x)

        # Down
        skip_connections = []
        for block in self.down_blocks:
            x, skips = block(x, time_emb)
            skip_connections.extend(skips)

        # Middle
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        # Up
        for block in self.up_blocks:
            skip = skip_connections.pop()
            x = block(x, skip, time_emb)

        return self.conv_out(x)


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Models scheduler.

    Implements the forward (noising) and reverse (denoising) process.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            steps = torch.arange(num_timesteps + 1) / num_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Precompute for forward process
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        # Precompute for reverse process
        self.sqrt_recip_alphas = torch.sqrt(1 / alphas)
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    def add_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x at timestep t."""
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x.device)

        noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        return noisy, noise

    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step."""
        alpha = self.alphas[t].to(x.device)
        alpha_cumprod = self.alphas_cumprod[t].to(x.device)
        beta = self.betas[t].to(x.device)

        # Predicted x0
        pred_x0 = (x - beta / self.sqrt_one_minus_alphas_cumprod[t].to(x.device) * model_output) / self.sqrt_alphas_cumprod[t].to(x.device)

        # Posterior mean
        coef1 = beta * torch.sqrt(self.alphas_cumprod_prev[t]).to(x.device) / (1 - alpha_cumprod)
        coef2 = (1 - self.alphas_cumprod_prev[t]).to(x.device) * torch.sqrt(alpha) / (1 - alpha_cumprod)
        mean = coef1 * pred_x0 + coef2 * x

        if t > 0:
            noise = torch.randn_like(x)
            variance = self.posterior_variance[t].to(x.device)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean


class DDIMScheduler(DDPMScheduler):
    """
    Denoising Diffusion Implicit Models scheduler.

    Allows for faster sampling with fewer steps.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        eta: float = 0.0,  # 0 = deterministic, 1 = DDPM
    ):
        super().__init__(num_timesteps, beta_start, beta_end)
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # Compute inference timesteps
        step_ratio = num_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_timesteps, step_ratio)

    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        t_prev: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """DDIM sampling step."""
        alpha_prod_t = self.alphas_cumprod[t].to(x.device)
        alpha_prod_t_prev = self.alphas_cumprod[t_prev].to(x.device) if t_prev >= 0 else torch.tensor(1.0).to(x.device)

        # Predicted x0
        pred_x0 = (x - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)

        # Direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_prod_t_prev - self.eta**2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)) * model_output

        # Random noise
        if self.eta > 0 and t_prev > 0:
            noise = torch.randn_like(x)
            variance = self.eta * torch.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev))
        else:
            noise = 0
            variance = 0

        x_prev = torch.sqrt(alpha_prod_t_prev) * pred_x0 + pred_dir + variance * noise
        return x_prev


class GenCastModel(ProbabilisticWeatherModel):
    """
    GenCast: Diffusion-based Ensemble Weather Forecasting.

    Based on Price et al. (2023), DeepMind.

    Uses conditional denoising diffusion to generate
    probabilistic weather forecasts. Multiple samples
    provide ensemble members for uncertainty quantification.

    Args:
        in_channels: Number of atmospheric variables
        base_channels: Base channel count for UNet
        num_timesteps: Diffusion timesteps
        num_inference_steps: Steps for sampling (DDIM)
    """

    def __init__(
        self,
        in_channels: int = 20,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_timesteps: int = 1000,
        num_inference_steps: int = 50,
        img_size: Tuple[int, int] = (64, 128),
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
        num_ensemble_members: int = 50,
    ):
        if input_variables is None:
            input_variables = self._get_default_variables()
        if output_variables is None:
            output_variables = self._get_default_variables()

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="1deg",
            forecast_hours=6,
            num_ensemble_members=num_ensemble_members,
        )

        self.in_channels = in_channels
        self.img_size = img_size
        self.num_timesteps = num_timesteps

        # Diffusion model
        self.model = DiffusionUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            cond_channels=in_channels,  # Condition on initial state
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
        )

        # Scheduler
        self.scheduler = DDIMScheduler(
            num_timesteps=num_timesteps,
            num_inference_steps=num_inference_steps,
        )

    def _get_default_variables(self) -> List[str]:
        return [
            "z_500", "z_850", "z_1000",
            "t_500", "t_850", "t_1000",
            "u_500", "u_850", "u_1000",
            "v_500", "v_850", "v_1000",
            "q_500", "q_850", "q_1000",
            "t2m", "u10", "v10", "msl", "tp",
        ]

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Deterministic forward (ensemble mean).
        """
        return self.get_mean(x, num_samples=10)

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate ensemble samples using diffusion.

        Args:
            x: Initial state (batch, channels, h, w)
            num_samples: Number of ensemble members

        Returns:
            Samples (batch, num_samples, channels, h, w)
        """
        batch = x.shape[0]
        samples = []

        for _ in range(num_samples):
            # Start from noise
            xt = torch.randn_like(x)

            # Reverse diffusion
            timesteps = self.scheduler.timesteps.tolist()
            for i, t in enumerate(reversed(timesteps)):
                t_tensor = torch.full((batch,), t, device=x.device, dtype=torch.long)

                # Predict noise
                noise_pred = self.model(xt, t_tensor, x)

                # Denoise step
                t_prev = timesteps[len(timesteps) - i - 2] if i < len(timesteps) - 1 else -1
                xt = self.scheduler.step(noise_pred, t, t_prev, xt)

            samples.append(xt)

        return torch.stack(samples, dim=1)

    def get_training_loss(
        self,
        x0: torch.Tensor,  # Target state
        cond: torch.Tensor,  # Initial state (conditioning)
    ) -> torch.Tensor:
        """
        Compute training loss (noise prediction).

        Args:
            x0: Target/clean state (batch, channels, h, w)
            cond: Conditioning state (batch, channels, h, w)

        Returns:
            Loss scalar
        """
        batch = x0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch,), device=x0.device)

        # Add noise
        xt, noise = self.scheduler.add_noise(x0, t)

        # Predict noise
        noise_pred = self.model(xt, t, cond)

        # MSE loss
        return F.mse_loss(noise_pred, noise)


# Register model
gencast_info = ModelInfo(
    name="GenCast",
    category=ModelCategory.DIFFUSION,
    scale=ModelScale.LARGE,
    description="Diffusion-based ensemble forecasting for probabilistic weather prediction",
    paper_title="GenCast: Diffusion-based ensemble forecasting for medium-range weather",
    paper_url="https://arxiv.org/abs/2312.15796",
    paper_year=2023,
    authors=["Ilan Price", "Alvaro Sanchez-Gonzalez", "Ferran Alet", "et al."],
    organization="Google DeepMind",
    input_variables=["z", "t", "u", "v", "q", "t2m", "u10", "v10", "msl", "tp"],
    output_variables=["z", "t", "u", "v", "q", "t2m", "u10", "v10", "msl", "tp"],
    supported_resolutions=["0.25deg"],
    forecast_range="0-15 days",
    temporal_resolution="12h",
    is_probabilistic=True,
    supports_ensemble=True,
    has_pretrained_weights=False,
    min_gpu_memory_gb=32.0,
    typical_training_time="~1 week on 32 TPUv4",
    inference_time_per_step="~30 seconds per sample on A100",
    tags=["diffusion", "probabilistic", "ensemble", "uncertainty"],
    related_models=["seedstorm", "stormcast"],
)

register_model("gencast", GenCastModel, gencast_info, {
    "in_channels": 20,
    "base_channels": 64,
    "num_timesteps": 1000,
    "num_inference_steps": 50,
    "img_size": (64, 128),
})
