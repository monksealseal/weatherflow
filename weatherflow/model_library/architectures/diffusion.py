"""
Diffusion Model Components for Weather Prediction

General-purpose diffusion model components that can be used
for various weather applications:
    - Probabilistic forecasting
    - Ensemble generation
    - Super-resolution
    - Data assimilation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherDiffusion(nn.Module):
    """
    General-purpose weather diffusion model.

    A flexible diffusion model for weather applications that
    can be configured for different tasks.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int = 128,
        out_channels: Optional[int] = None,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_heads: int = 8,
        num_timesteps: int = 1000,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # UNet-style architecture
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        ch = model_channels
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    self._make_res_block(ch, out_ch, time_embed_dim)
                )
                ch = out_ch
                input_block_chans.append(ch)

            if level < len(channel_mult) - 1:
                self.down_blocks.append(
                    nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
                )
                input_block_chans.append(ch)
                ds *= 2

        # Middle blocks
        self.middle_block = nn.Sequential(
            self._make_res_block(ch, ch, time_embed_dim),
            self._make_res_block(ch, ch, time_embed_dim),
        )

        # Upsampling
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                self.up_blocks.append(
                    self._make_res_block(ch + skip_ch, out_ch, time_embed_dim)
                )
                ch = out_ch

            if level > 0:
                self.up_blocks.append(
                    nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
                )

        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def _make_res_block(
        self,
        in_ch: int,
        out_ch: int,
        time_embed_dim: int,
    ) -> nn.Module:
        """Create a residual block with time embedding."""

        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.GroupNorm(32, in_ch)
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
                self.time_proj = nn.Linear(time_embed_dim, out_ch)
                self.norm2 = nn.GroupNorm(32, out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

                if in_ch != out_ch:
                    self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
                else:
                    self.skip = nn.Identity()

            def forward(self, x, t_emb):
                h = self.norm1(x)
                h = F.silu(h)
                h = self.conv1(h)
                h = h + self.time_proj(t_emb)[:, :, None, None]
                h = self.norm2(h)
                h = F.silu(h)
                h = self.conv2(h)
                return h + self.skip(x)

        return ResBlock()

    def _timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting noise.

        Args:
            x: Noisy input (batch, channels, height, width)
            t: Timestep (batch,)
            cond: Optional conditioning (batch, cond_channels, height, width)

        Returns:
            Predicted noise
        """
        # Concatenate conditioning if provided
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        # Time embedding
        t_emb = self._timestep_embedding(t, self.in_channels)
        t_emb = self.time_embed(t_emb)

        # Input projection
        h = self.input_proj(x)

        # Down path
        hs = [h]
        for module in self.down_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            else:
                h = module(h, t_emb)
            hs.append(h)

        # Middle
        for block in self.middle_block:
            h = block(h, t_emb)

        # Up path
        for module in self.up_blocks:
            if isinstance(module, nn.ConvTranspose2d):
                h = module(h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)

        return self.out(h)


class DDPMScheduler:
    """DDPM noise scheduler."""

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
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

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

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Sample from the diffusion model."""
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = model(x, t_batch, cond)

            # Compute coefficients
            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]

            # Posterior mean
            x = (x - beta / self.sqrt_one_minus_alphas_cumprod[t] * noise_pred) / torch.sqrt(alpha)

            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = x + sigma * noise

        return x


class DDIMScheduler(DDPMScheduler):
    """DDIM scheduler for faster sampling."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        **kwargs,
    ):
        super().__init__(num_timesteps=num_timesteps, **kwargs)
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # Compute inference timesteps
        step_ratio = num_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_timesteps, step_ratio)[::-1].clone()

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """DDIM sampling for faster inference."""
        x = torch.randn(shape, device=device)

        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = model(x, t_batch, cond)

            # Get alpha values
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_t_prev = self.alphas_cumprod[self.timesteps[i + 1]] if i < len(self.timesteps) - 1 else torch.tensor(1.0)

            # Predicted x0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            # Direction pointing to xt
            pred_dir = torch.sqrt(1 - alpha_bar_t_prev - self.eta**2 * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)) * noise_pred

            # Stochastic component
            if self.eta > 0 and i < len(self.timesteps) - 1:
                noise = torch.randn_like(x)
                sigma = self.eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            else:
                noise = 0
                sigma = 0

            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir + sigma * noise

        return x
