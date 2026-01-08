"""
Image-to-Image Translation Models for Weather Applications

Implementation of Pix2Pix and CycleGAN based on:
    "Image-to-Image Translation with Conditional Adversarial Networks"
    Isola et al., CVPR 2017

    "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
    Zhu et al., ICCV 2017

Applications:
    - Hurricane satellite imagery to wind field estimation
    - Radar reflectivity to precipitation
    - Visible satellite to IR satellite
    - Low-resolution to high-resolution weather fields
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from weatherflow.model_library.architectures.base import BaseWeatherModel
from weatherflow.model_library.registry import (
    ModelCategory,
    ModelInfo,
    ModelScale,
    register_model,
)


class UNetDown(nn.Module):
    """U-Net encoder block with convolution and downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    """U-Net decoder block with upsampling and skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = torch.cat([x, skip], dim=1)
        return x


class Pix2PixGenerator(nn.Module):
    """
    U-Net Generator for Pix2Pix.

    Classic encoder-decoder with skip connections.
    Used for paired image-to-image translation.

    Architecture:
        Encoder: 8 downsampling blocks
        Decoder: 8 upsampling blocks with skip connections
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,  # u, v wind components
        features: int = 64,
    ):
        super().__init__()

        # Encoder
        self.down1 = UNetDown(in_channels, features, normalize=False)  # 128
        self.down2 = UNetDown(features, features * 2)  # 64
        self.down3 = UNetDown(features * 2, features * 4)  # 32
        self.down4 = UNetDown(features * 4, features * 8)  # 16
        self.down5 = UNetDown(features * 8, features * 8)  # 8
        self.down6 = UNetDown(features * 8, features * 8)  # 4
        self.down7 = UNetDown(features * 8, features * 8)  # 2
        self.down8 = UNetDown(features * 8, features * 8, normalize=False)  # 1

        # Decoder
        self.up1 = UNetUp(features * 8, features * 8, dropout=0.5)
        self.up2 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up3 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up4 = UNetUp(features * 16, features * 8)
        self.up5 = UNetUp(features * 16, features * 4)
        self.up6 = UNetUp(features * 8, features * 2)
        self.up7 = UNetUp(features * 4, features)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Pix2PixDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix.

    Classifies overlapping patches as real or fake.
    More efficient than full image classification.
    """

    def __init__(
        self,
        in_channels: int = 5,  # Input + output concatenated
        features: int = 64,
    ):
        super().__init__()

        self.model = nn.Sequential(
            # No normalization on first layer
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_A: Condition image (satellite)
            img_B: Target/generated image (wind field)
        """
        x = torch.cat([img_A, img_B], dim=1)
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generator."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """
    ResNet-style Generator for CycleGAN.

    Architecture:
        - Initial convolution
        - 2 downsampling blocks
        - 9 residual blocks
        - 2 upsampling blocks
        - Final convolution

    Used for unpaired image-to-image translation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        features: int = 64,
        num_residual_blocks: int = 9,
    ):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, kernel_size=7, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(features * 4) for _ in range(num_residual_blocks)]
        )

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )

        # Final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, out_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residuals(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.final(x)


class CycleGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for CycleGAN.

    70x70 PatchGAN discriminator.
    """

    def __init__(
        self,
        in_channels: int = 3,
        features: int = 64,
    ):
        super().__init__()

        self.model = nn.Sequential(
            # No normalization on first layer
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HurricaneWindFieldModel(BaseWeatherModel):
    """
    Hurricane Wind Field Estimation from Satellite Imagery.

    Uses Pix2Pix architecture to translate satellite images
    (visible, IR, water vapor) to wind field estimates (u, v).

    Based on the approach of using conditional GANs for
    geophysical variable estimation from remote sensing.

    Input: Satellite imagery (multiple channels)
    Output: Wind field components (u, v) at multiple heights

    The model learns to map the visual structure of hurricanes
    (eye, eyewall, spiral bands) to corresponding wind patterns.
    """

    def __init__(
        self,
        in_channels: int = 3,  # RGB or multi-spectral satellite
        out_channels: int = 2,  # u, v wind components
        features: int = 64,
        img_size: Tuple[int, int] = (256, 256),
        max_wind_speed: float = 80.0,  # m/s for normalization
        use_cyclegan: bool = False,
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
    ):
        if input_variables is None:
            input_variables = ["satellite_ch1", "satellite_ch2", "satellite_ch3"]
        if output_variables is None:
            output_variables = ["u_wind_10m", "v_wind_10m"]

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="4km",
            forecast_hours=0,  # Analysis, not forecast
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.max_wind_speed = max_wind_speed
        self.use_cyclegan = use_cyclegan

        # Generator
        if use_cyclegan:
            self.generator = CycleGANGenerator(
                in_channels=in_channels,
                out_channels=out_channels,
                features=features,
            )
        else:
            self.generator = Pix2PixGenerator(
                in_channels=in_channels,
                out_channels=out_channels,
                features=features,
            )

        # Discriminator for training
        if use_cyclegan:
            self.discriminator = CycleGANDiscriminator(
                in_channels=out_channels,
                features=features,
            )
        else:
            self.discriminator = Pix2PixDiscriminator(
                in_channels=in_channels + out_channels,
                features=features,
            )

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate wind field from satellite image.

        Args:
            x: Satellite imagery (batch, channels, height, width)

        Returns:
            Wind field (batch, 2, height, width) with u, v components
        """
        # Generate wind field (normalized -1 to 1)
        wind_normalized = self.generator(x)

        # Scale to physical units (m/s)
        wind = wind_normalized * self.max_wind_speed

        return wind

    def get_wind_speed_and_direction(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get wind speed and direction from satellite image.

        Returns:
            speed: (batch, 1, height, width) in m/s
            direction: (batch, 1, height, width) in degrees (meteorological convention)
        """
        wind = self.forward(x)
        u, v = wind[:, 0:1], wind[:, 1:2]

        speed = torch.sqrt(u**2 + v**2)
        direction = torch.atan2(-u, -v) * 180 / math.pi  # Meteorological convention

        return speed, direction

    def get_training_loss(
        self,
        satellite: torch.Tensor,
        wind_true: torch.Tensor,
        lambda_l1: float = 100.0,
    ) -> dict:
        """
        Compute Pix2Pix training losses.

        Args:
            satellite: Input satellite image
            wind_true: Ground truth wind field
            lambda_l1: Weight for L1 reconstruction loss

        Returns:
            Dictionary with generator and discriminator losses
        """
        # Normalize wind to [-1, 1]
        wind_true_norm = wind_true / self.max_wind_speed

        # Generate fake wind field
        wind_fake = self.generator(satellite)

        # Discriminator loss
        if self.use_cyclegan:
            d_real = self.discriminator(wind_true_norm)
            d_fake = self.discriminator(wind_fake.detach())
        else:
            d_real = self.discriminator(satellite, wind_true_norm)
            d_fake = self.discriminator(satellite, wind_fake.detach())

        d_loss_real = F.mse_loss(d_real, torch.ones_like(d_real))
        d_loss_fake = F.mse_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_loss_real + d_loss_fake) / 2

        # Generator loss
        if self.use_cyclegan:
            g_adv = self.discriminator(wind_fake)
        else:
            g_adv = self.discriminator(satellite, wind_fake)

        g_loss_adv = F.mse_loss(g_adv, torch.ones_like(g_adv))
        g_loss_l1 = F.l1_loss(wind_fake, wind_true_norm)
        g_loss = g_loss_adv + lambda_l1 * g_loss_l1

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_loss_adv": g_loss_adv,
            "g_loss_l1": g_loss_l1,
        }


class RadarToPrecipitationModel(BaseWeatherModel):
    """
    Radar Reflectivity to Precipitation Rate Conversion.

    Uses Pix2Pix to learn the complex relationship between
    radar reflectivity (dBZ) and precipitation rate (mm/hr).

    This goes beyond simple Z-R relationships by learning
    from data, accounting for:
    - Drop size distribution variations
    - Beam blockage
    - Bright band effects
    - Precipitation type (rain, snow, hail)
    """

    def __init__(
        self,
        in_channels: int = 1,  # Radar reflectivity
        out_channels: int = 1,  # Precipitation rate
        features: int = 64,
        max_precip_rate: float = 100.0,  # mm/hr
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
    ):
        if input_variables is None:
            input_variables = ["radar_reflectivity"]
        if output_variables is None:
            output_variables = ["precipitation_rate"]

        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            resolution="1km",
            forecast_hours=0,
        )

        self.max_precip_rate = max_precip_rate

        self.generator = Pix2PixGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )

    def forward(
        self,
        x: torch.Tensor,
        lead_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Convert radar reflectivity to precipitation rate."""
        precip_norm = self.generator(x)
        # Use sigmoid to ensure non-negative precipitation
        precip = F.sigmoid(precip_norm) * self.max_precip_rate
        return precip


# Register models
pix2pix_info = ModelInfo(
    name="Pix2Pix",
    category=ModelCategory.IMAGE_TO_IMAGE,
    scale=ModelScale.MEDIUM,
    description="Conditional GAN for paired image-to-image translation",
    paper_title="Image-to-Image Translation with Conditional Adversarial Networks",
    paper_url="https://arxiv.org/abs/1611.07004",
    paper_year=2017,
    authors=["Phillip Isola", "Jun-Yan Zhu", "Tinghui Zhou", "Alexei A. Efros"],
    organization="UC Berkeley",
    input_variables=["image"],
    output_variables=["image"],
    supported_resolutions=["256x256", "512x512"],
    forecast_range="N/A (image translation)",
    temporal_resolution="N/A",
    is_probabilistic=False,
    supports_ensemble=False,
    has_pretrained_weights=False,
    min_gpu_memory_gb=4.0,
    typical_training_time="~1-2 days on single GPU",
    inference_time_per_step="~10ms on GPU",
    tags=["gan", "image-translation", "u-net", "patchgan"],
    related_models=["cyclegan", "pix2pixhd"],
)

cyclegan_info = ModelInfo(
    name="CycleGAN",
    category=ModelCategory.IMAGE_TO_IMAGE,
    scale=ModelScale.MEDIUM,
    description="Unpaired image-to-image translation with cycle consistency",
    paper_title="Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
    paper_url="https://arxiv.org/abs/1703.10593",
    paper_year=2017,
    authors=["Jun-Yan Zhu", "Taesung Park", "Phillip Isola", "Alexei A. Efros"],
    organization="UC Berkeley",
    input_variables=["image"],
    output_variables=["image"],
    supported_resolutions=["256x256", "512x512"],
    forecast_range="N/A (image translation)",
    temporal_resolution="N/A",
    is_probabilistic=False,
    supports_ensemble=False,
    has_pretrained_weights=False,
    min_gpu_memory_gb=6.0,
    typical_training_time="~2-3 days on single GPU",
    inference_time_per_step="~15ms on GPU",
    tags=["gan", "unpaired", "cycle-consistency", "resnet"],
    related_models=["pix2pix", "unit", "stargan"],
)

hurricane_info = ModelInfo(
    name="HurricaneWindField",
    category=ModelCategory.IMAGE_TO_IMAGE,
    scale=ModelScale.MEDIUM,
    description="Hurricane wind field estimation from satellite imagery using Pix2Pix/CycleGAN",
    paper_title="Image-to-Image Translation with Conditional Adversarial Networks (adapted for meteorology)",
    paper_url="https://arxiv.org/abs/1611.07004",
    paper_year=2017,
    authors=["Phillip Isola", "Jun-Yan Zhu", "Tinghui Zhou", "Alexei A. Efros"],
    organization="UC Berkeley (original); adapted for weather applications",
    input_variables=["satellite_visible", "satellite_ir", "satellite_wv"],
    output_variables=["u_wind_10m", "v_wind_10m"],
    supported_resolutions=["4km", "2km", "1km"],
    forecast_range="Analysis (t=0)",
    temporal_resolution="Instantaneous",
    is_probabilistic=False,
    supports_ensemble=False,
    has_pretrained_weights=False,
    min_gpu_memory_gb=4.0,
    typical_training_time="~1-2 days on single GPU",
    inference_time_per_step="~10ms on GPU",
    tags=["hurricane", "wind-field", "satellite", "gan", "tropical-cyclone"],
    related_models=["pix2pix", "cyclegan"],
)

register_model("pix2pix", Pix2PixGenerator, pix2pix_info, {
    "in_channels": 3,
    "out_channels": 3,
    "features": 64,
})

register_model("cyclegan", CycleGANGenerator, cyclegan_info, {
    "in_channels": 3,
    "out_channels": 3,
    "features": 64,
    "num_residual_blocks": 9,
})

register_model("hurricane_wind_field", HurricaneWindFieldModel, hurricane_info, {
    "in_channels": 3,
    "out_channels": 2,
    "features": 64,
    "img_size": (256, 256),
})
