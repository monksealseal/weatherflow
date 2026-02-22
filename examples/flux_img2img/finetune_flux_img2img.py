"""
Fine-tune FLUX.1-dev for image-to-image tasks using LoRA.

FLUX.1-dev by Black Forest Labs is the most advanced open-source image model,
built on a 12B-parameter rectified flow transformer -- the same flow matching
framework WeatherFlow uses.  This script fine-tunes it with LoRA on paired
image data from your existing GAN dataset (StyleTransferDataset).

Usage:
    # Single GPU
    python examples/flux_img2img/finetune_flux_img2img.py \
        --config examples/flux_img2img/config.yaml

    # Multi-GPU via accelerate
    accelerate launch examples/flux_img2img/finetune_flux_img2img.py \
        --config examples/flux_img2img/config.yaml

Requirements:
    pip install diffusers[torch] peft accelerate transformers safetensors
"""

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports – these pull in large dependencies so we defer them
# ---------------------------------------------------------------------------

_diffusers = None
_peft = None
_transformers = None


def _import_diffusers():
    global _diffusers
    if _diffusers is None:
        import diffusers
        _diffusers = diffusers
    return _diffusers


def _import_peft():
    global _peft
    if _peft is None:
        import peft
        _peft = peft
    return _peft


def _import_transformers():
    global _transformers
    if _transformers is None:
        import transformers
        _transformers = transformers
    return _transformers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FluxFineTuneConfig:
    """Configuration for FLUX.1 image-to-image LoRA fine-tuning."""

    # Model
    pretrained_model: str = "black-forest-labs/FLUX.1-dev"
    revision: Optional[str] = None
    variant: Optional[str] = "fp16"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
    ])

    # Dataset
    content_dir: Optional[str] = None
    target_dir: Optional[str] = None
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True
    val_split: float = 0.1

    # Training
    epochs: int = 20
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_ratio: float = 0.05
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Precision / memory
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = True
    enable_xformers: bool = True

    # Flow matching
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    strength: float = 0.75  # img2img denoising strength

    # Logging / checkpointing
    output_dir: str = "./flux_ft_output"
    logging_steps: int = 10
    save_epochs: int = 5
    seed: int = 42
    use_wandb: bool = False
    experiment_name: str = "flux_img2img_finetune"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FluxFineTuneConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

class FluxImg2ImgDataset(Dataset):
    """Wraps paired image directories into a dataset for FLUX fine-tuning.

    Also accepts an existing ``StyleTransferDataset`` instance directly via
    ``from_style_transfer_dataset``.
    """

    def __init__(
        self,
        content_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        from PIL import Image
        from torchvision import transforms

        self.resolution = resolution

        # Build list of paired paths
        if pairs is not None:
            self.pairs = pairs
        elif content_dir is not None and target_dir is not None:
            content_path = Path(content_dir)
            target_path = Path(target_dir)
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
            content_files = sorted(
                p for p in content_path.iterdir() if p.suffix.lower() in exts
            )
            self.pairs = []
            for cf in content_files:
                tf = target_path / cf.name
                if tf.exists():
                    self.pairs.append((str(cf), str(tf)))
            if not self.pairs:
                raise FileNotFoundError(
                    f"No matching image pairs found between {content_dir} and {target_dir}"
                )
        else:
            raise ValueError("Provide either (content_dir, target_dir) or pairs")

        # Build transforms
        aug = []
        if center_crop:
            aug.append(transforms.CenterCrop(resolution))
        aug.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS))
        if random_flip:
            aug.append(transforms.RandomHorizontalFlip())
        aug.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform = transforms.Compose(aug)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from PIL import Image

        content_path, target_path = self.pairs[idx]
        content_img = Image.open(content_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        # Apply deterministic seed so both images get the same random augmentation
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        content = self.transform(content_img)
        torch.manual_seed(seed)
        target = self.transform(target_img)

        return {"input": content, "target": target}

    @classmethod
    def from_style_transfer_dataset(
        cls,
        style_dataset,
        resolution: int = 512,
    ) -> "FluxImg2ImgDataset":
        """Create from an existing StyleTransferDataset.

        This extracts (content, target) tensors from the existing dataset
        and wraps them for FLUX fine-tuning.
        """
        instance = cls.__new__(cls)
        instance.resolution = resolution
        instance._wrapped = style_dataset
        instance.pairs = list(range(len(style_dataset)))
        return instance

    def _getitem_wrapped(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._wrapped[idx]
        content = sample["input"]
        target = sample["target"]

        # Ensure CHW float in [-1, 1]
        for name, tensor in [("input", content), ("target", target)]:
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            tensor = tensor * 2.0 - 1.0
            if name == "input":
                content = tensor
            else:
                target = tensor

        # Resize
        content = F.interpolate(
            content.unsqueeze(0), size=self.resolution, mode="bilinear", align_corners=False
        ).squeeze(0)
        target = F.interpolate(
            target.unsqueeze(0), size=self.resolution, mode="bilinear", align_corners=False
        ).squeeze(0)

        return {"input": content, "target": target}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FluxImg2ImgTrainer:
    """LoRA fine-tuning trainer for FLUX.1-dev image-to-image.

    Training loop:
      1. Encode target image into VAE latents  (z = E(target))
      2. Sample timestep t ~ U[0,1]
      3. Create noisy latents via flow interpolation:  z_t = (1 - t)*noise + t*z
      4. Encode conditioning (source image) via FLUX's img2img pipeline internals
      5. Predict velocity field and compute MSE loss against (z - noise)
      6. Update only LoRA weights
    """

    def __init__(self, config: FluxFineTuneConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_model()

    def _setup_model(self):
        """Load FLUX.1 pipeline and attach LoRA adapters."""
        diffusers = _import_diffusers()
        peft = _import_peft()

        cfg = self.config

        logger.info(f"Loading FLUX.1 pipeline from {cfg.pretrained_model}")

        # Load the FLUX pipeline
        pipe_cls = getattr(diffusers, "FluxImg2ImgPipeline", None)
        if pipe_cls is None:
            pipe_cls = getattr(diffusers, "FluxPipeline")

        self.pipe = pipe_cls.from_pretrained(
            cfg.pretrained_model,
            revision=cfg.revision,
            variant=cfg.variant,
            torch_dtype=self._dtype(),
        )

        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.text_encoder = getattr(self.pipe, "text_encoder", None)
        self.text_encoder_2 = getattr(self.pipe, "text_encoder_2", None)
        self.scheduler = self.pipe.scheduler

        # Freeze everything
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)

        # Gradient checkpointing
        if cfg.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        # xformers
        if cfg.enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.warning("xformers not available, continuing without it")

        # Attach LoRA to the transformer
        lora_config = peft.LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            init_lora_weights="gaussian",
        )
        self.transformer = peft.get_peft_model(self.transformer, lora_config)
        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.transformer.parameters())
        logger.info(
            f"LoRA attached: {trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%)"
        )

        # Move to device
        self.vae.to(self.device)
        self.transformer.to(self.device)
        if self.text_encoder is not None:
            self.text_encoder.to(self.device)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(self.device)

    def _dtype(self) -> torch.dtype:
        mp = self.config.mixed_precision
        if mp == "bf16":
            return torch.bfloat16
        elif mp == "fp16":
            return torch.float16
        return torch.float32

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to VAE latents."""
        images = images.to(device=self.device, dtype=self._dtype())
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def _encode_prompt(self, batch_size: int) -> torch.Tensor:
        """Encode an empty prompt for unconditional img2img fine-tuning.

        For pure image-to-image tasks the text conditioning is a null prompt.
        The source image itself provides the conditioning signal.
        """
        tokenizer = getattr(self.pipe, "tokenizer", None)
        tokenizer_2 = getattr(self.pipe, "tokenizer_2", None)

        # FLUX uses dual text encoders (CLIP + T5)
        prompt_embeds_list = []

        for enc, tok in [
            (self.text_encoder, tokenizer),
            (self.text_encoder_2, tokenizer_2),
        ]:
            if enc is None or tok is None:
                continue
            tokens = tok(
                [""] * batch_size,
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            embeds = enc(tokens)[0]
            prompt_embeds_list.append(embeds)

        if prompt_embeds_list:
            return torch.cat(prompt_embeds_list, dim=-1)
        # Fallback: zero embeddings
        return torch.zeros(batch_size, 1, 1, device=self.device, dtype=self._dtype())

    def _flow_matching_loss(
        self,
        model_pred: torch.Tensor,
        noise: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Rectified flow matching loss: MSE(v_pred, z - noise)."""
        target_velocity = target_latents - noise
        return F.mse_loss(model_pred, target_velocity)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> Dict[str, list]:
        """Run the full fine-tuning loop."""
        cfg = self.config
        torch.manual_seed(cfg.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            drop_last=True,
        )
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )

        # Optimizer (only LoRA params)
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_epsilon,
            weight_decay=cfg.weight_decay,
        )

        # LR scheduler
        total_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation_steps
        warmup_steps = int(total_steps * cfg.lr_warmup_ratio)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.learning_rate * 0.1
        )

        # AMP scaler
        use_amp = cfg.mixed_precision != "no" and self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        amp_dtype = self._dtype()

        # wandb
        if cfg.use_wandb:
            import wandb
            wandb.init(project=cfg.experiment_name, config=vars(cfg))

        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}
        global_step = 0
        best_val_loss = float("inf")

        logger.info(f"Starting FLUX.1 LoRA fine-tuning for {cfg.epochs} epochs")
        logger.info(f"  Train samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"  Val samples:   {len(val_dataset)}")
        logger.info(f"  Batch size:    {cfg.batch_size} x {cfg.gradient_accumulation_steps} accum")
        logger.info(f"  Total steps:   {total_steps}")

        for epoch in range(cfg.epochs):
            # -- Training --
            self.transformer.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
            optimizer.zero_grad()

            for step, batch in enumerate(pbar):
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    loss = self._training_step(batch)
                    loss = loss / cfg.gradient_accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # Warmup: linear ramp then cosine
                    if global_step < warmup_steps:
                        lr_scale = (global_step + 1) / warmup_steps
                        for pg in optimizer.param_groups:
                            pg["lr"] = cfg.learning_rate * lr_scale
                    else:
                        scheduler.step()

                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * cfg.gradient_accumulation_steps
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item() * cfg.gradient_accumulation_steps:.4f}")

                if cfg.use_wandb and global_step % cfg.logging_steps == 0:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item() * cfg.gradient_accumulation_steps,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                    })

            avg_train_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_train_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            # -- Validation --
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader, amp_dtype, use_amp)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_lora("best_lora")

            # Logging
            msg = f"Epoch {epoch + 1}/{cfg.epochs} — train_loss: {avg_train_loss:.4f}"
            if val_loss is not None:
                msg += f" — val_loss: {val_loss:.4f}"
            logger.info(msg)
            print(msg)

            # Periodic checkpoint
            if (epoch + 1) % cfg.save_epochs == 0:
                self._save_lora(f"lora_epoch_{epoch + 1}")

        # Final save
        self._save_lora("final_lora")
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

        return history

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step: encode, add noise, predict, compute loss."""
        content = batch["input"]  # source / conditioning image
        target = batch["target"]  # desired output image

        # Encode target into latent space
        target_latents = self._encode_images(target)

        # Encode source image as conditioning latents
        cond_latents = self._encode_images(content)

        # Sample random timestep t ~ U[0, 1] for each sample
        bsz = target_latents.shape[0]
        t = torch.rand(bsz, device=self.device, dtype=target_latents.dtype)

        # Sample noise
        noise = torch.randn_like(target_latents)

        # Flow interpolation: z_t = (1 - t) * noise + t * target_latents
        t_broadcast = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_broadcast) * noise + t_broadcast * target_latents

        # Encode null prompt (image-conditioned only)
        prompt_embeds = self._encode_prompt(bsz)

        # Predict velocity field
        # FLUX transformer expects (hidden_states, encoder_hidden_states, timestep, ...)
        # We concatenate conditioning latents channel-wise with noisy target latents
        model_input = torch.cat([noisy_latents, cond_latents], dim=1)

        # The transformer call depends on the diffusers version; use the
        # pipeline's internal interface when possible.
        try:
            model_pred = self.transformer(
                hidden_states=model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=t * 1000,  # scheduler expects integer-ish timesteps
                return_dict=False,
            )[0]
        except TypeError:
            # Fallback for simpler transformer signatures
            model_pred = self.transformer(
                model_input,
                prompt_embeds,
                t * 1000,
            )
            if isinstance(model_pred, tuple):
                model_pred = model_pred[0]

        # Trim prediction to match target latent channels if needed
        if model_pred.shape[1] != target_latents.shape[1]:
            model_pred = model_pred[:, :target_latents.shape[1]]

        loss = self._flow_matching_loss(model_pred, noise, target_latents)
        return loss

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        amp_dtype: torch.dtype,
        use_amp: bool,
    ) -> float:
        self.transformer.eval()
        total_loss = 0.0
        n = 0
        for batch in val_loader:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = self._training_step(batch)
            total_loss += loss.item()
            n += 1
        self.transformer.train()
        return total_loss / max(n, 1)

    def _save_lora(self, name: str) -> None:
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        peft = _import_peft()
        self.transformer.save_pretrained(save_dir)

        # Save config
        with open(save_dir / "finetune_config.yaml", "w") as f:
            yaml.dump(vars(self.config), f, default_flow_style=False)

        logger.info(f"LoRA weights saved to {save_dir}")

    def generate(
        self,
        source_image,
        prompt: str = "",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
    ):
        """Generate an image-to-image translation using the fine-tuned model.

        Args:
            source_image: PIL Image or path to source image.
            prompt: Optional text prompt (empty for pure img2img).
            num_inference_steps: Override config default.
            guidance_scale: Override config default.
            strength: Override config default.

        Returns:
            PIL Image result.
        """
        from PIL import Image

        cfg = self.config
        steps = num_inference_steps or cfg.num_inference_steps
        scale = guidance_scale or cfg.guidance_scale
        s = strength or cfg.strength

        if isinstance(source_image, (str, Path)):
            source_image = Image.open(source_image).convert("RGB")

        self.transformer.eval()

        # Use the pipeline for inference
        result = self.pipe(
            prompt=prompt,
            image=source_image,
            num_inference_steps=steps,
            guidance_scale=scale,
            strength=s,
        )
        return result.images[0]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune FLUX.1-dev for image-to-image with LoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument("--content-dir", type=str, help="Directory of source images")
    parser.add_argument("--target-dir", type=str, help="Directory of target images")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, help="LoRA rank")
    parser.add_argument("--resolution", type=int, help="Image resolution")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--use-wandb", action="store_true", help="Log to W&B")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = FluxFineTuneConfig.from_yaml(args.config)
    else:
        config = FluxFineTuneConfig()

    # CLI overrides
    for key in [
        "content_dir", "target_dir", "output_dir", "epochs",
        "batch_size", "learning_rate", "lora_rank", "resolution", "seed",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)
    if args.use_wandb:
        config.use_wandb = True

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build dataset
    if config.content_dir and config.target_dir:
        full_dataset = FluxImg2ImgDataset(
            content_dir=config.content_dir,
            target_dir=config.target_dir,
            resolution=config.resolution,
            center_crop=config.center_crop,
            random_flip=config.random_flip,
        )
    else:
        # Try loading from StyleTransferDataset via project imports
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from weatherflow.data import StyleTransferDataset

        raise ValueError(
            "Please provide --content-dir and --target-dir, or modify this "
            "script to pass your StyleTransferDataset instance to "
            "FluxImg2ImgDataset.from_style_transfer_dataset()."
        )

    # Train / val split
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    logger.info(f"Dataset: {len(full_dataset)} pairs → {train_size} train, {val_size} val")

    # Train
    trainer = FluxImg2ImgTrainer(config)
    history = trainer.train(train_dataset, val_dataset)

    # Print summary
    print("\n" + "=" * 60)
    print("FLUX.1-dev LoRA Fine-Tuning Complete")
    print("=" * 60)
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        print(f"  Best val loss:    {min(history['val_loss']):.4f}")
    print(f"  Output:           {config.output_dir}")
    print(f"  LoRA weights:     {config.output_dir}/final_lora/")
    print("=" * 60)


if __name__ == "__main__":
    main()
