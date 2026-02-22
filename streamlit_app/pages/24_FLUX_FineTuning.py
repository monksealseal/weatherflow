"""
FLUX.1-dev Image-to-Image LoRA Fine-Tuning

Configure and launch fine-tuning of the FLUX.1-dev rectified flow transformer
(12B params) with LoRA on your paired image dataset.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="FLUX Fine-Tuning - WeatherFlow",
    page_icon="🎨",
    layout="wide",
)

st.markdown("# FLUX.1-dev Image-to-Image Fine-Tuning")
st.markdown(
    "Fine-tune the most advanced open-source image model with LoRA on your "
    "paired dataset. Uses the same rectified flow matching framework as WeatherFlow."
)

st.markdown("---")

# ---------- Configuration form ----------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Model Settings")
    pretrained = st.text_input(
        "Pretrained model",
        value="black-forest-labs/FLUX.1-dev",
        help="HuggingFace model ID for the base FLUX checkpoint",
    )
    lora_rank = st.select_slider("LoRA rank", options=[4, 8, 16, 32, 64], value=16)
    lora_alpha = st.select_slider("LoRA alpha", options=[4, 8, 16, 32, 64], value=16)
    mixed_precision = st.selectbox("Mixed precision", ["fp16", "bf16", "no"], index=0)

with col_right:
    st.markdown("### Training Settings")
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=20)
    batch_size = st.number_input("Batch size", min_value=1, max_value=32, value=1)
    grad_accum = st.number_input("Gradient accumulation steps", min_value=1, max_value=64, value=4)
    learning_rate = st.number_input(
        "Learning rate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.1e"
    )
    resolution = st.select_slider("Image resolution", options=[256, 512, 768, 1024], value=512)

st.markdown("---")

st.markdown("### Dataset")
col_c, col_t = st.columns(2)
with col_c:
    content_dir = st.text_input(
        "Source images directory",
        placeholder="/data/satellite_images",
    )
with col_t:
    target_dir = st.text_input(
        "Target images directory",
        placeholder="/data/wind_fields",
    )

use_existing = st.checkbox(
    "Use existing StyleTransferDataset from session",
    value=False,
    help="If you already loaded a dataset via the Data Manager, check this box.",
)

st.markdown("---")

# ---------- Generate config / launch ----------
col_gen, col_run = st.columns(2)

with col_gen:
    if st.button("Generate Config YAML", use_container_width=True):
        import yaml

        config = {
            "pretrained_model": pretrained,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "mixed_precision": mixed_precision,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": learning_rate,
            "resolution": resolution,
            "content_dir": content_dir or None,
            "target_dir": target_dir or None,
            "output_dir": "./flux_ft_output",
        }
        yaml_str = yaml.dump(config, default_flow_style=False)
        st.code(yaml_str, language="yaml")
        st.download_button(
            "Download config.yaml",
            data=yaml_str,
            file_name="flux_finetune_config.yaml",
            mime="text/yaml",
        )

with col_run:
    st.markdown("**Launch training** (requires GPU)")
    effective_bs = batch_size * grad_accum
    st.markdown(f"Effective batch size: **{effective_bs}**")
    st.markdown(f"Trainable params: ~**{lora_rank * 2 * 4 / 1000:.1f}K** per layer")

    if st.button("Start Fine-Tuning", type="primary", use_container_width=True):
        import torch

        if not torch.cuda.is_available():
            st.error("CUDA GPU required for FLUX fine-tuning. No GPU detected.")
        elif not content_dir or not target_dir:
            st.error("Please provide both source and target image directories.")
        else:
            st.info(
                "To run training, execute in your terminal:\n\n"
                "```bash\n"
                f"python examples/flux_img2img/finetune_flux_img2img.py \\\n"
                f"    --content-dir {content_dir} \\\n"
                f"    --target-dir {target_dir} \\\n"
                f"    --epochs {epochs} \\\n"
                f"    --batch-size {batch_size} \\\n"
                f"    --learning-rate {learning_rate} \\\n"
                f"    --lora-rank {lora_rank} \\\n"
                f"    --resolution {resolution}\n"
                "```"
            )

st.markdown("---")

# ---------- Architecture info ----------
with st.expander("About FLUX.1-dev"):
    st.markdown("""
**FLUX.1-dev** by Black Forest Labs is a 12-billion parameter rectified flow
transformer — the most advanced open-source image model available.

**Key properties:**
- Uses the same **rectified flow matching** framework that WeatherFlow is built on
- **12B parameters** — fine-tuned efficiently with LoRA (only ~0.1% of params trained)
- Supports **image-to-image** translation with conditioning
- Native **fp16/bf16** mixed precision for memory efficiency

**LoRA fine-tuning** attaches lightweight rank-decomposed adapters to the
transformer's attention layers, enabling high-quality domain adaptation on a
single GPU with as little as 16 GB VRAM.

**How it connects to your GAN workflow:**
Your existing `StyleTransferDataset` provides paired (content, target) images —
the same data format used to train pix2pix and CycleGAN models.  FLUX replaces
the adversarial objective with flow matching, giving more stable training and
higher-quality outputs.
""")

# Sidebar
with st.sidebar:
    st.markdown("### FLUX Fine-Tuning")
    st.markdown("Fine-tune FLUX.1-dev for image-to-image tasks using LoRA.")
    st.markdown("---")
    st.markdown("**Requirements:**")
    st.markdown("- `diffusers >= 0.31.0`")
    st.markdown("- `peft >= 0.13.0`")
    st.markdown("- `accelerate >= 1.0.0`")
    st.markdown("- CUDA GPU with 16+ GB VRAM")
