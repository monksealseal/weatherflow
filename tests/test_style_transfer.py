import numpy as np
import torch
from PIL import Image

from weatherflow.data import StyleTransferDataset
from weatherflow.models import StyleFlowMatch


def test_style_transfer_dataset_accepts_multiple_modalities(tmp_path):
    content_np = (np.ones((4, 4, 3)) * 127).astype(np.uint8)
    target_tensor = torch.rand(3, 4, 4)

    style_img = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    style_path = tmp_path / "style.png"
    style_img.save(style_path)

    dataset = StyleTransferDataset(
        content_items=[content_np, target_tensor],
        target_items=[target_tensor, content_np],
        style_items=[style_path, style_img],
    )

    sample0 = dataset[0]
    assert set(sample0.keys()) == {"input", "target", "style", "metadata"}
    assert sample0["input"].shape == (3, 4, 4)
    assert sample0["target"].shape == (3, 4, 4)
    assert torch.isclose(sample0["style"].max(), torch.tensor(1.0))


def test_style_flow_match_forward_supports_style_conditioning():
    model = StyleFlowMatch(
        input_channels=3,
        style_channels=3,
        hidden_dim=32,
        n_layers=1,
        use_attention=False,
        physics_informed=False,
    )

    x = torch.randn(2, 3, 8, 8)
    style = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)

    output = model(x, t, style=style)
    assert output.shape == x.shape
