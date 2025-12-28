import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds across common libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rank_zero_only(is_rank_zero: bool) -> bool:
    """Utility guard for logging in distributed contexts."""
    return bool(is_rank_zero)


def get_world_rank() -> int:
    """Best-effort fetch of world rank for DDP contexts."""
    return int(os.environ.get("RANK", 0))
