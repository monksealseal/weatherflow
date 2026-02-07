from typing import Dict, Iterable, Optional, Sequence

import torch

from .era5 import ERA5Dataset, _coerce_levels, _coerce_years


class MultiStepERA5Dataset(ERA5Dataset):
    """
    Multi-step ERA5 dataset that returns a context window and future targets.

    This implementation reuses the normalization and loading logic from
    ``ERA5Dataset`` while providing temporally contiguous context/target
    slices for sequence modelling.

    By default, uses a persistent cache directory (~/.weatherflow/datasets/era5/)
    so you don't need to re-download data every time you log in.
    """

    def __init__(
        self,
        years: Iterable[int],
        variables: Sequence[str],
        levels: Iterable[int],
        root_dir: Optional[str] = None,
        context_length: int = 4,
        pred_length: int = 4,
        stride: int = 1,
        download: bool = False,
    ):
        self.context_length = int(context_length)
        self.pred_length = int(pred_length)
        self.stride = int(stride)

        # Persist level metadata for compatibility with callers expecting
        # ``pressure_levels`` in returned metadata.
        self.pressure_levels = _coerce_levels(levels)
        self.years_seq = _coerce_years(years)

        super().__init__(
            years=self.years_seq,
            variables=variables,
            levels=self.pressure_levels,
            root_dir=root_dir,
            download=download,
        )

        # Cache time coordinate for quick indexing
        self.times = self.ds.time

    def __len__(self) -> int:
        total_steps = len(self.times)
        sequence_len = self.context_length + self.pred_length
        if total_steps < sequence_len:
            return 0
        return (total_steps - sequence_len) // self.stride + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.context_length + self.pred_length
        if end > len(self.times):
            raise IndexError("Index out of range for requested sequence window.")

        # Reuse base-class normalization by pulling each timestep via the parent __getitem__
        _base_getitem = super().__getitem__
        slices = [_base_getitem(t_idx) for t_idx in range(start, end)]
        sequence = torch.stack(slices, dim=0)  # [T, V, L, H, W]

        context = sequence[: self.context_length]
        target = sequence[self.context_length :]

        times = self.times[start:end].values

        return {
            "context": context,
            "target": target,
            "metadata": {
                "t_start": times[0],
                "t_end": times[-1],
                "variables": self.variables,
                "pressure_levels": self.pressure_levels,
                "context_length": self.context_length,
                "pred_length": self.pred_length,
            },
        }
