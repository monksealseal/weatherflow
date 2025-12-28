from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .era5 import ERA5Dataset


class MultiStepERA5Dataset(ERA5Dataset):
    """
    Multi-step ERA5 dataset that returns a context window and future targets.

    This enables training on sequences (e.g., context_length=24, pred_length=24)
    instead of single-step t,t+1 pairs.
    """

    def __init__(
        self,
        variables: List[str] = ['z', 't', 'u', 'v'],
        pressure_levels: List[int] = [500],
        data_path: Optional[str] = None,
        time_slice: Union[slice, str, Tuple[str, str]] = slice('2015', '2016'),
        normalize: bool = True,
        add_physics_features: bool = False,
        cache_data: bool = False,
        verbose: bool = True,
        stats_path: Optional[Union[str, str]] = None,
        auto_compute_stats: bool = False,
        local_cache_dir: Optional[Union[str, str]] = None,
        context_length: int = 4,
        pred_length: int = 4,
        stride: int = 1,
    ):
        self.context_length = context_length
        self.pred_length = pred_length
        self.stride = stride
        super().__init__(
            variables=variables,
            pressure_levels=pressure_levels,
            data_path=data_path,
            time_slice=time_slice,
            normalize=normalize,
            add_physics_features=add_physics_features,
            cache_data=cache_data,
            verbose=verbose,
            stats_path=stats_path,
            auto_compute_stats=auto_compute_stats,
            local_cache_dir=local_cache_dir,
        )

    def __len__(self) -> int:
        return max(
            0, (len(self.times) - (self.context_length + self.pred_length)) // self.stride + 1
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.context_length + self.pred_length

        # Collect slices for all requested timesteps
        times = self.times[start:end].values
        data_seq = {var: [] for var in self.variables}

        for t_val in times:
            for var in self.variables:
                try:
                    if var in self.ds:
                        if 'level' in self.ds[var].dims:
                            arr = self.ds[var].sel(time=t_val, level=self.pressure_levels).values
                        else:
                            arr = self.ds[var].sel(time=t_val).values
                    else:
                        shape = (len(self.pressure_levels), self.ds.latitude.size, self.ds.longitude.size)
                        arr = np.zeros(shape)

                    if self.normalize and var in self.normalize_stats:
                        stats = self.normalize_stats[var]
                        arr = (arr - stats['mean']) / stats['std']
                except Exception:
                    shape = (len(self.pressure_levels), self.ds.latitude.size, self.ds.longitude.size)
                    arr = np.zeros(shape)

                data_seq[var].append(arr)

        # Stack into tensors: [time, vars, levels, lat, lon]
        seq = np.stack(
            [np.stack(data_seq[var]) for var in self.variables], axis=1
        )
        seq_tensor = torch.tensor(seq).float()

        context = seq_tensor[: self.context_length]  # [T_ctx, V, L, H, W]
        target = seq_tensor[self.context_length :]  # [T_pred, V, L, H, W]

        return {
            'context': context,
            'target': target,
            'metadata': {
                't_start': times[0],
                't_end': times[-1],
                'variables': self.variables,
                'pressure_levels': self.pressure_levels,
                'context_length': self.context_length,
                'pred_length': self.pred_length,
            },
        }
