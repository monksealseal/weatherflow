import logging
from typing import Iterable, List

import numpy as np
import torch
import xarray as xr
from torch.utils.data import IterableDataset, get_worker_info


class StreamingERA5Dataset(IterableDataset):
    """
    Streams ERA5 data sequentially instead of random access.
    Optimized for high-throughput GPU training.
    """

    def __init__(
        self,
        data_path: str,
        variables: List[str] = ['z', 't', 'u', 'v'],
        pressure_levels: List[int] = [500],
        chunk_size: int = 100,
    ):
        self.data_path = data_path
        self.variables = variables
        self.pressure_levels = pressure_levels
        self.chunk_size = chunk_size
        self.ds = None  # Lazy init

    def _init_dataset(self) -> None:
        if self.ds is None:
            self.ds = xr.open_zarr(
                self.data_path,
                chunks=None,  # disable dask chunking for streaming
                consolidated=True,
            )

    def __iter__(self) -> Iterable:
        self._init_dataset()
        total_steps = len(self.ds.time)

        worker_info = get_worker_info()
        if worker_info is None:
            iter_start, iter_end = 0, total_steps
        else:
            per_worker = int(np.ceil(total_steps / worker_info.num_workers))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, total_steps)

        current_step = iter_start
        # Track the last sample from previous chunk to handle chunk boundary transitions
        prev_chunk_last_sample = None

        while current_step < iter_end - 1:
            chunk_end = min(current_step + self.chunk_size, iter_end)
            time_slice = slice(current_step, chunk_end)

            chunk_data = {}
            try:
                for var in self.variables:
                    chunk_data[var] = self.ds[var].isel(
                        time=time_slice,
                        level=self.ds.level.isin(self.pressure_levels),
                    ).values

                chunk_len = chunk_data[self.variables[0]].shape[0]

                # Handle chunk boundary: yield transition from previous chunk's last sample
                # to this chunk's first sample
                if prev_chunk_last_sample is not None and chunk_len > 0:
                    x0 = prev_chunk_last_sample
                    x1 = []
                    for var in self.variables:
                        x1.append(chunk_data[var][0])
                    x1_tensor = torch.tensor(np.stack(x1), dtype=torch.float32)
                    yield x0, x1_tensor

                # Process pairs within the current chunk
                for i in range(chunk_len - 1):
                    x0 = []
                    x1 = []
                    for var in self.variables:
                        v_data = chunk_data[var]
                        x0.append(v_data[i])
                        x1.append(v_data[i + 1])

                    x0_tensor = torch.tensor(np.stack(x0), dtype=torch.float32)
                    x1_tensor = torch.tensor(np.stack(x1), dtype=torch.float32)
                    yield x0_tensor, x1_tensor

                # Store last sample for chunk boundary handling
                if chunk_len > 0:
                    last_sample = []
                    for var in self.variables:
                        last_sample.append(chunk_data[var][-1])
                    prev_chunk_last_sample = torch.tensor(np.stack(last_sample), dtype=torch.float32)

            except Exception as e:
                logging.error(f"Error streaming chunk {current_step}: {e}")
                prev_chunk_last_sample = None  # Reset on error to avoid stale data

            current_step += self.chunk_size
