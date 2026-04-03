from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_npz_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_dataloader(
    path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, int]:
    data = load_npz_dataset(path)
    x = torch.from_numpy(data["x"]).float()
    legal_mask = torch.from_numpy(data["legal_mask"]).float()
    policy_index = torch.from_numpy(data["policy_index"]).long()
    value_target = torch.from_numpy(data["value_target"]).float()
    dataset = TensorDataset(x, legal_mask, policy_index, value_target)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(**loader_kwargs)
    input_dim = int(x.shape[1]) if x.ndim == 2 and x.shape[0] >= 0 else 17
    return loader, input_dim
