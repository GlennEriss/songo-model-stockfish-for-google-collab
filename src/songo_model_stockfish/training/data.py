from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_npz_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    count = int(arrays["x"].shape[0]) if "x" in arrays and arrays["x"].ndim >= 1 else 0
    if "policy_target_full" not in arrays:
        policy_index = arrays.get("policy_index", np.zeros((count,), dtype=np.int64))
        full = np.zeros((count, 7), dtype=np.float32)
        if count > 0:
            rows = np.arange(count, dtype=np.int64)
            valid = np.logical_and(policy_index >= 0, policy_index < 7)
            full[rows[valid], policy_index[valid]] = 1.0
        arrays["policy_target_full"] = full
    for key in ("capture_move_mask", "safe_move_mask", "risky_move_mask"):
        if key not in arrays:
            arrays[key] = np.zeros((count, 7), dtype=np.float32)
    return arrays


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
    policy_target_full = torch.from_numpy(data["policy_target_full"]).float()
    value_target = torch.from_numpy(data["value_target"]).float()
    capture_move_mask = torch.from_numpy(data["capture_move_mask"]).float()
    safe_move_mask = torch.from_numpy(data["safe_move_mask"]).float()
    risky_move_mask = torch.from_numpy(data["risky_move_mask"]).float()
    dataset = TensorDataset(
        x,
        legal_mask,
        policy_index,
        policy_target_full,
        value_target,
        capture_move_mask,
        safe_move_mask,
        risky_move_mask,
    )
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
