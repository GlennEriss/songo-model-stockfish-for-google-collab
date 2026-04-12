from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


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
    if "hard_example_weight" not in arrays:
        arrays["hard_example_weight"] = np.ones((count,), dtype=np.float32)
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
    weighted_sampling: bool = False,
    weighted_sampling_exponent: float = 1.0,
    weighted_sampling_min_weight: float = 1.0,
    weighted_sampling_max_weight: float = 5.0,
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
    hard_example_weight_np = np.asarray(data.get("hard_example_weight", np.ones((x.shape[0],), dtype=np.float32)), dtype=np.float32)
    if hard_example_weight_np.shape[0] != int(x.shape[0]):
        hard_example_weight_np = np.ones((int(x.shape[0]),), dtype=np.float32)
    hard_example_weight_np = np.where(np.isfinite(hard_example_weight_np), hard_example_weight_np, 1.0).astype(np.float32)
    min_w = max(1e-6, float(weighted_sampling_min_weight))
    max_w = max(min_w, float(weighted_sampling_max_weight))
    exponent = max(0.0, float(weighted_sampling_exponent))
    hard_example_weight_np = np.clip(hard_example_weight_np, min_w, max_w)
    if exponent != 1.0:
        hard_example_weight_np = np.power(hard_example_weight_np, exponent).astype(np.float32)
        hard_example_weight_np = np.clip(hard_example_weight_np, min_w, max_w)

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
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
    }
    sampler = None
    if bool(weighted_sampling) and bool(shuffle) and int(x.shape[0]) > 0:
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(hard_example_weight_np.astype(np.float64)),
            num_samples=int(x.shape[0]),
            replacement=True,
        )
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = bool(shuffle)
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(**loader_kwargs)
    loader.songo_sampling_metadata = {
        "weighted_sampling_enabled": bool(sampler is not None),
        "requested_weighted_sampling": bool(weighted_sampling),
        "samples": int(x.shape[0]),
        "hard_example_weight_min": float(np.min(hard_example_weight_np)) if int(x.shape[0]) > 0 else 1.0,
        "hard_example_weight_max": float(np.max(hard_example_weight_np)) if int(x.shape[0]) > 0 else 1.0,
        "hard_example_weight_mean": float(np.mean(hard_example_weight_np)) if int(x.shape[0]) > 0 else 1.0,
        "hard_example_weight_exponent": float(exponent),
    }
    input_dim = int(x.shape[1]) if x.ndim == 2 and x.shape[0] >= 0 else 17
    return loader, input_dim
