from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from songo_model_stockfish.ops.job import JobContext
from songo_model_stockfish.ops.logging import utc_now_iso
from songo_model_stockfish.ops.model_registry import next_model_version, promote_best_model, promoted_best_dir, upsert_model_record
from songo_model_stockfish.training.data import build_dataloader
from songo_model_stockfish.training.model import PolicyValueMLP


def _resolve_storage_path(base: Path, configured: str | None, fallback: Path) -> Path:
    if not configured:
        return fallback
    path = Path(configured)
    if path.is_absolute():
        return path
    return base / path


def _masked_policy_logits(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    # Mixed precision safe value for fp16/bf16/fp32.
    mask_value = torch.finfo(policy_logits.dtype).min
    return policy_logits.masked_fill(legal_mask <= 0, mask_value)


def _masked_policy_loss(policy_logits: torch.Tensor, legal_mask: torch.Tensor, policy_index: torch.Tensor) -> torch.Tensor:
    masked_logits = _masked_policy_logits(policy_logits, legal_mask)
    return F.cross_entropy(masked_logits, policy_index)


def _run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
    value_loss_weight: float = 1.0,
    on_batch_end: Callable[[dict[str, float | int | str]], None] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_correct = 0
    total_count = 0

    total_batches = len(loader)
    stage = "train" if training else "validation"

    for batch_index, (x, legal_mask, policy_index, value_target) in enumerate(loader, start=1):
        x = x.to(device)
        legal_mask = legal_mask.to(device)
        policy_index = policy_index.to(device)
        value_target = value_target.to(device)

        with autocast(device_type=device.type, enabled=amp_enabled):
            policy_logits, value_pred = model(x)
            loss_policy = _masked_policy_loss(policy_logits, legal_mask, policy_index)
            loss_value = F.mse_loss(value_pred, value_target)
            loss_total = loss_policy + (value_loss_weight * loss_value)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and amp_enabled:
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                optimizer.step()

        preds = _masked_policy_logits(policy_logits, legal_mask).argmax(dim=1)
        total_correct += int((preds == policy_index).sum().item())
        batch_count = int(x.shape[0])
        total_count += batch_count
        total_loss += float(loss_total.item()) * batch_count
        total_policy += float(loss_policy.item()) * batch_count
        total_value += float(loss_value.item()) * batch_count

        if on_batch_end is not None:
            on_batch_end(
                {
                    "stage": stage,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "examples_in_batch": batch_count,
                    "examples_seen": total_count,
                    "loss_total": float(loss_total.item()),
                    "loss_policy": float(loss_policy.item()),
                    "loss_value": float(loss_value.item()),
                }
            )

    denom = max(1, total_count)
    return {
        "loss_total": total_loss / denom,
        "loss_policy": total_policy / denom,
        "loss_value": total_value / denom,
        "policy_accuracy": total_correct / denom,
        "examples": total_count,
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def run_train(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("train", {})
    runtime_cfg = job.config.get("runtime", {})
    dataset_id = str(cfg.get("dataset_id", "dataset_v1"))
    dataset_path = _resolve_storage_path(job.paths.drive_root, cfg.get("dataset_path"), job.job_dir / "train.npz")
    validation_path = _resolve_storage_path(job.paths.drive_root, cfg.get("validation_path"), job.job_dir / "validation.npz")
    checkpoint_dir = _resolve_storage_path(job.paths.drive_root, cfg.get("checkpoint_dir"), job.job_dir / "checkpoints")
    final_dir = _resolve_storage_path(job.paths.drive_root, cfg.get("final_dir"), job.job_dir / "final")
    init_checkpoint_path_cfg = cfg.get("init_checkpoint_path")
    init_checkpoint_path = _resolve_storage_path(job.paths.drive_root, init_checkpoint_path_cfg, final_dir / "parent_model.pt") if init_checkpoint_path_cfg else None
    lineage_dir = _resolve_storage_path(job.paths.drive_root, cfg.get("lineage_dir"), job.paths.models_root / "lineage")
    promoted_best_checkpoint_path = _resolve_storage_path(
        job.paths.drive_root,
        cfg.get("promoted_best_checkpoint_path"),
        promoted_best_dir(job.paths.models_root) / "model.pt",
    )
    init_from_promoted_best = bool(cfg.get("init_from_promoted_best", True))
    model_id = str(cfg.get("model_id", "auto"))
    model_id_prefix = str(cfg.get("model_id_prefix", "songo_policy_value_colab_pro"))
    if model_id in {"", "auto"}:
        model_id = f"{model_id_prefix}_v{next_model_version(job.paths.models_root, model_id_prefix)}"
    hidden_sizes = list(cfg.get("hidden_sizes", [256, 256, 128]))
    batch_size = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("epochs", 20))
    learning_rate = float(cfg.get("learning_rate", 0.0003))
    value_loss_weight = float(cfg.get("value_loss_weight", 1.0))
    log_every_n_batches = max(1, int(cfg.get("log_every_n_batches", 1)))
    requested_device = str(runtime_cfg.get("device", "cpu"))
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    num_workers = int(runtime_cfg.get("num_workers", 0))
    pin_memory = bool(runtime_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(runtime_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = runtime_cfg.get("prefetch_factor")
    amp_enabled = bool(runtime_cfg.get("mixed_precision", False) and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, input_dim = build_dataloader(
        dataset_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    validation_loader, _ = build_dataloader(
        validation_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model = PolicyValueMLP(input_dim=input_dim, hidden_sizes=hidden_sizes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    state = job.read_state()
    start_epoch = int(state.get("epoch", 0))
    global_step = int(state.get("global_step", 0))
    best_metric = float(state.get("best_metric", float("-inf")))
    if best_metric == float("-inf"):
        best_metric = 0.0
    checkpoint_path_value = state.get("checkpoint_path")
    parent_checkpoint_snapshot_path = None
    if checkpoint_path_value:
        checkpoint_path = Path(str(checkpoint_path_value))
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = int(checkpoint.get("epoch", start_epoch))
            global_step = int(checkpoint.get("global_step", global_step))
            best_metric = float(checkpoint.get("best_metric", best_metric))
    elif init_checkpoint_path is None and init_from_promoted_best and promoted_best_checkpoint_path.exists():
        init_checkpoint_path = promoted_best_checkpoint_path
    elif init_checkpoint_path:
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"init_checkpoint_path introuvable: {init_checkpoint_path}")
        parent_checkpoint = torch.load(init_checkpoint_path, map_location=device)
        parent_model_config = parent_checkpoint.get("model_config", {})
        parent_hidden_sizes = list(parent_model_config.get("hidden_sizes", hidden_sizes))
        parent_input_dim = int(parent_model_config.get("input_dim", input_dim))
        parent_policy_dim = int(parent_model_config.get("policy_dim", 7))
        if parent_hidden_sizes != hidden_sizes or parent_input_dim != input_dim or parent_policy_dim != 7:
            raise ValueError(
                "Le checkpoint parent n'est pas compatible avec la config d'entrainement actuelle "
                f"(parent hidden_sizes={parent_hidden_sizes}, current hidden_sizes={hidden_sizes})"
            )
        model.load_state_dict(parent_checkpoint["model_state"])
        lineage_dir.mkdir(parents=True, exist_ok=True)
        parent_checkpoint_snapshot_path = lineage_dir / f"{model_id}_parent_snapshot.pt"
        shutil.copy2(init_checkpoint_path, parent_checkpoint_snapshot_path)
        best_metric = 0.0
        job.logger.info(
            "training init checkpoint loaded | model=%s | parent_checkpoint=%s",
            model_id,
            init_checkpoint_path,
        )
        job.write_event(
            "train_init_checkpoint_loaded",
            model_id=model_id,
            init_checkpoint_path=str(init_checkpoint_path),
            parent_snapshot_path=str(parent_checkpoint_snapshot_path),
        )

    if start_epoch > 0:
        job.logger.info(
            "training resume detected | dataset=%s | model=%s | next_epoch=%s/%s | checkpoint=%s",
            dataset_id,
            model_id,
            start_epoch + 1,
            epochs,
            checkpoint_path_value,
        )
        job.write_event(
            "train_resume_detected",
            dataset_id=dataset_id,
            model_id=model_id,
            resume_from_epoch=start_epoch,
            total_epochs=epochs,
            checkpoint_path=str(checkpoint_path_value),
        )

    job.logger.info(
        "training started | dataset=%s | model=%s | device=%s | mixed_precision=%s | epochs=%s | batch_size=%s | train_examples=%s | validation_examples=%s",
        dataset_id,
        model_id,
        device,
        amp_enabled,
        epochs,
        batch_size,
        len(train_loader.dataset),
        len(validation_loader.dataset),
    )
    job.set_phase("training")
    job.write_event(
        "train_started",
        dataset_id=dataset_id,
        model_id=model_id,
        init_checkpoint_path=str(init_checkpoint_path) if init_checkpoint_path else "",
        init_from_promoted_best=init_from_promoted_best,
        promoted_best_checkpoint_path=str(promoted_best_checkpoint_path),
        device=str(device),
        mixed_precision=amp_enabled,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        train_examples=len(train_loader.dataset),
        validation_examples=len(validation_loader.dataset),
        resume_granularity="epoch",
    )
    job.write_metric(
        {
            "metric_type": "train_started",
            "dataset_id": dataset_id,
            "model_id": model_id,
            "train_examples": len(train_loader.dataset),
            "validation_examples": len(validation_loader.dataset),
            "mixed_precision": amp_enabled,
        }
    )

    best_checkpoint_path = checkpoint_dir / f"{model_id}_best.pt"
    last_checkpoint_path = checkpoint_dir / f"{model_id}_last.pt"

    history: list[dict[str, float | int]] = []
    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        job.set_phase(f"training:epoch_{epoch_number:03d}")
        job.logger.info("training epoch started | epoch=%s/%s", epoch_number, epochs)
        job.write_event(
            "train_epoch_started",
            epoch=epoch_number,
            total_epochs=epochs,
            train_batches=len(train_loader),
            validation_batches=len(validation_loader),
        )
        job.write_state(
            {
                "epoch": epoch,
                "epochs_total": epochs,
                "stage": "train",
                "batch_index": 0,
                "total_batches": len(train_loader),
                "global_step": global_step,
                "best_metric": best_metric,
                "last_completed_phase": "epoch_initialized",
                "checkpoint_path": str(last_checkpoint_path),
            }
        )
        job.write_status(
            "running",
            phase=f"epoch_{epoch_number:03d}_train",
            extra={
                "dataset_id": dataset_id,
                "model_id": model_id,
                "resume_granularity": "epoch",
                "current_epoch": epoch_number,
                "total_epochs": epochs,
                "last_checkpoint_path": str(last_checkpoint_path),
                "last_state_path": str(job.state_path),
            },
        )

        def on_train_batch_end(batch_payload: dict[str, float | int | str]) -> None:
            batch_index = int(batch_payload["batch_index"])
            total_batches = int(batch_payload["total_batches"])
            if batch_index == 1 or batch_index == total_batches or batch_index % log_every_n_batches == 0:
                job.logger.info(
                    "training batch | epoch=%s/%s | stage=train | batch=%s/%s | seen=%s | loss=%.4f | policy=%.4f | value=%.4f",
                    epoch_number,
                    epochs,
                    batch_index,
                    total_batches,
                    int(batch_payload["examples_seen"]),
                    float(batch_payload["loss_total"]),
                    float(batch_payload["loss_policy"]),
                    float(batch_payload["loss_value"]),
                )
                job.write_event(
                    "train_batch_progress",
                    epoch=epoch_number,
                    total_epochs=epochs,
                    stage="train",
                    batch_index=batch_index,
                    total_batches=total_batches,
                    examples_seen=int(batch_payload["examples_seen"]),
                    loss_total=float(batch_payload["loss_total"]),
                    loss_policy=float(batch_payload["loss_policy"]),
                    loss_value=float(batch_payload["loss_value"]),
                )
                job.write_status(
                    "running",
                    phase=f"epoch_{epoch_number:03d}_train",
                    extra={
                        "dataset_id": dataset_id,
                        "model_id": model_id,
                        "resume_granularity": "epoch",
                        "current_epoch": epoch_number,
                        "total_epochs": epochs,
                        "stage": "train",
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "last_checkpoint_path": str(last_checkpoint_path),
                        "last_state_path": str(job.state_path),
                    },
                )
                job.write_state(
                    {
                        "epoch": epoch,
                        "epochs_total": epochs,
                        "stage": "train",
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "global_step": global_step,
                        "best_metric": best_metric,
                        "last_completed_phase": "train_batch_progress",
                        "checkpoint_path": str(last_checkpoint_path),
                    }
                )

        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            value_loss_weight=value_loss_weight,
            on_batch_end=on_train_batch_end,
        )
        job.logger.info(
            "training stage completed | epoch=%s/%s | stage=train | loss=%.4f | acc=%.4f",
            epoch_number,
            epochs,
            train_metrics["loss_total"],
            train_metrics["policy_accuracy"],
        )
        job.write_status(
            "running",
            phase=f"epoch_{epoch_number:03d}_validation",
            extra={
                "dataset_id": dataset_id,
                "model_id": model_id,
                "resume_granularity": "epoch",
                "current_epoch": epoch_number,
                "total_epochs": epochs,
                "stage": "validation",
                "batch_index": 0,
                "total_batches": len(validation_loader),
                "last_checkpoint_path": str(last_checkpoint_path),
                "last_state_path": str(job.state_path),
            },
        )

        def on_validation_batch_end(batch_payload: dict[str, float | int | str]) -> None:
            batch_index = int(batch_payload["batch_index"])
            total_batches = int(batch_payload["total_batches"])
            if batch_index == 1 or batch_index == total_batches or batch_index % log_every_n_batches == 0:
                job.logger.info(
                    "training batch | epoch=%s/%s | stage=validation | batch=%s/%s | seen=%s | loss=%.4f | policy=%.4f | value=%.4f",
                    epoch_number,
                    epochs,
                    batch_index,
                    total_batches,
                    int(batch_payload["examples_seen"]),
                    float(batch_payload["loss_total"]),
                    float(batch_payload["loss_policy"]),
                    float(batch_payload["loss_value"]),
                )
                job.write_event(
                    "train_batch_progress",
                    epoch=epoch_number,
                    total_epochs=epochs,
                    stage="validation",
                    batch_index=batch_index,
                    total_batches=total_batches,
                    examples_seen=int(batch_payload["examples_seen"]),
                    loss_total=float(batch_payload["loss_total"]),
                    loss_policy=float(batch_payload["loss_policy"]),
                    loss_value=float(batch_payload["loss_value"]),
                )
                job.write_status(
                    "running",
                    phase=f"epoch_{epoch_number:03d}_validation",
                    extra={
                        "dataset_id": dataset_id,
                        "model_id": model_id,
                        "resume_granularity": "epoch",
                        "current_epoch": epoch_number,
                        "total_epochs": epochs,
                        "stage": "validation",
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "last_checkpoint_path": str(last_checkpoint_path),
                        "last_state_path": str(job.state_path),
                    },
                )
                job.write_state(
                    {
                        "epoch": epoch,
                        "epochs_total": epochs,
                        "stage": "validation",
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "global_step": global_step,
                        "best_metric": best_metric,
                        "last_completed_phase": "validation_batch_progress",
                        "checkpoint_path": str(last_checkpoint_path),
                    }
                )

        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
            value_loss_weight=value_loss_weight,
            on_batch_end=on_validation_batch_end,
        )
        global_step += int(train_metrics["examples"])

        epoch_payload = {
            "epoch": epoch_number,
            "train_loss_total": train_metrics["loss_total"],
            "train_loss_policy": train_metrics["loss_policy"],
            "train_loss_value": train_metrics["loss_value"],
            "train_policy_accuracy": train_metrics["policy_accuracy"],
            "validation_loss_total": validation_metrics["loss_total"],
            "validation_loss_policy": validation_metrics["loss_policy"],
            "validation_loss_value": validation_metrics["loss_value"],
            "validation_policy_accuracy": validation_metrics["policy_accuracy"],
        }
        history.append(epoch_payload)
        job.write_metric({"metric_type": "train_epoch", **epoch_payload})
        job.write_event("train_epoch_completed", total_epochs=epochs, **epoch_payload)
        job.logger.info(
            "training epoch completed | epoch=%s/%s | train_loss=%.4f | val_loss=%.4f | train_acc=%.4f | val_acc=%.4f",
            epoch_number,
            epochs,
            train_metrics["loss_total"],
            validation_metrics["loss_total"],
            train_metrics["policy_accuracy"],
            validation_metrics["policy_accuracy"],
        )

        checkpoint_payload = {
            "epoch": epoch_number,
            "global_step": global_step,
            "best_metric": best_metric,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "amp_enabled": amp_enabled,
            "model_config": {
                "input_dim": input_dim,
                "hidden_sizes": hidden_sizes,
                "policy_dim": 7,
            },
        }
        _save_checkpoint(last_checkpoint_path, checkpoint_payload)
        job.write_event(
            "last_checkpoint_updated",
            epoch=epoch_number,
            checkpoint_path=str(last_checkpoint_path),
        )

        validation_score = float(validation_metrics["policy_accuracy"])
        if validation_score >= best_metric:
            best_metric = validation_score
            checkpoint_payload["best_metric"] = best_metric
            _save_checkpoint(best_checkpoint_path, checkpoint_payload)
            job.write_event("best_checkpoint_updated", epoch=epoch_number, checkpoint_path=str(best_checkpoint_path))

        job.write_state(
            {
                "epoch": epoch_number,
                "global_step": global_step,
                "best_metric": best_metric,
                "last_completed_phase": "validation",
                "checkpoint_path": str(last_checkpoint_path),
            }
        )
        job.write_status(
            "running",
            phase=f"epoch_{epoch_number:03d}_validation",
            extra={
                "dataset_id": dataset_id,
                "last_checkpoint_path": str(last_checkpoint_path),
                "last_state_path": str(job.state_path),
            },
        )

    final_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_dir / f"{model_id}.pt"
    final_payload = {
        "model_state": model.state_dict(),
        "model_config": {
            "input_dim": input_dim,
            "hidden_sizes": hidden_sizes,
            "policy_dim": 7,
        },
        "dataset_id": dataset_id,
        "best_metric": best_metric,
        "amp_enabled": amp_enabled,
    }
    torch.save(final_payload, final_model_path)

    training_summary = {
        "job_id": job.job_id,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path else "",
        "init_from_promoted_best": init_from_promoted_best,
        "promoted_best_checkpoint_path": str(promoted_best_checkpoint_path),
        "parent_checkpoint_snapshot_path": str(parent_checkpoint_snapshot_path) if parent_checkpoint_snapshot_path else "",
        "device": str(device),
        "mixed_precision": amp_enabled,
        "epochs": epochs,
        "best_validation_metric": best_metric,
        "final_model_path": str(final_model_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "history": history,
    }
    _write_json(job.job_dir / "training_summary.json", training_summary)

    model_card = {
        "model_id": model_id,
        "created_at": utc_now_iso(),
        "git_commit": job.config.get("project", {}).get("git_commit", "auto"),
        "dataset_id": dataset_id,
        "training_job_id": job.job_id,
        "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path else "",
        "init_from_promoted_best": init_from_promoted_best,
        "promoted_best_checkpoint_path": str(promoted_best_checkpoint_path),
        "parent_checkpoint_snapshot_path": str(parent_checkpoint_snapshot_path) if parent_checkpoint_snapshot_path else "",
        "architecture": {
            "family": str(cfg.get("model_family", "policy_value")),
            "backbone": str(cfg.get("backbone", "mlp")),
            "hidden_sizes": hidden_sizes,
        },
        "checkpoint_path": str(final_model_path),
        "best_validation_metric": best_metric,
        "benchmark_summary_path": "",
        "runtime": {
            "device": str(device),
            "mixed_precision": amp_enabled,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
    }
    _write_json(final_dir / f"{model_id}.model_card.json", model_card)
    upsert_model_record(
        job.paths.models_root,
        {
            "model_id": model_id,
            "sort_ts": time.time(),
            "dataset_id": dataset_id,
            "training_job_id": job.job_id,
            "checkpoint_path": str(final_model_path),
            "best_validation_metric": best_metric,
            "evaluation_top1": model_card.get("evaluation_top1", -1.0),
            "benchmark_score": model_card.get("benchmark_score", -1.0),
            "parent_model_path": str(init_checkpoint_path) if init_checkpoint_path else "",
            "model_card_path": str(final_dir / f"{model_id}.model_card.json"),
        },
    )
    promote_best_model(job.paths.models_root)
    job.write_metric({"metric_type": "train_completed", "model_id": model_id, "best_validation_metric": best_metric})
    job.write_event("train_completed", model_id=model_id, final_model_path=str(final_model_path))
    job.logger.info(
        "training completed | model=%s | final_model=%s | best_validation_metric=%.4f",
        model_id,
        final_model_path,
        best_metric,
    )
    return training_summary
