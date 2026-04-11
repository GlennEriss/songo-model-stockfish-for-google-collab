from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import autocast

from songo_model_stockfish.ops.io_utils import read_json_dict, write_json_atomic
from songo_model_stockfish.ops.job import JobContext
from songo_model_stockfish.ops.logging import utc_now_iso
from songo_model_stockfish.ops.model_registry import (
    latest_model_record,
    load_registry,
    promote_best_model,
    promoted_best_metadata,
    upsert_model_record,
)
from songo_model_stockfish.training.data import build_dataloader
from songo_model_stockfish.training.jobs import (
    _masked_policy_logits,
    _resolve_built_dataset_by_id,
    _select_largest_built_dataset,
    _soft_policy_loss,
    _tactical_mask_regularization,
)
from songo_model_stockfish.training.model import PolicyValueMLP


def _resolve_storage_path(base: Path, configured: str | None, fallback: Path) -> Path:
    if not configured:
        return fallback
    path = Path(configured)
    if path.is_absolute():
        return path
    return base / path


def _write_json(path: Path, payload: dict) -> None:
    write_json_atomic(path, payload, ensure_ascii=True, indent=2)


def _ensure_checkpoint_exists(*, model_id: str, checkpoint_path: Path, mode: str) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path
    raise FileNotFoundError(
        "Checkpoint d'evaluation introuvable "
        f"(mode={mode}, model_id={model_id}, checkpoint_path={checkpoint_path})."
    )


def _resolve_evaluation_target(job: JobContext, cfg: dict[str, Any]) -> tuple[str, Path]:
    requested_model_id = str(cfg.get("model_id", "auto_latest")).strip() or "auto_latest"
    requested_checkpoint = str(cfg.get("checkpoint_path", "")).strip()

    if requested_model_id in {"auto_best", "auto_promoted_best"} or requested_checkpoint in {"auto_best", "auto_promoted_best"}:
        metadata = promoted_best_metadata(job.paths.models_root)
        if not metadata:
            raise FileNotFoundError("Aucun modele promu disponible pour l'evaluation auto_best.")
        checkpoint_path = job.paths.models_root / "promoted" / "best" / "model.pt"
        resolved_model_id = str(metadata.get("model_id", "promoted_best"))
        return resolved_model_id, _ensure_checkpoint_exists(
            model_id=resolved_model_id,
            checkpoint_path=checkpoint_path,
            mode="auto_best",
        )

    if requested_model_id in {"auto", "auto_latest"} or requested_checkpoint in {"", "auto", "auto_latest"}:
        latest = latest_model_record(job.paths.models_root)
        if not latest:
            raise FileNotFoundError("Aucun modele disponible dans le registre pour l'evaluation auto_latest.")
        checkpoint_path = Path(str(latest.get("checkpoint_path", "")).strip())
        resolved_model_id = str(latest.get("model_id", ""))
        return resolved_model_id, _ensure_checkpoint_exists(
            model_id=resolved_model_id,
            checkpoint_path=checkpoint_path,
            mode="auto_latest",
        )

    checkpoint_path = _resolve_storage_path(job.paths.drive_root, cfg.get("checkpoint_path"), job.job_dir / f"{requested_model_id}.pt")
    return requested_model_id, _ensure_checkpoint_exists(
        model_id=requested_model_id,
        checkpoint_path=checkpoint_path,
        mode="explicit",
    )


def _update_model_card_after_evaluation(models_root: Path, model_id: str, summary: dict[str, object]) -> None:
    model_card_path = models_root / "final" / f"{model_id}.model_card.json"
    if not model_card_path.exists():
        return
    payload = read_json_dict(model_card_path, default={})
    if not payload:
        return
    payload["evaluation_summary_path"] = str(summary["evaluation_summary_path"])
    payload["evaluation_top1"] = float(summary["policy_accuracy_top1"])
    payload["evaluation_top3"] = float(summary["policy_accuracy_top3"])
    payload["evaluation_loss_total"] = float(summary["loss_total"])
    payload["evaluation_value_mae"] = float(summary["value_mae"])
    write_json_atomic(model_card_path, payload, ensure_ascii=True, indent=2)


def run_evaluation(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("evaluation", {})
    runtime_cfg = job.config.get("runtime", {})
    firestore_cfg = {
        "backend": str(cfg.get("dataset_registry_backend", cfg.get("global_progress_backend", "file"))).strip().lower() or "file",
        "project_id": str(cfg.get("dataset_registry_firestore_project_id", cfg.get("global_progress_firestore_project_id", ""))).strip(),
        "collection": str(cfg.get("dataset_registry_firestore_collection", "dataset_registry")).strip() or "dataset_registry",
        "document": str(cfg.get("dataset_registry_firestore_document", "primary")).strip() or "primary",
        "credentials_path": str(cfg.get("dataset_registry_firestore_credentials_path", cfg.get("global_progress_firestore_credentials_path", ""))).strip(),
        "api_key": str(cfg.get("dataset_registry_firestore_api_key", cfg.get("global_progress_firestore_api_key", ""))).strip(),
    }
    model_id, checkpoint_path = _resolve_evaluation_target(job, cfg)
    dataset_selection_mode = str(cfg.get("dataset_selection_mode", "configured")).strip().lower() or "configured"
    if dataset_selection_mode == "largest_built":
        selected_dataset = _select_largest_built_dataset(job.paths.data_root, firestore_cfg=firestore_cfg)
        dataset_id = str(selected_dataset.get("dataset_id", "dataset_v1"))
        selected_output_dir = Path(str(selected_dataset["output_dir"]))
        test_dataset_path = selected_output_dir / "test.npz"
        dataset_resolution = {
            "selection_mode": dataset_selection_mode,
            "resolved_from_registry": True,
            "selected_labeled_samples": int(selected_dataset.get("labeled_samples", 0)),
            "selected_build_mode": str(selected_dataset.get("build_mode", "teacher_label")),
            "selected_teacher_engine": str(selected_dataset.get("teacher_engine", "")),
            "selected_teacher_level": str(selected_dataset.get("teacher_level", "")),
            "selected_output_dir": str(selected_output_dir),
        }
    else:
        dataset_id = str(cfg.get("dataset_id", "dataset_v1"))
        configured_test_dataset_path = str(cfg.get("test_dataset_path", "")).strip()
        if dataset_id not in {"", "auto"} and not configured_test_dataset_path:
            selected_dataset = _resolve_built_dataset_by_id(job.paths.data_root, dataset_id, firestore_cfg=firestore_cfg)
            selected_output_dir = Path(str(selected_dataset["output_dir"]))
            test_dataset_path = selected_output_dir / "test.npz"
            dataset_resolution = {
                "selection_mode": dataset_selection_mode,
                "resolved_from_registry": True,
                "selected_labeled_samples": int(selected_dataset.get("labeled_samples", 0)),
                "selected_build_mode": str(selected_dataset.get("build_mode", "teacher_label")),
                "selected_teacher_engine": str(selected_dataset.get("teacher_engine", "")),
                "selected_teacher_level": str(selected_dataset.get("teacher_level", "")),
                "selected_output_dir": str(selected_output_dir),
            }
        else:
            test_dataset_path = _resolve_storage_path(job.paths.drive_root, cfg.get("test_dataset_path"), job.job_dir / "test.npz")
            dataset_resolution = {
                "selection_mode": dataset_selection_mode,
                "resolved_from_registry": False,
            }
    output_dir = _resolve_storage_path(job.paths.drive_root, cfg.get("output_dir"), job.job_dir / "reports")
    batch_size = int(cfg.get("batch_size", 128))
    soft_policy_loss_weight = float(cfg.get("soft_policy_loss_weight", 0.35))
    tactical_aux_loss_weight = float(cfg.get("tactical_aux_loss_weight", 0.05))
    log_every_n_batches = max(1, int(cfg.get("log_every_n_batches", 1)))
    requested_device = str(runtime_cfg.get("device", "cpu")).strip().lower() or "cpu"
    if requested_device in {"tpu", "xla"}:
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

            device = xm.xla_device()
        except Exception:
            job.logger.warning("TPU/XLA requested but unavailable, falling back to CPU")
            device = torch.device("cpu")
    elif requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    num_workers = int(runtime_cfg.get("num_workers", 0))
    pin_memory = bool(runtime_cfg.get("pin_memory", device.type == "cuda"))
    if device.type == "xla":
        pin_memory = False
    persistent_workers = bool(runtime_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = runtime_cfg.get("prefetch_factor")
    amp_enabled = bool(runtime_cfg.get("mixed_precision", False) and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    model = PolicyValueMLP(
        input_dim=int(model_config.get("input_dim", 17)),
        hidden_sizes=list(model_config.get("hidden_sizes", [256, 256, 128])),
        policy_dim=int(model_config.get("policy_dim", 7)),
        use_layer_norm=bool(model_config.get("use_layer_norm", False)),
        dropout=float(model_config.get("dropout", 0.0)),
        residual_connections=bool(model_config.get("residual_connections", False)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    test_loader, _input_dim = build_dataloader(
        test_dataset_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    model_input_dim = int(model_config.get("input_dim", 17))
    if int(_input_dim) != model_input_dim:
        raise ValueError(
            "Incompatibilite modele/dataset detectee avant evaluation "
            f"(model_id={model_id}, checkpoint_input_dim={model_input_dim}, dataset_input_dim={_input_dim}, dataset_id={dataset_id}, test_dataset_path={test_dataset_path})."
        )

    state = job.read_state()
    completed_batches = int(state.get("completed_batches", 0))
    total_loss = float(state.get("total_loss", 0.0))
    total_policy = float(state.get("total_policy", 0.0))
    total_value = float(state.get("total_value", 0.0))
    total_abs_value_error = float(state.get("total_abs_value_error", 0.0))
    total_correct = int(state.get("total_correct", 0))
    total_top3 = int(state.get("total_top3", 0))
    total_examples = int(state.get("total_examples", 0))

    if completed_batches > 0:
        job.logger.info(
            "evaluation resume detected | model=%s | dataset=%s | selection_mode=%s | completed_batches=%s/%s | completed_examples=%s",
            model_id,
            dataset_id,
            dataset_selection_mode,
            completed_batches,
            len(test_loader),
            total_examples,
        )
        job.write_event(
            "evaluation_resume_detected",
            model_id=model_id,
            dataset_id=dataset_id,
            dataset_selection_mode=dataset_selection_mode,
            completed_batches=completed_batches,
            total_batches=len(test_loader),
            completed_examples=total_examples,
        )

    job.logger.info(
        "evaluation started | model=%s | checkpoint=%s | dataset=%s | selection_mode=%s | dataset_path=%s | device=%s | mixed_precision=%s | batch_size=%s | examples=%s",
        model_id,
        checkpoint_path,
        dataset_id,
        dataset_selection_mode,
        test_dataset_path,
        device,
        amp_enabled,
        batch_size,
        len(test_loader.dataset),
    )
    job.set_phase("evaluation")
    job.write_event(
        "evaluation_started",
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_selection_mode=dataset_selection_mode,
        dataset_resolution=dataset_resolution,
        checkpoint_path=str(checkpoint_path),
        test_dataset_path=str(test_dataset_path),
        device=str(device),
        mixed_precision=amp_enabled,
        batch_size=batch_size,
        total_examples=len(test_loader.dataset),
        total_batches=len(test_loader),
        resume_granularity="batch",
    )
    job.write_metric(
        {
            "metric_type": "evaluation_started",
            "model_id": model_id,
            "dataset_id": dataset_id,
            "dataset_selection_mode": dataset_selection_mode,
            "total_examples": len(test_loader.dataset),
            "total_batches": len(test_loader),
            "mixed_precision": amp_enabled,
        }
    )
    job.write_status(
        "running",
        phase="evaluation",
        extra={
            "model_id": model_id,
            "dataset_id": dataset_id,
            "resume_granularity": "batch",
            "batch_index": completed_batches,
            "total_batches": len(test_loader),
            "last_state_path": str(job.state_path),
        },
    )

    with torch.no_grad():
        for batch_index, (x, legal_mask, policy_index, policy_target_full, value_target, capture_move_mask, safe_move_mask, risky_move_mask) in enumerate(test_loader, start=1):
            if batch_index <= completed_batches:
                continue

            x = x.to(device)
            legal_mask = legal_mask.to(device)
            policy_index = policy_index.to(device)
            policy_target_full = policy_target_full.to(device)
            value_target = value_target.to(device)
            capture_move_mask = capture_move_mask.to(device)
            safe_move_mask = safe_move_mask.to(device)
            risky_move_mask = risky_move_mask.to(device)

            with autocast(device_type=device.type, enabled=amp_enabled):
                policy_logits, value_pred = model(x)
                masked_logits = _masked_policy_logits(policy_logits, legal_mask)
                loss_policy_hard = F.cross_entropy(masked_logits, policy_index)
                loss_policy_soft = _soft_policy_loss(policy_logits, legal_mask, policy_target_full)
                loss_policy = ((1.0 - soft_policy_loss_weight) * loss_policy_hard) + (soft_policy_loss_weight * loss_policy_soft)
                loss_value = F.mse_loss(value_pred, value_target)
                tactical_aux_loss = _tactical_mask_regularization(
                    policy_logits,
                    legal_mask,
                    capture_move_mask,
                    safe_move_mask,
                    risky_move_mask,
                )
                loss_total = loss_policy + loss_value + (tactical_aux_loss_weight * tactical_aux_loss)
            if not bool(torch.isfinite(loss_total).item()):
                error_message = (
                    "non-finite loss detected during evaluation "
                    f"(batch={batch_index}/{len(test_loader)}, model_id={model_id}, dataset_id={dataset_id})."
                )
                job.write_event(
                    "evaluation_non_finite_detected",
                    model_id=model_id,
                    dataset_id=dataset_id,
                    batch_index=batch_index,
                    total_batches=len(test_loader),
                    error=error_message,
                )
                job.write_metric(
                    {
                        "metric_type": "evaluation_non_finite_detected",
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "batch_index": batch_index,
                        "total_batches": len(test_loader),
                    }
                )
                raise FloatingPointError(error_message)

            preds = masked_logits.argmax(dim=1)
            topk = min(3, masked_logits.shape[1])
            top3 = masked_logits.topk(topk, dim=1).indices
            total_correct += int((preds == policy_index).sum().item())
            total_top3 += int((top3 == policy_index.unsqueeze(1)).any(dim=1).sum().item())

            batch_count = int(x.shape[0])
            total_examples += batch_count
            total_loss += float(loss_total.item()) * batch_count
            total_policy += float(loss_policy.item()) * batch_count
            total_value += float(loss_value.item()) * batch_count
            total_abs_value_error += float(torch.abs(value_pred - value_target).sum().item())
            if batch_index == 1 or batch_index == len(test_loader) or batch_index % log_every_n_batches == 0:
                job.logger.info(
                    "evaluation batch | batch=%s/%s | examples=%s | loss=%.4f | policy=%.4f | value=%.4f | value_mae=%.4f | top1=%.4f | top3=%.4f",
                    batch_index,
                    len(test_loader),
                    total_examples,
                    float(loss_total.item()),
                    float(loss_policy.item()),
                    float(loss_value.item()),
                    total_abs_value_error / max(1, total_examples),
                    total_correct / max(1, total_examples),
                    total_top3 / max(1, total_examples),
                )
                job.write_event(
                    "evaluation_batch_completed",
                    model_id=model_id,
                    dataset_id=dataset_id,
                    batch_index=batch_index,
                    total_batches=len(test_loader),
                    total_examples=total_examples,
                    batch_examples=batch_count,
                    loss_total=float(loss_total.item()),
                    loss_policy=float(loss_policy.item()),
                    loss_value=float(loss_value.item()),
                    value_mae=total_abs_value_error / max(1, total_examples),
                    accuracy_top1=total_correct / max(1, total_examples),
                    accuracy_top3=total_top3 / max(1, total_examples),
                )
            job.write_state(
                {
                    "completed_batches": batch_index,
                    "total_batches": len(test_loader),
                    "total_examples": total_examples,
                    "total_loss": total_loss,
                    "total_policy": total_policy,
                    "total_value": total_value,
                    "total_abs_value_error": total_abs_value_error,
                    "total_correct": total_correct,
                    "total_top3": total_top3,
                    "last_completed_phase": "evaluation_batch_completed",
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            job.write_status(
                "running",
                phase="evaluation",
                extra={
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "resume_granularity": "batch",
                    "batch_index": batch_index,
                    "total_batches": len(test_loader),
                    "last_state_path": str(job.state_path),
                },
            )

    denom = max(1, total_examples)
    summary = {
        "job_id": job.job_id,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "dataset_selection_mode": dataset_selection_mode,
        "dataset_resolution": dataset_resolution,
        "device": str(device),
        "mixed_precision": amp_enabled,
        "examples": total_examples,
        "loss_total": total_loss / denom,
        "loss_policy": total_policy / denom,
        "loss_value": total_value / denom,
        "value_mae": total_abs_value_error / denom,
        "policy_accuracy_top1": total_correct / denom,
        "policy_accuracy_top3": total_top3 / denom,
        "checkpoint_path": str(checkpoint_path),
        "evaluated_at": utc_now_iso(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{model_id}_evaluation_summary.json"
    _write_json(report_path, summary)
    summary["evaluation_summary_path"] = str(report_path)
    _write_json(job.job_dir / "evaluation_summary.json", summary)
    _update_model_card_after_evaluation(job.paths.models_root, model_id, summary)
    registry = load_registry(job.paths.models_root)
    existing = next((item for item in registry.get("models", []) if str(item.get("model_id")) == model_id), {})
    upsert_model_record(
        job.paths.models_root,
        {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "checkpoint_path": str(checkpoint_path),
            "training_job_id": existing.get("training_job_id", ""),
            "best_validation_metric": existing.get("best_validation_metric", -1.0),
            "evaluation_top1": float(summary["policy_accuracy_top1"]),
            "evaluation_top3": float(summary["policy_accuracy_top3"]),
            "evaluated_at": str(summary.get("evaluated_at", utc_now_iso())),
            "evaluation_summary_path": str(report_path),
            "benchmark_score": float(existing.get("benchmark_score", -1.0)),
            "model_card_path": str(job.paths.models_root / "final" / f"{model_id}.model_card.json"),
        },
    )
    promote_best_model(job.paths.models_root)
    job.write_state(
        {
            "completed_batches": len(test_loader),
            "total_batches": len(test_loader),
            "total_examples": total_examples,
            "total_loss": total_loss,
            "total_policy": total_policy,
            "total_value": total_value,
            "total_abs_value_error": total_abs_value_error,
            "total_correct": total_correct,
            "total_top3": total_top3,
            "last_completed_phase": "evaluation_completed",
            "checkpoint_path": str(checkpoint_path),
        }
    )
    job.write_metric({"metric_type": "evaluation_completed", **summary})
    job.write_event("evaluation_completed", model_id=model_id)
    job.logger.info(
        "evaluation completed | model=%s | dataset=%s | selection_mode=%s | examples=%s | top1=%.4f | top3=%.4f | value_mae=%.4f | loss=%.4f | summary=%s",
        model_id,
        dataset_id,
        dataset_selection_mode,
        total_examples,
        summary["policy_accuracy_top1"],
        summary["policy_accuracy_top3"],
        summary["value_mae"],
        summary["loss_total"],
        report_path,
    )
    return summary
