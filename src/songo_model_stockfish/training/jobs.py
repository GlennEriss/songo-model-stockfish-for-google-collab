from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from songo_model_stockfish.ops.io_utils import read_json_dict, write_json_atomic
from songo_model_stockfish.ops.job import JobContext
from songo_model_stockfish.ops.logging import utc_now_iso
from songo_model_stockfish.ops.model_registry import next_model_version, promote_best_model, promoted_best_dir, upsert_model_record
from songo_model_stockfish.training.data import build_dataloader
from songo_model_stockfish.training.model import PolicyValueMLP


def _build_model_config(
    input_dim: int,
    hidden_sizes: list[int],
    *,
    policy_dim: int = 7,
    use_layer_norm: bool = False,
    dropout: float = 0.0,
    residual_connections: bool = False,
) -> dict[str, Any]:
    return {
        "input_dim": input_dim,
        "hidden_sizes": hidden_sizes,
        "policy_dim": policy_dim,
        "use_layer_norm": use_layer_norm,
        "dropout": dropout,
        "residual_connections": residual_connections,
    }


def _resolve_storage_path(base: Path, configured: str | None, fallback: Path) -> Path:
    if not configured:
        return fallback
    path = Path(configured)
    if path.is_absolute():
        return path
    return base / path


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if text in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _read_dataset_registry(data_root: Path, firestore_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = firestore_cfg if isinstance(firestore_cfg, dict) else {}
    backend = str(cfg.get("backend", "file")).strip().lower() or "file"
    if backend == "firestore":
        project_id = str(cfg.get("project_id", "")).strip()
        collection = str(cfg.get("collection", "dataset_registry")).strip() or "dataset_registry"
        document = str(cfg.get("document", "primary")).strip() or "primary"
        try:
            from google.cloud import firestore

            credentials_path = str(cfg.get("credentials_path", "")).strip()
            api_key = str(cfg.get("api_key", "")).strip()
            credentials = None
            client_options = None
            if credentials_path:
                if not Path(credentials_path).exists():
                    raise FileNotFoundError(f"Fichier credentials Firestore introuvable: {credentials_path}")
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(credentials_path)
            elif api_key:
                raise RuntimeError(
                    "Mode API key non supporte avec google-cloud-firestore; "
                    "configure `dataset_registry_firestore_credentials_path`."
                )
            else:
                raise RuntimeError(
                    "Credentials Firestore absents; configure `dataset_registry_firestore_credentials_path`."
                )
            client = firestore.Client(project=(project_id or None), credentials=credentials, client_options=client_options)
            snap = client.collection(collection).document(document).get()
            payload = snap.to_dict() if snap.exists else {"dataset_sources": [], "built_datasets": []}
        except Exception as exc:
            raise RuntimeError(
                "Lecture dataset_registry Firestore impossible | "
                f"project_id={project_id or '<empty>'} | collection={collection} | document={document} | "
                f"cause={type(exc).__name__}: {exc}"
            ) from exc
    else:
        registry_path = data_root / "dataset_registry.json"
        payload = read_json_dict(registry_path, default={"dataset_sources": [], "built_datasets": []})
    if not isinstance(payload, dict):
        return {"dataset_sources": [], "built_datasets": []}
    payload.setdefault("dataset_sources", [])
    payload.setdefault("built_datasets", [])
    return payload


def _normalize_registry_output_dir(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return None
    return candidate.resolve(strict=False)


def _select_largest_built_dataset(data_root: Path, firestore_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = _read_dataset_registry(data_root, firestore_cfg=firestore_cfg)
    candidates: list[dict[str, Any]] = []
    for entry in registry.get("built_datasets", []):
        output_dir = _normalize_registry_output_dir(entry.get("output_dir"))
        if output_dir is None:
            continue
        if not (output_dir / "train.npz").exists():
            continue
        if not (output_dir / "validation.npz").exists():
            continue
        normalized_entry = dict(entry)
        normalized_entry["output_dir"] = str(output_dir)
        candidates.append(normalized_entry)
    if not candidates:
        raise FileNotFoundError("Aucun built dataset complet avec train.npz et validation.npz trouve dans le registre")
    candidates.sort(
        key=lambda entry: (
            int(entry.get("labeled_samples", 0)),
            str(entry.get("updated_at", "")),
        ),
        reverse=True,
    )
    return candidates[0]


def _resolve_built_dataset_by_id(data_root: Path, dataset_id: str, firestore_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    requested_id = str(dataset_id).strip()
    if not requested_id:
        raise ValueError("dataset_id vide pour la resolution du built dataset.")
    registry = _read_dataset_registry(data_root, firestore_cfg=firestore_cfg)
    for entry in registry.get("built_datasets", []):
        if str(entry.get("dataset_id", "")).strip() == requested_id:
            output_dir = _normalize_registry_output_dir(entry.get("output_dir"))
            if output_dir is None:
                raise ValueError(
                    f"Built dataset invalide dans le registre (output_dir vide/non-absolu): {requested_id}"
                )
            normalized_entry = dict(entry)
            normalized_entry["output_dir"] = str(output_dir)
            return normalized_entry
    raise FileNotFoundError(f"Built dataset introuvable dans le registre: {requested_id}")


def _masked_policy_logits(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    # Mixed precision safe value for fp16/bf16/fp32.
    mask_value = torch.finfo(policy_logits.dtype).min
    return policy_logits.masked_fill(legal_mask <= 0, mask_value)


def _masked_policy_loss(policy_logits: torch.Tensor, legal_mask: torch.Tensor, policy_index: torch.Tensor) -> torch.Tensor:
    masked_logits = _masked_policy_logits(policy_logits, legal_mask)
    return F.cross_entropy(masked_logits, policy_index)


def _soft_policy_loss(policy_logits: torch.Tensor, legal_mask: torch.Tensor, policy_target_full: torch.Tensor) -> torch.Tensor:
    masked_logits = _masked_policy_logits(policy_logits, legal_mask)
    log_probs = F.log_softmax(masked_logits, dim=1)
    return -(policy_target_full * log_probs).sum(dim=1).mean()


def _tactical_mask_regularization(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    capture_move_mask: torch.Tensor,
    safe_move_mask: torch.Tensor,
    risky_move_mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = _masked_policy_logits(policy_logits, legal_mask)
    probs = F.softmax(masked_logits, dim=1)
    eps = 1e-6

    def _mass(mask: torch.Tensor) -> torch.Tensor:
        active = mask.sum(dim=1) > 0
        if not bool(active.any().item()):
            return torch.zeros((), device=policy_logits.device, dtype=policy_logits.dtype)
        return (probs * mask).sum(dim=1)[active].mean()

    capture_mass = _mass(capture_move_mask)
    safe_mass = _mass(safe_move_mask)
    risky_mass = _mass(risky_move_mask)

    return (-torch.log(capture_mass + eps)) + (-torch.log(safe_mass + eps)) + (-torch.log(1.0 - risky_mass + eps))


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any], epochs: int):
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "none")).strip().lower()
    if scheduler_type in {"", "none"}:
        return None, scheduler_type, {}
    if scheduler_type == "cosine":
        min_lr = float(scheduler_cfg.get("min_lr", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=min_lr,
        )
        return scheduler, scheduler_type, {"min_lr": min_lr}
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def _run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
    value_loss_weight: float = 1.0,
    soft_policy_loss_weight: float = 0.35,
    tactical_aux_loss_weight: float = 0.05,
    gradient_clip_norm: float = 0.0,
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

    for batch_index, (x, legal_mask, policy_index, policy_target_full, value_target, capture_move_mask, safe_move_mask, risky_move_mask) in enumerate(loader, start=1):
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
            loss_policy_hard = _masked_policy_loss(policy_logits, legal_mask, policy_index)
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
            loss_total = loss_policy + (value_loss_weight * loss_value) + (tactical_aux_loss_weight * tactical_aux_loss)
        if not bool(torch.isfinite(loss_total).item()):
            raise FloatingPointError(
                f"non-finite loss detected | stage={stage} | batch={batch_index}/{total_batches} | "
                f"loss_total={float(loss_total.detach().cpu().item())}"
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and amp_enabled:
                scaler.scale(loss_total).backward()
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                if device.type == "xla":
                    import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

                    xm.optimizer_step(optimizer, barrier=False)
                else:
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
                    "loss_policy_hard": float(loss_policy_hard.item()),
                    "loss_policy_soft": float(loss_policy_soft.item()),
                    "loss_tactical_aux": float(tactical_aux_loss.item()),
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
    write_json_atomic(path, payload, ensure_ascii=True, indent=2)


def run_train(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("train", {})
    runtime_cfg = job.config.get("runtime", {})
    firestore_cfg = {
        "backend": str(cfg.get("dataset_registry_backend", cfg.get("global_progress_backend", "file"))).strip().lower() or "file",
        "project_id": str(cfg.get("dataset_registry_firestore_project_id", cfg.get("global_progress_firestore_project_id", ""))).strip(),
        "collection": str(cfg.get("dataset_registry_firestore_collection", "dataset_registry")).strip() or "dataset_registry",
        "document": str(cfg.get("dataset_registry_firestore_document", "primary")).strip() or "primary",
        "credentials_path": str(cfg.get("dataset_registry_firestore_credentials_path", cfg.get("global_progress_firestore_credentials_path", ""))).strip(),
        "api_key": str(cfg.get("dataset_registry_firestore_api_key", cfg.get("global_progress_firestore_api_key", ""))).strip(),
    }
    dataset_selection_mode = str(cfg.get("dataset_selection_mode", "configured")).strip().lower() or "configured"
    expected_dataset_input_dim: int | None = None
    selected_dataset_feature_schema_version = ""
    if dataset_selection_mode == "largest_built":
        selected_dataset = _select_largest_built_dataset(job.paths.data_root, firestore_cfg=firestore_cfg)
        dataset_id = str(selected_dataset.get("dataset_id", "dataset_v1"))
        selected_output_dir = Path(str(selected_dataset["output_dir"]))
        dataset_path = selected_output_dir / "train.npz"
        validation_path = selected_output_dir / "validation.npz"
        if selected_dataset.get("input_dim") is not None:
            expected_dataset_input_dim = int(selected_dataset.get("input_dim", 0) or 0) or None
        selected_dataset_feature_schema_version = str(selected_dataset.get("feature_schema_version", "")).strip()
        dataset_resolution = {
            "selection_mode": dataset_selection_mode,
            "resolved_from_registry": True,
            "selected_labeled_samples": int(selected_dataset.get("labeled_samples", 0)),
            "selected_build_mode": str(selected_dataset.get("build_mode", "teacher_label")),
            "selected_teacher_engine": str(selected_dataset.get("teacher_engine", "")),
            "selected_teacher_level": str(selected_dataset.get("teacher_level", "")),
            "selected_output_dir": str(selected_output_dir),
            "selected_input_dim": expected_dataset_input_dim,
            "selected_feature_schema_version": selected_dataset_feature_schema_version,
        }
    else:
        dataset_id = str(cfg.get("dataset_id", "dataset_v1"))
        configured_dataset_path = str(cfg.get("dataset_path", "")).strip()
        configured_validation_path = str(cfg.get("validation_path", "")).strip()
        if dataset_id not in {"", "auto"} and not configured_dataset_path and not configured_validation_path:
            selected_dataset = _resolve_built_dataset_by_id(job.paths.data_root, dataset_id, firestore_cfg=firestore_cfg)
            selected_output_dir = Path(str(selected_dataset["output_dir"]))
            dataset_path = selected_output_dir / "train.npz"
            validation_path = selected_output_dir / "validation.npz"
            if selected_dataset.get("input_dim") is not None:
                expected_dataset_input_dim = int(selected_dataset.get("input_dim", 0) or 0) or None
            selected_dataset_feature_schema_version = str(selected_dataset.get("feature_schema_version", "")).strip()
            dataset_resolution = {
                "selection_mode": dataset_selection_mode,
                "resolved_from_registry": True,
                "selected_labeled_samples": int(selected_dataset.get("labeled_samples", 0)),
                "selected_build_mode": str(selected_dataset.get("build_mode", "teacher_label")),
                "selected_teacher_engine": str(selected_dataset.get("teacher_engine", "")),
                "selected_teacher_level": str(selected_dataset.get("teacher_level", "")),
                "selected_output_dir": str(selected_output_dir),
                "selected_input_dim": expected_dataset_input_dim,
                "selected_feature_schema_version": selected_dataset_feature_schema_version,
            }
        else:
            dataset_path = _resolve_storage_path(job.paths.drive_root, cfg.get("dataset_path"), job.job_dir / "train.npz")
            validation_path = _resolve_storage_path(job.paths.drive_root, cfg.get("validation_path"), job.job_dir / "validation.npz")
            dataset_resolution = {
                "selection_mode": dataset_selection_mode,
                "resolved_from_registry": False,
            }
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
    init_from_promoted_best = _as_bool(cfg.get("init_from_promoted_best", True), default=True)
    model_id = str(cfg.get("model_id", "auto"))
    model_id_prefix = str(cfg.get("model_id_prefix", "songo_policy_value_colab_pro"))
    if model_id in {"", "auto"}:
        model_id = f"{model_id_prefix}_v{next_model_version(job.paths.models_root, model_id_prefix)}"
    hidden_sizes = list(cfg.get("hidden_sizes", [256, 256, 128]))
    use_layer_norm = bool(cfg.get("use_layer_norm", False))
    dropout = float(cfg.get("dropout", 0.0))
    residual_connections = bool(cfg.get("residual_connections", False))
    batch_size = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("epochs", 20))
    learning_rate = float(cfg.get("learning_rate", 0.0003))
    value_loss_weight = float(cfg.get("value_loss_weight", 1.0))
    soft_policy_loss_weight = float(cfg.get("soft_policy_loss_weight", 0.35))
    tactical_aux_loss_weight = float(cfg.get("tactical_aux_loss_weight", 0.05))
    log_every_n_batches = max(1, int(cfg.get("log_every_n_batches", 1)))
    gradient_clip_norm = float(cfg.get("gradient_clip_norm", 0.0))
    early_stopping_patience = max(0, int(cfg.get("early_stopping_patience", 0)))
    hard_example_oversampling_enabled = _as_bool(cfg.get("hard_example_oversampling_enabled", True), default=True)
    hard_example_weight_exponent = max(0.0, float(cfg.get("hard_example_weight_exponent", 1.0)))
    hard_example_weight_min = max(1e-6, float(cfg.get("hard_example_weight_min", 1.0)))
    hard_example_weight_max = max(hard_example_weight_min, float(cfg.get("hard_example_weight_max", 5.0)))
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

    train_loader, input_dim = build_dataloader(
        dataset_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        weighted_sampling=hard_example_oversampling_enabled,
        weighted_sampling_exponent=hard_example_weight_exponent,
        weighted_sampling_min_weight=hard_example_weight_min,
        weighted_sampling_max_weight=hard_example_weight_max,
    )
    validation_loader, _ = build_dataloader(
        validation_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        weighted_sampling=False,
    )
    train_sampling_metadata = getattr(train_loader, "songo_sampling_metadata", {})

    if expected_dataset_input_dim is not None and int(expected_dataset_input_dim) != int(input_dim):
        raise ValueError(
            "Incoherence de schema detectee entre le registre dataset et le NPZ charge "
            f"(dataset_id={dataset_id}, expected_input_dim={expected_dataset_input_dim}, loaded_input_dim={input_dim}, "
            f"feature_schema_version={selected_dataset_feature_schema_version or '<unknown>'})."
        )

    model = PolicyValueMLP(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
        residual_connections=residual_connections,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler, scheduler_type, scheduler_params = _build_scheduler(optimizer, cfg, epochs)
    scaler = GradScaler(device="cuda", enabled=amp_enabled)
    model_config = _build_model_config(
        input_dim,
        hidden_sizes,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
        residual_connections=residual_connections,
    )

    state = job.read_state()
    start_epoch = int(state.get("epoch", 0))
    global_step = int(state.get("global_step", 0))
    best_metric = float(state.get("best_metric", float("-inf")))
    best_epoch = int(state.get("best_epoch", 0))
    last_val_acc = float(state.get("last_val_acc", 0.0))
    epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
    if best_metric == float("-inf"):
        best_metric = 0.0
    checkpoint_path_value = state.get("checkpoint_path")
    parent_checkpoint_snapshot_path = None
    resumed_from_checkpoint = False
    if checkpoint_path_value:
        checkpoint_path = Path(str(checkpoint_path_value))
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            checkpoint_model_config = checkpoint.get("model_config", {})
            checkpoint_input_dim = int(checkpoint_model_config.get("input_dim", input_dim))
            if checkpoint_input_dim != int(input_dim):
                raise ValueError(
                    "Checkpoint de reprise incompatible avec le dataset courant "
                    f"(checkpoint_input_dim={checkpoint_input_dim}, dataset_input_dim={input_dim}, "
                    f"checkpoint={checkpoint_path}, dataset_id={dataset_id})."
                )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if scaler is not None and checkpoint.get("scaler_state") is not None:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint.get("epoch", start_epoch))
            global_step = int(checkpoint.get("global_step", global_step))
            best_metric = float(checkpoint.get("best_metric", best_metric))
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", epochs_without_improvement))
            resumed_from_checkpoint = True
    if (not resumed_from_checkpoint) and init_checkpoint_path is None and init_from_promoted_best and promoted_best_checkpoint_path.exists():
        init_checkpoint_path = promoted_best_checkpoint_path
    if (not resumed_from_checkpoint) and init_checkpoint_path:
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"init_checkpoint_path introuvable: {init_checkpoint_path}")
        parent_checkpoint = torch.load(init_checkpoint_path, map_location=device)
        parent_model_config = parent_checkpoint.get("model_config", {})
        parent_hidden_sizes = list(parent_model_config.get("hidden_sizes", hidden_sizes))
        parent_input_dim = int(parent_model_config.get("input_dim", input_dim))
        parent_policy_dim = int(parent_model_config.get("policy_dim", 7))
        parent_use_layer_norm = bool(parent_model_config.get("use_layer_norm", False))
        parent_dropout = float(parent_model_config.get("dropout", 0.0))
        parent_residual_connections = bool(parent_model_config.get("residual_connections", False))
        incompatible = (
            parent_hidden_sizes != hidden_sizes
            or parent_input_dim != input_dim
            or parent_policy_dim != 7
            or parent_use_layer_norm != use_layer_norm
            or parent_dropout != dropout
            or parent_residual_connections != residual_connections
        )
        if incompatible:
            compatibility_message = (
                "Le checkpoint parent n'est pas compatible avec la config d'entrainement actuelle "
                f"(parent hidden_sizes={parent_hidden_sizes}, current hidden_sizes={hidden_sizes}, "
                f"parent input_dim={parent_input_dim}, current input_dim={input_dim}, "
                f"parent policy_dim={parent_policy_dim}, current policy_dim=7, "
                f"parent use_layer_norm={parent_use_layer_norm}, current use_layer_norm={use_layer_norm}, "
                f"parent dropout={parent_dropout}, current dropout={dropout}, "
                f"parent residual_connections={parent_residual_connections}, current residual_connections={residual_connections})"
            )
            # If parent comes from auto promoted-best fallback, degrade gracefully to scratch.
            if init_from_promoted_best and init_checkpoint_path == promoted_best_checkpoint_path:
                job.logger.warning(
                    "training parent checkpoint incompatible, fallback to scratch | model=%s | parent_checkpoint=%s | reason=%s",
                    model_id,
                    init_checkpoint_path,
                    compatibility_message,
                )
                job.write_event(
                    "train_init_checkpoint_incompatible_fallback_scratch",
                    model_id=model_id,
                    init_checkpoint_path=str(init_checkpoint_path),
                    reason=compatibility_message,
                )
                init_checkpoint_path = None
            else:
                raise ValueError(compatibility_message)
        if init_checkpoint_path is not None:
            model.load_state_dict(parent_checkpoint["model_state"])
            lineage_dir.mkdir(parents=True, exist_ok=True)
            parent_checkpoint_snapshot_path = lineage_dir / f"{model_id}_parent_snapshot.pt"
            shutil.copy2(init_checkpoint_path, parent_checkpoint_snapshot_path)
            best_metric = 0.0
            job.logger.info(
                "training init checkpoint loaded | model=%s | parent_checkpoint=%s | promoted_best=%s",
                model_id,
                init_checkpoint_path,
                init_from_promoted_best,
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
        "training started | dataset=%s | selection_mode=%s | dataset_path=%s | validation_path=%s | model=%s | init_checkpoint=%s | promoted_best=%s | device=%s | mixed_precision=%s | epochs=%s | batch_size=%s | train_examples=%s | validation_examples=%s | weighted_sampling=%s | hard_weight_min=%.3f | hard_weight_max=%.3f | hard_weight_mean=%.3f | hard_weight_exponent=%.3f",
        dataset_id,
        dataset_selection_mode,
        dataset_path,
        validation_path,
        model_id,
        init_checkpoint_path if init_checkpoint_path else "",
        init_from_promoted_best,
        device,
        amp_enabled,
        epochs,
        batch_size,
        len(train_loader.dataset),
        len(validation_loader.dataset),
        bool(train_sampling_metadata.get("weighted_sampling_enabled", False)),
        float(train_sampling_metadata.get("hard_example_weight_min", 1.0)),
        float(train_sampling_metadata.get("hard_example_weight_max", 1.0)),
        float(train_sampling_metadata.get("hard_example_weight_mean", 1.0)),
        float(train_sampling_metadata.get("hard_example_weight_exponent", hard_example_weight_exponent)),
    )
    job.set_phase("training")
    job.write_event(
        "train_started",
        dataset_id=dataset_id,
        dataset_selection_mode=dataset_selection_mode,
        dataset_resolution=dataset_resolution,
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
        scheduler_type=scheduler_type,
        gradient_clip_norm=gradient_clip_norm,
        early_stopping_patience=early_stopping_patience,
        hard_example_oversampling_enabled=hard_example_oversampling_enabled,
        hard_example_weight_exponent=hard_example_weight_exponent,
        hard_example_weight_min=hard_example_weight_min,
        hard_example_weight_max=hard_example_weight_max,
        train_sampling_metadata=train_sampling_metadata,
    )
    job.write_metric(
        {
            "metric_type": "train_started",
            "dataset_id": dataset_id,
            "dataset_selection_mode": dataset_selection_mode,
            "model_id": model_id,
            "train_examples": len(train_loader.dataset),
            "validation_examples": len(validation_loader.dataset),
            "mixed_precision": amp_enabled,
            "hard_example_oversampling_enabled": hard_example_oversampling_enabled,
            "hard_example_weight_mean": float(train_sampling_metadata.get("hard_example_weight_mean", 1.0)),
        }
    )

    best_checkpoint_path = checkpoint_dir / f"{model_id}_best.pt"
    last_checkpoint_path = checkpoint_dir / f"{model_id}_last.pt"

    history: list[dict[str, float | int]] = []
    early_stopped = False
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
                "best_epoch": best_epoch,
                "last_val_acc": last_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
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
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "last_val_acc": last_val_acc,
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
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "last_val_acc": last_val_acc,
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
                        "best_epoch": best_epoch,
                        "last_val_acc": last_val_acc,
                        "epochs_without_improvement": epochs_without_improvement,
                        "last_completed_phase": "train_batch_progress",
                        "checkpoint_path": str(last_checkpoint_path),
                    }
                )

        try:
            train_metrics = _run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                value_loss_weight=value_loss_weight,
                soft_policy_loss_weight=soft_policy_loss_weight,
                tactical_aux_loss_weight=tactical_aux_loss_weight,
                gradient_clip_norm=gradient_clip_norm,
                on_batch_end=on_train_batch_end,
            )
        except FloatingPointError as exc:
            job.write_event(
                "train_non_finite_detected",
                epoch=epoch_number,
                stage="train",
                error=f"{type(exc).__name__}: {exc}",
            )
            job.write_metric(
                {
                    "metric_type": "train_non_finite_detected",
                    "epoch": epoch_number,
                    "stage": "train",
                }
            )
            raise
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
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "last_val_acc": last_val_acc,
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
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "last_val_acc": last_val_acc,
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
                        "best_epoch": best_epoch,
                        "last_val_acc": last_val_acc,
                        "epochs_without_improvement": epochs_without_improvement,
                        "last_completed_phase": "validation_batch_progress",
                        "checkpoint_path": str(last_checkpoint_path),
                    }
                )

        try:
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                optimizer=None,
                scaler=None,
                amp_enabled=amp_enabled,
                value_loss_weight=value_loss_weight,
                soft_policy_loss_weight=soft_policy_loss_weight,
                tactical_aux_loss_weight=tactical_aux_loss_weight,
                on_batch_end=on_validation_batch_end,
            )
        except FloatingPointError as exc:
            job.write_event(
                "train_non_finite_detected",
                epoch=epoch_number,
                stage="validation",
                error=f"{type(exc).__name__}: {exc}",
            )
            job.write_metric(
                {
                    "metric_type": "train_non_finite_detected",
                    "epoch": epoch_number,
                    "stage": "validation",
                }
            )
            raise
        global_step += int(train_metrics["examples"])

        epoch_payload = {
            "epoch": epoch_number,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
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
            "training epoch completed | epoch=%s/%s | lr=%.6f | train_loss=%.4f | val_loss=%.4f | train_acc=%.4f | val_acc=%.4f | best_metric=%.4f | best_epoch=%s",
            epoch_number,
            epochs,
            float(optimizer.param_groups[0]["lr"]),
            train_metrics["loss_total"],
            validation_metrics["loss_total"],
            train_metrics["policy_accuracy"],
            validation_metrics["policy_accuracy"],
            best_metric,
            best_epoch,
        )

        validation_score = float(validation_metrics["policy_accuracy"])
        last_val_acc = validation_score
        improved = validation_score >= best_metric
        if improved:
            best_metric = validation_score
            best_epoch = epoch_number
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        checkpoint_payload = {
            "epoch": epoch_number,
            "global_step": global_step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "amp_enabled": amp_enabled,
            "model_config": model_config,
        }
        _save_checkpoint(last_checkpoint_path, checkpoint_payload)
        job.write_event(
            "last_checkpoint_updated",
            epoch=epoch_number,
            checkpoint_path=str(last_checkpoint_path),
        )

        if improved:
            checkpoint_payload["best_metric"] = best_metric
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["epochs_without_improvement"] = epochs_without_improvement
            _save_checkpoint(best_checkpoint_path, checkpoint_payload)
            job.write_event("best_checkpoint_updated", epoch=epoch_number, checkpoint_path=str(best_checkpoint_path))

        job.write_state(
            {
                "epoch": epoch_number,
                "global_step": global_step,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "last_val_acc": last_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
                "last_completed_phase": "validation",
                "checkpoint_path": str(last_checkpoint_path),
            }
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
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "last_val_acc": last_val_acc,
                "last_checkpoint_path": str(last_checkpoint_path),
                "last_state_path": str(job.state_path),
            },
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            early_stopped = True
            job.write_event(
                "train_early_stopped",
                epoch=epoch_number,
                best_epoch=best_epoch,
                best_metric=best_metric,
                early_stopping_patience=early_stopping_patience,
            )
            job.logger.info(
                "training early stopped | epoch=%s/%s | best_epoch=%s | best_validation_metric=%.4f | patience=%s",
                epoch_number,
                epochs,
                best_epoch,
                best_metric,
                early_stopping_patience,
            )
            break

    final_dir.mkdir(parents=True, exist_ok=True)
    restored_best_checkpoint = False
    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state"])
        restored_best_checkpoint = True
    final_model_path = final_dir / f"{model_id}.pt"
    final_payload = {
        "model_state": model.state_dict(),
        "model_config": model_config,
        "dataset_id": dataset_id,
        "best_metric": best_metric,
        "amp_enabled": amp_enabled,
    }
    torch.save(final_payload, final_model_path)

    training_summary = {
        "job_id": job.job_id,
        "dataset_id": dataset_id,
        "dataset_selection_mode": dataset_selection_mode,
        "dataset_resolution": dataset_resolution,
        "model_id": model_id,
        "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path else "",
        "init_from_promoted_best": init_from_promoted_best,
        "promoted_best_checkpoint_path": str(promoted_best_checkpoint_path),
        "parent_checkpoint_snapshot_path": str(parent_checkpoint_snapshot_path) if parent_checkpoint_snapshot_path else "",
        "device": str(device),
        "mixed_precision": amp_enabled,
        "epochs": epochs,
        "completed_epochs": len(history),
        "best_validation_metric": best_metric,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "early_stopping_patience": early_stopping_patience,
        "scheduler_type": scheduler_type,
        "scheduler": scheduler_params,
        "gradient_clip_norm": gradient_clip_norm,
        "hard_example_oversampling_enabled": hard_example_oversampling_enabled,
        "hard_example_weight_exponent": hard_example_weight_exponent,
        "hard_example_weight_min": hard_example_weight_min,
        "hard_example_weight_max": hard_example_weight_max,
        "train_sampling_metadata": train_sampling_metadata,
        "restored_best_checkpoint_for_export": restored_best_checkpoint,
        "final_model_path": str(final_model_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "history": history,
    }
    job.write_artifact_json("training_summary.json", training_summary, ensure_ascii=True, indent=2)

    model_card = {
        "model_id": model_id,
        "created_at": utc_now_iso(),
        "git_commit": job.config.get("project", {}).get("git_commit", "auto"),
        "dataset_id": dataset_id,
        "dataset_selection_mode": dataset_selection_mode,
        "dataset_resolution": dataset_resolution,
        "training_job_id": job.job_id,
        "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path else "",
        "init_from_promoted_best": init_from_promoted_best,
        "promoted_best_checkpoint_path": str(promoted_best_checkpoint_path),
        "parent_checkpoint_snapshot_path": str(parent_checkpoint_snapshot_path) if parent_checkpoint_snapshot_path else "",
        "architecture": {
            "family": str(cfg.get("model_family", "policy_value")),
            "backbone": str(cfg.get("backbone", "mlp")),
            "input_dim": int(input_dim),
            "hidden_sizes": hidden_sizes,
            "use_layer_norm": use_layer_norm,
            "dropout": dropout,
            "residual_connections": residual_connections,
        },
        "checkpoint_path": str(final_model_path),
        "best_validation_metric": best_metric,
        "best_epoch": best_epoch,
        "benchmark_summary_path": "",
        "runtime": {
            "device": str(device),
            "mixed_precision": amp_enabled,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
        "training_controls": {
            "gradient_clip_norm": gradient_clip_norm,
            "early_stopping_patience": early_stopping_patience,
            "scheduler_type": scheduler_type,
            "scheduler": scheduler_params,
            "hard_example_oversampling_enabled": hard_example_oversampling_enabled,
            "hard_example_weight_exponent": hard_example_weight_exponent,
            "hard_example_weight_min": hard_example_weight_min,
            "hard_example_weight_max": hard_example_weight_max,
            "train_sampling_metadata": train_sampling_metadata,
        },
    }
    _write_json(final_dir / f"{model_id}.model_card.json", model_card)
    upsert_model_record(
        job.paths.models_root,
        {
            "model_id": model_id,
            "sort_ts": time.time(),
            "dataset_id": dataset_id,
            "input_dim": int(input_dim),
            "dataset_feature_schema_version": selected_dataset_feature_schema_version,
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
    job.write_event(
        "train_completed",
        model_id=model_id,
        final_model_path=str(final_model_path),
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        restored_best_checkpoint_for_export=restored_best_checkpoint,
    )
    job.logger.info(
        "training completed | model=%s | dataset=%s | selection_mode=%s | final_model=%s | best_checkpoint=%s | best_validation_metric=%.4f | best_epoch=%s | early_stopped=%s",
        model_id,
        dataset_id,
        dataset_selection_mode,
        final_model_path,
        best_checkpoint_path,
        best_metric,
        best_epoch,
        early_stopped,
    )
    return training_summary
