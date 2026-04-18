from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from songo_model_stockfish.ops.io_utils import guard_write_path, write_json_atomic


def registry_path(models_root: Path) -> Path:
    return models_root / "model_registry.json"


def load_registry(models_root: Path) -> dict[str, Any]:
    path = registry_path(models_root)
    if not path.exists():
        return {"models": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(models_root: Path, payload: dict[str, Any]) -> None:
    path = registry_path(models_root)
    write_json_atomic(path, payload, ensure_ascii=True, indent=2)


def promoted_best_dir(models_root: Path) -> Path:
    return models_root / "promoted" / "best"


def best_model_record(models_root: Path) -> dict[str, Any] | None:
    registry = load_registry(models_root)
    models = list(registry.get("models", []))
    return models[0] if models else None


def latest_model_record(models_root: Path) -> dict[str, Any] | None:
    registry = load_registry(models_root)
    models = list(registry.get("models", []))
    if not models:
        return None
    return max(models, key=lambda item: float(item.get("sort_ts", 0.0)))


def promoted_best_metadata(models_root: Path) -> dict[str, Any] | None:
    path = promoted_best_dir(models_root) / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def next_model_version(models_root: Path, prefix: str) -> int:
    registry = load_registry(models_root)
    versions: list[int] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)$")
    for item in registry.get("models", []):
        model_id = str(item.get("model_id", ""))
        match = pattern.match(model_id)
        if match:
            versions.append(int(match.group(1)))
    return (max(versions) + 1) if versions else 1


def _sort_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        bench = float(item.get("benchmark_score", -1.0))
        eval_top1 = float(item.get("evaluation_top1", -1.0))
        created = float(item.get("sort_ts", 0.0))
        return (bench, eval_top1, created)

    ranked = sorted(models, key=sort_key, reverse=True)
    for idx, model in enumerate(ranked, start=1):
        model["rank"] = idx
    return ranked


def upsert_model_record(models_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    registry = load_registry(models_root)
    models = list(registry.get("models", []))
    model_id = str(record["model_id"])
    replaced = False
    for idx, item in enumerate(models):
        if str(item.get("model_id")) == model_id:
            merged = dict(item)
            merged.update(record)
            models[idx] = merged
            replaced = True
            break
    if not replaced:
        models.append(record)
    registry["models"] = _sort_models(models)
    save_registry(models_root, registry)
    return registry


def promote_best_model(models_root: Path) -> dict[str, Any] | None:
    registry = load_registry(models_root)
    models = list(registry.get("models", []))
    if not models:
        return None
    best = models[0]
    checkpoint_path_value = str(best.get("checkpoint_path", "")).strip()
    if not checkpoint_path_value:
        return best
    checkpoint_path = Path(checkpoint_path_value)
    if not checkpoint_path.exists():
        return best

    dest_dir = promoted_best_dir(models_root)
    guard_write_path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_checkpoint = dest_dir / "model.pt"
    guard_write_path(dest_checkpoint)
    shutil.copy2(checkpoint_path, dest_checkpoint)

    model_card_path_value = str(best.get("model_card_path", "")).strip()
    if model_card_path_value:
        model_card_path = Path(model_card_path_value)
        if model_card_path.exists():
            guard_write_path(dest_dir / "model_card.json")
            shutil.copy2(model_card_path, dest_dir / "model_card.json")

    metadata = {
        "model_id": str(best.get("model_id", "")),
        "rank": int(best.get("rank", 1)),
        "checkpoint_path": str(checkpoint_path),
        "promoted_checkpoint_path": str(dest_checkpoint),
        "best_validation_metric": float(best.get("best_validation_metric", -1.0)),
        "evaluation_top1": float(best.get("evaluation_top1", -1.0)),
        "benchmark_score": float(best.get("benchmark_score", -1.0)),
    }
    write_json_atomic(dest_dir / "metadata.json", metadata, ensure_ascii=True, indent=2)
    return metadata
