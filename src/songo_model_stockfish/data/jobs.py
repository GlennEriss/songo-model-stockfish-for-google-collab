from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from songo_model_stockfish.adapters import songo_ai_game
from songo_model_stockfish.ops.job import JobContext
from songo_model_stockfish.ops.logging import utc_now_iso
from songo_model_stockfish.training.features import encode_raw_state


def _slugify_matchup(matchup_spec: str) -> str:
    return matchup_spec.replace(":", "_").replace(" ", "").replace("/", "_")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _resolve_storage_path(base: Path, configured: str | None, fallback: Path) -> Path:
    if not configured:
        return fallback
    path = Path(configured)
    if path.is_absolute():
        return path
    return base / path


def _default_raw_dir_name_for_dataset_source(dataset_source_id: str) -> str:
    if dataset_source_id.startswith("sampled_"):
        return "raw_" + dataset_source_id[len("sampled_") :]
    if dataset_source_id.startswith("data/"):
        leaf = Path(dataset_source_id).name
        return _default_raw_dir_name_for_dataset_source(leaf)
    return f"raw_{dataset_source_id}"


def _dataset_registry_path(job: JobContext) -> Path:
    return job.paths.data_root / "dataset_registry.json"


def _read_dataset_registry(job: JobContext) -> dict[str, Any]:
    path = _dataset_registry_path(job)
    if not path.exists():
        return {"dataset_sources": [], "built_datasets": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"dataset_sources": [], "built_datasets": []}
    payload.setdefault("dataset_sources", [])
    payload.setdefault("built_datasets", [])
    return payload


def _write_dataset_registry(job: JobContext, payload: dict[str, Any]) -> None:
    path = _dataset_registry_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _upsert_registry_entry(entries: list[dict[str, Any]], *, key: str, value: str, payload: dict[str, Any]) -> None:
    for index, entry in enumerate(entries):
        if str(entry.get(key, "")) == value:
            entries[index] = payload
            return
    entries.append(payload)


def _count_jsonl_files(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jsonl")) if root.exists() else 0


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for count, _line in enumerate(handle, start=1):
            pass
    return count


def _count_json_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.json") if not path.name.startswith("_"))


def _copy_tree_incremental(source_root: Path, target_root: Path, *, pattern: str) -> int:
    copied = 0
    if not source_root.exists():
        return copied
    for source_path in sorted(source_root.rglob(pattern)):
        if source_path.name.startswith("_"):
            continue
        relative_path = source_path.relative_to(source_root)
        target_path = target_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            continue
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def _resolve_dataset_source(job: JobContext, dataset_source_id: str) -> dict[str, Any]:
    registry = _read_dataset_registry(job)
    for entry in registry.get("dataset_sources", []):
        if str(entry.get("dataset_source_id", "")) == dataset_source_id:
            return entry
    job.logger.info(
        "dataset source missing from registry, probing legacy paths | dataset_source_id=%s",
        dataset_source_id,
    )
    legacy_entry = _discover_legacy_dataset_source(job, dataset_source_id)
    if legacy_entry is not None:
        job.logger.info(
            "dataset source legacy auto-registered | dataset_source_id=%s | sampled_dir=%s | raw_dir=%s",
            dataset_source_id,
            legacy_entry["sampled_dir"],
            legacy_entry["raw_dir"],
        )
        job.write_event(
            "dataset_source_legacy_autoregistered",
            dataset_source_id=dataset_source_id,
            sampled_dir=str(legacy_entry["sampled_dir"]),
            raw_dir=str(legacy_entry["raw_dir"]),
        )
        return legacy_entry
    raise FileNotFoundError(f"Dataset source introuvable dans le registre: {dataset_source_id}")


def _discover_legacy_dataset_source(job: JobContext, dataset_source_id: str) -> dict[str, Any] | None:
    candidate_sampled_dirs = [
        job.paths.data_root / dataset_source_id,
        job.paths.drive_root / dataset_source_id,
    ]
    seen_paths: set[Path] = set()

    for sampled_dir in candidate_sampled_dirs:
        sampled_dir = sampled_dir.resolve()
        if sampled_dir in seen_paths:
            continue
        seen_paths.add(sampled_dir)
        if not sampled_dir.exists() or not sampled_dir.is_dir():
            continue
        sampled_files = _count_jsonl_files(sampled_dir)
        if sampled_files <= 0:
            continue

        metadata_path = sampled_dir / "_dataset_source_metadata.json"
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            registry = _read_dataset_registry(job)
            _upsert_registry_entry(
                registry["dataset_sources"],
                key="dataset_source_id",
                value=str(payload.get("dataset_source_id", dataset_source_id)),
                payload=payload,
            )
            _write_dataset_registry(job, registry)
            return payload

        raw_dir_name = dataset_source_id
        if dataset_source_id.startswith("sampled_"):
            raw_dir_name = "raw_" + dataset_source_id[len("sampled_") :]
        raw_dir = job.paths.data_root / raw_dir_name
        if not raw_dir.exists():
            raw_dir = sampled_dir.parent / raw_dir_name

        # Legacy sources may be large; avoid an expensive full line count during
        # auto-registration and let dataset-build scan the files directly.
        payload = {
            "dataset_source_id": dataset_source_id,
            "source_mode": "legacy_import",
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "target_samples": 0,
            "games_per_matchup": 0,
            "sample_every_n_plies": 0,
            "matchups": [],
            "raw_files": _count_json_files(raw_dir),
            "sampled_files": sampled_files,
            "sampled_positions": 0,
            "source_dataset_id": "",
            "source_dataset_ids": [],
            "derivation_strategy": "",
            "derivation_params": {},
            "dataset_version": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        registry = _read_dataset_registry(job)
        _upsert_registry_entry(
            registry["dataset_sources"],
            key="dataset_source_id",
            value=dataset_source_id,
            payload=payload,
        )
        _write_dataset_registry(job, registry)
        _write_json(sampled_dir / "_dataset_source_metadata.json", payload)
        if raw_dir.exists():
            _write_json(raw_dir / "_dataset_source_metadata.json", payload)
        return payload
    return None


def _resolve_built_dataset(job: JobContext, dataset_id: str) -> dict[str, Any]:
    registry = _read_dataset_registry(job)
    for entry in registry.get("built_datasets", []):
        if str(entry.get("dataset_id", "")) == dataset_id:
            return entry
    raise FileNotFoundError(f"Built dataset introuvable dans le registre: {dataset_id}")


def _register_dataset_source(
    job: JobContext,
    *,
    dataset_source_id: str,
    source_mode: str,
    raw_dir: Path,
    sampled_dir: Path,
    target_samples: int,
    games_per_matchup: int,
    sample_every_n_plies: int,
    matchups: list[str],
    source_dataset_id: str = "",
    source_dataset_ids: list[str] | None = None,
    derivation_strategy: str = "",
    derivation_params: dict[str, Any] | None = None,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    resolved_source_dataset_ids = source_dataset_ids or ([source_dataset_id] if source_dataset_id else [])
    payload = {
        "dataset_source_id": dataset_source_id,
        "source_mode": source_mode,
        "raw_dir": str(raw_dir),
        "sampled_dir": str(sampled_dir),
        "target_samples": target_samples,
        "games_per_matchup": games_per_matchup,
        "sample_every_n_plies": sample_every_n_plies,
        "matchups": matchups,
        "raw_files": _count_json_files(raw_dir),
        "sampled_files": _count_jsonl_files(sampled_dir),
        "sampled_positions": _count_total_jsonl_lines(sampled_dir) if sampled_dir.exists() else 0,
        "source_dataset_id": source_dataset_id,
        "source_dataset_ids": resolved_source_dataset_ids,
        "derivation_strategy": derivation_strategy,
        "derivation_params": derivation_params or {},
        "dataset_version": dataset_version or utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    registry = _read_dataset_registry(job)
    _upsert_registry_entry(
        registry["dataset_sources"],
        key="dataset_source_id",
        value=dataset_source_id,
        payload=payload,
    )
    _write_dataset_registry(job, registry)
    _write_json(sampled_dir / "_dataset_source_metadata.json", payload)
    _write_json(raw_dir / "_dataset_source_metadata.json", payload)
    return payload


def _register_built_dataset(
    job: JobContext,
    *,
    dataset_id: str,
    source_dataset_id: str,
    source_dataset_ids: list[str] | None = None,
    sampled_root: Path,
    output_root: Path,
    label_cache_dir: Path,
    teacher_engine: str,
    teacher_level: str,
    split_summary: dict[str, dict[str, int]],
    labeled_samples: int,
    target_labeled_samples: int,
    build_mode: str = "teacher_label",
    parent_dataset_ids: list[str] | None = None,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    resolved_source_dataset_ids = source_dataset_ids or ([source_dataset_id] if source_dataset_id else [])
    payload = {
        "dataset_id": dataset_id,
        "source_dataset_id": source_dataset_id,
        "source_dataset_ids": resolved_source_dataset_ids,
        "sampled_root": str(sampled_root),
        "output_dir": str(output_root),
        "label_cache_dir": str(label_cache_dir),
        "teacher_engine": teacher_engine,
        "teacher_level": teacher_level,
        "splits": split_summary,
        "labeled_samples": labeled_samples,
        "target_labeled_samples": target_labeled_samples,
        "build_mode": build_mode,
        "parent_dataset_ids": parent_dataset_ids or [],
        "dataset_version": dataset_version or utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    registry = _read_dataset_registry(job)
    _upsert_registry_entry(
        registry["built_datasets"],
        key="dataset_id",
        value=dataset_id,
        payload=payload,
    )
    _write_dataset_registry(job, registry)
    _write_json(output_root / "dataset_metadata.json", payload)
    return payload


def _sample_position_signature(sample: dict[str, Any]) -> str:
    state = sample["state"]
    scores = state["scores"]
    signature_payload = {
        "board": list(state["board"]),
        "south": int(scores["south"]),
        "north": int(scores["north"]),
        "player_to_move": str(state["player_to_move"]),
    }
    encoded = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _derive_existing_dataset_source(
    *,
    source_entry: dict[str, Any],
    target_raw_dir: Path,
    target_sampled_dir: Path,
    target_samples: int,
    derivation_strategy: str,
    derivation_params: dict[str, Any],
) -> dict[str, Any]:
    source_raw_dir = Path(str(source_entry["raw_dir"]))
    source_sampled_dir = Path(str(source_entry["sampled_dir"]))
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    target_sampled_dir.mkdir(parents=True, exist_ok=True)

    if derivation_strategy not in {"unique_positions", "endgame_focus", "high_branching"}:
        raise ValueError(f"Unsupported derivation_strategy: {derivation_strategy}")

    seen_signatures: set[str] = set()
    selected_files = 0
    selected_samples = 0
    scanned_files = 0
    scanned_samples = 0
    copied_raw_files = 0

    endgame_max_board_seeds = int(derivation_params.get("endgame_max_board_seeds", 24))
    high_branching_min_legal_moves = int(derivation_params.get("high_branching_min_legal_moves", 4))

    def _keep_sample(sample: dict[str, Any]) -> bool:
        nonlocal selected_samples
        if target_samples > 0 and selected_samples >= target_samples:
            return False
        if derivation_strategy == "unique_positions":
            signature = _sample_position_signature(sample)
            if signature in seen_signatures:
                return False
            seen_signatures.add(signature)
            return True
        if derivation_strategy == "endgame_focus":
            board_seeds = int(sum(int(value) for value in sample["state"]["board"]))
            return board_seeds <= endgame_max_board_seeds
        if derivation_strategy == "high_branching":
            return len(sample.get("legal_moves", [])) >= high_branching_min_legal_moves
        return False

    for source_sampled_file in sorted(source_sampled_dir.rglob("*.jsonl")):
        scanned_files += 1
        relative_path = source_sampled_file.relative_to(source_sampled_dir)
        kept_samples: list[dict[str, Any]] = []
        for sample in _iter_jsonl(source_sampled_file):
            scanned_samples += 1
            if _keep_sample(sample):
                kept_samples.append(sample)
                selected_samples += 1
                if target_samples > 0 and selected_samples >= target_samples:
                    break

        if kept_samples:
            target_sampled_file = target_sampled_dir / relative_path
            target_sampled_file.parent.mkdir(parents=True, exist_ok=True)
            with target_sampled_file.open("w", encoding="utf-8") as handle:
                for sample in kept_samples:
                    handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
            selected_files += 1

            if source_raw_dir.exists():
                source_raw_file = source_raw_dir / relative_path.with_suffix(".json")
                target_raw_file = target_raw_dir / relative_path.with_suffix(".json")
                if source_raw_file.exists() and not target_raw_file.exists():
                    target_raw_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_raw_file, target_raw_file)
                    copied_raw_files += 1

        if target_samples > 0 and selected_samples >= target_samples:
            break

    return {
        "scanned_files": scanned_files,
        "scanned_samples": scanned_samples,
        "selected_files": selected_files,
        "selected_samples": selected_samples,
        "copied_raw_files": copied_raw_files,
    }


def _augment_existing_dataset_source(
    *,
    job: JobContext,
    source_entry: dict[str, Any],
    target_raw_dir: Path,
    target_sampled_dir: Path,
    target_samples: int,
    augmentation_params: dict[str, Any],
) -> dict[str, Any]:
    source_raw_dir = Path(str(source_entry["raw_dir"]))
    source_sampled_dir = Path(str(source_entry["sampled_dir"]))
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    target_sampled_dir.mkdir(parents=True, exist_ok=True)

    include_original_samples = bool(augmentation_params.get("include_original_samples", True))
    max_depth = max(1, int(augmentation_params.get("max_depth", 2)))
    max_branching = max(1, int(augmentation_params.get("max_branching", 3)))
    max_generated_per_source_sample = max(1, int(augmentation_params.get("max_generated_per_source_sample", 8)))

    seen_signatures: set[str] = set()
    scanned_files = 0
    scanned_samples = 0
    selected_files = 0
    selected_samples = 0
    selected_original_samples = 0
    selected_augmented_samples = 0
    duplicate_samples = 0
    copied_raw_files = 0
    next_augmented_sample_index = 0
    augmentation_depth_breakdown: dict[str, int] = {}
    source_file_breakdown: dict[str, dict[str, int]] = {}

    for source_sampled_file in sorted(source_sampled_dir.rglob("*.jsonl")):
        if target_samples > 0 and selected_samples >= target_samples:
            break
        scanned_files += 1
        relative_path = source_sampled_file.relative_to(source_sampled_dir)
        relative_name = str(relative_path)
        kept_samples: list[dict[str, Any]] = []
        file_stats = {
            "scanned_samples": 0,
            "selected_original_samples": 0,
            "selected_augmented_samples": 0,
            "duplicate_samples": 0,
            "max_depth_reached": 0,
        }

        for sample in _iter_jsonl(source_sampled_file):
            if target_samples > 0 and selected_samples >= target_samples:
                break
            scanned_samples += 1
            file_stats["scanned_samples"] += 1

            sample_signature = _sample_position_signature(sample)
            if include_original_samples:
                if sample_signature in seen_signatures:
                    duplicate_samples += 1
                    file_stats["duplicate_samples"] += 1
                else:
                    kept_samples.append(sample)
                    seen_signatures.add(sample_signature)
                    selected_samples += 1
                    selected_original_samples += 1
                    file_stats["selected_original_samples"] += 1
                    if target_samples > 0 and selected_samples >= target_samples:
                        break

            runtime_state = _raw_state_to_runtime_state(sample["state"])
            frontier: list[tuple[Any, int, list[int]]] = [(runtime_state, 0, [])]
            generated_from_sample = 0
            local_signatures: set[str] = {sample_signature}

            while frontier and generated_from_sample < max_generated_per_source_sample:
                if target_samples > 0 and selected_samples >= target_samples:
                    break
                current_state, current_depth, lineage_moves = frontier.pop(0)
                if current_depth >= max_depth:
                    continue
                legal_moves = list(songo_ai_game.legal_moves(current_state))
                if not legal_moves:
                    continue

                for move in legal_moves[:max_branching]:
                    if target_samples > 0 and selected_samples >= target_samples:
                        break
                    if generated_from_sample >= max_generated_per_source_sample:
                        break

                    next_state = songo_ai_game.simulate_move(current_state, int(move))
                    augmented_sample = _sample_position(
                        game_id=f"aug_{sample['game_id']}",
                        matchup_id=f"{sample.get('matchup_id', 'derived')}_aug",
                        sample_index=next_augmented_sample_index,
                        ply=int(sample.get("ply", 0)) + current_depth + 1,
                        seed=int(sample.get("seed", 0)),
                        state=next_state,
                    )
                    next_augmented_sample_index += 1

                    if bool(augmented_sample["state"].get("is_terminal", False)) or not augmented_sample["legal_moves"]:
                        continue

                    augmented_signature = _sample_position_signature(augmented_sample)
                    if augmented_signature in seen_signatures or augmented_signature in local_signatures:
                        duplicate_samples += 1
                        file_stats["duplicate_samples"] += 1
                        continue

                    local_signatures.add(augmented_signature)
                    seen_signatures.add(augmented_signature)
                    augmented_sample["source_engine"] = "augment_existing"
                    augmented_sample["source_level"] = "derived"
                    augmented_sample["augmented_from_sample_id"] = str(sample.get("sample_id", ""))
                    augmented_sample["augmentation_depth"] = current_depth + 1
                    augmented_sample["augmentation_lineage_moves"] = lineage_moves + [int(move)]
                    kept_samples.append(augmented_sample)
                    selected_samples += 1
                    selected_augmented_samples += 1
                    file_stats["selected_augmented_samples"] += 1
                    generated_from_sample += 1
                    depth_key = str(current_depth + 1)
                    augmentation_depth_breakdown[depth_key] = augmentation_depth_breakdown.get(depth_key, 0) + 1
                    file_stats["max_depth_reached"] = max(file_stats["max_depth_reached"], current_depth + 1)

                    if current_depth + 1 < max_depth:
                        frontier.append((next_state, current_depth + 1, lineage_moves + [int(move)]))

        if kept_samples:
            target_sampled_file = target_sampled_dir / relative_path
            target_sampled_file.parent.mkdir(parents=True, exist_ok=True)
            with target_sampled_file.open("w", encoding="utf-8") as handle:
                for kept_sample in kept_samples:
                    handle.write(json.dumps(kept_sample, ensure_ascii=True) + "\n")
            selected_files += 1

            if source_raw_dir.exists():
                source_raw_file = source_raw_dir / relative_path.with_suffix(".json")
                target_raw_file = target_raw_dir / relative_path.with_suffix(".json")
                if source_raw_file.exists() and not target_raw_file.exists():
                    target_raw_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_raw_file, target_raw_file)
                    copied_raw_files += 1

        source_file_breakdown[relative_name] = file_stats
        if scanned_files <= 3 or scanned_files % 100 == 0:
            job.logger.info(
                "dataset generation augment progress | files=%s | scanned_samples=%s | selected_samples=%s | selected_augmented_samples=%s | duplicate_samples=%s | last_file=%s",
                scanned_files,
                scanned_samples,
                selected_samples,
                selected_augmented_samples,
                duplicate_samples,
                relative_name,
            )

    return {
        "scanned_files": scanned_files,
        "scanned_samples": scanned_samples,
        "selected_files": selected_files,
        "selected_samples": selected_samples,
        "selected_original_samples": selected_original_samples,
        "selected_augmented_samples": selected_augmented_samples,
        "duplicate_samples": duplicate_samples,
        "copied_raw_files": copied_raw_files,
        "augmentation_depth_breakdown": augmentation_depth_breakdown,
        "source_file_breakdown": source_file_breakdown,
        "augmentation_params": {
            "include_original_samples": include_original_samples,
            "max_depth": max_depth,
            "max_branching": max_branching,
            "max_generated_per_source_sample": max_generated_per_source_sample,
        },
    }


def _merge_existing_dataset_sources(
    *,
    source_entries: list[dict[str, Any]],
    target_raw_dir: Path,
    target_sampled_dir: Path,
    target_samples: int,
    dedupe_sample_ids: bool,
) -> dict[str, Any]:
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    target_sampled_dir.mkdir(parents=True, exist_ok=True)

    seen_sample_ids: set[str] = set()
    scanned_files = 0
    scanned_samples = 0
    selected_files = 0
    selected_samples = 0
    duplicate_samples = 0
    copied_raw_files = 0
    source_breakdown: dict[str, dict[str, int]] = {}

    for source_entry in source_entries:
        source_dataset_id = str(source_entry["dataset_source_id"])
        source_raw_dir = Path(str(source_entry["raw_dir"]))
        source_sampled_dir = Path(str(source_entry["sampled_dir"]))
        source_stats = {
            "scanned_files": 0,
            "scanned_samples": 0,
            "selected_files": 0,
            "selected_samples": 0,
            "duplicate_samples": 0,
            "copied_raw_files": 0,
        }

        for source_sampled_file in sorted(source_sampled_dir.rglob("*.jsonl")):
            if target_samples > 0 and selected_samples >= target_samples:
                break
            scanned_files += 1
            source_stats["scanned_files"] += 1
            relative_path = source_sampled_file.relative_to(source_sampled_dir)
            target_sampled_file = target_sampled_dir / source_dataset_id / relative_path
            target_sampled_file.parent.mkdir(parents=True, exist_ok=True)

            kept_samples: list[dict[str, Any]] = []
            for sample in _iter_jsonl(source_sampled_file):
                scanned_samples += 1
                source_stats["scanned_samples"] += 1
                if target_samples > 0 and selected_samples >= target_samples:
                    break
                sample_id = str(sample.get("sample_id", ""))
                if dedupe_sample_ids and sample_id:
                    if sample_id in seen_sample_ids:
                        duplicate_samples += 1
                        source_stats["duplicate_samples"] += 1
                        continue
                    seen_sample_ids.add(sample_id)
                kept_samples.append(sample)
                selected_samples += 1
                source_stats["selected_samples"] += 1

            if kept_samples:
                with target_sampled_file.open("w", encoding="utf-8") as handle:
                    for sample in kept_samples:
                        handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
                selected_files += 1
                source_stats["selected_files"] += 1

                source_raw_file = source_raw_dir / relative_path.with_suffix(".json")
                target_raw_file = target_raw_dir / source_dataset_id / relative_path.with_suffix(".json")
                if source_raw_file.exists() and not target_raw_file.exists():
                    target_raw_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_raw_file, target_raw_file)
                    copied_raw_files += 1
                    source_stats["copied_raw_files"] += 1

        if target_samples > 0 and selected_samples >= target_samples:
            source_breakdown[source_dataset_id] = source_stats
            break
        source_breakdown[source_dataset_id] = source_stats

    return {
        "scanned_files": scanned_files,
        "scanned_samples": scanned_samples,
        "selected_files": selected_files,
        "selected_samples": selected_samples,
        "duplicate_samples": duplicate_samples,
        "copied_raw_files": copied_raw_files,
        "source_breakdown": source_breakdown,
    }


def _append_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _count_total_jsonl_lines(root: Path) -> int:
    total = 0
    for path in sorted(root.rglob("*.jsonl")):
        total += _count_jsonl_lines(path)
    return total


def _format_eta_seconds(total_seconds: float | None) -> str:
    if total_seconds is None:
        return "unknown"
    remaining = max(0, int(round(total_seconds)))
    hours, remainder = divmod(remaining, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _merge_npz_splits(
    split_paths: list[Path],
    *,
    dedupe_sample_ids: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    merged_chunks: dict[str, list[np.ndarray]] = {
        "x": [],
        "legal_mask": [],
        "policy_index": [],
        "value_target": [],
        "sample_ids": [],
        "game_ids": [],
    }
    seen_sample_ids: set[str] = set()
    input_samples = 0
    kept_samples = 0
    duplicate_samples = 0

    for split_path in split_paths:
        arrays = _load_npz_arrays(split_path)
        sample_ids = arrays["sample_ids"]
        keep_indices: list[int] = []
        for index, sample_id in enumerate(sample_ids.tolist()):
            input_samples += 1
            sample_id_str = str(sample_id)
            if dedupe_sample_ids and sample_id_str in seen_sample_ids:
                duplicate_samples += 1
                continue
            seen_sample_ids.add(sample_id_str)
            keep_indices.append(index)

        if not keep_indices:
            continue

        keep = np.asarray(keep_indices, dtype=np.int64)
        for key in merged_chunks:
            merged_chunks[key].append(arrays[key][keep])
        kept_samples += len(keep_indices)

    if kept_samples == 0:
        merged = {
            "x": np.zeros((0, 17), dtype=np.float32),
            "legal_mask": np.zeros((0, 7), dtype=np.float32),
            "policy_index": np.zeros((0,), dtype=np.int64),
            "value_target": np.zeros((0,), dtype=np.float32),
            "sample_ids": np.asarray([], dtype=object),
            "game_ids": np.asarray([], dtype=object),
        }
    else:
        merged = {}
        for key, chunks in merged_chunks.items():
            merged[key] = np.concatenate(chunks, axis=0)

    summary = {
        "input_samples": input_samples,
        "kept_samples": kept_samples,
        "duplicate_samples": duplicate_samples,
        "unique_games": len({str(value) for value in merged["game_ids"].tolist()}),
    }
    return merged, summary


def _merge_npz_splits_with_source_breakdown(
    split_items: list[tuple[str, Path]],
    *,
    dedupe_sample_ids: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, dict[str, int]]]:
    merged_chunks: dict[str, list[np.ndarray]] = {
        "x": [],
        "legal_mask": [],
        "policy_index": [],
        "value_target": [],
        "sample_ids": [],
        "game_ids": [],
    }
    seen_sample_ids: set[str] = set()
    source_breakdown: dict[str, dict[str, int]] = {}
    input_samples = 0
    kept_samples = 0
    duplicate_samples = 0

    for source_dataset_id, split_path in split_items:
        arrays = _load_npz_arrays(split_path)
        sample_ids = arrays["sample_ids"]
        keep_indices: list[int] = []
        stats = {
            "input_samples": int(len(sample_ids)),
            "kept_samples": 0,
            "duplicate_samples": 0,
            "unique_games": 0,
        }
        source_games: set[str] = set()
        for index, sample_id in enumerate(sample_ids.tolist()):
            input_samples += 1
            sample_id_str = str(sample_id)
            if dedupe_sample_ids and sample_id_str in seen_sample_ids:
                duplicate_samples += 1
                stats["duplicate_samples"] += 1
                continue
            seen_sample_ids.add(sample_id_str)
            keep_indices.append(index)
            source_games.add(str(arrays["game_ids"][index]))

        if keep_indices:
            keep = np.asarray(keep_indices, dtype=np.int64)
            for key in merged_chunks:
                merged_chunks[key].append(arrays[key][keep])
            kept_samples += len(keep_indices)
            stats["kept_samples"] = len(keep_indices)
            stats["unique_games"] = len(source_games)
        source_breakdown[source_dataset_id] = stats

    if kept_samples == 0:
        merged = {
            "x": np.zeros((0, 17), dtype=np.float32),
            "legal_mask": np.zeros((0, 7), dtype=np.float32),
            "policy_index": np.zeros((0,), dtype=np.int64),
            "value_target": np.zeros((0,), dtype=np.float32),
            "sample_ids": np.asarray([], dtype=object),
            "game_ids": np.asarray([], dtype=object),
        }
    else:
        merged = {}
        for key, chunks in merged_chunks.items():
            merged[key] = np.concatenate(chunks, axis=0)

    summary = {
        "input_samples": input_samples,
        "kept_samples": kept_samples,
        "duplicate_samples": duplicate_samples,
        "unique_games": len({str(value) for value in merged["game_ids"].tolist()}),
    }
    return merged, summary, source_breakdown


def _existing_game_numbers(raw_dir: Path, sampled_dir: Path, matchup_id: str) -> set[int]:
    raw_matchup_dir = raw_dir / matchup_id
    sampled_matchup_dir = sampled_dir / matchup_id
    raw_stems = {path.stem for path in raw_matchup_dir.glob("*.json")} if raw_matchup_dir.exists() else set()
    sampled_stems = {path.stem for path in sampled_matchup_dir.glob("*.jsonl")} if sampled_matchup_dir.exists() else set()
    completed = raw_stems & sampled_stems
    game_numbers: set[int] = set()
    prefix = f"{matchup_id}_game_"
    for stem in completed:
        if stem.startswith(prefix):
            suffix = stem[len(prefix) :]
            if suffix.isdigit():
                game_numbers.add(int(suffix))
    return game_numbers


def _build_pending_incremental_games(
    *,
    raw_dir: Path,
    sampled_dir: Path,
    matchup_id: str,
    games_to_add: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    existing_numbers = _existing_game_numbers(raw_dir, sampled_dir, matchup_id)
    pending: list[dict[str, Any]] = []
    candidate = 1
    while len(pending) < games_to_add:
        game_id = f"{matchup_id}_game_{candidate:06d}"
        raw_game_path = raw_dir / matchup_id / f"{game_id}.json"
        sampled_game_path = sampled_dir / matchup_id / f"{game_id}.jsonl"
        if candidate not in existing_numbers or not (raw_game_path.exists() and sampled_game_path.exists()):
            pending.append(
                {
                    "game_index": candidate - 1,
                    "game_no": candidate,
                    "starter": (candidate - 1) % 2,
                    "seed": seed_base + (candidate - 1),
                    "game_id": game_id,
                    "raw_game_path": raw_game_path,
                    "sampled_game_path": sampled_game_path,
                }
            )
        candidate += 1
    return pending


def _build_external_agent(spec: str):
    from songo_model_stockfish.reference_songo.agents import MCTSAgent, MinimaxAgent

    kind, level = spec.split(":", 1)
    if kind == "minimax":
        return MinimaxAgent(level)
    if kind == "mcts":
        return MCTSAgent(level)
    raise ValueError(f"Unsupported dataset generation agent: {spec}")


def _teacher_choose(state: Any, *, engine: str, level: str) -> tuple[int, dict[str, Any]]:
    if engine == "minimax":
        from songo_model_stockfish.reference_songo.levels import get_config
        from songo_model_stockfish.reference_songo.minimax import choose_move

        return choose_move(songo_ai_game.clone_state(state), get_config(level))
    if engine == "mcts":
        from songo_model_stockfish.reference_songo.levels import get_mcts_config
        from songo_model_stockfish.reference_songo.mcts import choose_move

        return choose_move(songo_ai_game.clone_state(state), get_mcts_config(level))
    raise ValueError(f"Unsupported teacher engine: {engine}")


def _parse_matchup(matchup_spec: str) -> tuple[str, str]:
    parts = matchup_spec.split(" vs ")
    if len(parts) != 2:
        raise ValueError(f"Invalid matchup spec: {matchup_spec}")
    return parts[0].strip(), parts[1].strip()


def _sample_position(
    *,
    game_id: str,
    matchup_id: str,
    sample_index: int,
    ply: int,
    seed: int,
    state: Any,
) -> dict[str, Any]:
    raw_state = songo_ai_game.to_raw_state(state)
    raw_state["turn_index"] = ply
    legal_moves = songo_ai_game.legal_moves(state)
    effective_terminal = bool(raw_state.get("is_terminal", False)) or not legal_moves
    raw_state["is_terminal"] = effective_terminal
    return {
        "sample_id": f"{game_id}_sample_{sample_index:06d}",
        "game_id": game_id,
        "matchup_id": matchup_id,
        "ply": ply,
        "seed": seed,
        "player_to_move": raw_state["player_to_move"],
        "state": raw_state,
        "legal_moves": [] if effective_terminal else legal_moves,
        "source_engine": "match_replay",
        "source_level": "mixed",
    }


def _raw_state_to_runtime_state(raw_state: dict[str, Any]) -> Any:
    board_flat = list(raw_state["board"])
    if len(board_flat) != 14:
        raise ValueError("Invalid raw_state board length")
    current_player = 0 if raw_state["player_to_move"] == "south" else 1
    scores = raw_state["scores"]
    return {
        "board": [board_flat[:7], board_flat[7:]],
        "scores": [int(scores["south"]), int(scores["north"])],
        "current_player": current_player,
        "finished": bool(raw_state.get("is_terminal", False)),
        "winner": None,
        "reason": "",
        "turn_index": int(raw_state.get("turn_index", 0)),
    }


def _normalize_value(value: float) -> float:
    clipped = max(-1_000_000.0, min(1_000_000.0, float(value)))
    return float(np.tanh(clipped / 200.0))


def _build_policy_distribution_from_scores(
    legal_moves: list[int],
    move_scores: dict[int, float],
    *,
    temperature: float = 0.35,
) -> dict[str, float]:
    distribution = {str(move): 0.0 for move in legal_moves}
    if not legal_moves:
        return distribution
    if len(legal_moves) == 1:
        distribution[str(legal_moves[0])] = 1.0
        return distribution

    logits = np.asarray(
        [_normalize_value(move_scores.get(int(move), 0.0)) / max(temperature, 1e-6) for move in legal_moves],
        dtype=np.float64,
    )
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        uniform = 1.0 / float(len(legal_moves))
        for move in legal_moves:
            distribution[str(move)] = uniform
        return distribution

    probs = weights / total
    for move, prob in zip(legal_moves, probs.tolist()):
        distribution[str(move)] = float(prob)
    return distribution


def _build_tactical_analysis(
    runtime_state: Any,
    legal_moves: list[int],
    move_scores: dict[int, float],
    *,
    best_move: int,
) -> dict[str, Any]:
    root_player = int(runtime_state["current_player"])
    opponent = 1 - root_player
    root_scores = list(runtime_state["scores"])
    root_player_score = int(root_scores[root_player])
    root_opponent_score = int(root_scores[opponent])

    per_move: dict[str, dict[str, Any]] = {}
    capture_moves_count = 0
    safe_moves_count = 0
    risky_moves_count = 0
    max_immediate_score_gain = 0
    min_opponent_best_gain: int | None = None

    for move in legal_moves:
        next_state = songo_ai_game.simulate_move(runtime_state, int(move))
        next_scores = list(next_state["scores"])
        immediate_score_gain = int(next_scores[root_player]) - root_player_score
        immediate_score_loss = int(next_scores[opponent]) - root_opponent_score
        opponent_legal_moves = list(songo_ai_game.legal_moves(next_state))

        opponent_best_immediate_gain = 0
        opponent_capture_moves = 0
        for opponent_move in opponent_legal_moves:
            reply_state = songo_ai_game.simulate_move(next_state, int(opponent_move))
            reply_scores = list(reply_state["scores"])
            gain = int(reply_scores[opponent]) - int(next_scores[opponent])
            if gain > 0:
                opponent_capture_moves += 1
            if gain > opponent_best_immediate_gain:
                opponent_best_immediate_gain = gain

        net_score_swing = int(immediate_score_gain) - int(opponent_best_immediate_gain)
        has_immediate_capture = immediate_score_gain > 0
        exposes_to_immediate_capture = opponent_best_immediate_gain > 0
        if has_immediate_capture:
            capture_moves_count += 1
        if exposes_to_immediate_capture:
            risky_moves_count += 1
        else:
            safe_moves_count += 1

        max_immediate_score_gain = max(max_immediate_score_gain, int(immediate_score_gain))
        if min_opponent_best_gain is None or opponent_best_immediate_gain < min_opponent_best_gain:
            min_opponent_best_gain = int(opponent_best_immediate_gain)

        per_move[str(move)] = {
            "teacher_score": float(move_scores.get(int(move), 0.0)),
            "immediate_score_gain": int(immediate_score_gain),
            "immediate_score_loss": int(immediate_score_loss),
            "has_immediate_capture": bool(has_immediate_capture),
            "opponent_legal_moves_count": int(len(opponent_legal_moves)),
            "opponent_capture_moves_count": int(opponent_capture_moves),
            "opponent_best_immediate_gain": int(opponent_best_immediate_gain),
            "exposes_to_immediate_capture": bool(exposes_to_immediate_capture),
            "net_score_swing_next_ply": int(net_score_swing),
        }

    best_move_stats = per_move.get(str(best_move), {})
    return {
        "summary": {
            "capture_moves_count": int(capture_moves_count),
            "safe_moves_count": int(safe_moves_count),
            "risky_moves_count": int(risky_moves_count),
            "max_immediate_score_gain": int(max_immediate_score_gain),
            "min_opponent_best_immediate_gain": int(min_opponent_best_gain or 0),
            "best_move_has_immediate_capture": bool(best_move_stats.get("has_immediate_capture", False)),
            "best_move_exposes_to_immediate_capture": bool(best_move_stats.get("exposes_to_immediate_capture", False)),
            "best_move_net_score_swing_next_ply": int(best_move_stats.get("net_score_swing_next_ply", 0)),
        },
        "per_move": per_move,
    }


def _label_sample(sample: dict[str, Any], *, teacher_engine: str, teacher_level: str) -> dict[str, Any]:
    if bool(sample["state"].get("is_terminal", False)):
        raise ValueError(f"Cannot label terminal sample: {sample['sample_id']}")

    if not sample["legal_moves"]:
        raise ValueError(f"Cannot label sample without legal moves: {sample['sample_id']}")

    runtime_state = _raw_state_to_runtime_state(sample["state"])
    best_move, info = _teacher_choose(runtime_state, engine=teacher_engine, level=teacher_level)
    legal_moves = list(sample["legal_moves"])
    move_scores: dict[int, float] = {}
    if teacher_engine == "minimax":
        move_scores = {int(move): float(score) for move, score in dict(info.get("root_scores", {})).items() if int(move) in legal_moves}
    elif teacher_engine == "mcts":
        root_q = info.get("root_q", {})
        move_scores = {int(move): float(score) for move, score in dict(root_q).items() if int(move) in legal_moves}

    if not move_scores:
        fallback_score = float(info.get("score", 0.0))
        move_scores = {int(move): fallback_score for move in legal_moves}

    if best_move not in legal_moves and legal_moves:
        best_move = legal_moves[0]
    if legal_moves:
        best_move = max(legal_moves, key=lambda move: float(move_scores.get(int(move), float("-inf"))))

    teacher_score = float(move_scores.get(int(best_move), info.get("score", 0.0)))

    labeled = dict(sample)
    labeled["teacher_engine"] = teacher_engine
    labeled["teacher_level"] = teacher_level
    labeled["teacher_move_scores"] = {str(move): float(move_scores.get(int(move), 0.0)) for move in legal_moves}
    labeled["policy_target"] = {
        "best_move": int(best_move),
        "distribution": _build_policy_distribution_from_scores(legal_moves, move_scores),
    }
    labeled["tactical_analysis"] = _build_tactical_analysis(
        runtime_state,
        legal_moves,
        move_scores,
        best_move=int(best_move),
    )
    labeled["value_target"] = _normalize_value(teacher_score)
    return labeled


def _encode_features(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, float]:
    features, legal_mask = encode_raw_state(sample["state"], sample["legal_moves"])
    policy_index = int(sample["policy_target"]["best_move"]) - 1
    value = float(sample["value_target"])
    return features, legal_mask, policy_index, value


def _label_samples_from_file(
    sampled_file_path: str,
    *,
    teacher_engine: str,
    teacher_level: str,
) -> tuple[int, list[dict[str, Any]], int, int]:
    source_samples = list(_iter_jsonl(Path(sampled_file_path)))
    source_count = len(source_samples)
    labeled_samples: list[dict[str, Any]] = []
    skipped_terminal = 0
    skipped_no_legal = 0
    for sample in source_samples:
        if bool(sample["state"].get("is_terminal", False)):
            skipped_terminal += 1
            continue
        if not sample["legal_moves"]:
            skipped_no_legal += 1
            continue
        labeled_samples.append(_label_sample(sample, teacher_engine=teacher_engine, teacher_level=teacher_level))
    return source_count, labeled_samples, skipped_terminal, skipped_no_legal


def _play_and_sample_game(
    agent_a,
    agent_b,
    *,
    matchup_id: str,
    game_id: str,
    seed: int,
    starter: int,
    sample_every_n_plies: int,
    max_moves: int = 300,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = songo_ai_game.create_state()
    agents = [agent_a, agent_b] if starter == 0 else [agent_b, agent_a]
    moves: list[int] = []
    samples: list[dict[str, Any]] = []
    ply = 0
    sample_index = 0
    started_at = utc_now_iso()
    end_reason = "finished"

    while not songo_ai_game.is_terminal(state) and ply < max_moves:
        if ply % sample_every_n_plies == 0:
            sample_index += 1
            samples.append(
                _sample_position(
                    game_id=game_id,
                    matchup_id=matchup_id,
                    sample_index=sample_index,
                    ply=ply,
                    seed=seed,
                    state=state,
                )
            )

        legal = songo_ai_game.legal_moves(state)
        if not legal:
            end_reason = "no_legal_moves_available"
            break

        current = songo_ai_game.current_player(state)
        try:
            move, _info = agents[current].choose(songo_ai_game.clone_state(state))
        except Exception:
            if legal:
                move = legal[0]
            else:
                end_reason = "agent_failed_no_legal_moves"
                break
        if move not in legal:
            move = legal[0]
        moves.append(int(move))
        state = songo_ai_game.simulate_move(state, int(move))
        ply += 1

    if sample_every_n_plies > 0 and (not samples or samples[-1]["ply"] != ply):
        sample_index += 1
        samples.append(
            _sample_position(
                game_id=game_id,
                matchup_id=matchup_id,
                sample_index=sample_index,
                ply=ply,
                seed=seed,
                state=state,
            )
        )

    winner = songo_ai_game.winner(state)
    if starter == 1 and winner is not None:
        logical_winner = "player_a" if winner == 1 else "player_b"
    elif winner is not None:
        logical_winner = "player_a" if winner == 0 else "player_b"
    else:
        logical_winner = None

    raw_log = {
        "game_id": game_id,
        "matchup_id": matchup_id,
        "seed": seed,
        "starter": starter,
        "player_a": agent_a.display_name,
        "player_b": agent_b.display_name,
        "winner": logical_winner,
        "moves": moves,
        "ply_count": ply,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "scores": list(songo_ai_game.scores(state)),
        "reason": "finished" if songo_ai_game.is_terminal(state) else (end_reason if end_reason != "finished" else f"max_moves_reached:{max_moves}"),
    }
    return raw_log, samples


def _play_and_sample_game_from_specs(
    matchup_a: str,
    matchup_b: str,
    *,
    matchup_id: str,
    game_id: str,
    seed: int,
    starter: int,
    sample_every_n_plies: int,
    max_moves: int = 300,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    agent_a = _build_external_agent(matchup_a)
    agent_b = _build_external_agent(matchup_b)
    return _play_and_sample_game(
        agent_a,
        agent_b,
        matchup_id=matchup_id,
        game_id=game_id,
        seed=seed,
        starter=starter,
        sample_every_n_plies=sample_every_n_plies,
        max_moves=max_moves,
    )


def _materialize_completed_game(
    job: JobContext,
    *,
    pending: dict[str, Any],
    raw_payload: dict[str, Any],
    samples: list[dict[str, Any]],
    matchup_id: str,
    games: int,
    execution_mode: str,
) -> tuple[int, int]:
    _write_json(Path(pending["raw_game_path"]), raw_payload)
    sampled_game_path = Path(pending["sampled_game_path"])
    sampled_game_path.parent.mkdir(parents=True, exist_ok=True)
    with sampled_game_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")

    sample_count = len(samples)
    job.logger.info(
        "dataset game completed | matchup=%s | game=%s/%s | moves=%s | samples=%s | winner=%s | mode=%s",
        matchup_id,
        pending["game_no"],
        games,
        raw_payload["ply_count"],
        sample_count,
        raw_payload["winner"],
        execution_mode,
    )
    job.write_event(
        "dataset_game_completed",
        matchup=matchup_id,
        game_id=pending["game_id"],
        game_index=pending["game_no"],
        samples=sample_count,
        winner=raw_payload["winner"],
        execution_mode=execution_mode,
    )
    job.write_metric(
        {
            "metric_type": "dataset_game_completed",
            "matchup_id": matchup_id,
            "game_id": pending["game_id"],
            "samples": sample_count,
            "moves": raw_payload["ply_count"],
        }
    )
    return 1, sample_count


def _run_pending_games_sequential(
    job: JobContext,
    *,
    pending_games: list[dict[str, Any]],
    matchup_a: str,
    matchup_b: str,
    matchup_id: str,
    games: int,
    sample_every_n_plies: int,
    max_moves: int,
) -> tuple[int, int, str]:
    agent_a = _build_external_agent(matchup_a)
    agent_b = _build_external_agent(matchup_b)
    completed_games = 0
    completed_samples = 0
    for pending in pending_games:
        job.logger.info(
            "dataset game running | matchup=%s | game=%s/%s | seed=%s | starter=%s | mode=sequential",
            matchup_id,
            pending["game_no"],
            games,
            pending["seed"],
            pending["starter"],
        )
        raw_payload, samples = _play_and_sample_game(
            agent_a,
            agent_b,
            matchup_id=matchup_id,
            game_id=str(pending["game_id"]),
            seed=int(pending["seed"]),
            starter=int(pending["starter"]),
            sample_every_n_plies=sample_every_n_plies,
            max_moves=max_moves,
        )
        games_inc, samples_inc = _materialize_completed_game(
            job,
            pending=pending,
            raw_payload=raw_payload,
            samples=samples,
            matchup_id=matchup_id,
            games=games,
            execution_mode="sequential",
        )
        completed_games += games_inc
        completed_samples += samples_inc
    return completed_games, completed_samples, "sequential"


def run_dataset_generation(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("dataset_generation", {})
    runtime_cfg = job.config.get("runtime", {})
    source_mode = str(cfg.get("source_mode", "benchmatch")).strip().lower() or "benchmatch"
    dataset_source_id = str(cfg.get("dataset_source_id", "")).strip() or Path(str(cfg.get("output_sampled_dir", "sampled_positions"))).name
    source_dataset_id = str(cfg.get("source_dataset_id", "")).strip()
    source_dataset_ids = [str(value).strip() for value in cfg.get("source_dataset_ids", []) if str(value).strip()]
    derivation_strategy = str(cfg.get("derivation_strategy", "unique_positions")).strip().lower() or "unique_positions"
    derivation_params = dict(cfg.get("derivation_params", {}))
    merge_dedupe_sample_ids = bool(cfg.get("merge_dedupe_sample_ids", True))
    games = int(cfg.get("games", 20))
    matchups = list(cfg.get("matchups", []))
    sample_every_n_plies = int(cfg.get("sample_every_n_plies", 2))
    if sample_every_n_plies <= 0:
        raise ValueError("`sample_every_n_plies` doit etre strictement positif")
    base_seed = int(job.config.get("runtime", {}).get("seed", 42))
    max_moves = int(cfg.get("max_moves", 300))
    num_workers = max(1, int(runtime_cfg.get("num_workers", 1)))
    max_pending_futures = max(1, int(cfg.get("max_pending_futures", num_workers * 2)))
    multiprocessing_start_method = str(runtime_cfg.get("multiprocessing_start_method", "spawn")).strip().lower() or "spawn"
    max_tasks_per_child = int(runtime_cfg.get("max_tasks_per_child", 25))
    target_samples = int(cfg.get("target_samples", 0))

    dataset_dir = job.job_dir / "dataset_generation"
    configured_output_raw_dir = cfg.get("output_raw_dir")
    configured_output_sampled_dir = cfg.get("output_sampled_dir")
    raw_dir = _resolve_storage_path(job.paths.drive_root, configured_output_raw_dir, dataset_dir / "raw_match_logs")
    sampled_dir = _resolve_storage_path(job.paths.drive_root, configured_output_sampled_dir, dataset_dir / "sampled_positions")

    if source_mode != "benchmatch":
        sampled_dir = _resolve_storage_path(
            job.paths.drive_root,
            None,
            job.paths.data_root / dataset_source_id,
        )
        raw_dir = _resolve_storage_path(
            job.paths.drive_root,
            None,
            job.paths.data_root / _default_raw_dir_name_for_dataset_source(dataset_source_id),
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir.mkdir(parents=True, exist_ok=True)

    if source_mode != "benchmatch":
        job.logger.info(
            "dataset generation output isolation enabled | dataset_source_id=%s | isolated_raw_dir=%s | isolated_sampled_dir=%s",
            dataset_source_id,
            raw_dir,
            sampled_dir,
        )

    job.logger.info(
        "dataset generation startup | source_mode=%s | dataset_source_id=%s | source_dataset_id=%s | source_dataset_ids=%s | target_samples=%s | output_raw_dir=%s | output_sampled_dir=%s | workers=%s | max_pending_futures=%s",
        source_mode,
        dataset_source_id,
        source_dataset_id or "<none>",
        source_dataset_ids,
        target_samples,
        raw_dir,
        sampled_dir,
        num_workers,
        max_pending_futures,
    )

    if source_mode not in {"benchmatch", "clone_existing", "derive_existing", "augment_existing", "merge_existing"}:
        raise ValueError(f"Unsupported dataset generation source_mode: {source_mode}")

    if source_mode == "clone_existing":
        if not source_dataset_id:
            raise ValueError("`source_dataset_id` est requis quand `source_mode=clone_existing`")
        source_entry = _resolve_dataset_source(job, source_dataset_id)
        source_raw_dir = Path(str(source_entry["raw_dir"]))
        source_sampled_dir = Path(str(source_entry["sampled_dir"]))
        copied_raw_files = _copy_tree_incremental(source_raw_dir, raw_dir, pattern="*.json")
        copied_sampled_files = _copy_tree_incremental(source_sampled_dir, sampled_dir, pattern="*.jsonl")
        metadata = _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_id,
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "source_dataset_id": source_dataset_id,
            "copied_raw_files": copied_raw_files,
            "copied_sampled_files": copied_sampled_files,
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "total_samples": int(metadata["sampled_positions"]),
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation cloned existing source | source_dataset_id=%s | dataset_source_id=%s | copied_raw_files=%s | copied_sampled_files=%s | total_samples=%s",
            source_dataset_id,
            dataset_source_id,
            copied_raw_files,
            copied_sampled_files,
            metadata["sampled_positions"],
        )
        job.write_event(
            "dataset_generation_cloned_existing",
            source_dataset_id=source_dataset_id,
            dataset_source_id=dataset_source_id,
            copied_raw_files=copied_raw_files,
            copied_sampled_files=copied_sampled_files,
        )
        job.write_metric(
            {
                "metric_type": "dataset_generation_cloned_existing",
                "source_dataset_id": source_dataset_id,
                "dataset_source_id": dataset_source_id,
                "copied_raw_files": copied_raw_files,
                "copied_sampled_files": copied_sampled_files,
            }
        )
        return summary

    if source_mode == "derive_existing":
        if not source_dataset_id:
            raise ValueError("`source_dataset_id` est requis quand `source_mode=derive_existing`")
        source_entry = _resolve_dataset_source(job, source_dataset_id)
        derived_summary = _derive_existing_dataset_source(
            source_entry=source_entry,
            target_raw_dir=raw_dir,
            target_sampled_dir=sampled_dir,
            target_samples=target_samples,
            derivation_strategy=derivation_strategy,
            derivation_params=derivation_params,
        )
        metadata = _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_id,
            derivation_strategy=derivation_strategy,
            derivation_params=derivation_params,
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "source_dataset_id": source_dataset_id,
            "derivation_strategy": derivation_strategy,
            "derivation_params": derivation_params,
            **derived_summary,
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "total_samples": int(metadata["sampled_positions"]),
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation derived existing source | source_dataset_id=%s | dataset_source_id=%s | strategy=%s | selected_files=%s | selected_samples=%s",
            source_dataset_id,
            dataset_source_id,
            derivation_strategy,
            derived_summary["selected_files"],
            derived_summary["selected_samples"],
        )
        job.write_event(
            "dataset_generation_derived_existing",
            source_dataset_id=source_dataset_id,
            dataset_source_id=dataset_source_id,
            derivation_strategy=derivation_strategy,
            selected_files=derived_summary["selected_files"],
            selected_samples=derived_summary["selected_samples"],
        )
        job.write_metric(
            {
                "metric_type": "dataset_generation_derived_existing",
                "source_dataset_id": source_dataset_id,
                "dataset_source_id": dataset_source_id,
                "derivation_strategy": derivation_strategy,
                "selected_files": derived_summary["selected_files"],
                "selected_samples": derived_summary["selected_samples"],
            }
        )
        return summary

    if source_mode == "augment_existing":
        if not source_dataset_id:
            raise ValueError("`source_dataset_id` est requis quand `source_mode=augment_existing`")
        source_entry = _resolve_dataset_source(job, source_dataset_id)
        augmented_summary = _augment_existing_dataset_source(
            job=job,
            source_entry=source_entry,
            target_raw_dir=raw_dir,
            target_sampled_dir=sampled_dir,
            target_samples=target_samples,
            augmentation_params=derivation_params,
        )
        metadata = _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_id,
            derivation_params=augmented_summary["augmentation_params"],
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "source_dataset_id": source_dataset_id,
            **augmented_summary,
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "total_samples": int(metadata["sampled_positions"]),
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation augmented existing source | source_dataset_id=%s | dataset_source_id=%s | selected_original_samples=%s | selected_augmented_samples=%s | duplicate_samples=%s | total_samples=%s",
            source_dataset_id,
            dataset_source_id,
            augmented_summary["selected_original_samples"],
            augmented_summary["selected_augmented_samples"],
            augmented_summary["duplicate_samples"],
            metadata["sampled_positions"],
        )
        job.write_event(
            "dataset_generation_augmented_existing",
            source_dataset_id=source_dataset_id,
            dataset_source_id=dataset_source_id,
            selected_original_samples=augmented_summary["selected_original_samples"],
            selected_augmented_samples=augmented_summary["selected_augmented_samples"],
            duplicate_samples=augmented_summary["duplicate_samples"],
        )
        job.write_metric(
            {
                "metric_type": "dataset_generation_augmented_existing",
                "source_dataset_id": source_dataset_id,
                "dataset_source_id": dataset_source_id,
                "selected_original_samples": augmented_summary["selected_original_samples"],
                "selected_augmented_samples": augmented_summary["selected_augmented_samples"],
                "duplicate_samples": augmented_summary["duplicate_samples"],
                "total_samples": int(metadata["sampled_positions"]),
            }
        )
        return summary

    if source_mode == "merge_existing":
        if not source_dataset_ids:
            raise ValueError("`source_dataset_ids` est requis quand `source_mode=merge_existing`")
        source_entries = [_resolve_dataset_source(job, value) for value in source_dataset_ids]
        merged_summary = _merge_existing_dataset_sources(
            source_entries=source_entries,
            target_raw_dir=raw_dir,
            target_sampled_dir=sampled_dir,
            target_samples=target_samples,
            dedupe_sample_ids=merge_dedupe_sample_ids,
        )
        metadata = _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_ids[0],
            source_dataset_ids=source_dataset_ids,
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "source_dataset_ids": source_dataset_ids,
            "merge_dedupe_sample_ids": merge_dedupe_sample_ids,
            **merged_summary,
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "total_samples": int(metadata["sampled_positions"]),
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation merged existing sources | dataset_source_id=%s | source_datasets=%s | selected_files=%s | selected_samples=%s | duplicate_samples=%s",
            dataset_source_id,
            len(source_dataset_ids),
            merged_summary["selected_files"],
            merged_summary["selected_samples"],
            merged_summary["duplicate_samples"],
        )
        for merged_source_id, merged_stats in merged_summary["source_breakdown"].items():
            job.logger.info(
                "dataset generation merged source breakdown | dataset_source_id=%s | source_dataset_id=%s | scanned_files=%s | scanned_samples=%s | selected_files=%s | selected_samples=%s | duplicate_samples=%s | copied_raw_files=%s",
                dataset_source_id,
                merged_source_id,
                merged_stats["scanned_files"],
                merged_stats["scanned_samples"],
                merged_stats["selected_files"],
                merged_stats["selected_samples"],
                merged_stats["duplicate_samples"],
                merged_stats["copied_raw_files"],
            )
        job.write_event(
            "dataset_generation_merged_existing",
            dataset_source_id=dataset_source_id,
            source_dataset_ids=source_dataset_ids,
            selected_files=merged_summary["selected_files"],
            selected_samples=merged_summary["selected_samples"],
            duplicate_samples=merged_summary["duplicate_samples"],
            source_breakdown=merged_summary["source_breakdown"],
        )
        job.write_metric(
            {
                "metric_type": "dataset_generation_merged_existing",
                "dataset_source_id": dataset_source_id,
                "source_datasets": len(source_dataset_ids),
                "selected_files": merged_summary["selected_files"],
                "selected_samples": merged_summary["selected_samples"],
                "duplicate_samples": merged_summary["duplicate_samples"],
            }
        )
        return summary

    state = job.read_state()
    summaries: list[dict[str, Any]] = []
    initial_total_samples = _count_total_jsonl_lines(sampled_dir)
    total_samples_after_run = initial_total_samples

    job.logger.info("dataset generation started")
    job.set_phase("dataset_generation")
    job.write_event(
        "dataset_generation_started",
        existing_samples=initial_total_samples,
        target_samples=target_samples,
    )
    job.write_metric(
        {
            "metric_type": "dataset_generation_started",
            "existing_samples": initial_total_samples,
            "target_samples": target_samples,
        }
    )

    if target_samples > 0 and initial_total_samples >= target_samples:
        _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_id,
            derivation_strategy=derivation_strategy,
            derivation_params=derivation_params,
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "matchups": [],
            "existing_samples": initial_total_samples,
            "added_samples": 0,
            "total_samples": initial_total_samples,
            "target_samples": target_samples,
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation skipped | target already reached | existing_samples=%s | target_samples=%s",
            initial_total_samples,
            target_samples,
        )
        job.write_state(
            {
                "current_matchup": None,
                "completed_games": 0,
                "remaining_games": 0,
                "last_completed_game_id": state.get("last_completed_game_id"),
                "sample_count": initial_total_samples,
                "target_samples": target_samples,
            }
        )
        return summary

    for matchup_index, matchup_spec in enumerate(matchups):
        matchup_a, matchup_b = _parse_matchup(str(matchup_spec))
        matchup_id = _slugify_matchup(str(matchup_spec))
        job.set_phase(f"dataset_generation:{matchup_id}")
        summary_path = dataset_dir / f"{matchup_id}_summary.json"
        if target_samples > 0 and total_samples_after_run >= target_samples:
            job.logger.info(
                "dataset generation target reached during run | current_samples=%s | target_samples=%s",
                total_samples_after_run,
                target_samples,
            )
            break

        matchup_game_count = 0
        matchup_sample_count = 0
        existing_numbers = _existing_game_numbers(raw_dir, sampled_dir, matchup_id)
        existing_game_count = len(existing_numbers)
        job.logger.info(
            "dataset matchup started | %s/%s | %s vs %s | add_games=%s | existing_games=%s | sample_every=%s | workers=%s | target_samples=%s",
            matchup_index + 1,
            len(matchups),
            matchup_a,
            matchup_b,
            games,
            existing_game_count,
            sample_every_n_plies,
            num_workers,
            target_samples,
        )
        job.write_event(
            "dataset_matchup_started",
            matchup=matchup_id,
            matchup_index=matchup_index + 1,
            total_matchups=len(matchups),
            player_a=matchup_a,
            player_b=matchup_b,
            add_games=games,
            existing_games=existing_game_count,
            sample_every_n_plies=sample_every_n_plies,
            num_workers=num_workers,
        )

        pending_games = _build_pending_incremental_games(
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            matchup_id=matchup_id,
            games_to_add=games,
            seed_base=base_seed + (matchup_index * 1_000_000),
        )

        if num_workers <= 1:
            completed_games, completed_samples, _execution_mode = _run_pending_games_sequential(
                job,
                pending_games=pending_games,
                matchup_a=matchup_a,
                matchup_b=matchup_b,
                matchup_id=matchup_id,
                games=games,
                sample_every_n_plies=sample_every_n_plies,
                max_moves=max_moves,
            )
            matchup_game_count += completed_games
            matchup_sample_count += completed_samples
            if pending_games:
                job.write_state(
                    {
                        "current_matchup": matchup_id,
                        "completed_games": matchup_game_count,
                        "remaining_games": games - matchup_game_count,
                        "last_completed_game_id": pending_games[-1]["game_id"],
                        "sample_count": total_samples_after_run + matchup_sample_count,
                        "target_samples": target_samples,
                    }
                )
        else:
            job.logger.info(
                "dataset matchup parallel execution | matchup=%s | workers=%s | pending_games=%s | max_pending=%s | start_method=%s | max_tasks_per_child=%s",
                matchup_id,
                num_workers,
                len(pending_games),
                max_pending_futures,
                multiprocessing_start_method,
                max_tasks_per_child,
            )
            job.write_event(
                "dataset_parallel_execution_started",
                matchup=matchup_id,
                workers=num_workers,
                pending_games=len(pending_games),
                max_pending_futures=max_pending_futures,
                multiprocessing_start_method=multiprocessing_start_method,
                max_tasks_per_child=max_tasks_per_child,
            )

            future_map: dict[concurrent.futures.Future, dict[str, Any]] = {}
            pending_queue = list(pending_games)
            try:
                mp_context = multiprocessing.get_context(multiprocessing_start_method)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=mp_context,
                    max_tasks_per_child=max_tasks_per_child,
                ) as executor:
                    while pending_queue or future_map:
                        while pending_queue and len(future_map) < max_pending_futures:
                            pending = pending_queue.pop(0)
                            job.logger.info(
                                "dataset game scheduled | matchup=%s | game=%s/%s | seed=%s | starter=%s | mode=parallel",
                                matchup_id,
                                pending["game_no"],
                                games,
                                pending["seed"],
                                pending["starter"],
                            )
                            future = executor.submit(
                                _play_and_sample_game_from_specs,
                                matchup_a,
                                matchup_b,
                                matchup_id=matchup_id,
                                game_id=str(pending["game_id"]),
                                seed=int(pending["seed"]),
                                starter=int(pending["starter"]),
                                sample_every_n_plies=sample_every_n_plies,
                                max_moves=max_moves,
                            )
                            future_map[future] = pending

                        done, _not_done = concurrent.futures.wait(
                            future_map.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            pending = future_map.pop(future)
                            raw_payload, samples = future.result()
                            games_inc, samples_inc = _materialize_completed_game(
                                job,
                                pending=pending,
                                raw_payload=raw_payload,
                                samples=samples,
                                matchup_id=matchup_id,
                                games=games,
                                execution_mode="parallel",
                            )
                            matchup_game_count += games_inc
                            matchup_sample_count += samples_inc
                            total_samples_after_run += samples_inc
                            job.write_state(
                                {
                                    "current_matchup": matchup_id,
                                    "completed_games": matchup_game_count,
                                    "remaining_games": games - matchup_game_count,
                                    "last_completed_game_id": pending["game_id"],
                                    "sample_count": total_samples_after_run,
                                    "target_samples": target_samples,
                                }
                            )
            except concurrent.futures.process.BrokenProcessPool as exc:
                failed_pending = list(future_map.values()) + list(pending_queue)
                job.logger.warning(
                    "dataset parallel pool broken | matchup=%s | completed_games=%s | remaining_fallback=%s | error=%s",
                    matchup_id,
                    matchup_game_count,
                    len(failed_pending),
                    exc,
                )
                job.write_event(
                    "dataset_parallel_execution_broken",
                    matchup=matchup_id,
                    completed_games=matchup_game_count,
                    remaining_fallback=len(failed_pending),
                    error=str(exc),
                )
                completed_games, completed_samples, _execution_mode = _run_pending_games_sequential(
                    job,
                    pending_games=failed_pending,
                    matchup_a=matchup_a,
                    matchup_b=matchup_b,
                    matchup_id=matchup_id,
                    games=games,
                    sample_every_n_plies=sample_every_n_plies,
                    max_moves=max_moves,
                )
                matchup_game_count += completed_games
                matchup_sample_count += completed_samples
                total_samples_after_run += completed_samples
                if failed_pending:
                    job.write_state(
                        {
                            "current_matchup": matchup_id,
                            "completed_games": matchup_game_count,
                            "remaining_games": games - matchup_game_count,
                            "last_completed_game_id": failed_pending[-1]["game_id"],
                            "sample_count": total_samples_after_run,
                            "target_samples": target_samples,
                        }
                    )

        if num_workers <= 1 and pending_games:
            total_samples_after_run += matchup_sample_count

        summary_payload = {
            "matchup_id": matchup_id,
            "player_a": matchup_a,
            "player_b": matchup_b,
            "existing_games": existing_game_count,
            "games_added": matchup_game_count,
            "samples_added": matchup_sample_count,
            "sample_every_n_plies": sample_every_n_plies,
            "num_workers": num_workers,
        }
        _write_json(summary_path, summary_payload)
        summaries.append(summary_payload)
        job.logger.info(
            "dataset matchup completed | %s vs %s | games_added=%s | samples_added=%s | total_samples=%s",
            matchup_a,
            matchup_b,
            matchup_game_count,
            matchup_sample_count,
            total_samples_after_run,
        )
        job.write_state(
            {
                "current_matchup": matchup_id,
                "completed_games": matchup_game_count,
                "remaining_games": 0,
                "last_completed_game_id": pending_games[-1]["game_id"] if pending_games else state.get("last_completed_game_id"),
                "sample_count": total_samples_after_run,
                "target_samples": target_samples,
            }
        )
        job.write_metric({"metric_type": "dataset_matchup_completed", **summary_payload})
        job.write_event("dataset_matchup_completed", matchup=matchup_id, samples=matchup_sample_count)

    added_games = sum(int(item["games_added"]) for item in summaries)
    added_samples = sum(int(item["samples_added"]) for item in summaries)
    summary = {
        "job_id": job.job_id,
        "dataset_source_id": dataset_source_id,
        "source_mode": source_mode,
        "matchups": summaries,
        "existing_samples": initial_total_samples,
        "added_games": added_games,
        "added_samples": added_samples,
        "total_samples": initial_total_samples + added_samples,
        "target_samples": target_samples,
    }
    _register_dataset_source(
        job,
        dataset_source_id=dataset_source_id,
        source_mode=source_mode,
        raw_dir=raw_dir,
        sampled_dir=sampled_dir,
        target_samples=target_samples,
        games_per_matchup=games,
        sample_every_n_plies=sample_every_n_plies,
        matchups=[str(matchup) for matchup in matchups],
        source_dataset_id=source_dataset_id,
        derivation_strategy=derivation_strategy,
        derivation_params=derivation_params,
    )
    _write_json(dataset_dir / "dataset_generation_summary.json", summary)
    job.logger.info(
        "dataset generation completed | matchups=%s | added_games=%s | added_samples=%s | total_samples=%s | target_samples=%s",
        len(summaries),
        added_games,
        added_samples,
        initial_total_samples + added_samples,
        target_samples,
    )
    job.write_state(
        {
            "current_matchup": None,
            "completed_games": added_games,
            "remaining_games": 0,
            "last_completed_game_id": state.get("last_completed_game_id"),
            "sample_count": initial_total_samples + added_samples,
            "target_samples": target_samples,
        }
    )
    return summary


def run_dataset_build(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("dataset_build", {})
    runtime_cfg = job.config.get("runtime", {})
    teacher_cfg = cfg.get("teacher", {})
    teacher_engine = str(teacher_cfg.get("engine", "minimax"))
    teacher_level = str(teacher_cfg.get("level", "hard"))
    dataset_id = str(cfg.get("dataset_id", "dataset_v1"))
    source_dataset_id = str(cfg.get("source_dataset_id", "")).strip()
    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.8))
    validation_ratio = float(split_cfg.get("validation", 0.1))
    num_workers = max(1, int(cfg.get("num_workers", runtime_cfg.get("num_workers", 1))))
    max_pending_futures = max(1, int(cfg.get("max_pending_futures", num_workers * 2)))
    multiprocessing_start_method = str(runtime_cfg.get("multiprocessing_start_method", "spawn")).strip().lower() or "spawn"
    max_tasks_per_child = int(runtime_cfg.get("max_tasks_per_child", 25))

    job.logger.info(
        "dataset build startup | dataset=%s | source_dataset_id=%s | configured_input_sampled_dir=%s | workers=%s | max_pending_futures=%s",
        dataset_id,
        source_dataset_id or "<auto>",
        str(cfg.get("input_sampled_dir", "")),
        num_workers,
        max_pending_futures,
    )

    dataset_dir = job.job_dir / "dataset_build"
    if source_dataset_id:
        source_entry = _resolve_dataset_source(job, source_dataset_id)
        sampled_root = Path(str(source_entry["sampled_dir"]))
    else:
        sampled_root = _resolve_storage_path(job.paths.drive_root, cfg.get("input_sampled_dir"), dataset_dir.parent / "dataset_generation" / "sampled_positions")
        source_dataset_id = Path(sampled_root).name
    configured_label_cache_dir = cfg.get("label_cache_dir")
    if configured_label_cache_dir:
        configured_label_cache_path = Path(str(configured_label_cache_dir))
        if configured_label_cache_path.name != dataset_id:
            job.logger.info(
                "dataset build label cache isolation enabled | dataset=%s | configured_label_cache_dir=%s | isolated_label_cache_dir=%s",
                dataset_id,
                configured_label_cache_path,
                job.paths.data_root / "label_cache" / dataset_id,
            )
            configured_label_cache_dir = None
    label_cache_dir = _resolve_storage_path(
        job.paths.drive_root,
        configured_label_cache_dir,
        job.paths.data_root / "label_cache" / dataset_id,
    )
    labeled_root = label_cache_dir / "labeled_positions"
    configured_output_dir = cfg.get("output_dir")
    if configured_output_dir:
        configured_output_path = Path(str(configured_output_dir))
        if configured_output_path.name != dataset_id:
            job.logger.info(
                "dataset build output isolation enabled | dataset=%s | configured_output_dir=%s | isolated_output_dir=%s",
                dataset_id,
                configured_output_path,
                job.paths.data_root / "datasets" / dataset_id,
            )
            configured_output_dir = None
    output_root = _resolve_storage_path(job.paths.drive_root, configured_output_dir, job.paths.data_root / "datasets" / dataset_id)
    labeled_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    target_labeled_samples = int(cfg.get("target_labeled_samples", 0))
    label_cache_metadata_path = label_cache_dir / "metadata.json"
    _write_json(
        label_cache_metadata_path,
        {
            "dataset_id": dataset_id,
            "source_dataset_id": source_dataset_id,
            "teacher_engine": teacher_engine,
            "teacher_level": teacher_level,
            "label_cache_dir": str(label_cache_dir),
        },
    )

    sampled_files = sorted(sampled_root.rglob("*.jsonl"))
    if not sampled_files:
        raise FileNotFoundError(f"Aucun fichier sampled_positions trouve dans {sampled_root}")

    state = job.read_state()
    completed_files = set(state.get("completed_files", []))
    file_sample_counts: dict[str, int] = {}
    sampled_relative_names = [str(path.relative_to(sampled_root)) for path in sampled_files]
    contiguous_completed_prefix = 0
    for relative_name in sampled_relative_names:
        if relative_name not in completed_files:
            break
        if not (labeled_root / relative_name).exists():
            break
        contiguous_completed_prefix += 1

    processed_count = contiguous_completed_prefix
    labeled_samples_total = int(state.get("labeled_samples", 0)) if contiguous_completed_prefix > 0 else 0
    skipped_terminal_samples = 0
    skipped_no_legal_samples = 0
    log_every_n_files = max(1, int(cfg.get("log_every_n_files", 1)))
    last_logged_progress_count = -1
    build_started_monotonic = time.monotonic()

    def _estimate_remaining_seconds() -> float | None:
        if processed_count <= 0:
            return None
        elapsed = max(0.001, time.monotonic() - build_started_monotonic)
        files_per_second = processed_count / elapsed
        if files_per_second <= 0:
            return None
        remaining_files = len(sampled_files) - processed_count
        return remaining_files / files_per_second

    def _throughput_metrics() -> tuple[float | None, float | None]:
        if processed_count <= 0:
            return None, None
        elapsed = max(0.001, time.monotonic() - build_started_monotonic)
        files_per_second = processed_count / elapsed
        samples_per_second = labeled_samples_total / elapsed
        return files_per_second, samples_per_second

    def _write_build_state() -> None:
        job.write_state(
            {
                "completed_files": sorted(completed_files),
                "processed_files": processed_count,
                "remaining_files": len(sampled_files) - processed_count,
                "labeled_samples": labeled_samples_total,
                "skipped_terminal_samples": skipped_terminal_samples,
                "skipped_no_legal_samples": skipped_no_legal_samples,
                "target_labeled_samples": target_labeled_samples,
                "contiguous_completed_prefix": contiguous_completed_prefix,
                "last_completed_file": sampled_relative_names[processed_count - 1] if processed_count > 0 else "",
            }
        )

    def _log_build_progress_if_needed() -> None:
        nonlocal last_logged_progress_count
        if processed_count == last_logged_progress_count:
            return
        if processed_count % log_every_n_files != 0 and processed_count != len(sampled_files):
            return
        files_per_second, samples_per_second = _throughput_metrics()
        job.logger.info(
            "dataset build progress | files=%s/%s | labeled_samples=%s | skipped_terminal=%s | skipped_no_legal=%s | files_per_sec=%.2f | samples_per_sec=%.2f | eta=%s",
            processed_count,
            len(sampled_files),
            labeled_samples_total,
            skipped_terminal_samples,
            skipped_no_legal_samples,
            files_per_second or 0.0,
            samples_per_second or 0.0,
            _format_eta_seconds(_estimate_remaining_seconds()),
        )
        last_logged_progress_count = processed_count

    job.logger.info(
        "dataset build started | dataset=%s | source_dataset_id=%s | teacher=%s:%s | sampled_root=%s | label_cache=%s | output_dir=%s | files=%s | target_labeled_samples=%s | workers=%s",
        dataset_id,
        source_dataset_id,
        teacher_engine,
        teacher_level,
        sampled_root,
        label_cache_dir,
        output_root,
        len(sampled_files),
        target_labeled_samples,
        num_workers,
    )
    job.set_phase("dataset_build")
    job.write_event(
        "dataset_build_started",
        dataset_id=dataset_id,
        source_dataset_id=source_dataset_id,
        teacher_engine=teacher_engine,
        teacher_level=teacher_level,
        sampled_root=str(sampled_root),
        label_cache_dir=str(label_cache_dir),
        sampled_files=len(sampled_files),
        target_labeled_samples=target_labeled_samples,
        num_workers=num_workers,
    )
    job.write_metric({"metric_type": "dataset_build_started"})

    if contiguous_completed_prefix > 0:
        job.logger.info(
            "dataset build resume shortcut | contiguous_completed_prefix=%s | skipped_rescan_files=%s | labeled_samples=%s",
            contiguous_completed_prefix,
            contiguous_completed_prefix,
            labeled_samples_total,
        )
        job.write_event(
            "dataset_build_resume_shortcut",
            contiguous_completed_prefix=contiguous_completed_prefix,
            skipped_rescan_files=contiguous_completed_prefix,
            labeled_samples=labeled_samples_total,
        )

    pending_files: list[dict[str, Any]] = []
    for file_index, sampled_file in enumerate(sampled_files[contiguous_completed_prefix:], start=contiguous_completed_prefix + 1):
        relative_name = str(sampled_file.relative_to(sampled_root))
        output_labeled = labeled_root / relative_name
        output_labeled.parent.mkdir(parents=True, exist_ok=True)

        if output_labeled.exists():
            source_count = _count_jsonl_lines(output_labeled)
            job.logger.info(
                "dataset build file reused | %s/%s | file=%s | labeled_samples=%s",
                file_index,
                len(sampled_files),
                relative_name,
                source_count,
            )
            completed_files.add(relative_name)
            file_sample_counts[relative_name] = source_count
            processed_count += 1
            labeled_samples_total += source_count
        else:
            pending_files.append(
                {
                    "file_index": file_index,
                    "relative_name": relative_name,
                    "sampled_file": sampled_file,
                    "output_labeled": output_labeled,
                }
            )

        _log_build_progress_if_needed()
        _write_build_state()

    reused_count = processed_count
    pending_count = len(pending_files)
    files_per_second, samples_per_second = _throughput_metrics()
    job.logger.info(
        "dataset build scan completed | reused_files=%s | pending_files=%s | total_files=%s | labeled_samples=%s | files_per_sec=%.2f | samples_per_sec=%.2f | eta=%s",
        reused_count,
        pending_count,
        len(sampled_files),
        labeled_samples_total,
        files_per_second or 0.0,
        samples_per_second or 0.0,
        _format_eta_seconds(_estimate_remaining_seconds()),
    )
    job.write_event(
        "dataset_build_scan_completed",
        reused_files=reused_count,
        pending_files=pending_count,
        total_files=len(sampled_files),
        labeled_samples=labeled_samples_total,
    )

    def _materialize_labeled_file(
        file_item: dict[str, Any],
        source_count: int,
        file_samples: list[dict[str, Any]],
        skipped_terminal: int,
        skipped_no_legal: int,
        *,
        mode: str,
    ) -> None:
        nonlocal processed_count, skipped_terminal_samples, skipped_no_legal_samples, labeled_samples_total
        output_labeled = Path(file_item["output_labeled"])
        output_labeled.parent.mkdir(parents=True, exist_ok=True)
        with output_labeled.open("w", encoding="utf-8") as handle:
            for sample in file_samples:
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")

        skipped_terminal_samples += skipped_terminal
        skipped_no_legal_samples += skipped_no_legal
        completed_files.add(str(file_item["relative_name"]))
        file_sample_counts[str(file_item["relative_name"])] = len(file_samples)
        processed_count += 1
        labeled_samples_total += len(file_samples)
        job.logger.info(
            "dataset build file completed | %s/%s | file=%s | mode=%s | source_samples=%s | labeled_samples=%s | skipped_terminal=%s | skipped_no_legal=%s",
            file_item["file_index"],
            len(sampled_files),
            file_item["relative_name"],
            mode,
            source_count,
            len(file_samples),
            skipped_terminal_samples,
            skipped_no_legal_samples,
        )
        job.write_event("dataset_labeled_file_completed", file=str(file_item["relative_name"]), samples=len(file_samples))
        job.write_metric({"metric_type": "dataset_labeled_file_completed", "file": str(file_item["relative_name"]), "samples": len(file_samples)})
        _log_build_progress_if_needed()
        _write_build_state()

    if pending_files:
        if num_workers <= 1:
            for file_item in pending_files:
                job.logger.info(
                    "dataset build file started | %s/%s | file=%s | mode=sequential",
                    file_item["file_index"],
                    len(sampled_files),
                    file_item["relative_name"],
                )
                source_count, file_samples, skipped_terminal, skipped_no_legal = _label_samples_from_file(
                    str(file_item["sampled_file"]),
                    teacher_engine=teacher_engine,
                    teacher_level=teacher_level,
                )
                _materialize_labeled_file(
                    file_item,
                    source_count,
                    file_samples,
                    skipped_terminal,
                    skipped_no_legal,
                    mode="sequential",
                )
        else:
            job.logger.info(
                "dataset build parallel execution | workers=%s | pending_files=%s | max_pending=%s | start_method=%s | max_tasks_per_child=%s",
                num_workers,
                len(pending_files),
                max_pending_futures,
                multiprocessing_start_method,
                max_tasks_per_child,
            )
            job.write_event(
                "dataset_build_parallel_execution_started",
                workers=num_workers,
                pending_files=len(pending_files),
                max_pending_futures=max_pending_futures,
                multiprocessing_start_method=multiprocessing_start_method,
                max_tasks_per_child=max_tasks_per_child,
            )
            future_map: dict[concurrent.futures.Future, dict[str, Any]] = {}
            pending_queue = list(pending_files)
            try:
                mp_context = multiprocessing.get_context(multiprocessing_start_method)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=mp_context,
                    max_tasks_per_child=max_tasks_per_child,
                ) as executor:
                    while pending_queue or future_map:
                        while pending_queue and len(future_map) < max_pending_futures:
                            file_item = pending_queue.pop(0)
                            job.logger.info(
                                "dataset build file started | %s/%s | file=%s | mode=parallel",
                                file_item["file_index"],
                                len(sampled_files),
                                file_item["relative_name"],
                            )
                            future = executor.submit(
                                _label_samples_from_file,
                                str(file_item["sampled_file"]),
                                teacher_engine=teacher_engine,
                                teacher_level=teacher_level,
                            )
                            future_map[future] = file_item

                        done, _not_done = concurrent.futures.wait(
                            future_map.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            file_item = future_map.pop(future)
                            source_count, file_samples, skipped_terminal, skipped_no_legal = future.result()
                            _materialize_labeled_file(
                                file_item,
                                source_count,
                                file_samples,
                                skipped_terminal,
                                skipped_no_legal,
                                mode="parallel",
                            )
            except concurrent.futures.process.BrokenProcessPool as exc:
                failed_pending = list(future_map.values()) + list(pending_queue)
                job.logger.warning(
                    "dataset build parallel pool broken | completed_files=%s | remaining_fallback=%s | error=%s",
                    processed_count,
                    len(failed_pending),
                    exc,
                )
                job.write_event(
                    "dataset_build_parallel_execution_broken",
                    completed_files=processed_count,
                    remaining_fallback=len(failed_pending),
                    error=str(exc),
                )
                for file_item in failed_pending:
                    job.logger.info(
                        "dataset build file started | %s/%s | file=%s | mode=sequential_fallback",
                        file_item["file_index"],
                        len(sampled_files),
                        file_item["relative_name"],
                    )
                    source_count, file_samples, skipped_terminal, skipped_no_legal = _label_samples_from_file(
                        str(file_item["sampled_file"]),
                        teacher_engine=teacher_engine,
                        teacher_level=teacher_level,
                    )
                    _materialize_labeled_file(
                        file_item,
                        source_count,
                        file_samples,
                        skipped_terminal,
                        skipped_no_legal,
                        mode="sequential_fallback",
                    )

    game_files = [str(path.relative_to(sampled_root)) for path in sampled_files]
    split_train_end = int(len(game_files) * train_ratio)
    split_validation_end = split_train_end + int(len(game_files) * validation_ratio)
    split_files = {
        "train": game_files[:split_train_end],
        "validation": game_files[split_train_end:split_validation_end],
        "test": game_files[split_validation_end:],
    }

    split_summary: dict[str, dict[str, int]] = {}
    for split_name, selected_files in split_files.items():
        selected_sample_count = 0
        for relative_name in selected_files:
            sample_count = file_sample_counts.get(relative_name)
            if sample_count is None:
                labeled_path = labeled_root / relative_name
                sample_count = _count_jsonl_lines(labeled_path) if labeled_path.exists() else 0
                file_sample_counts[relative_name] = sample_count
            selected_sample_count += sample_count
        job.logger.info(
            "dataset build split export started | split=%s | games=%s | samples=%s",
            split_name,
            len(selected_files),
            selected_sample_count,
        )

        features_list = []
        masks_list = []
        policy_list = []
        value_list = []
        sample_ids = []
        game_id_list = []
        for relative_name in selected_files:
            labeled_path = labeled_root / relative_name
            for sample in _iter_jsonl(labeled_path):
                features, legal_mask, policy_index, value = _encode_features(sample)
                features_list.append(features)
                masks_list.append(legal_mask)
                policy_list.append(policy_index)
                value_list.append(value)
                sample_ids.append(str(sample["sample_id"]))
                game_id_list.append(str(sample["game_id"]))

        x = np.asarray(features_list, dtype=np.float32) if features_list else np.zeros((0, 17), dtype=np.float32)
        legal_mask = np.asarray(masks_list, dtype=np.float32) if masks_list else np.zeros((0, 7), dtype=np.float32)
        policy_index = np.asarray(policy_list, dtype=np.int64) if policy_list else np.zeros((0,), dtype=np.int64)
        value_target = np.asarray(value_list, dtype=np.float32) if value_list else np.zeros((0,), dtype=np.float32)
        np.savez_compressed(
            output_root / f"{split_name}.npz",
            x=x,
            legal_mask=legal_mask,
            policy_index=policy_index,
            value_target=value_target,
            sample_ids=np.asarray(sample_ids, dtype=object),
            game_ids=np.asarray(game_id_list, dtype=object),
        )
        split_summary[split_name] = {"games": len(selected_files), "samples": selected_sample_count}
        job.logger.info(
            "dataset build split export completed | split=%s | games=%s | samples=%s | path=%s",
            split_name,
            len(selected_files),
            selected_sample_count,
            output_root / f"{split_name}.npz",
        )

    summary = {
        "job_id": job.job_id,
        "dataset_id": dataset_id,
        "teacher_engine": teacher_engine,
        "teacher_level": teacher_level,
        "source_dataset_id": source_dataset_id,
        "label_cache_dir": str(label_cache_dir),
        "splits": split_summary,
        "output_dir": str(output_root),
        "labeled_samples": labeled_samples_total,
        "target_labeled_samples": target_labeled_samples,
        "skipped_terminal_samples": skipped_terminal_samples,
        "skipped_no_legal_samples": skipped_no_legal_samples,
    }
    _register_built_dataset(
        job,
        dataset_id=dataset_id,
        source_dataset_id=source_dataset_id,
        source_dataset_ids=[source_dataset_id] if source_dataset_id else [],
        sampled_root=sampled_root,
        output_root=output_root,
        label_cache_dir=label_cache_dir,
        teacher_engine=teacher_engine,
        teacher_level=teacher_level,
        split_summary=split_summary,
        labeled_samples=labeled_samples_total,
        target_labeled_samples=target_labeled_samples,
        build_mode="teacher_label",
        parent_dataset_ids=[],
    )
    _write_json(dataset_dir / "dataset_build_summary.json", summary)
    job.logger.info(
        "dataset build completed | dataset=%s | build_mode=teacher_label | source_dataset_id=%s | train=%s | validation=%s | test=%s | skipped_terminal=%s | skipped_no_legal=%s | output_dir=%s",
        dataset_id,
        source_dataset_id,
        split_summary.get("train", {}).get("samples", 0),
        split_summary.get("validation", {}).get("samples", 0),
        split_summary.get("test", {}).get("samples", 0),
        skipped_terminal_samples,
        skipped_no_legal_samples,
        output_root,
    )
    job.write_metric({"metric_type": "dataset_build_completed", "dataset_id": dataset_id})
    job.write_event("dataset_build_completed", dataset_id=dataset_id)
    return summary


def run_dataset_merge_final(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("dataset_merge_final", {})
    dataset_id = str(cfg.get("dataset_id", "dataset_merged_final")).strip()
    if not dataset_id:
        raise ValueError("`dataset_id` est requis pour `dataset_merge_final`")

    source_dataset_ids = [str(value).strip() for value in cfg.get("source_dataset_ids", []) if str(value).strip()]
    include_all_built = bool(cfg.get("include_all_built", False))
    dedupe_sample_ids = bool(cfg.get("dedupe_sample_ids", True))

    registry = _read_dataset_registry(job)
    built_entries = registry.get("built_datasets", [])
    if include_all_built:
        source_dataset_ids = [str(entry.get("dataset_id", "")).strip() for entry in built_entries if str(entry.get("dataset_id", "")).strip()]
    if not source_dataset_ids:
        raise ValueError("Aucun `source_dataset_ids` fourni pour `dataset_merge_final`")

    source_entries = [_resolve_built_dataset(job, source_dataset_id) for source_dataset_id in source_dataset_ids]
    teacher_pairs = {(str(entry.get("teacher_engine", "")), str(entry.get("teacher_level", ""))) for entry in source_entries}
    if len(teacher_pairs) != 1:
        raise ValueError("Tous les datasets finaux a fusionner doivent partager le meme teacher")
    teacher_engine, teacher_level = next(iter(teacher_pairs))

    output_root = _resolve_storage_path(
        job.paths.drive_root,
        cfg.get("output_dir"),
        job.paths.data_root / "datasets" / dataset_id,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    merge_dir = job.job_dir / "dataset_merge_final"
    merge_dir.mkdir(parents=True, exist_ok=True)

    job.logger.info(
        "dataset final merge started | dataset=%s | source_datasets=%s | teacher=%s:%s | dedupe_sample_ids=%s | output_dir=%s",
        dataset_id,
        len(source_entries),
        teacher_engine,
        teacher_level,
        dedupe_sample_ids,
        output_root,
    )
    job.write_event(
        "dataset_merge_final_started",
        dataset_id=dataset_id,
        source_dataset_ids=source_dataset_ids,
        teacher_engine=teacher_engine,
        teacher_level=teacher_level,
        dedupe_sample_ids=dedupe_sample_ids,
        output_dir=str(output_root),
    )

    split_summary: dict[str, dict[str, int]] = {}
    merge_breakdown: dict[str, dict[str, int]] = {}
    source_breakdown: dict[str, dict[str, dict[str, int]]] = {}
    total_labeled_samples = 0
    source_sampled_roots = [str(entry.get("sampled_root", "")) for entry in source_entries if str(entry.get("sampled_root", "")).strip()]
    label_cache_dirs = [str(entry.get("label_cache_dir", "")) for entry in source_entries if str(entry.get("label_cache_dir", "")).strip()]

    for split_name in ("train", "validation", "test"):
        split_items: list[tuple[str, Path]] = []
        for entry in source_entries:
            output_dir = Path(str(entry["output_dir"]))
            split_path = output_dir / f"{split_name}.npz"
            if not split_path.exists():
                raise FileNotFoundError(f"Split introuvable pour {entry['dataset_id']}: {split_path}")
            split_items.append((str(entry["dataset_id"]), split_path))

        job.logger.info(
            "dataset final merge split started | split=%s | source_files=%s",
            split_name,
            len(split_items),
        )
        merged_arrays, split_metrics, split_source_breakdown = _merge_npz_splits_with_source_breakdown(
            split_items,
            dedupe_sample_ids=dedupe_sample_ids,
        )
        np.savez_compressed(output_root / f"{split_name}.npz", **merged_arrays)
        split_summary[split_name] = {
            "games": int(split_metrics["unique_games"]),
            "samples": int(split_metrics["kept_samples"]),
        }
        merge_breakdown[split_name] = split_metrics
        source_breakdown[split_name] = split_source_breakdown
        total_labeled_samples += int(split_metrics["kept_samples"])
        job.logger.info(
            "dataset final merge split completed | split=%s | kept_samples=%s | duplicate_samples=%s | unique_games=%s | path=%s",
            split_name,
            split_metrics["kept_samples"],
            split_metrics["duplicate_samples"],
            split_metrics["unique_games"],
            output_root / f"{split_name}.npz",
        )
        for source_dataset_id, stats in split_source_breakdown.items():
            job.logger.info(
                "dataset final merge source breakdown | split=%s | source_dataset_id=%s | input_samples=%s | kept_samples=%s | duplicate_samples=%s | unique_games=%s",
                split_name,
                source_dataset_id,
                stats["input_samples"],
                stats["kept_samples"],
                stats["duplicate_samples"],
                stats["unique_games"],
            )

    metadata = _register_built_dataset(
        job,
        dataset_id=dataset_id,
        source_dataset_id=source_dataset_ids[0],
        source_dataset_ids=source_dataset_ids,
        sampled_root=Path(source_sampled_roots[0]) if source_sampled_roots else output_root,
        output_root=output_root,
        label_cache_dir=Path(label_cache_dirs[0]) if label_cache_dirs else output_root,
        teacher_engine=teacher_engine,
        teacher_level=teacher_level,
        split_summary=split_summary,
        labeled_samples=total_labeled_samples,
        target_labeled_samples=total_labeled_samples,
        build_mode="merged_final",
        parent_dataset_ids=source_dataset_ids,
    )
    metadata["merge_breakdown"] = merge_breakdown
    metadata["source_breakdown"] = source_breakdown
    metadata["dedupe_sample_ids"] = dedupe_sample_ids
    metadata["source_dataset_ids"] = source_dataset_ids
    _write_json(output_root / "dataset_metadata.json", metadata)

    summary = {
        "job_id": job.job_id,
        "dataset_id": dataset_id,
        "build_mode": "merged_final",
        "source_dataset_ids": source_dataset_ids,
        "teacher_engine": teacher_engine,
        "teacher_level": teacher_level,
        "dedupe_sample_ids": dedupe_sample_ids,
        "splits": split_summary,
        "merge_breakdown": merge_breakdown,
        "source_breakdown": source_breakdown,
        "output_dir": str(output_root),
        "labeled_samples": total_labeled_samples,
    }
    _write_json(merge_dir / "dataset_merge_final_summary.json", summary)
    job.logger.info(
        "dataset final merge completed | dataset=%s | sources=%s | train=%s | validation=%s | test=%s | labeled_samples=%s",
        dataset_id,
        len(source_dataset_ids),
        split_summary.get("train", {}).get("samples", 0),
        split_summary.get("validation", {}).get("samples", 0),
        split_summary.get("test", {}).get("samples", 0),
        total_labeled_samples,
    )
    job.write_event(
        "dataset_merge_final_completed",
        dataset_id=dataset_id,
        source_dataset_ids=source_dataset_ids,
        labeled_samples=total_labeled_samples,
        source_breakdown=source_breakdown,
    )
    job.write_metric({"metric_type": "dataset_merge_final_completed", "dataset_id": dataset_id, "labeled_samples": total_labeled_samples})
    return summary
