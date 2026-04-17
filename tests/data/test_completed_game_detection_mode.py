from __future__ import annotations

from pathlib import Path

from songo_model_stockfish.data.jobs import (
    _build_pending_incremental_games,
    _existing_game_numbers,
)


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_existing_game_numbers_by_completion_mode(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    sampled_dir = tmp_path / "sampled"
    matchup_id = "agent_avsagent_b"

    _touch(raw_dir / matchup_id / f"{matchup_id}_game_000001.json", "{}")
    _touch(sampled_dir / matchup_id / f"{matchup_id}_game_000001.jsonl", "{}\n")
    _touch(sampled_dir / matchup_id / f"{matchup_id}_game_000002.jsonl", "{}\n")
    _touch(raw_dir / matchup_id / f"{matchup_id}_game_000003.json", "{}")

    assert _existing_game_numbers(raw_dir, sampled_dir, matchup_id, completion_mode="raw_and_sampled") == {1}
    assert _existing_game_numbers(raw_dir, sampled_dir, matchup_id, completion_mode="sampled_only") == {1, 2}
    assert _existing_game_numbers(raw_dir, sampled_dir, matchup_id, completion_mode="raw_only") == {1, 3}
    # Mode invalide => fallback raw_and_sampled
    assert _existing_game_numbers(raw_dir, sampled_dir, matchup_id, completion_mode="unknown_mode") == {1}


def test_pending_incremental_games_respects_sampled_only_completion(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    sampled_dir = tmp_path / "sampled"
    matchup_id = "agent_xvsagent_y"

    # game 1 complete raw+sampled
    _touch(raw_dir / matchup_id / f"{matchup_id}_game_000001.json", "{}")
    _touch(sampled_dir / matchup_id / f"{matchup_id}_game_000001.jsonl", "{}\n")
    # game 2 sampled-only (cas volatile VM: raw absent)
    _touch(sampled_dir / matchup_id / f"{matchup_id}_game_000002.jsonl", "{}\n")

    pending_sampled_only = _build_pending_incremental_games(
        matchup_id=matchup_id,
        raw_dir=raw_dir,
        sampled_dir=sampled_dir,
        games_to_add=2,
        seed_base=10,
        sample_every_n_plies=1,
        completion_mode="sampled_only",
    )
    # games 1 et 2 sont deja consideres completes, la prochaine est 3.
    assert [item["game_id"] for item in pending_sampled_only] == [f"{matchup_id}_game_000003", f"{matchup_id}_game_000004"]

    pending_strict = _build_pending_incremental_games(
        matchup_id=matchup_id,
        raw_dir=raw_dir,
        sampled_dir=sampled_dir,
        games_to_add=2,
        seed_base=10,
        sample_every_n_plies=1,
        completion_mode="raw_and_sampled",
    )
    # En mode strict raw+sampled, game 2 est incomplete (raw absent), elle doit etre reprise.
    assert [item["game_id"] for item in pending_strict] == [f"{matchup_id}_game_000002", f"{matchup_id}_game_000003"]
