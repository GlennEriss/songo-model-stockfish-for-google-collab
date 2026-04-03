from __future__ import annotations

from songo_model_stockfish.reference_songo.mcts import MCTSConfig
from songo_model_stockfish.reference_songo.minimax import SearchConfig


class Difficulty:
    BEGINNER = "beginner"
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    INSANE = "insane"
    INSANE_BIDOUA = "insane_bidoua"
    EXTREME = "extreme"
    EXTREME_BIDOUA = "extreme_bidoua"
    EXTREME_BIDOUA_MATH = "extreme_bidoua_math"


PRESETS: dict[str, SearchConfig] = {
    Difficulty.BEGINNER: SearchConfig(time_ms=20, max_depth=2, use_tt=False, order_moves=False),
    Difficulty.VERY_EASY: SearchConfig(time_ms=40, max_depth=4, use_tt=False, order_moves=False),
    Difficulty.EASY: SearchConfig(time_ms=80, max_depth=6, use_tt=True, order_moves=False),
    Difficulty.MEDIUM: SearchConfig(time_ms=150, max_depth=8, use_tt=True, order_moves=True),
    Difficulty.HARD: SearchConfig(time_ms=300, max_depth=12, use_tt=True, order_moves=True),
    Difficulty.VERY_HARD: SearchConfig(time_ms=700, max_depth=18, use_tt=True, order_moves=True),
    Difficulty.INSANE: SearchConfig(time_ms=1200, max_depth=22, use_tt=True, order_moves=True),
    Difficulty.INSANE_BIDOUA: SearchConfig(time_ms=1200, max_depth=22, use_tt=True, order_moves=True, eval_mode="bidoua"),
    Difficulty.EXTREME: SearchConfig(time_ms=2200, max_depth=28, use_tt=True, order_moves=True),
    Difficulty.EXTREME_BIDOUA: SearchConfig(time_ms=2200, max_depth=28, use_tt=True, order_moves=True, eval_mode="bidoua"),
    Difficulty.EXTREME_BIDOUA_MATH: SearchConfig(
        time_ms=2200,
        max_depth=28,
        use_tt=True,
        order_moves=True,
        eval_mode="bidoua_math",
    ),
}


MCTS_PRESETS: dict[str, MCTSConfig] = {
    Difficulty.BEGINNER: MCTSConfig(time_ms=20, sims=250, c_uct=1.35, max_rollout_depth=0),
    Difficulty.VERY_EASY: MCTSConfig(time_ms=40, sims=700, c_uct=1.30, max_rollout_depth=0),
    Difficulty.EASY: MCTSConfig(time_ms=80, sims=1800, c_uct=1.25, max_rollout_depth=0),
    Difficulty.MEDIUM: MCTSConfig(time_ms=150, sims=4500, c_uct=1.20, max_rollout_depth=0),
    Difficulty.HARD: MCTSConfig(time_ms=300, sims=12000, c_uct=1.15, max_rollout_depth=0),
    Difficulty.VERY_HARD: MCTSConfig(time_ms=700, sims=28000, c_uct=1.10, max_rollout_depth=0),
    Difficulty.INSANE: MCTSConfig(time_ms=2000, sims=90000, c_uct=1.06, max_rollout_depth=0),
    Difficulty.INSANE_BIDOUA: MCTSConfig(time_ms=2000, sims=90000, c_uct=1.06, max_rollout_depth=0, eval_mode="bidoua"),
    Difficulty.EXTREME: MCTSConfig(time_ms=3500, sims=180000, c_uct=1.04, max_rollout_depth=0),
    Difficulty.EXTREME_BIDOUA: MCTSConfig(time_ms=3500, sims=180000, c_uct=1.04, max_rollout_depth=0, eval_mode="bidoua"),
    Difficulty.EXTREME_BIDOUA_MATH: MCTSConfig(
        time_ms=3500,
        sims=180000,
        c_uct=1.04,
        max_rollout_depth=0,
        eval_mode="bidoua_math",
    ),
}


def get_config(level: str) -> SearchConfig:
    level = (level or "").strip().lower()
    return PRESETS.get(level, PRESETS[Difficulty.MEDIUM])


def get_mcts_config(level: str) -> MCTSConfig:
    level = (level or "").strip().lower()
    return MCTS_PRESETS.get(level, MCTS_PRESETS[Difficulty.MEDIUM])

