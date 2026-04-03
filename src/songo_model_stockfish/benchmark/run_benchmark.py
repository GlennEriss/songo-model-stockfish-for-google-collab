from __future__ import annotations

from dataclasses import asdict, dataclass

from songo_model_stockfish.benchmark.play_match import AgentLike, MatchResult, play_match


@dataclass
class BenchmarkSummary:
    games: int
    wins_a: int
    wins_b: int
    draws: int
    avg_moves: float


def run_benchmark(agent_a: AgentLike, agent_b: AgentLike, *, games: int = 20, max_moves: int = 300) -> tuple[BenchmarkSummary, list[MatchResult]]:
    results: list[MatchResult] = []
    wins_a = 0
    wins_b = 0
    draws = 0
    for game_index in range(games):
        starter = game_index % 2
        result = play_match(agent_a, agent_b, max_moves=max_moves, starter=starter)
        results.append(result)
        if result.winner == 0:
            wins_a += 1
        elif result.winner == 1:
            wins_b += 1
        else:
            draws += 1
    avg_moves = sum(result.moves for result in results) / max(1, len(results))
    return BenchmarkSummary(games=games, wins_a=wins_a, wins_b=wins_b, draws=draws, avg_moves=avg_moves), results


def benchmark_summary_to_dict(summary: BenchmarkSummary) -> dict[str, float | int]:
    return asdict(summary)
