from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    max_depth: int = 4
    time_ms: int | None = 250
    use_iterative_deepening: bool = True
    use_transposition_table: bool = False
    use_move_ordering: bool = True
