from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineState:
    raw_state: Any


@dataclass
class SearchInfo:
    best_score: float
    depth_reached: int
    nodes_searched: int
    elapsed_ms: float
    pv: list[int] = field(default_factory=list)
