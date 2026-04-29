from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from songo_model_stockfish.adapters import songo_ai_game
from songo_model_stockfish.training.features import adapt_feature_dim, encode_model_features
from songo_model_stockfish.training.jobs import _masked_policy_logits
from songo_model_stockfish.training.model import PolicyValueMLP


class ModelAgent:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        display_name: str | None = None,
        device: str = "cpu",
        search_enabled: bool = True,
        search_top_k: int = 6,
        search_top_k_child: int | None = 4,
        search_depth: int = 3,
        search_policy_weight: float = 0.35,
        search_value_weight: float = 1.0,
        search_profile: str = "fort_plusplus",
        search_alpha_beta: bool = True,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {self._checkpoint_path}")
        self._device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self._checkpoint_path, map_location=self._device)
        model_config = checkpoint.get("model_config", {})
        self._input_dim = int(model_config.get("input_dim", 17))
        self._model = PolicyValueMLP(
            input_dim=self._input_dim,
            hidden_sizes=list(model_config.get("hidden_sizes", [256, 256, 128])),
            policy_dim=int(model_config.get("policy_dim", 7)),
            use_layer_norm=bool(model_config.get("use_layer_norm", False)),
            dropout=float(model_config.get("dropout", 0.0)),
            residual_connections=bool(model_config.get("residual_connections", False)),
        )
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.to(self._device)
        self._model.eval()
        self._display_name = display_name or self._checkpoint_path.stem
        self._search_enabled = bool(search_enabled)
        self._search_top_k = max(1, int(search_top_k))
        self._search_top_k_child = max(1, int(search_top_k_child if search_top_k_child is not None else 4))
        self._search_depth = max(1, int(search_depth))
        self._search_policy_weight = float(search_policy_weight)
        self._search_value_weight = float(search_value_weight)
        self._search_profile = str(search_profile or "fort_plusplus").strip().lower() or "fort_plusplus"
        self._search_alpha_beta = bool(search_alpha_beta)
        if self._search_profile not in {"fort_plusplus"}:
            raise ValueError(
                "ModelAgent search_profile non supporte. "
                f"Valeur recue: {self._search_profile}. Valeur autorisee: fort_plusplus."
            )

    @property
    def display_name(self) -> str:
        return self._display_name

    def _infer_state(self, state: Any) -> tuple[list[int], dict[int, float], float]:
        raw_state = songo_ai_game.to_raw_state(state)
        legal_moves = songo_ai_game.legal_moves(state)
        if not legal_moves:
            return [], {}, 0.0
        features, legal_mask = encode_model_features(raw_state, legal_moves, tactical_analysis=None)
        features = adapt_feature_dim(features, self._input_dim)
        x = torch.from_numpy(features).unsqueeze(0).to(self._device)
        mask = torch.from_numpy(legal_mask).unsqueeze(0).to(self._device)
        with torch.no_grad():
            policy_logits, value = self._model(x)
            masked_logits = _masked_policy_logits(policy_logits, mask)
            policy_probs = torch.softmax(masked_logits, dim=1).squeeze(0).detach().cpu()
            root_value = float(value.item())
        policy_by_move = {int(move): float(policy_probs[int(move) - 1].item()) for move in legal_moves}
        prior_total = float(sum(policy_by_move.values()))
        if prior_total > 0.0:
            policy_by_move = {int(move): float(score / prior_total) for move, score in policy_by_move.items()}
        else:
            uniform = 1.0 / float(len(legal_moves))
            policy_by_move = {int(move): float(uniform) for move in legal_moves}
        return list(legal_moves), policy_by_move, root_value

    @staticmethod
    def _state_signature(state: Any) -> tuple[Any, ...]:
        raw_state = songo_ai_game.to_raw_state(state)
        scores = raw_state.get("scores", {})
        return (
            tuple(int(value) for value in raw_state.get("board", [])),
            str(raw_state.get("player_to_move", "")),
            int(scores.get("south", 0)),
            int(scores.get("north", 0)),
            bool(raw_state.get("is_terminal", False)),
            int(state.get("turn_index", 0)),
        )

    @staticmethod
    def _terminal_value_for_root_player(state: Any, *, root_player: int) -> float:
        winner = songo_ai_game.winner(state)
        if winner is None:
            return 0.0
        return 1.0 if int(winner) == int(root_player) else -1.0

    @staticmethod
    def _rank_candidate_moves(legal_moves: list[int], policy_by_move: dict[int, float], *, top_k: int) -> list[int]:
        ranked = sorted(legal_moves, key=lambda m: float(policy_by_move.get(int(m), 0.0)), reverse=True)
        return ranked[: min(len(ranked), max(1, int(top_k)))]

    @staticmethod
    def _value_from_root_pov(
        state_value: float,
        *,
        side_to_move: int,
        root_player: int,
    ) -> float:
        # Network value is always predicted from side-to-move perspective.
        if int(side_to_move) == int(root_player):
            return float(state_value)
        return -float(state_value)

    def _minimax_search(
        self,
        state: Any,
        *,
        depth_left: int,
        root_player: int,
        alpha: float,
        beta: float,
        infer_cache: dict[tuple[Any, ...], tuple[list[int], dict[int, float], float]],
        stats: dict[str, int],
    ) -> float:
        stats["nodes"] = int(stats.get("nodes", 0)) + 1
        if songo_ai_game.is_terminal(state):
            stats["terminal_nodes"] = int(stats.get("terminal_nodes", 0)) + 1
            return self._terminal_value_for_root_player(state, root_player=int(root_player))

        signature = self._state_signature(state)
        cached = infer_cache.get(signature)
        if cached is None:
            cached = self._infer_state(state)
            infer_cache[signature] = cached
        else:
            stats["cache_hits"] = int(stats.get("cache_hits", 0)) + 1
        legal_moves, policy_by_move, state_value = cached
        if depth_left <= 0 or not legal_moves:
            stats["leaf_nodes"] = int(stats.get("leaf_nodes", 0)) + 1
            side_to_move = int(songo_ai_game.current_player(state))
            return self._value_from_root_pov(
                float(state_value),
                side_to_move=side_to_move,
                root_player=int(root_player),
            )

        candidate_moves = self._rank_candidate_moves(
            legal_moves,
            policy_by_move,
            top_k=self._search_top_k_child,
        )
        use_alpha_beta = bool(self._search_alpha_beta)
        side_to_move = int(songo_ai_game.current_player(state))
        maximizing = int(side_to_move) == int(root_player)
        local_alpha = float(alpha)
        local_beta = float(beta)

        if maximizing:
            best_score = float("-inf")
            for move in candidate_moves:
                child_state = songo_ai_game.simulate_move(state, int(move))
                child_score = self._minimax_search(
                    child_state,
                    depth_left=int(depth_left) - 1,
                    root_player=int(root_player),
                    alpha=local_alpha,
                    beta=local_beta,
                    infer_cache=infer_cache,
                    stats=stats,
                )
                if child_score > best_score:
                    best_score = float(child_score)
                if not use_alpha_beta:
                    continue
                if child_score > local_alpha:
                    local_alpha = float(child_score)
                if local_alpha >= local_beta:
                    stats["cutoffs"] = int(stats.get("cutoffs", 0)) + 1
                    break
            return float(best_score)

        best_score = float("inf")
        for move in candidate_moves:
            child_state = songo_ai_game.simulate_move(state, int(move))
            child_score = self._minimax_search(
                child_state,
                depth_left=int(depth_left) - 1,
                root_player=int(root_player),
                alpha=local_alpha,
                beta=local_beta,
                infer_cache=infer_cache,
                stats=stats,
            )
            if child_score < best_score:
                best_score = float(child_score)
            if not use_alpha_beta:
                continue
            if child_score < local_beta:
                local_beta = float(child_score)
            if local_alpha >= local_beta:
                stats["cutoffs"] = int(stats.get("cutoffs", 0)) + 1
                break
        return float(best_score)

    def choose(self, state):
        infer_cache: dict[tuple[Any, ...], tuple[list[int], dict[int, float], float]] = {}
        root_signature = self._state_signature(state)
        legal_moves, policy_by_move, root_value = self._infer_state(state)
        infer_cache[root_signature] = (legal_moves, policy_by_move, root_value)
        if not legal_moves:
            raise ValueError("Aucun coup legal disponible pour ModelAgent.choose")

        if not self._search_enabled:
            move = max(legal_moves, key=lambda m: float(policy_by_move.get(int(m), 0.0)))
            if move not in legal_moves:
                raise RuntimeError(
                    "ModelAgent produced an illegal argmax move in strict mode | "
                    f"move={move} | legal_moves={legal_moves} | checkpoint={self._checkpoint_path}"
                )
            return move, {"value": root_value, "search_enabled": False}

        candidate_moves = self._rank_candidate_moves(legal_moves, policy_by_move, top_k=self._search_top_k)
        best_move = candidate_moves[0]
        best_blended_score = float("-inf")
        best_search_value = float("-inf")
        search_stats = {"nodes": 0, "leaf_nodes": 0, "terminal_nodes": 0, "cache_hits": 0, "cutoffs": 0}
        depth_for_children = max(1, int(self._search_depth) - 1)
        root_player = int(songo_ai_game.current_player(state))
        for move in candidate_moves:
            child_state = songo_ai_game.simulate_move(state, int(move))
            child_value = self._minimax_search(
                child_state,
                depth_left=depth_for_children,
                root_player=root_player,
                alpha=float("-inf"),
                beta=float("inf"),
                infer_cache=infer_cache,
                stats=search_stats,
            )
            prior = float(policy_by_move.get(move, 0.0))
            score = (self._search_value_weight * child_value) + (self._search_policy_weight * prior)
            if score > best_blended_score:
                best_blended_score = score
                best_move = int(move)
                best_search_value = float(child_value)

        return best_move, {
            "value": root_value,
            "search_enabled": True,
            "search_profile": self._search_profile,
            "search_depth": self._search_depth,
            "search_top_k": self._search_top_k,
            "search_top_k_child": self._search_top_k_child,
            "search_alpha_beta": bool(self._search_alpha_beta),
            "search_score": float(best_blended_score),
            "search_child_value": float(best_search_value),
            "search_nodes": int(search_stats.get("nodes", 0)),
            "search_leaf_nodes": int(search_stats.get("leaf_nodes", 0)),
            "search_terminal_nodes": int(search_stats.get("terminal_nodes", 0)),
            "search_cache_hits": int(search_stats.get("cache_hits", 0)),
            "search_cutoffs": int(search_stats.get("cutoffs", 0)),
        }
