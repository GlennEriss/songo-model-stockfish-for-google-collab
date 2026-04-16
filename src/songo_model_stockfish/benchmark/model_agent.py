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
        search_top_k: int = 4,
        search_policy_weight: float = 0.35,
        search_value_weight: float = 1.0,
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
        self._search_policy_weight = float(search_policy_weight)
        self._search_value_weight = float(search_value_weight)

    @property
    def display_name(self) -> str:
        return self._display_name

    def _infer_state(self, state: Any) -> tuple[list[int], torch.Tensor, float]:
        raw_state = songo_ai_game.to_raw_state(state)
        legal_moves = songo_ai_game.legal_moves(state)
        if not legal_moves:
            return [], torch.zeros((0,), dtype=torch.float32), 0.0
        features, legal_mask = encode_model_features(raw_state, legal_moves, tactical_analysis=None)
        features = adapt_feature_dim(features, self._input_dim)
        x = torch.from_numpy(features).unsqueeze(0).to(self._device)
        mask = torch.from_numpy(legal_mask).unsqueeze(0).to(self._device)
        with torch.no_grad():
            policy_logits, value = self._model(x)
            masked_logits = _masked_policy_logits(policy_logits, mask)
            policy_probs = torch.softmax(masked_logits, dim=1).squeeze(0).detach().cpu()
            root_value = float(value.item())
        return list(legal_moves), policy_probs, root_value

    def _child_value_from_root_pov(self, child_state: Any, *, root_player: int) -> float:
        # Terminal fast-path: exact outcome from root player's perspective.
        if songo_ai_game.is_terminal(child_state):
            winner = songo_ai_game.winner(child_state)
            if winner is None:
                return 0.0
            return 1.0 if int(winner) == int(root_player) else -1.0

        _legal_moves, _policy_probs, child_value = self._infer_state(child_state)
        # Value is learned from side-to-move perspective; after one ply it's opponent to move.
        return -float(child_value)

    def choose(self, state):
        legal_moves, policy_probs, root_value = self._infer_state(state)
        if not legal_moves:
            raise ValueError("Aucun coup legal disponible pour ModelAgent.choose")

        if not self._search_enabled:
            move_index = int(policy_probs.argmax().item())
            move = move_index + 1
            if move not in legal_moves:
                raise RuntimeError(
                    "ModelAgent produced an illegal argmax move in strict mode | "
                    f"move={move} | legal_moves={legal_moves} | checkpoint={self._checkpoint_path}"
                )
            return move, {"value": root_value, "search_enabled": False}

        policy_by_move = {move: float(policy_probs[move - 1].item()) for move in legal_moves}
        ranked_moves = sorted(legal_moves, key=lambda m: policy_by_move.get(m, 0.0), reverse=True)
        candidate_moves = ranked_moves[: min(len(ranked_moves), self._search_top_k)]

        root_player = int(songo_ai_game.current_player(state))
        best_move = candidate_moves[0]
        best_score = float("-inf")
        best_child_value = 0.0
        for move in candidate_moves:
            child_state = songo_ai_game.simulate_move(state, int(move))
            child_value = self._child_value_from_root_pov(child_state, root_player=root_player)
            prior = float(policy_by_move.get(move, 0.0))
            score = (self._search_value_weight * child_value) + (self._search_policy_weight * prior)
            if score > best_score:
                best_score = score
                best_move = int(move)
                best_child_value = float(child_value)

        return best_move, {
            "value": root_value,
            "search_enabled": True,
            "search_top_k": self._search_top_k,
            "search_score": float(best_score),
            "search_child_value": float(best_child_value),
        }
