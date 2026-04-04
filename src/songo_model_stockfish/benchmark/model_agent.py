from __future__ import annotations

from pathlib import Path

import torch

from songo_model_stockfish.adapters import songo_ai_game
from songo_model_stockfish.training.features import encode_raw_state
from songo_model_stockfish.training.jobs import _masked_policy_logits
from songo_model_stockfish.training.model import PolicyValueMLP


class ModelAgent:
    def __init__(self, checkpoint_path: str, *, display_name: str | None = None, device: str = "cpu") -> None:
        self._checkpoint_path = Path(checkpoint_path)
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {self._checkpoint_path}")
        self._device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self._checkpoint_path, map_location=self._device)
        model_config = checkpoint.get("model_config", {})
        self._model = PolicyValueMLP(
            input_dim=int(model_config.get("input_dim", 17)),
            hidden_sizes=list(model_config.get("hidden_sizes", [256, 256, 128])),
            policy_dim=int(model_config.get("policy_dim", 7)),
        )
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.to(self._device)
        self._model.eval()
        self._display_name = display_name or self._checkpoint_path.stem

    @property
    def display_name(self) -> str:
        return self._display_name

    def choose(self, state):
        raw_state = songo_ai_game.to_raw_state(state)
        legal_moves = songo_ai_game.legal_moves(state)
        features, legal_mask = encode_raw_state(raw_state, legal_moves)
        x = torch.from_numpy(features).unsqueeze(0).to(self._device)
        mask = torch.from_numpy(legal_mask).unsqueeze(0).to(self._device)
        with torch.no_grad():
            policy_logits, value = self._model(x)
            masked_logits = _masked_policy_logits(policy_logits, mask)
            move_index = int(masked_logits.argmax(dim=1).item())
        move = move_index + 1
        if move not in legal_moves and legal_moves:
            move = legal_moves[0]
        return move, {"value": float(value.item())}
