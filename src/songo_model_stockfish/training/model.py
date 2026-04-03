from __future__ import annotations

import torch
from torch import nn


class PolicyValueMLP(nn.Module):
    def __init__(self, input_dim: int = 17, hidden_sizes: list[int] | None = None, policy_dim: int = 7) -> None:
        super().__init__()
        hidden = hidden_sizes or [256, 256, 128]
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            prev = size
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, policy_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return policy_logits, value
