from __future__ import annotations

import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_residual = residual and input_dim == output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        if self.use_residual:
            h = h + x
        return h


class PolicyValueMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 17,
        hidden_sizes: list[int] | None = None,
        policy_dim: int = 7,
        *,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        residual_connections: bool = False,
    ) -> None:
        super().__init__()
        hidden = hidden_sizes or [256, 256, 128]
        prev = input_dim
        if not use_layer_norm and dropout <= 0.0 and not residual_connections:
            layers: list[nn.Module] = []
            for size in hidden:
                layers.append(nn.Linear(prev, size))
                layers.append(nn.ReLU())
                prev = size
            self.backbone = nn.Sequential(*layers)
        else:
            blocks: list[nn.Module] = []
            for size in hidden:
                blocks.append(
                    MLPBlock(
                        prev,
                        size,
                        use_layer_norm=use_layer_norm,
                        dropout=dropout,
                        residual=residual_connections,
                    )
                )
                prev = size
            self.backbone = nn.Sequential(*blocks)
        self.policy_head = nn.Linear(prev, policy_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return policy_logits, value
