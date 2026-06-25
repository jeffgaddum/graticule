from __future__ import annotations

import torch

NEG = float("-1e30")


def temporal_causal_bias(length: int, device: torch.device) -> torch.Tensor:
    future = torch.triu(torch.ones(length, length, device=device), diagonal=1)
    return future * NEG


def spillover_bias(adjacency: torch.Tensor, threshold: float) -> torch.Tensor:
    b, u, _ = adjacency.shape
    eye = torch.eye(u, device=adjacency.device, dtype=torch.bool).unsqueeze(0)
    connected = adjacency.abs() > threshold
    allow = connected | eye
    bias = torch.where(allow, adjacency, torch.full_like(adjacency, NEG))
    return bias.masked_fill(eye, 0.0)
