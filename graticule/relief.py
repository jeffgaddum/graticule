from __future__ import annotations

import math

import torch
from torch import nn

from .cartouche import ModelSpec
from .hachure import spillover_bias, temporal_causal_bias


class MultiHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        n, length, _ = x.shape
        return x.view(n, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + bias
        weights = self.drop(torch.softmax(scores, dim=-1))
        pooled = torch.matmul(weights, v)
        merged = pooled.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out(merged)


class ReliefBlock(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.temporal = MultiHead(spec.d_model, spec.n_heads, spec.dropout)
        self.spatial = MultiHead(spec.d_model, spec.n_heads, spec.dropout)
        self.norm_t = nn.LayerNorm(spec.d_model)
        self.norm_s = nn.LayerNorm(spec.d_model)
        self.norm_f = nn.LayerNorm(spec.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(spec.d_model, spec.ffn_mult * spec.d_model),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.ffn_mult * spec.d_model, spec.d_model),
        )

    def forward(self, tokens: torch.Tensor, spatial_bias: torch.Tensor) -> torch.Tensor:
        b, u, t, d = tokens.shape
        causal = temporal_causal_bias(t, tokens.device).view(1, 1, t, t)
        temporal_in = self.norm_t(tokens).reshape(b * u, t, d)
        temporal_out = self.temporal(temporal_in, causal).reshape(b, u, t, d)
        tokens = tokens + temporal_out
        spatial_in = self.norm_s(tokens).permute(0, 2, 1, 3).reshape(b * t, u, d)
        spat_bias = spatial_bias.unsqueeze(1).expand(b, t, u, u).reshape(b * t, 1, u, u)
        spatial_out = self.spatial(spatial_in, spat_bias)
        spatial_out = spatial_out.reshape(b, t, u, d).permute(0, 2, 1, 3)
        tokens = tokens + spatial_out
        tokens = tokens + self.ffn(self.norm_f(tokens))
        return tokens


class PanelBackbone(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.blocks = nn.ModuleList([ReliefBlock(spec) for _ in range(spec.n_layers)])
        self.final = nn.LayerNorm(spec.d_model)

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        bias = spillover_bias(adjacency, self.spec.spillover_threshold)
        for block in self.blocks:
            tokens = block(tokens, bias)
        return self.final(tokens)
