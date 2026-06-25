from __future__ import annotations

import math

import torch
from torch import nn

from .cartouche import ModelSpec


def sinusoidal_table(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    span = torch.arange(0, dim, 2, dtype=torch.float32)
    divisor = torch.exp(span * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * divisor)
    table[:, 1::2] = torch.cos(position * divisor)
    return table


class PanelTokenizer(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.covariate = nn.Sequential(
            nn.Linear(spec.n_cov, spec.d_model),
            nn.GELU(),
            nn.Linear(spec.d_model, spec.d_model),
        )
        self.unit = nn.Embedding(spec.max_units, spec.d_model)
        self.register_buffer("clock", sinusoidal_table(spec.max_time, spec.d_model))
        self.clock: torch.Tensor
        self.scale = nn.Parameter(torch.ones(3))
        self.norm = nn.LayerNorm(spec.d_model)

    def forward(
        self,
        covariates: torch.Tensor,
        unit_index: torch.Tensor,
        time_index: torch.Tensor,
    ) -> torch.Tensor:
        b, u, t, _ = covariates.shape
        cov = self.covariate(covariates)
        clock = self.clock[time_index].unsqueeze(1)
        gate = torch.softmax(self.scale, dim=0)
        token = gate[0] * cov + gate[2] * clock
        if self.spec.use_unit_fe:
            unit = self.unit(unit_index).unsqueeze(2)
            token = token + gate[1] * unit
        return self.norm(token)
