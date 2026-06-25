from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .gazetteer import PanelBatch
from .legend import CounterfactualAtlas


@dataclass(frozen=True)
class ContourWeights:
    factual: float = 1.0
    potential: float = 0.5
    anchor: float = 0.1


def pretraining_objective(
    atlas: CounterfactualAtlas, batch: PanelBatch, weights: ContourWeights
) -> tuple[torch.Tensor, dict[str, float]]:
    state = atlas.represent(batch)
    factual = atlas.factual_nll(batch, state)
    mean0 = atlas.potential_mean(state, 0.0)
    mean1 = atlas.potential_mean(state, 1.0)
    potential = F.mse_loss(mean0, batch.y0) + F.mse_loss(mean1, batch.y1)
    log_pi, _, _ = atlas.head(state, batch.treatment)
    anchor = (log_pi.exp().mean(dim=(0, 1, 2)) - 1.0 / log_pi.shape[-1]).pow(2).sum()
    total = weights.factual * factual + weights.potential * potential + weights.anchor * anchor
    parts = {
        "factual": float(factual.detach()),
        "potential": float(potential.detach()),
        "anchor": float(anchor.detach()),
        "total": float(total.detach()),
    }
    return total, parts


def coherence_penalty(effects: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    total = effects.new_zeros(())
    for c in torch.unique(labels):
        member = effects[labels == c]
        if member.numel() > 1:
            total = total + member.var(unbiased=False)
    return total
