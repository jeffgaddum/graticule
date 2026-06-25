from __future__ import annotations

import math

import torch
from torch import nn

from .cartouche import ModelSpec
from .gazetteer import PanelBatch
from .projection import PanelTokenizer
from .relief import PanelBackbone

LOG_2PI = math.log(2.0 * math.pi)


class MixtureHead(nn.Module):
    def __init__(self, d_model: int, components: int, dropout: float) -> None:
        super().__init__()
        self.components = components
        self.treatment = nn.Embedding(2, d_model)
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.logits = nn.Linear(d_model, components)
        self.location = nn.Linear(d_model, components)
        self.log_scale = nn.Linear(d_model, components)

    def forward(
        self, state: torch.Tensor, treatment: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conditioned = state + self.treatment(treatment.long())
        hidden = self.trunk(conditioned)
        log_pi = torch.log_softmax(self.logits(hidden), dim=-1)
        mu = self.location(hidden)
        log_var = self.log_scale(hidden).clamp(-8.0, 8.0)
        return log_pi, mu, log_var

    @staticmethod
    def expected(log_pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return (log_pi.exp() * mu).sum(dim=-1)

    @staticmethod
    def nll(
        target: torch.Tensor,
        log_pi: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        diff = target.unsqueeze(-1) - mu
        comp = -0.5 * (LOG_2PI + log_var + diff.pow(2) / log_var.exp())
        return -torch.logsumexp(log_pi + comp, dim=-1)


class CounterfactualAtlas(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.tokenizer = PanelTokenizer(spec)
        self.backbone = PanelBackbone(spec)
        self.head = MixtureHead(spec.d_model, spec.mixture_components, spec.dropout)

    def represent(self, batch: PanelBatch) -> torch.Tensor:
        tokens = self.tokenizer(batch.covariates, batch.unit_index, batch.time_index)
        return self.backbone(tokens, batch.adjacency)

    def potential_mean(self, state: torch.Tensor, value: float) -> torch.Tensor:
        arm = torch.full(state.shape[:-1], int(value), device=state.device)
        log_pi, mu, _ = self.head(state, arm)
        return MixtureHead.expected(log_pi, mu)

    def cate(self, state: torch.Tensor) -> torch.Tensor:
        return self.potential_mean(state, 1.0) - self.potential_mean(state, 0.0)

    def factual_nll(self, batch: PanelBatch, state: torch.Tensor) -> torch.Tensor:
        log_pi, mu, log_var = self.head(state, batch.treatment)
        return MixtureHead.nll(batch.factual, log_pi, mu, log_var).mean()


def _farthest_init(points: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=points.device).manual_seed(seed)
    first = int(torch.randint(points.shape[0], (1,), generator=generator).item())
    chosen = [first]
    distance = torch.cdist(points, points[first : first + 1]).squeeze(1)
    for _ in range(1, k):
        nxt = int(torch.argmax(distance).item())
        chosen.append(nxt)
        fresh = torch.cdist(points, points[nxt : nxt + 1]).squeeze(1)
        distance = torch.minimum(distance, fresh)
    return points[torch.tensor(chosen, device=points.device)]


def _coherence(effects: torch.Tensor) -> float:
    if effects.numel() < 2:
        return 1.0
    spread = effects.var(unbiased=False)
    return float(1.0 / (1.0 + spread))


def discover_subgroups(
    embedding: torch.Tensor,
    effects: torch.Tensor,
    k_max: int,
    delta: float,
    iters: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    n = embedding.shape[0]
    k = min(k_max, n)
    mean = embedding.mean(dim=0, keepdim=True)
    std = embedding.std(dim=0, keepdim=True) + 1e-6
    points = (embedding - mean) / std
    centres = _farthest_init(points, k, seed)
    labels = torch.zeros(n, dtype=torch.long, device=embedding.device)
    for _ in range(iters):
        labels = torch.argmin(torch.cdist(points, centres), dim=1)
        for c in range(centres.shape[0]):
            members = points[labels == c]
            if members.shape[0] > 0:
                centres[c] = members.mean(dim=0)
    keep: list[int] = []
    for c in range(centres.shape[0]):
        member_mask = labels == c
        if int(member_mask.sum().item()) == 0:
            continue
        if _coherence(effects[member_mask]) >= delta:
            keep.append(c)
    if not keep:
        return torch.zeros(n, dtype=torch.long, device=embedding.device), 1
    survivors = centres[torch.tensor(keep, device=embedding.device)]
    final = torch.argmin(torch.cdist(points, survivors), dim=1)
    return final, len(keep)
