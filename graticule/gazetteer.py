from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .cartouche import DataSpec


@dataclass
class PanelBatch:
    covariates: torch.Tensor
    treatment: torch.Tensor
    factual: torch.Tensor
    y0: torch.Tensor
    y1: torch.Tensor
    adjacency: torch.Tensor
    tau: torch.Tensor
    unit_index: torch.Tensor
    time_index: torch.Tensor

    def to(self, device: torch.device) -> "PanelBatch":
        return PanelBatch(
            covariates=self.covariates.to(device),
            treatment=self.treatment.to(device),
            factual=self.factual.to(device),
            y0=self.y0.to(device),
            y1=self.y1.to(device),
            adjacency=self.adjacency.to(device),
            tau=self.tau.to(device),
            unit_index=self.unit_index.to(device),
            time_index=self.time_index.to(device),
        )


def _family_effect(family: int, modifier: np.ndarray, signal: float) -> np.ndarray:
    centred = modifier - modifier.mean()
    if family % 4 == 0:
        base = centred
    elif family % 4 == 1:
        base = np.tanh(2.0 * centred)
    elif family % 4 == 2:
        base = np.where(centred > 0.0, centred, 0.5 * centred)
    else:
        base = centred * np.abs(centred)
    swing = 1.0 if family < 4 else -1.0
    return signal * swing * base.astype(np.float64)


def _adjacency(rng: np.random.Generator, n_units: int) -> np.ndarray:
    raw = rng.normal(size=(n_units, n_units))
    sym = 0.5 * (raw + raw.T)
    np.fill_diagonal(sym, 0.0)
    keep = np.abs(sym) > 1.0
    weighted = np.where(keep, np.tanh(sym), 0.0)
    return weighted.astype(np.float64)


def synthesize_panel(
    rng: np.random.Generator, spec: DataSpec, family: int
) -> dict[str, np.ndarray]:
    u, t, d = spec.n_units, spec.n_time, spec.n_cov
    cov = rng.normal(size=(u, t, d)).astype(np.float64)
    modifier = cov[:, :, spec.modifier_cov].mean(axis=1)
    unit_fe = rng.normal(scale=0.7, size=u)
    trend = np.linspace(-0.5, 0.5, t)
    weights = rng.normal(scale=0.4, size=d)
    structural = cov @ weights
    adjacency = _adjacency(rng, u)
    spill = adjacency @ structural
    ar = np.zeros((u, t), dtype=np.float64)
    noise = rng.normal(scale=0.3, size=(u, t))
    for k in range(1, t):
        ar[:, k] = 0.5 * ar[:, k - 1] + noise[:, k]
    y0 = structural + unit_fe[:, None] + trend[None, :] + 0.2 * spill + ar
    effect = _family_effect(family, modifier, spec.signal)
    tau = np.broadcast_to(effect[:, None], (u, t)).astype(np.float64)
    y1 = y0 + tau
    propensity = 1.0 / (1.0 + np.exp(-(0.8 * modifier + unit_fe)))
    adopt = rng.uniform(size=u) < (propensity * spec.treat_fraction * 2.0).clip(0.0, 1.0)
    start = rng.integers(low=t // 3, high=t, size=u)
    treatment = np.zeros((u, t), dtype=np.float64)
    for i in range(u):
        if adopt[i]:
            treatment[i, start[i] :] = 1.0
    factual = treatment * y1 + (1.0 - treatment) * y0
    return {
        "covariates": cov,
        "treatment": treatment,
        "factual": factual,
        "y0": y0,
        "y1": y1,
        "tau": tau,
        "adjacency": adjacency,
    }


class SyntheticPanels(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, spec: DataSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed

    def __len__(self) -> int:
        return self.spec.panels_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        family = int(rng.integers(0, self.spec.n_families))
        arrays = synthesize_panel(rng, self.spec, family)
        return {k: torch.from_numpy(v).float() for k, v in arrays.items()}


def collate_panels(samples: list[dict[str, torch.Tensor]]) -> PanelBatch:
    stack = {key: torch.stack([s[key] for s in samples], dim=0) for key in samples[0]}
    b, u, t = stack["factual"].shape
    return PanelBatch(
        covariates=stack["covariates"],
        treatment=stack["treatment"],
        factual=stack["factual"],
        y0=stack["y0"],
        y1=stack["y1"],
        adjacency=stack["adjacency"],
        tau=stack["tau"],
        unit_index=torch.arange(u).unsqueeze(0).expand(b, u).clone(),
        time_index=torch.arange(t).unsqueeze(0).expand(b, t).clone(),
    )


def load_ihdp(path: str) -> PanelBatch:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"IHDP array not found at {path}; see data-sources.txt to obtain it"
        )
    blob = np.load(path)
    cov = blob["x"].astype(np.float64)
    n, d = cov.shape
    treatment = blob["t"].astype(np.float64).reshape(n, 1)
    y0 = blob["mu0"].astype(np.float64).reshape(n, 1)
    y1 = blob["mu1"].astype(np.float64).reshape(n, 1)
    factual = treatment * y1 + (1.0 - treatment) * y0
    adjacency = np.zeros((n, n), dtype=np.float64)
    pack = {
        "covariates": cov.reshape(n, 1, d),
        "treatment": treatment,
        "factual": factual,
        "y0": y0,
        "y1": y1,
        "tau": (y1 - y0),
        "adjacency": adjacency,
    }
    tensors = {k: torch.from_numpy(v).float() for k, v in pack.items()}
    return collate_panels([tensors])
