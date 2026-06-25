from __future__ import annotations

import numpy as np
import torch

from graticule.cartouche import DataSpec, ModelSpec
from graticule.gazetteer import PanelBatch, collate_panels, synthesize_panel


def tiny_model(**overrides: object) -> ModelSpec:
    base = dict(
        n_cov=6,
        d_model=16,
        n_layers=2,
        n_heads=2,
        ffn_mult=2,
        dropout=0.0,
        mixture_components=3,
        max_units=16,
        max_time=8,
    )
    base.update(overrides)
    return ModelSpec(**base)


def tiny_data(**overrides: object) -> DataSpec:
    base = dict(n_units=5, n_time=4, n_cov=6, n_families=3, panels_per_epoch=4)
    base.update(overrides)
    return DataSpec(**base)


def make_batch(panels: int = 2, seed: int = 11, **data_overrides: object) -> PanelBatch:
    spec = tiny_data(**data_overrides)
    samples = []
    for i in range(panels):
        rng = np.random.default_rng(seed + i)
        family = int(rng.integers(0, spec.n_families))
        arrays = synthesize_panel(rng, spec, family)
        samples.append({k: torch.from_numpy(v).float() for k, v in arrays.items()})
    return collate_panels(samples)
