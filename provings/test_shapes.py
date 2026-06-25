from __future__ import annotations

import torch
from _fixtures import make_batch, tiny_model

from graticule.legend import CounterfactualAtlas
from graticule.projection import PanelTokenizer


def test_tokenizer_shape() -> None:
    spec = tiny_model()
    batch = make_batch()
    tok = PanelTokenizer(spec)
    tokens = tok(batch.covariates, batch.unit_index, batch.time_index)
    assert tokens.shape == (*batch.covariates.shape[:3], spec.d_model)


def test_representation_and_cate_shape() -> None:
    spec = tiny_model()
    batch = make_batch()
    atlas = CounterfactualAtlas(spec)
    state = atlas.represent(batch)
    assert state.shape == (*batch.factual.shape, spec.d_model)
    tau = atlas.cate(state)
    assert tau.shape == batch.factual.shape
    assert torch.isfinite(tau).all()


def test_mixture_outputs() -> None:
    spec = tiny_model()
    batch = make_batch()
    atlas = CounterfactualAtlas(spec)
    state = atlas.represent(batch)
    log_pi, mu, log_var = atlas.head(state, batch.treatment)
    assert log_pi.shape[-1] == spec.mixture_components
    assert torch.allclose(log_pi.exp().sum(dim=-1), torch.ones_like(log_pi[..., 0]))
    assert torch.isfinite(mu).all() and torch.isfinite(log_var).all()
