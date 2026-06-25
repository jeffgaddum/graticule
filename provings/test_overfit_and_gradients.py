from __future__ import annotations

import torch
from _fixtures import make_batch, tiny_model

from graticule.contour import ContourWeights, pretraining_objective
from graticule.legend import CounterfactualAtlas


def test_overfit_single_batch() -> None:
    torch.manual_seed(0)
    spec = tiny_model()
    batch = make_batch(panels=2)
    atlas = CounterfactualAtlas(spec)
    optimizer = torch.optim.Adam(atlas.parameters(), lr=5e-3)
    first, _ = pretraining_objective(atlas, batch, ContourWeights())
    start = float(first.detach())
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = pretraining_objective(atlas, batch, ContourWeights())
        loss.backward()
        optimizer.step()
    end, _ = pretraining_objective(atlas, batch, ContourWeights())
    assert float(end.detach()) < start - 0.4


def test_all_parameters_receive_gradient() -> None:
    spec = tiny_model()
    batch = make_batch()
    atlas = CounterfactualAtlas(spec)
    loss, _ = pretraining_objective(atlas, batch, ContourWeights())
    loss.backward()
    missing = [n for n, p in atlas.named_parameters() if p.grad is None]
    assert missing == []
