from __future__ import annotations

import torch

from graticule.legend import discover_subgroups


def test_discovery_is_seed_deterministic() -> None:
    embedding = torch.randn(40, 8, generator=torch.Generator().manual_seed(1))
    effects = embedding[:, 0]
    a, ka = discover_subgroups(embedding, effects, 5, 0.4, 20, seed=9)
    b, kb = discover_subgroups(embedding, effects, 5, 0.4, 20, seed=9)
    assert ka == kb
    assert torch.equal(a, b)


def test_separated_clusters_are_recovered() -> None:
    gen = torch.Generator().manual_seed(4)
    left = torch.randn(30, 4, generator=gen) * 0.05 - 3.0
    right = torch.randn(30, 4, generator=gen) * 0.05 + 3.0
    embedding = torch.cat([left, right], dim=0)
    effects = torch.cat([torch.full((30,), -2.0), torch.full((30,), 2.0)])
    labels, count = discover_subgroups(embedding, effects, 6, 0.5, 30, seed=2)
    assert count >= 2
    top = labels[:30].mode().values
    bottom = labels[30:].mode().values
    assert top != bottom
