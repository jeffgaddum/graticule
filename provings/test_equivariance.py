from __future__ import annotations

import torch
from _fixtures import make_batch, tiny_model

from graticule.projection import PanelTokenizer
from graticule.relief import PanelBackbone


def test_cross_unit_permutation_equivariance() -> None:
    spec = tiny_model(use_unit_fe=False)
    batch = make_batch(panels=1)
    tok = PanelTokenizer(spec).eval()
    backbone = PanelBackbone(spec).eval()
    perm = torch.tensor([2, 0, 4, 1, 3])
    with torch.no_grad():
        base = backbone(
            tok(batch.covariates, batch.unit_index, batch.time_index),
            batch.adjacency,
        )
        cov_p = batch.covariates[:, perm]
        adj_p = batch.adjacency[:, perm][:, :, perm]
        permuted = backbone(tok(cov_p, batch.unit_index, batch.time_index), adj_p)
    assert torch.allclose(permuted, base[:, perm], atol=1e-5)
