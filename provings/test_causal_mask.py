from __future__ import annotations

import torch
from _fixtures import make_batch, tiny_model

from graticule.projection import PanelTokenizer
from graticule.relief import PanelBackbone


def test_future_does_not_leak_into_past() -> None:
    spec = tiny_model(max_units=2)
    batch = make_batch(panels=1, n_units=1)
    tok = PanelTokenizer(spec).eval()
    backbone = PanelBackbone(spec).eval()
    with torch.no_grad():
        tokens = tok(batch.covariates, batch.unit_index, batch.time_index)
        before = backbone(tokens, batch.adjacency)
        perturbed = batch.covariates.clone()
        perturbed[:, :, -1, :] += 5.0
        tokens2 = tok(perturbed, batch.unit_index, batch.time_index)
        after = backbone(tokens2, batch.adjacency)
    early = before.shape[2] - 1
    assert torch.allclose(before[:, :, :early], after[:, :, :early], atol=1e-5)
    assert not torch.allclose(before[:, :, -1], after[:, :, -1], atol=1e-5)
