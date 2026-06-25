from __future__ import annotations

import math

import torch

from graticule.burin import assemble, stream
from graticule.cartouche import load_sheet
from graticule.chart import Trainer


def test_two_step_smoke(tmp_path: object) -> None:
    cfg = load_sheet("sheets/_smoke.cfg")
    atlas = assemble(cfg)
    trainer = Trainer(atlas, cfg)
    batches = stream(cfg, rounds=1)
    history = trainer.fit(batches, total=2)
    assert len(history) == 2
    assert all(math.isfinite(h["total"]) for h in history)
    target = f"{tmp_path}/atlas.pt"
    trainer.save(target)
    blob = torch.load(target, map_location="cpu")
    assert blob["seed"] == cfg.seed
    assert "ema" in blob and "model" in blob
