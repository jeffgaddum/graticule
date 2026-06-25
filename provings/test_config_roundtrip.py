from __future__ import annotations

import pytest

from graticule.cartouche import RunConfig, from_mapping, load_sheet, write_sheet


def test_smoke_sheet_loads() -> None:
    cfg = load_sheet("sheets/_smoke.cfg")
    assert cfg.tag == "smoke_test_only"
    assert cfg.model.d_model == 16
    assert cfg.optim.epochs == 2


def test_main_sheet_hostile_defaults() -> None:
    cfg = load_sheet("sheets/main.cfg")
    assert cfg.optim.world_size == 16
    assert cfg.optim.lr < 1e-5
    assert cfg.optim.epochs >= 700


def test_roundtrip(tmp_path: object) -> None:
    cfg = RunConfig()
    target = f"{tmp_path}/copy.cfg"
    write_sheet(cfg, target)
    again = load_sheet(target)
    assert again.model.d_model == cfg.model.d_model
    assert again.optim.lr == cfg.optim.lr
    assert again.hte.coherence_delta == cfg.hte.coherence_delta


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError):
        from_mapping({"model": {"made_up": 1}})
