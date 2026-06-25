from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from confection import Config


@dataclass(frozen=True)
class ModelSpec:
    n_cov: int = 52
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 16
    ffn_mult: int = 4
    dropout: float = 0.1
    mixture_components: int = 5
    max_units: int = 256
    max_time: int = 64
    spillover_threshold: float = 0.0
    use_unit_fe: bool = True


@dataclass(frozen=True)
class DataSpec:
    kind: str = "synthetic"
    n_units: int = 48
    n_time: int = 24
    n_cov: int = 52
    n_families: int = 8
    panels_per_epoch: int = 256
    treat_fraction: float = 0.32
    signal: float = 1.0
    modifier_cov: int = 0
    source_path: str = ""


@dataclass(frozen=True)
class OptimSpec:
    lr: float = 2e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 30
    epochs: int = 700
    batch_size: int = 96
    grad_accum: int = 6
    world_size: int = 16
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    precision: str = "fp32"


@dataclass(frozen=True)
class HTESpec:
    k_max: int = 8
    coherence_delta: float = 0.6
    prototype_iters: int = 25


@dataclass(frozen=True)
class EvalSpec:
    alpha: float = 0.05
    n_bootstrap: int = 2000
    n_seeds: int = 10
    init_window: int = 10


@dataclass(frozen=True)
class RunConfig:
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    hte: HTESpec = field(default_factory=HTESpec)
    evaluation: EvalSpec = field(default_factory=EvalSpec)
    seed: int = 20240101
    tag: str = "main"
    out_dir: str = "engravings"


_SECTIONS: dict[str, Any] = {
    "model": ModelSpec,
    "data": DataSpec,
    "optim": OptimSpec,
    "hte": HTESpec,
    "evaluation": EvalSpec,
}


def _coerce(section: Any, payload: Mapping[str, Any]) -> Any:
    fields = set(section.__dataclass_fields__)
    unknown = set(payload) - fields
    if unknown:
        raise ValueError(f"{section.__name__} rejects keys {sorted(unknown)}")
    return section(**{k: payload[k] for k in payload})


def from_mapping(raw: Mapping[str, Any]) -> RunConfig:
    parts: dict[str, Any] = {}
    for name, section in _SECTIONS.items():
        block = raw.get(name, {})
        parts[name if name != "evaluation" else "evaluation"] = _coerce(section, block)
    top = raw.get("run", {})
    return RunConfig(
        model=parts["model"],
        data=parts["data"],
        optim=parts["optim"],
        hte=parts["hte"],
        evaluation=parts["evaluation"],
        seed=int(top.get("seed", 20240101)),
        tag=str(top.get("tag", "main")),
        out_dir=str(top.get("out_dir", "engravings")),
    )


def load_sheet(path: str) -> RunConfig:
    raw = Config().from_disk(path)
    return from_mapping(raw)


def to_config(cfg: RunConfig) -> Config:
    payload: dict[str, Any] = {
        "run": {"seed": cfg.seed, "tag": cfg.tag, "out_dir": cfg.out_dir},
        "model": asdict(cfg.model),
        "data": asdict(cfg.data),
        "optim": asdict(cfg.optim),
        "hte": asdict(cfg.hte),
        "evaluation": asdict(cfg.evaluation),
    }
    return Config(payload)


def write_sheet(cfg: RunConfig, path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    to_config(cfg).to_disk(path)
