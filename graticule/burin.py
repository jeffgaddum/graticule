from __future__ import annotations

import os

import cyclopts
import torch
from torch.utils.data import DataLoader

from ._orient import get_logger, set_seed
from .cartouche import RunConfig, load_sheet
from .chart import Trainer
from .compass import interval_coverage, mixture_interval, root_pehe
from .gazetteer import PanelBatch, SyntheticPanels, collate_panels
from .legend import CounterfactualAtlas, discover_subgroups

app = cyclopts.App(name="graticule")
_log = get_logger("graticule.burin")


def assemble(cfg: RunConfig) -> CounterfactualAtlas:
    set_seed(cfg.seed)
    return CounterfactualAtlas(cfg.model)


def stream(cfg: RunConfig, rounds: int) -> list[PanelBatch]:
    dataset = SyntheticPanels(cfg.data, cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        collate_fn=collate_panels,
        drop_last=True,
    )
    batches: list[PanelBatch] = []
    for _ in range(rounds):
        for sample in loader:
            batches.append(sample)
    return batches


@app.command
def engrave(sheet: str, steps: int = 4, out: str = "engravings/atlas.pt") -> None:
    cfg = load_sheet(sheet)
    atlas = assemble(cfg)
    trainer = Trainer(atlas, cfg)
    history = trainer.fit(stream(cfg, rounds=2), total=steps)
    trainer.save(out)
    last = history[-1] if history else {}
    _log.info("engraved %s steps total=%.4f", trainer.step, last.get("total", 0.0))


@app.command
def read(sheet: str, ckpt: str = "engravings/atlas.pt") -> None:
    cfg = load_sheet(sheet)
    atlas = assemble(cfg)
    if os.path.exists(ckpt):
        atlas.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    atlas.eval()
    batch = stream(cfg, rounds=1)[0]
    with torch.no_grad():
        state = atlas.represent(batch)
        tau = atlas.cate(state)
        log_pi, mu, log_var = atlas.head(state, batch.treatment)
        lo, hi = mixture_interval(log_pi, mu, log_var, cfg.evaluation.alpha)
    pehe = root_pehe(tau, batch.tau)
    cover = interval_coverage(lo, hi, batch.factual)
    _log.info("sheet=%s pehe=%.4f coverage=%.3f", cfg.tag, pehe, cover)


@app.command
def proof(sheet: str, ckpt: str = "engravings/atlas.pt") -> None:
    cfg = load_sheet(sheet)
    atlas = assemble(cfg)
    if os.path.exists(ckpt):
        atlas.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    atlas.eval()
    batch = stream(cfg, rounds=1)[0]
    with torch.no_grad():
        state = atlas.represent(batch)
        tau = atlas.cate(state)
        pooled = state.mean(dim=(0, 2))
        labels, count = discover_subgroups(
            pooled,
            tau.mean(dim=(0, 2)),
            cfg.hte.k_max,
            cfg.hte.coherence_delta,
            cfg.hte.prototype_iters,
            cfg.seed,
        )
    _log.info("subgroups=%s mean_cate=%.4f", count, float(tau.mean()))


@app.command
def plate(sheet: str, ckpt: str = "engravings/atlas.pt", out: str = "atlas.onnx") -> None:
    cfg = load_sheet(sheet)
    atlas = assemble(cfg)
    if os.path.exists(ckpt):
        atlas.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    atlas.eval()
    batch = stream(cfg, rounds=1)[0]
    wrapper = _ExportArm(atlas)
    torch.onnx.export(
        wrapper,
        (batch.covariates, batch.unit_index, batch.time_index, batch.adjacency),
        out,
        input_names=["covariates", "unit_index", "time_index", "adjacency"],
        output_names=["cate"],
        dynamo=False,
    )
    _log.info("plated %s", out)


class _ExportArm(torch.nn.Module):
    def __init__(self, atlas: CounterfactualAtlas) -> None:
        super().__init__()
        self.atlas = atlas

    def forward(
        self,
        covariates: torch.Tensor,
        unit_index: torch.Tensor,
        time_index: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.atlas.tokenizer(covariates, unit_index, time_index)
        state = self.atlas.backbone(tokens, adjacency)
        return self.atlas.cate(state)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
