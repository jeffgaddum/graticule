from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from ._orient import atomic_save
from .cartouche import RunConfig
from .contour import ContourWeights, pretraining_objective
from .gazetteer import PanelBatch
from .legend import CounterfactualAtlas


def cosine_warmup(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup and warmup > 0:
        return base_lr * float(step + 1) / float(warmup)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def wrap_for_distribution(module: nn.Module) -> nn.Module:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return nn.parallel.DistributedDataParallel(module)
    return module


class Trainer:
    def __init__(self, atlas: CounterfactualAtlas, cfg: RunConfig) -> None:
        self.atlas = atlas
        self.cfg = cfg
        self.weights = ContourWeights()
        self.optimizer = torch.optim.AdamW(
            atlas.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
        self.shadow = {n: p.detach().clone() for n, p in atlas.named_parameters()}
        self.step = 0

    def _lr(self, total: int) -> float:
        return cosine_warmup(self.step, self.cfg.optim.warmup_epochs, total, self.cfg.optim.lr)

    def _update_ema(self) -> None:
        decay = self.cfg.optim.ema_decay
        for name, param in self.atlas.named_parameters():
            self.shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def train_step(self, batch: PanelBatch, total: int) -> dict[str, float]:
        lr = self._lr(total)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.atlas.train()
        loss, parts = pretraining_objective(self.atlas, batch, self.weights)
        accum = max(self.cfg.optim.grad_accum, 1)
        (loss / accum).backward()
        if (self.step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(self.atlas.parameters(), self.cfg.optim.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._update_ema()
        self.step += 1
        parts["lr"] = lr
        return parts

    def fit(self, batches: Iterable[PanelBatch], total: int) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for batch in batches:
            history.append(self.train_step(batch, total))
            if self.step >= total:
                break
        return history

    def save(self, path: str) -> None:
        atomic_save(
            {
                "model": self.atlas.state_dict(),
                "ema": self.shadow,
                "optimizer": self.optimizer.state_dict(),
                "seed": self.cfg.seed,
                "step": self.step,
            },
            path,
        )
