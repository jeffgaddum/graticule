from __future__ import annotations

import logging
import os
import random
import tempfile
from typing import Any

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def atomic_save(payload: dict[str, Any], path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=folder, suffix=".part")
    os.close(handle)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def count_parameters(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)
