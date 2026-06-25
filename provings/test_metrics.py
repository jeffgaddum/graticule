from __future__ import annotations

import math

import numpy as np
import torch

from graticule.compass import (
    ate_error,
    bonferroni,
    bootstrap_ci,
    cohens_d,
    interval_coverage,
    loco_gap,
    root_pehe,
    walk_forward_rmse,
    wilcoxon_p,
)


def test_root_pehe_matches_manual() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.5, 1.5, 4.0])
    manual = math.sqrt(((a - b) ** 2).mean().item())
    assert abs(root_pehe(a, b) - manual) < 1e-6


def test_ate_error_and_coverage() -> None:
    a = torch.tensor([2.0, 4.0])
    b = torch.tensor([1.0, 1.0])
    assert abs(ate_error(a, b) - 2.0) < 1e-6
    lo = torch.tensor([0.0, 0.0])
    hi = torch.tensor([3.0, 3.0])
    target = torch.tensor([1.0, 5.0])
    assert abs(interval_coverage(lo, hi, target) - 0.5) < 1e-6


def test_statistics_helpers() -> None:
    same = np.array([0.1, 0.2, 0.3, 0.4])
    assert wilcoxon_p(same, same) == 1.0
    high = np.array([3.0, 3.1, 2.9, 3.2, 3.0, 2.8, 3.3, 3.1])
    low = np.array([1.0, 0.9, 1.1, 1.0, 0.8, 1.2, 0.7, 1.05])
    assert wilcoxon_p(high, low) < 0.05
    assert cohens_d(high, low) > 1.0
    assert bonferroni([0.001, 0.2], 0.05) == [True, False]


def test_bootstrap_and_walk_forward() -> None:
    values = np.linspace(0.0, 1.0, 200)
    lo, hi = bootstrap_ci(values, n_resamples=200, alpha=0.05, seed=3)
    assert lo < values.mean() < hi
    pred = torch.zeros(2, 3, 6)
    target = torch.ones(2, 3, 6)
    scores = walk_forward_rmse(pred, target, init_window=2)
    assert len(scores) == 4 and all(abs(s - 1.0) < 1e-6 for s in scores)
    assert abs(loco_gap(0.4, 0.5) - 0.25) < 1e-6
