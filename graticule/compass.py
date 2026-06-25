from __future__ import annotations

import numpy as np
import torch
from scipy import stats


def root_pehe(tau_hat: torch.Tensor, tau_true: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((tau_hat - tau_true) ** 2)))


def ate_error(tau_hat: torch.Tensor, tau_true: torch.Tensor) -> float:
    return float(torch.abs(tau_hat.mean() - tau_true.mean()))


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((prediction - target) ** 2)))


def mixture_interval(
    log_pi: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = log_pi.exp()
    mean = (weight * mu).sum(dim=-1)
    second = (weight * (mu.pow(2) + log_var.exp())).sum(dim=-1)
    variance = (second - mean.pow(2)).clamp_min(1e-8)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    spread = z * torch.sqrt(variance)
    return mean - spread, mean + spread


def interval_coverage(lower: torch.Tensor, upper: torch.Tensor, target: torch.Tensor) -> float:
    inside = (target >= lower) & (target <= upper)
    return float(inside.float().mean())


def policy_regret(tau_hat: torch.Tensor, tau_true: torch.Tensor, scale: float) -> float:
    recommended = (tau_hat > 0).float()
    realised = recommended * tau_true
    oracle = (tau_true > 0).float() * tau_true
    return float((oracle - realised).mean() / scale)


def wilcoxon_p(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, right):
        return 1.0
    return float(stats.wilcoxon(left, right).pvalue)


def cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    diff = left - right
    spread = diff.std(ddof=1)
    if spread == 0.0:
        return 0.0
    return float(diff.mean() / spread)


def bonferroni(pvalues: list[float], alpha: float) -> list[bool]:
    threshold = alpha / max(len(pvalues), 1)
    return [p < threshold for p in pvalues]


def bootstrap_ci(
    values: np.ndarray, n_resamples: int, alpha: float, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(values, size=values.shape[0], replace=True)
        means[i] = float(sample.mean())
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def walk_forward_rmse(
    prediction: torch.Tensor, target: torch.Tensor, init_window: int
) -> list[float]:
    horizon = prediction.shape[-1]
    scores: list[float] = []
    for step in range(init_window, horizon):
        scores.append(rmse(prediction[..., step], target[..., step]))
    return scores


def loco_gap(in_sample: float, held_out: float) -> float:
    if in_sample <= 0.0:
        return float("nan")
    return (held_out - in_sample) / in_sample
