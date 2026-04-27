from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    threshold: float | None = None
    budget: float | None = None


def _quantile_index(pmf: torch.Tensor, alpha: float) -> int:
    cdf = torch.cumsum(pmf, dim=0)
    idx = int(torch.searchsorted(cdf, torch.tensor(alpha, device=pmf.device)).item())
    return min(idx + 1, int(pmf.size(0)))


def stopping_rule_decision(
    strategy: str,
    pmf_k: torch.Tensor,
    k: int,
    *,
    threshold: float | None = None,
) -> bool:
    strat = strategy.lower()
    K = int(pmf_k.size(0))
    kk = max(1, min(int(k), K))
    cdf_k = float(pmf_k[:kk].sum().item())
    sf_k = float(pmf_k[kk:].sum().item()) if kk < K else 0.0
    if strat == "cumulative_probability":
        c = 0.8 if threshold is None else float(threshold)
        return cdf_k >= c
    if strat == "future_improvement":
        eps = 0.1 if threshold is None else float(threshold)
        return sf_k <= eps
    if strat == "hazard":
        c_h = 0.5 if threshold is None else float(threshold)
        denom = float(pmf_k[kk - 1 :].sum().item())
        if denom <= 1e-8:
            return True
        h_k = float(pmf_k[kk - 1].item()) / denom
        return h_k >= c_h
    if strat == "quantile":
        alpha = 0.8 if threshold is None else float(threshold)
        q = _quantile_index(pmf_k, alpha)
        return q <= kk
    raise ValueError(f"Unknown stopping strategy: {strategy}")


def apply_stopping_strategy(
    strategy: str,
    pmf_per_step: list[torch.Tensor],
    *,
    threshold: float | None = None,
) -> int:
    if not pmf_per_step:
        return 1
    K = len(pmf_per_step)
    for k in range(1, K + 1):
        pmf_k = pmf_per_step[k - 1]
        if stopping_rule_decision(strategy, pmf_k, k, threshold=threshold):
            return k
    return K
