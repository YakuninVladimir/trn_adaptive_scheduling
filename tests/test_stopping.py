from __future__ import annotations

import torch

from diplom.runner.stopping import apply_stopping_strategy, stopping_rule_decision


def test_stopping_rules_smoke():
    pmf = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    assert stopping_rule_decision("cumulative_probability", pmf, 3, threshold=0.5)
    assert stopping_rule_decision("future_improvement", pmf, 4, threshold=0.01)
    assert stopping_rule_decision("hazard", pmf, 4, threshold=0.5)
    assert stopping_rule_decision("quantile", pmf, 3, threshold=0.7)


def test_apply_stopping_strategy_returns_valid_step():
    per_step = [
        torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32),
        torch.tensor([0.8, 0.1, 0.1], dtype=torch.float32),
        torch.tensor([0.9, 0.05, 0.05], dtype=torch.float32),
    ]
    s = apply_stopping_strategy("cumulative_probability", per_step, threshold=0.6)
    assert 1 <= s <= len(per_step)
