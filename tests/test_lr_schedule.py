from __future__ import annotations

import pytest

from diplom.runner.train import _lr_multiplier


def test_warmup_then_flat():
    m1 = _lr_multiplier(1, warmup_steps=4, max_steps=100, lr_schedule="none", lr_min_ratio=0.1)
    m4 = _lr_multiplier(4, warmup_steps=4, max_steps=100, lr_schedule="none", lr_min_ratio=0.1)
    m5 = _lr_multiplier(5, warmup_steps=4, max_steps=100, lr_schedule="none", lr_min_ratio=0.1)
    assert abs(m1 - 0.25) < 1e-6
    assert abs(m4 - 1.0) < 1e-6
    assert abs(m5 - 1.0) < 1e-6


def test_cosine_endpoints():
    m_end = _lr_multiplier(100, warmup_steps=0, max_steps=100, lr_schedule="cosine", lr_min_ratio=0.1)
    m_start = _lr_multiplier(1, warmup_steps=0, max_steps=100, lr_schedule="cosine", lr_min_ratio=0.1)
    assert abs(m_end - 0.1) < 1e-5
    assert m_start > m_end


def test_warmup_then_cosine():
    m_w = _lr_multiplier(2, warmup_steps=2, max_steps=4, lr_schedule="cosine", lr_min_ratio=0.0)
    assert abs(m_w - 1.0) < 1e-6
    m_last = _lr_multiplier(4, warmup_steps=2, max_steps=4, lr_schedule="cosine", lr_min_ratio=0.0)
    assert abs(m_last) < 1e-5


def test_bad_schedule():
    with pytest.raises(ValueError):
        _lr_multiplier(1, warmup_steps=0, max_steps=10, lr_schedule="bogus", lr_min_ratio=0.1)
