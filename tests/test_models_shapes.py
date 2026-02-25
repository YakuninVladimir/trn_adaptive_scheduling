from __future__ import annotations

import torch

from diplom.models.hrm import HRM, HRMConfig
from diplom.models.trm import TRM, TRMConfig


def test_hrm_forward_shapes():
    cfg = HRMConfig(
        vocab_size=11,
        seq_len=81,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers_L=1,
        n_layers_H=1,
        L_cycles=2,
        H_cycles=2,
        one_step_grad_approx=True,
        halt_head=True,
    )
    m = HRM(cfg)
    x = torch.randint(0, 10, (3, 81), dtype=torch.long)
    out = m(x)
    assert out.logits.shape == (3, 81, 11)
    assert out.aux_tensor is not None and out.aux_tensor.shape == (3, 32)
    assert isinstance(out.state, tuple)
    zH, zL = out.state
    assert zH.shape == (3, 81, 32)
    assert zL.shape == (3, 81, 32)
    assert "halt_prob" in out.loss_parts


def test_trm_forward_shapes():
    cfg = TRMConfig(
        vocab_size=11,
        seq_len=81,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        L_cycles=2,
        H_cycles=2,
        halt_head=True,
        use_attention=True,
    )
    m = TRM(cfg)
    x = torch.randint(0, 10, (2, 81), dtype=torch.long)
    out = m(x)
    assert out.logits.shape == (2, 81, 11)
    assert out.aux_tensor is not None and out.aux_tensor.shape == (2, 32)
    assert isinstance(out.state, tuple)
    y, z = out.state
    assert y.shape == (2, 81, 32)
    assert z.shape == (2, 81, 32)
    assert "halt_prob" in out.loss_parts

