from __future__ import annotations

import torch

from diplom.models.trm_oracle import TRMOracle, TRMOracleConfig


def test_trm_oracle_head_shapes_and_loss():
    cfg = TRMOracleConfig(
        vocab_size=11,
        seq_len=81,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        L_cycles=1,
        H_cycles=1,
        N_sup=4,
        halt_head=False,
        oracle_max_steps=4,
        oracle_horizon=4,
        oracle_n_heads=2,
        oracle_n_layers=1,
        oracle_d_ff=64,
    )
    m = TRMOracle(cfg)
    x = torch.randint(0, 10, (3, 81), dtype=torch.long)

    state = None
    aux_hist = []
    for _ in range(4):
        out = m(x, state=state, recursion_n=1, recursion_T=1)
        state = tuple(s.detach() for s in out.state)
        aux_hist.append(out.state[0].detach())

    aux = torch.stack(aux_hist, dim=1)  # [B, T, L, D]
    logits = m.oracle_logits(aux)
    assert logits.shape == (3, 4)

    target = torch.tensor([0, 1, 2], dtype=torch.long)
    loss = m.oracle_loss(aux, target)
    assert loss.dim() == 0
    assert float(loss.item()) >= 0.0

    per_step_psloss = torch.rand(4, 3)
    rollout_loss = m.oracle_loss_from_rollout(aux, per_step_psloss)
    assert rollout_loss.dim() == 0
    assert float(rollout_loss.item()) >= 0.0

    # Current-step-inclusive target (delta=0 allowed): T=1 should still produce valid loss.
    aux_one = aux[:, :1]
    per_step_one = torch.rand(1, 3)
    rollout_loss_one = m.oracle_loss_from_rollout(aux_one, per_step_one)
    assert rollout_loss_one.dim() == 0
    assert float(rollout_loss_one.item()) >= 0.0

    d_greedy = m.choose_delta(aux, valid_horizon=4, policy="greedy")
    assert 0 <= d_greedy < 4
    d_sampling = m.choose_delta(aux, valid_horizon=4, policy="sampling", temperature=1.0)
    assert 0 <= d_sampling < 4

