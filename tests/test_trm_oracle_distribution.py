from __future__ import annotations

import torch

from diplom.models.trm_oracle import TRMOracle, TRMOracleConfig


def test_distribution_models_return_normalized_pmf():
    cfg = TRMOracleConfig(
        vocab_size=11,
        seq_len=16,
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
        oracle_use_full_y=False,
    )
    m = TRMOracle(cfg)
    aux = torch.randn(2, 3, 32)
    models = [
        "finite_discrete",
        "smoothed_loss",
        "mixture_geometric",
        "mixture_exponential",
        "power",
        "negative_binomial",
        "lognormal",
        "hybrid",
    ]
    for name in models:
        pmf, _ = m.oracle_distribution(aux, valid_horizon=4, distribution_model=name)
        assert pmf.shape == (2, 4)
        sums = pmf.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
