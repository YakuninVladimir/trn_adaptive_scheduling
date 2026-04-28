from __future__ import annotations

from pathlib import Path

import torch

from diplom.models.trm_oracle import TRMOracle, TRMOracleConfig
from diplom.runner.oracle_trace import (
    SCHEMA_VERSION_SHARD,
    build_oracle_trace_batch_payload,
    save_oracle_trace_batch,
    save_oracle_trace_shard,
)


def test_save_oracle_trace_roundtrip_matches_oracle_loss(tmp_path: Path) -> None:
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
        halt_head=True,
        oracle_max_steps=4,
        oracle_horizon=4,
        oracle_n_heads=2,
        oracle_n_layers=1,
        oracle_d_ff=64,
    )
    m = TRMOracle(cfg)
    x = torch.randint(0, 10, (3, 81), dtype=torch.long)

    aux_hist: list[torch.Tensor] = []
    per_step: list[torch.Tensor] = []
    logits_hist: list[torch.Tensor] = []
    sy: list[torch.Tensor] = []
    sz: list[torch.Tensor] = []
    halt_hist: list[torch.Tensor | None] = []
    state = None
    for _ in range(4):
        out = m(x, state=state, recursion_n=1, recursion_T=1)
        state = tuple(s.detach() for s in out.state)
        aux_hist.append(out.state[0].detach())
        per_step.append(torch.rand(3))
        logits_hist.append(out.logits.detach())
        sy.append(out.state[0].detach())
        sz.append(out.state[1].detach())
        hp = out.loss_parts.get("halt_prob")
        halt_hist.append(hp.detach() if isinstance(hp, torch.Tensor) else None)

    y = torch.randint(0, 10, (3, 81), dtype=torch.long)
    path = save_oracle_trace_batch(
        tmp_path,
        epoch=1,
        global_step=2,
        batch_idx=0,
        x_tokens=x,
        y=y,
        y_mask=None,
        y_weight=None,
        aux_hist=aux_hist,
        per_step_psloss=per_step,
        logits_hist=logits_hist,
        state_y_hist=sy,
        state_z_hist=sz,
        halt_hist=halt_hist,
        schedule_rows=[
            {
                "supervision_step": i + 1,
                "recursion_n": 1,
                "recursion_T": 1,
                "supervision_weight": 1.0,
                "halt_threshold": 0.5,
            }
            for i in range(4)
        ],
        n_sup=4,
        used_sup=4,
        config_path="dummy.yaml",
        model_name="TRMOracle",
        model_config={"N_sup": 4},
        oracle_cfg={"oracle_horizon": 4},
        fp16=False,
    )

    blob = torch.load(path, map_location="cpu", weights_only=False)
    aux_seq = blob["aux_seq"]
    if aux_seq.dim() == 4:
        aux_btd = aux_seq.permute(1, 0, 2, 3)
    else:
        aux_btd = aux_seq.permute(1, 0, 2)
    ce_tb = blob["per_sample_ce"]
    loss_disk = m.oracle_loss_from_rollout(aux_btd, ce_tb)
    loss_direct = m.oracle_loss_from_rollout(torch.stack(aux_hist, dim=1), torch.stack(per_step, dim=0))
    assert torch.allclose(loss_disk, loss_direct)


def test_oracle_trace_shard_file(tmp_path) -> None:
    r0 = build_oracle_trace_batch_payload(
        epoch=1,
        global_step=1,
        batch_idx=0,
        x_tokens=torch.zeros(1, 2, dtype=torch.long),
        y=torch.zeros(1, 2, dtype=torch.long),
        y_mask=None,
        y_weight=None,
        aux_hist=[torch.zeros(1, 4), torch.zeros(1, 4)],
        per_step_psloss=[torch.zeros(1), torch.zeros(1)],
        logits_hist=None,
        state_y_hist=None,
        state_z_hist=None,
        halt_hist=[],
        schedule_rows=[{"supervision_step": 1}, {"supervision_step": 2}],
        n_sup=2,
        used_sup=2,
        config_path="c.yaml",
        model_name="M",
        model_config={"k": 1},
        oracle_cfg=None,
        fp16=False,
        embed_shared_meta=False,
        embed_notes=False,
    )
    r1 = build_oracle_trace_batch_payload(
        epoch=1,
        global_step=2,
        batch_idx=1,
        x_tokens=torch.ones(1, 2, dtype=torch.long),
        y=torch.ones(1, 2, dtype=torch.long),
        y_mask=None,
        y_weight=None,
        aux_hist=[torch.ones(1, 4), torch.ones(1, 4)],
        per_step_psloss=[torch.ones(1), torch.ones(1)],
        logits_hist=None,
        state_y_hist=None,
        state_z_hist=None,
        halt_hist=[],
        schedule_rows=[{"supervision_step": 1}, {"supervision_step": 2}],
        n_sup=2,
        used_sup=2,
        config_path="c.yaml",
        model_name="M",
        model_config={"k": 1},
        oracle_cfg=None,
        fp16=False,
        embed_shared_meta=False,
        embed_notes=False,
    )
    path = save_oracle_trace_shard(
        tmp_path,
        0,
        [r0, r1],
        model_name="M",
        model_config={"k": 1},
        oracle_cfg=None,
        config_path="c.yaml",
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["schema_version"] == SCHEMA_VERSION_SHARD
    assert len(blob["batches"]) == 2
    assert blob["batches"][0]["global_step"] == 1
    assert blob["batches"][1]["global_step"] == 2
