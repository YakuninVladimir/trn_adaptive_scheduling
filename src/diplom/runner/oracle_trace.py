"""
Save per-batch tensors needed to reproduce / train the TRMOracle head offline.

Oracle forward uses prefix sequences per supervision step: either full ``y``
``[B, L, d_model]`` (default ``oracle_use_full_y``) or mean-pooled ``aux_tensor``
``[B, d_model]``. Stacked history is ``[B, T, L, D]`` or ``[B, T, D]``.

Training uses ``oracle_loss_from_rollout(aux_history_full, per_step_psloss, per_step_acc=...)`` with
``per_step_psloss`` ``[T, B]`` (weighted per-sample CE) and optional ``per_step_acc`` ``[T, B]`` (masked token
accuracy). When ``per_step_acc`` is provided, distribution-mode oracle targets use ``tau* = argmax_t acc[t]``.

Storage:
  - schema_version 1: one batch per file (legacy), or single-batch dict shape.
  - schema_version 2: shard file with ``batches`` list of batch records (shared metadata once).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION_BATCH = 1
SCHEMA_VERSION_SHARD = 2

NOTES_ORACLE_LAYOUT = (
    "aux_seq is [T,B,D] (mean_y) or [T,B,L,D] (full_y). For oracle_loss_from_rollout use "
    "permute(1,0,2) or permute(1,0,2,3) to get [B,T,...]."
)


def _fcpu(x: torch.Tensor, fp16: bool) -> torch.Tensor:
    y = x.detach().cpu()
    if y.dtype.is_floating_point and fp16:
        return y.half()
    return y


def build_oracle_trace_batch_payload(
    *,
    epoch: int,
    global_step: int,
    batch_idx: int,
    x_tokens: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor | None,
    y_weight: torch.Tensor | None,
    aux_hist: list[torch.Tensor],
    per_step_psloss: list[torch.Tensor],
    logits_hist: list[torch.Tensor] | None,
    state_y_hist: list[torch.Tensor] | None,
    state_z_hist: list[torch.Tensor] | None,
    halt_hist: list[torch.Tensor | None],
    schedule_rows: list[dict[str, Any]],
    n_sup: int,
    used_sup: int,
    config_path: str,
    model_name: str,
    model_config: dict[str, Any],
    oracle_cfg: dict[str, Any] | None,
    fp16: bool,
    embed_shared_meta: bool = True,
    embed_notes: bool = True,
) -> dict[str, Any]:
    """
    One batch record. If ``embed_shared_meta`` is False, omit fields duplicated in a shard header
    (model_name, model_config, oracle_cfg, config_path).
    """
    if len(aux_hist) != len(per_step_psloss):
        raise ValueError(
            f"aux_hist ({len(aux_hist)}) and per_step_psloss ({len(per_step_psloss)}) length mismatch"
        )
    if schedule_rows and len(schedule_rows) != len(aux_hist):
        raise ValueError(
            f"schedule_rows ({len(schedule_rows)}) and aux_hist ({len(aux_hist)}) length mismatch"
        )

    el0 = aux_hist[0]
    if el0.dim() == 2:
        aux_seq = torch.stack([_fcpu(t, fp16) for t in aux_hist], dim=0)  # [T, B, D]
        layout = "mean_y"
        d_aux = int(aux_seq.shape[2])
    elif el0.dim() == 3:
        aux_seq = torch.stack([_fcpu(t, fp16) for t in aux_hist], dim=0)  # [T, B, L, D]
        layout = "full_y"
        d_aux = int(aux_seq.shape[3])
    else:
        raise ValueError(f"aux_hist step tensor must be [B,D] or [B,L,D], got dim {el0.dim()}")
    ps_ce = torch.stack([_fcpu(t, fp16) for t in per_step_psloss], dim=0)  # [T, B]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_BATCH,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "batch_idx": int(batch_idx),
        "n_sup": int(n_sup),
        "used_supervision_steps": int(used_sup),
        "T": int(aux_seq.shape[0]),
        "B": int(aux_seq.shape[1]),
        "oracle_history_layout": layout,
        "d_aux": d_aux,
        **({"L_aux": int(aux_seq.shape[2])} if layout == "full_y" else {}),
        "aux_seq": aux_seq,
        "per_sample_ce": ps_ce,
        "x_tokens": x_tokens.detach().cpu(),
        "y": y.detach().cpu(),
        "schedule": schedule_rows,
    }

    if embed_shared_meta:
        payload["model_name"] = model_name
        payload["model_config"] = dict(model_config)
        payload["config_path"] = str(config_path)
        if oracle_cfg is not None:
            payload["oracle_cfg"] = dict(oracle_cfg)

    if embed_notes:
        payload["notes"] = NOTES_ORACLE_LAYOUT

    if y_mask is not None:
        payload["y_mask"] = y_mask.detach().cpu()
    if y_weight is not None:
        payload["y_weight"] = _fcpu(y_weight, fp16)

    if logits_hist is not None:
        if len(logits_hist) != len(aux_hist):
            raise ValueError("logits_hist length mismatch")
        payload["logits_seq"] = torch.stack([_fcpu(t, fp16) for t in logits_hist], dim=0)  # [T,B,L,V]

    if state_y_hist is not None and state_z_hist is not None:
        if len(state_y_hist) != len(aux_hist) or len(state_z_hist) != len(aux_hist):
            raise ValueError("state hist length mismatch")
        payload["state_y_seq"] = torch.stack([_fcpu(t, fp16) for t in state_y_hist], dim=0)  # [T,B,L,D]
        payload["state_z_seq"] = torch.stack([_fcpu(t, fp16) for t in state_z_hist], dim=0)

    if halt_hist:
        if len(halt_hist) != len(aux_hist):
            raise ValueError(f"halt_hist ({len(halt_hist)}) vs aux_hist ({len(aux_hist)}) length mismatch")
        cleaned: list[torch.Tensor | None] = []
        for h in halt_hist:
            if h is None:
                cleaned.append(None)
            else:
                cleaned.append(_fcpu(h, fp16))
        payload["halt_prob_seq"] = cleaned  # list length T, each [B] or None

    return payload


def save_oracle_trace_batch(
    dump_dir: Path,
    *,
    epoch: int,
    global_step: int,
    batch_idx: int,
    x_tokens: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor | None,
    y_weight: torch.Tensor | None,
    aux_hist: list[torch.Tensor],
    per_step_psloss: list[torch.Tensor],
    logits_hist: list[torch.Tensor] | None,
    state_y_hist: list[torch.Tensor] | None,
    state_z_hist: list[torch.Tensor] | None,
    halt_hist: list[torch.Tensor | None],
    schedule_rows: list[dict[str, Any]],
    n_sup: int,
    used_sup: int,
    config_path: str,
    model_name: str,
    model_config: dict[str, Any],
    oracle_cfg: dict[str, Any] | None,
    fp16: bool,
) -> Path:
    """Writes one ``.pt`` file with a single batch (schema_version 1, full metadata)."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    fn = f"trace_e{epoch:04d}_step{global_step:08d}_b{batch_idx:05d}.pt"
    path = dump_dir / fn
    payload = build_oracle_trace_batch_payload(
        epoch=epoch,
        global_step=global_step,
        batch_idx=batch_idx,
        x_tokens=x_tokens,
        y=y,
        y_mask=y_mask,
        y_weight=y_weight,
        aux_hist=aux_hist,
        per_step_psloss=per_step_psloss,
        logits_hist=logits_hist,
        state_y_hist=state_y_hist,
        state_z_hist=state_z_hist,
        halt_hist=halt_hist,
        schedule_rows=schedule_rows,
        n_sup=n_sup,
        used_sup=used_sup,
        config_path=config_path,
        model_name=model_name,
        model_config=model_config,
        oracle_cfg=oracle_cfg,
        fp16=fp16,
        embed_shared_meta=True,
        embed_notes=True,
    )
    torch.save(payload, path)
    return path


def save_oracle_trace_shard(
    dump_dir: Path,
    shard_index: int,
    batches: list[dict[str, Any]],
    *,
    model_name: str,
    model_config: dict[str, Any],
    oracle_cfg: dict[str, Any] | None,
    config_path: str,
) -> Path:
    """
    Writes one ``.pt`` with ``schema_version`` 2 and a list of batch records
    (each typically built with ``embed_shared_meta=False``).
    """
    if not batches:
        raise ValueError("save_oracle_trace_shard: empty batches")
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"oracle_trace_shard_{shard_index:05d}.pt"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_SHARD,
        "shard_index": int(shard_index),
        "n_batches": len(batches),
        "model_name": model_name,
        "model_config": dict(model_config),
        "config_path": str(config_path),
        "batch_schema_version": SCHEMA_VERSION_BATCH,
        "notes": NOTES_ORACLE_LAYOUT,
        "batches": batches,
    }
    if oracle_cfg is not None:
        payload["oracle_cfg"] = dict(oracle_cfg)
    torch.save(payload, path)
    return path
