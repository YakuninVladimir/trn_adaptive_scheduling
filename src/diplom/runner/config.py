from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    device: str = "auto"  # auto|cpu|cuda
    epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    # Optimizer LR: linear warmup then optional cosine decay to lr * lr_min_ratio.
    warmup_steps: int = 0
    lr_schedule: str = "none"  # none|cosine (cosine applies after warmup)
    lr_min_ratio: float = 0.1
    max_steps: int | None = None
    log_every: int = 10
    eval_every: int = 100
    ckpt_every: int = 200
    save_best_only: bool = True
    best_metric: str = "auto"  # auto|val_loss|train_loss|<metric key>
    best_metric_mode: str = "auto"  # auto|min|max
    run_dir: str = "runs/dev"
    beta_halt: float = 0.5
    use_halt_loss: bool = True
    progress_bar: bool = True
    live_plots: bool = True
    live_plot_every: int = 100
    # Dump per-batch tensors for offline oracle-head training (aux sequence + per-step CE, optional logits/state).
    dump_oracle_trace: bool = False
    dump_oracle_trace_dir: str | None = None  # relative to run_dir if not absolute; default run_dir/oracle_traces
    dump_oracle_trace_every: int = 1  # every N global optimizer steps (1 = each batch)
    dump_oracle_trace_shard_batches: int = 64  # pack this many batch records into one .pt (fewer files, faster to read)
    dump_oracle_trace_max_batches: int | None = None  # stop recording after this many batches total (None = no limit)
    dump_oracle_trace_include_logits: bool = True
    dump_oracle_trace_include_state: bool = True  # TRM/HRM (y, z) latents per step
    dump_oracle_trace_fp16: bool = True  # store float tensors on CPU as float16 where applicable
    # CUDA only: autocast + GradScaler (fp16 or bf16). No effect on CPU.
    amp: bool = False
    amp_dtype: str = "float16"  # float16 | bfloat16 (bf16 if hardware supports it)


@dataclass(frozen=True)
class ExperimentConfig:
    task: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_experiment_config(path: str) -> ExperimentConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping")

    train_raw = raw.get("train", {}) or {}
    train_cfg = TrainConfig(**train_raw)
    return ExperimentConfig(
        task=raw.get("task", {}) or {},
        model=raw.get("model", {}) or {},
        scheduler=raw.get("scheduler", {}) or {},
        train=train_cfg,
    )

