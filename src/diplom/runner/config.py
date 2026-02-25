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
    max_steps: int | None = None
    log_every: int = 10
    eval_every: int = 100
    ckpt_every: int = 200
    run_dir: str = "runs/dev"
    beta_halt: float = 0.5


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

