from __future__ import annotations

from pathlib import Path

import torch

from diplom.runner.config import load_experiment_config
from diplom.runner.factory import build_model, build_scheduler, build_task, resolve_device
from diplom.runner.train import _validate_loop


def validate_from_yaml(config_path: str, checkpoint_path: str | None = None) -> None:
    exp = load_experiment_config(config_path)
    device = resolve_device(exp.train.device)

    task = build_task(exp.task)
    model = build_model(exp.model).to(device)
    scheduler = build_scheduler(exp.scheduler)

    n_sup = int(exp.model.get("N_sup", 16))

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)

    _, val_dl = task.build_dataloaders(batch_size=exp.train.batch_size)
    if val_dl is None:
        raise SystemExit("No validation dataloader configured for this task.")

    metrics = _validate_loop(
        model=model,
        task=task,
        val_dl=val_dl,
        device=device,
        n_sup=n_sup,
        scheduler=scheduler,
        max_epochs=exp.train.epochs,
        max_steps=exp.train.max_steps or exp.train.epochs * len(val_dl),
    )
    ckpt_info = f"checkpoint={checkpoint_path}" if checkpoint_path else "checkpoint=None"
    print(f"[validate] {Path(config_path).name} {ckpt_info} metrics={metrics}")

