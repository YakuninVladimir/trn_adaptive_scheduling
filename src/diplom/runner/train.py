from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from diplom.runner.config import load_experiment_config
from diplom.runner.factory import build_model, build_scheduler, build_task, resolve_device
from diplom.schedulers.base import SchedulerState
from diplom.utils.jsonl import JsonlLogger
from diplom.utils.seed import seed_everything


def train_from_yaml(config_path: str) -> None:
    exp = load_experiment_config(config_path)

    seed_everything(exp.train.seed)
    device = resolve_device(exp.train.device)

    task = build_task(exp.task)
    model = build_model(exp.model).to(device)
    scheduler = build_scheduler(exp.scheduler)

    n_sup = int(exp.model.get("N_sup", 16))
    run_dir = Path(exp.train.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(run_dir / "metrics.jsonl")

    train_dl, val_dl = task.build_dataloaders(batch_size=exp.train.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay)

    global_step = 0
    max_steps = exp.train.max_steps or (exp.train.epochs * len(train_dl))

    model.train()
    start_time = time.time()
    for epoch in range(1, exp.train.epochs + 1):
        for batch_idx, batch in enumerate(train_dl):
            global_step += 1
            x_tokens = batch.x_tokens.to(device)
            y = batch.y.to(device)
            y_mask = batch.y_mask.to(device) if batch.y_mask is not None else None
            batch_on_device = type(batch)(x_tokens=x_tokens, y=y, y_mask=y_mask)

            state = None
            prev_aux = None
            total = torch.zeros((), device=device)
            last_metrics: dict[str, float] = {}
            used_sup = 0

            for sup_step in range(1, n_sup + 1):
                used_sup = sup_step
                st = SchedulerState(
                    epoch=epoch,
                    max_epochs=exp.train.epochs,
                    global_step=global_step,
                    max_steps=max_steps,
                    supervision_step=sup_step,
                    max_supervision_steps=n_sup,
                    task_name=getattr(task, "name", None),
                )
                sch = scheduler.get_schedule(st, model_aux=prev_aux)
                out = model(
                    x_tokens,
                    state=state,
                    recursion_n=sch.recursion_n,
                    recursion_T=sch.recursion_T,
                )

                main_loss = task.compute_loss(out.logits, batch_on_device)
                loss = main_loss * float(sch.supervision_weight)

                # optional halting head supervision (BCE on correctness)
                if "halt_prob" in out.loss_parts:
                    targets = task.halt_targets(out.logits, batch_on_device)
                    if targets is not None:
                        q = out.loss_parts["halt_prob"]
                        halt_loss = F.binary_cross_entropy(q, targets)
                        loss = loss + exp.train.beta_halt * halt_loss * float(sch.supervision_weight)

                total = total + loss
                last_metrics = task.compute_metrics(out.logits, batch_on_device)

                prev_aux = out.aux_tensor.detach() if out.aux_tensor is not None else None
                if out.state is not None:
                    if isinstance(out.state, tuple):
                        state = tuple(s.detach() for s in out.state)
                    else:
                        state = out.state

                if sch.halt_threshold is not None and "halt_prob" in out.loss_parts:
                    q = out.loss_parts["halt_prob"]
                    if bool((q > sch.halt_threshold).all().item()):
                        break

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

            if global_step % exp.train.log_every == 0:
                elapsed = time.time() - start_time
                rec = {
                    "kind": "train",
                    "time_s": elapsed,
                    "epoch": epoch,
                    "step": global_step,
                    "batch_idx": batch_idx,
                    "loss": float(total.detach().item()),
                    "used_sup": used_sup,
                    **last_metrics,
                }
                logger.log(rec)

            if val_dl is not None and global_step % exp.train.eval_every == 0:
                val = _validate_loop(model, task, val_dl, device=device, n_sup=n_sup, scheduler=scheduler, max_epochs=exp.train.epochs, max_steps=max_steps)
                logger.log({"kind": "val", "epoch": epoch, "step": global_step, **val})
                model.train()

            if global_step % exp.train.ckpt_every == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "config_path": str(config_path),
                    },
                    ckpt_path,
                )

            if exp.train.max_steps is not None and global_step >= exp.train.max_steps:
                break
        if exp.train.max_steps is not None and global_step >= exp.train.max_steps:
            break


@torch.no_grad()
def _validate_loop(model, task, val_dl, device: torch.device, n_sup: int, scheduler, max_epochs: int, max_steps: int) -> dict[str, float]:
    model.eval()
    metrics_sum: dict[str, float] = {}
    count = 0
    global_step = 0
    for batch in val_dl:
        global_step += 1
        x_tokens = batch.x_tokens.to(device)
        y = batch.y.to(device)
        y_mask = batch.y_mask.to(device) if batch.y_mask is not None else None
        batch_on_device = type(batch)(x_tokens=x_tokens, y=y, y_mask=y_mask)

        state = None
        prev_aux = None
        logits = None
        for sup_step in range(1, n_sup + 1):
            st = SchedulerState(
                epoch=max_epochs,
                max_epochs=max_epochs,
                global_step=global_step,
                max_steps=max_steps,
                supervision_step=sup_step,
                max_supervision_steps=n_sup,
                task_name=getattr(task, "name", None),
            )
            sch = scheduler.get_schedule(st, model_aux=prev_aux)
            out = model(x_tokens, state=state, recursion_n=sch.recursion_n, recursion_T=sch.recursion_T)
            logits = out.logits
            prev_aux = out.aux_tensor
            state = out.state

            if sch.halt_threshold is not None and "halt_prob" in out.loss_parts:
                q = out.loss_parts["halt_prob"]
                if bool((q > sch.halt_threshold).all().item()):
                    break

        assert logits is not None
        m = task.compute_metrics(logits, batch_on_device)
        for k, v in m.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v)
        count += 1
        if count >= 50:
            # keep validation cheap by default
            break

    if count == 0:
        return {}
    return {k: v / count for k, v in metrics_sum.items()}

