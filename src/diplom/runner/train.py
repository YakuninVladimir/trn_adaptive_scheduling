from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diplom.runner.config import load_experiment_config
from diplom.runner.factory import build_model, build_scheduler, build_task, resolve_device
from diplom.schedulers.base import SchedulerState
from diplom.utils.jsonl import JsonlLogger
from diplom.utils.seed import seed_everything


def _per_sample_token_ce(logits: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Per-sample CE for token tasks.
    logits: [B, L, V], y: [B, L]
    Returns: [B]
    """
    B, L, V = logits.shape
    ce = F.cross_entropy(logits.view(B * L, V), y.view(B * L), reduction="none").view(B, L)
    if y_mask is None:
        return ce.mean(dim=1)
    denom = y_mask.float().sum(dim=1).clamp_min(1.0)
    return (ce * y_mask.float()).sum(dim=1) / denom


def train_from_yaml(
    config_path: str,
    live_plots_override: bool | None = None,
    live_plot_every_override: int | None = None,
) -> None:
    exp = load_experiment_config(config_path)

    seed_everything(exp.train.seed)
    device = resolve_device(exp.train.device)

    task = build_task(exp.task)
    model = build_model(exp.model).to(device)
    scheduler = build_scheduler(exp.scheduler)
    live_plots = exp.train.live_plots if live_plots_override is None else live_plots_override
    live_plot_every = exp.train.live_plot_every if live_plot_every_override is None else live_plot_every_override
    live_plot_every = max(int(live_plot_every), 1)

    n_sup = int(exp.model.get("N_sup", 16))
    run_dir = Path(exp.train.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Start a fresh metrics stream for each run invocation in this run_dir.
    logger = JsonlLogger(run_dir / "metrics.jsonl", truncate=True)

    train_dl, val_dl = task.build_dataloaders(batch_size=exp.train.batch_size)

    has_oracle_head = hasattr(model, "oracle_parameters") and callable(getattr(model, "oracle_parameters"))
    if has_oracle_head:
        opt_main = torch.optim.AdamW(model.backbone_parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay)
        opt_oracle = torch.optim.AdamW(model.oracle_parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay)
    else:
        opt_main = torch.optim.AdamW(model.parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay)
        opt_oracle = None

    global_step = 0
    max_steps = exp.train.max_steps or (exp.train.epochs * len(train_dl))

    model.train()
    start_time = time.time()
    epoch_iter = range(1, exp.train.epochs + 1)
    if exp.train.progress_bar:
        epoch_iter = tqdm(epoch_iter, total=exp.train.epochs, desc="Epochs")

    for epoch in epoch_iter:
        batch_iter = enumerate(train_dl)
        if exp.train.progress_bar:
            batch_iter = tqdm(
                batch_iter,
                total=len(train_dl),
                desc=f"Epoch {epoch}/{exp.train.epochs}",
                leave=False,
            )

        for batch_idx, batch in batch_iter:
            global_step += 1
            x_tokens = batch.x_tokens.to(device)
            y = batch.y.to(device)
            y_mask = batch.y_mask.to(device) if batch.y_mask is not None else None
            batch_on_device = type(batch)(x_tokens=x_tokens, y=y, y_mask=y_mask)

            state = None
            prev_aux = None
            total = torch.zeros((), device=device)
            total_main = torch.zeros((), device=device)
            total_oracle = torch.zeros((), device=device)
            last_metrics: dict[str, float] = {}
            used_sup = 0
            aux_hist: list[torch.Tensor] = []
            per_step_psloss: list[torch.Tensor] = []

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
                total_main = total_main + main_loss.detach()

                # optional halting head supervision (BCE on correctness)
                if "halt_prob" in out.loss_parts:
                    targets = task.halt_targets(out.logits, batch_on_device)
                    if targets is not None:
                        q = out.loss_parts["halt_prob"]
                        halt_loss = F.binary_cross_entropy(q, targets)
                        loss = loss + exp.train.beta_halt * halt_loss * float(sch.supervision_weight)
                        total_oracle = total_oracle + halt_loss.detach()

                total = total + loss
                last_metrics = task.compute_metrics(out.logits, batch_on_device)

                prev_aux = out.aux_tensor.detach() if out.aux_tensor is not None else None
                if prev_aux is not None:
                    aux_hist.append(prev_aux)
                if y.dim() == 2 and out.logits.dim() == 3:
                    per_step_psloss.append(_per_sample_token_ce(out.logits.detach(), y, y_mask))
                if out.state is not None:
                    if isinstance(out.state, tuple):
                        state = tuple(s.detach() for s in out.state)
                    else:
                        state = out.state

                # For oracle model: always perform full rollout to T_max, then train oracle head.
                allow_early_halt = not bool(getattr(model, "requires_full_rollout", False))
                if allow_early_halt and sch.halt_threshold is not None and "halt_prob" in out.loss_parts:
                    q = out.loss_parts["halt_prob"]
                    if bool((q > sch.halt_threshold).all().item()):
                        break

            opt_main.zero_grad(set_to_none=True)
            total.backward()
            opt_main.step()

            # Oracle-step training: full rollout already done, now train
            # prefix-conditioned oracle on future deltas.
            if has_oracle_head and aux_hist and per_step_psloss:
                aux_history = torch.stack(aux_hist, dim=1)
                per_step = torch.stack(per_step_psloss, dim=0)
                oracle_loss = model.oracle_loss_from_rollout(
                    aux_history.detach(),
                    per_step.detach(),
                )
                weighted_oracle = oracle_loss * float(getattr(model.cfg_oracle, "oracle_loss_weight", 1.0))

                opt_oracle.zero_grad(set_to_none=True)
                weighted_oracle.backward()
                opt_oracle.step()
                total_oracle = total_oracle + weighted_oracle.detach()

            if global_step % exp.train.log_every == 0:
                elapsed = time.time() - start_time
                rec = {
                    "kind": "train",
                    "time_s": elapsed,
                    "epoch": epoch,
                    "step": global_step,
                    "batch_idx": batch_idx,
                    "loss": float(total.detach().item()),
                    "main_loss": float(total_main.detach().item() / max(used_sup, 1)),
                    "oracle_loss": float(total_oracle.detach().item() / max(used_sup, 1)),
                    "used_sup": used_sup,
                    **last_metrics,
                }
                logger.log(rec)
                if exp.train.progress_bar:
                    postfix = {
                        "loss": f"{rec['loss']:.4f}",
                        "main": f"{rec['main_loss']:.4f}",
                        "oracle": f"{rec['oracle_loss']:.4f}",
                        "sup": rec["used_sup"],
                        "step": global_step,
                    }
                    # Show current train accuracy if task reports it.
                    if "exact_acc" in rec:
                        postfix["exact"] = f"{rec['exact_acc']:.4f}"
                    if "token_acc" in rec:
                        postfix["token"] = f"{rec['token_acc']:.4f}"
                    if "nextbin_acc" in rec:
                        postfix["nextbin"] = f"{rec['nextbin_acc']:.4f}"
                    # `batch_iter` is tqdm only when progress_bar is enabled.
                    try:
                        batch_iter.set_postfix(postfix)
                    except Exception:
                        pass
                if live_plots and global_step % live_plot_every == 0:
                    try:
                        from diplom.viz.plot_run import plot_run

                        plot_run(str(run_dir), out_path=str(run_dir / "plots.png"))
                    except Exception as e:
                        # Keep training robust even if plotting fails in environment-specific setups.
                        print(f"[train] live plot update failed at step={global_step}: {e}")

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

    if live_plots:
        try:
            from diplom.viz.plot_run import plot_run

            plot_run(str(run_dir), out_path=str(run_dir / "plots.png"))
        except Exception as e:
            print(f"[train] final live plot update failed: {e}")


@torch.no_grad()
def _validate_loop(
    model,
    task,
    val_dl,
    device: torch.device,
    n_sup: int,
    scheduler,
    max_epochs: int,
    max_steps: int,
    oracle_policy: str = "none",
    oracle_max_steps: int | None = None,
    oracle_temperature: float = 1.0,
) -> dict[str, float]:
    model.eval()
    metrics_sum: dict[str, float] = {}
    count = 0
    global_step = 0
    oracle_enabled = oracle_policy in {"greedy", "sampling"} and hasattr(model, "choose_delta")
    infer_step_cap = int(oracle_max_steps) if oracle_max_steps is not None else int(n_sup)
    infer_step_cap = max(infer_step_cap, 1)
    for batch in val_dl:
        global_step += 1
        x_tokens = batch.x_tokens.to(device)
        y = batch.y.to(device)
        y_mask = batch.y_mask.to(device) if batch.y_mask is not None else None
        batch_on_device = type(batch)(x_tokens=x_tokens, y=y, y_mask=y_mask)

        state = None
        prev_aux = None
        logits = None
        if not oracle_enabled:
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
        else:
            # Oracle-guided iterative inference:
            # run current step -> choose delta (0 means stop now) -> advance delta -> repeat.
            aux_hist: list[torch.Tensor] = []
            sup_step = 0
            halted = False
            while sup_step < infer_step_cap:
                st = SchedulerState(
                    epoch=max_epochs,
                    max_epochs=max_epochs,
                    global_step=global_step,
                    max_steps=max_steps,
                    supervision_step=sup_step + 1,
                    max_supervision_steps=max(infer_step_cap, 1),
                    task_name=getattr(task, "name", None),
                )
                sch = scheduler.get_schedule(st, model_aux=prev_aux)
                out = model(x_tokens, state=state, recursion_n=sch.recursion_n, recursion_T=sch.recursion_T)
                logits = out.logits
                prev_aux = out.aux_tensor
                state = out.state
                sup_step += 1
                if out.aux_tensor is not None:
                    aux_hist.append(out.aux_tensor.detach())

                if sch.halt_threshold is not None and "halt_prob" in out.loss_parts:
                    q = out.loss_parts["halt_prob"]
                    if bool((q > sch.halt_threshold).all().item()):
                        halted = True
                        break

                if not aux_hist:
                    break
                valid_horizon = min(int(getattr(model.cfg_oracle, "oracle_horizon", n_sup)), infer_step_cap - sup_step + 1)
                if valid_horizon <= 0:
                    break
                aux_prefix = torch.stack(aux_hist, dim=1)
                delta = model.choose_delta(
                    aux_prefix,
                    valid_horizon=valid_horizon,
                    policy=oracle_policy,
                    temperature=oracle_temperature,
                )
                # delta==0 means stop at current step.
                if delta <= 0:
                    break

                # Advance fixed delta steps, then re-query oracle.
                for _ in range(delta):
                    if sup_step >= infer_step_cap:
                        break
                    st2 = SchedulerState(
                        epoch=max_epochs,
                        max_epochs=max_epochs,
                        global_step=global_step,
                        max_steps=max_steps,
                        supervision_step=sup_step + 1,
                        max_supervision_steps=max(infer_step_cap, 1),
                        task_name=getattr(task, "name", None),
                    )
                    sch2 = scheduler.get_schedule(st2, model_aux=prev_aux)
                    out2 = model(x_tokens, state=state, recursion_n=sch2.recursion_n, recursion_T=sch2.recursion_T)
                    logits = out2.logits
                    prev_aux = out2.aux_tensor
                    state = out2.state
                    sup_step += 1
                    if out2.aux_tensor is not None:
                        aux_hist.append(out2.aux_tensor.detach())
                    if sch2.halt_threshold is not None and "halt_prob" in out2.loss_parts:
                        q2 = out2.loss_parts["halt_prob"]
                        if bool((q2 > sch2.halt_threshold).all().item()):
                            halted = True
                            break
                if halted:
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

