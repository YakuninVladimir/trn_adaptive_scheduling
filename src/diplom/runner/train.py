from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diplom.models.trm_oracle import oracle_history_tensor_from_output
from diplom.runner.config import load_experiment_config
from diplom.runner.factory import build_model, build_scheduler, build_task, resolve_device
from diplom.runner.oracle_trace import build_oracle_trace_batch_payload, save_oracle_trace_shard
from diplom.schedulers.base import SchedulerState
from diplom.utils.jsonl import JsonlLogger
from diplom.utils.seed import seed_everything


def _resolve_amp_dtype(name: str) -> torch.dtype:
    n = (name or "float16").strip().lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def _lr_multiplier(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    lr_schedule: str,
    lr_min_ratio: float,
) -> float:
    """
    Per-step multiplier applied to each optimizer param group base LR.
    `step` is 1-based global training step (first batch => 1).
    """
    sched = (lr_schedule or "none").strip().lower()
    min_r = float(lr_min_ratio)
    min_r = max(0.0, min(min_r, 1.0))

    if warmup_steps > 0:
        if step <= warmup_steps:
            return float(step) / float(max(warmup_steps, 1))
        post = step - warmup_steps
        span = max(max_steps - warmup_steps, 1)
    else:
        post = step
        span = max(max_steps, 1)

    if sched in ("none", "constant", ""):
        return 1.0
    if sched == "cosine":
        t = float(post) / float(span)
        t = min(1.0, max(0.0, t))
        c = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_r + (1.0 - min_r) * c
    raise ValueError(f"Unknown train.lr_schedule: {lr_schedule!r} (use none|cosine)")


def _apply_lr_multiplier(
    opt: torch.optim.Optimizer,
    base_lrs: list[float],
    mult: float,
) -> None:
    for i, pg in enumerate(opt.param_groups):
        pg["lr"] = float(base_lrs[i]) * mult


def _pick_best_metric_name(train_cfg, has_val: bool, val_metrics: dict[str, float] | None) -> str:
    raw = str(getattr(train_cfg, "best_metric", "auto") or "auto").strip().lower()
    if raw != "auto":
        return raw
    if has_val:
        if val_metrics:
            if "val_loss" in val_metrics:
                return "val_loss"
            if "loss" in val_metrics:
                return "loss"
            for k in ("token_acc", "exact_acc", "nextbin_acc", "mae"):
                if k in val_metrics:
                    return k
        return "val_loss"
    return "train_loss"


def _pick_metric_mode(metric_name: str, train_cfg) -> str:
    raw = str(getattr(train_cfg, "best_metric_mode", "auto") or "auto").strip().lower()
    if raw in ("min", "max"):
        return raw
    # auto
    if "loss" in metric_name or metric_name in ("mae", "mse"):
        return "min"
    return "max"


def _per_sample_token_ce(
    logits: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor | None,
    y_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Per-sample CE for token tasks.
    logits: [B, L, V], y: [B, L]
    Returns: [B]
    """
    B, L, V = logits.shape
    ce = F.cross_entropy(logits.view(B * L, V), y.view(B * L), reduction="none").view(B, L)
    if y_mask is None:
        return ce.mean(dim=1)
    m = y_mask.float()
    w = y_weight.float() if y_weight is not None else torch.ones_like(m)
    denom = (m * w).sum(dim=1).clamp_min(1.0)
    return (ce * m * w).sum(dim=1) / denom


def train_from_yaml(
    config_path: str,
    init_checkpoint: str | None = None,
    oracle_only: bool = False,
    live_plots_override: bool | None = None,
    live_plot_every_override: int | None = None,
) -> None:
    exp = load_experiment_config(config_path)

    seed_everything(exp.train.seed)
    device = resolve_device(exp.train.device)

    task = build_task(exp.task)
    model = build_model(exp.model).to(device)
    if init_checkpoint is not None:
        ckpt = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[train] loaded init checkpoint: {init_checkpoint}")
    if oracle_only:
        if not (hasattr(model, "oracle_parameters") and callable(model.oracle_parameters)):
            raise ValueError("--oracle-only requires a model with oracle_parameters().")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.oracle_parameters():
            p.requires_grad = True
        print("[train] oracle-only mode: backbone frozen, training oracle head only")
    n_trainable = _count_trainable_parameters(model)
    print(f"[train] model={model.__class__.__name__} trainable_parameters={n_trainable:,}")
    scheduler = build_scheduler(exp.scheduler)
    live_plots = exp.train.live_plots if live_plots_override is None else live_plots_override
    live_plot_every = exp.train.live_plot_every if live_plot_every_override is None else live_plot_every_override
    live_plot_every = max(int(live_plot_every), 1)

    n_sup = int(exp.model.get("N_sup", 16))
    run_dir = Path(exp.train.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    oracle_trace_dir: Path | None = None
    oracle_trace_state: dict[str, object] | None = None
    if exp.train.dump_oracle_trace:
        td = exp.train.dump_oracle_trace_dir or "oracle_traces"
        oracle_trace_dir = Path(td)
        if not oracle_trace_dir.is_absolute():
            oracle_trace_dir = run_dir / oracle_trace_dir
        oracle_trace_dir.mkdir(parents=True, exist_ok=True)
        shard_sz = max(int(exp.train.dump_oracle_trace_shard_batches), 1)
        oracle_trace_state = {"shard_idx": 0, "buffer": [], "batches_recorded": 0}
        print(
            f"[train] oracle trace dump -> {oracle_trace_dir} "
            f"(every {max(int(exp.train.dump_oracle_trace_every), 1)} step(s), "
            f"{shard_sz} batch(es) per shard file)"
        )
    # Start a fresh metrics stream for each run invocation in this run_dir.
    logger = JsonlLogger(run_dir / "metrics.jsonl", truncate=True)

    train_dl, val_dl = task.build_dataloaders(batch_size=exp.train.batch_size)

    has_oracle_head = hasattr(model, "oracle_parameters") and callable(model.oracle_parameters)
    if oracle_only and not has_oracle_head:
        raise ValueError("--oracle-only requires a model with oracle_parameters().")
    if has_oracle_head:
        opt_main = (
            None
            if oracle_only
            else torch.optim.AdamW(model.backbone_parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay)
        )
        opt_oracle = torch.optim.AdamW(model.oracle_parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay)
    else:
        if oracle_only:
            raise ValueError("--oracle-only cannot be used with non-oracle models.")
        opt_main = torch.optim.AdamW(model.parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay)
        opt_oracle = None

    base_lr_main = [float(pg["lr"]) for pg in opt_main.param_groups] if opt_main is not None else None
    base_lr_oracle = [float(pg["lr"]) for pg in opt_oracle.param_groups] if opt_oracle is not None else None

    use_amp = bool(exp.train.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(str(exp.train.amp_dtype))
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[train] amp_dtype=bfloat16 not supported on this GPU; using float16")
        amp_dtype = torch.float16
    scaler_main = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_oracle = torch.amp.GradScaler("cuda", enabled=use_amp and has_oracle_head)
    if use_amp:
        print(f"[train] AMP enabled (dtype={amp_dtype})")

    global_step = 0
    max_steps = exp.train.max_steps or (exp.train.epochs * len(train_dl))
    max_steps = max(int(max_steps), 1)
    if exp.train.warmup_steps > 0 or (exp.train.lr_schedule or "").lower() not in ("none", "", "constant"):
        print(
            f"[train] lr warmup_steps={exp.train.warmup_steps} schedule={exp.train.lr_schedule!r} "
            f"lr_min_ratio={exp.train.lr_min_ratio} max_steps={max_steps}"
        )

    model.train()
    start_time = time.time()
    best_metric_name = _pick_best_metric_name(exp.train, has_val=val_dl is not None, val_metrics=None)
    best_metric_mode = _pick_metric_mode(best_metric_name, exp.train)
    best_metric_value: float | None = None
    last_val_metrics: dict[str, float] | None = None
    if bool(getattr(exp.train, "save_best_only", True)):
        print(
            f"[train] checkpoint policy: best-only "
            f"(metric={best_metric_name!r}, mode={best_metric_mode})"
        )
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
            lr_mult = _lr_multiplier(
                global_step,
                warmup_steps=int(exp.train.warmup_steps),
                max_steps=max_steps,
                lr_schedule=str(exp.train.lr_schedule),
                lr_min_ratio=float(exp.train.lr_min_ratio),
            )
            if opt_main is not None and base_lr_main is not None:
                _apply_lr_multiplier(opt_main, base_lr_main, lr_mult)
            if opt_oracle is not None and base_lr_oracle is not None:
                _apply_lr_multiplier(opt_oracle, base_lr_oracle, lr_mult)

            x_tokens = batch.x_tokens.to(device)
            y = batch.y.to(device)
            y_mask = batch.y_mask.to(device) if batch.y_mask is not None else None
            y_weight = batch.y_weight.to(device) if batch.y_weight is not None else None
            batch_on_device = type(batch)(x_tokens=x_tokens, y=y, y_mask=y_mask, y_weight=y_weight)

            state = None
            prev_aux = None
            total_main_f = 0.0
            total_halt_f = 0.0
            halt_loss_used = False
            loss_sum_f = 0.0
            total_oracle = torch.zeros((), device=device)
            last_metrics: dict[str, float] = {}
            used_sup = 0
            aux_hist: list[torch.Tensor] = []
            per_step_psloss: list[torch.Tensor] = []
            dump_every = max(int(exp.train.dump_oracle_trace_every), 1)
            dump_cap = exp.train.dump_oracle_trace_max_batches
            _tr = oracle_trace_state or {}
            _nrec = int(_tr.get("batches_recorded", 0))
            should_dump_trace = (
                oracle_trace_dir is not None
                and oracle_trace_state is not None
                and (global_step % dump_every == 0)
                and (dump_cap is None or _nrec < int(dump_cap))
            )
            schedule_rows: list[dict] = []
            logits_trace: list[torch.Tensor] | None = (
                [] if should_dump_trace and exp.train.dump_oracle_trace_include_logits else None
            )
            state_y_trace: list[torch.Tensor] | None = (
                [] if should_dump_trace and exp.train.dump_oracle_trace_include_state else None
            )
            state_z_trace: list[torch.Tensor] | None = (
                [] if should_dump_trace and exp.train.dump_oracle_trace_include_state else None
            )
            halt_trace: list[torch.Tensor | None] = [] if should_dump_trace else []

            # One backward per supervision step so the autograd graph depth is O(1) instead of O(N_sup)
            # (avoids holding activations for all refinement steps until a single backward).
            if opt_main is not None:
                opt_main.zero_grad(set_to_none=True)
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
                with torch.set_grad_enabled(not oracle_only):
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(
                            x_tokens,
                            state=state,
                            recursion_n=sch.recursion_n,
                            recursion_T=sch.recursion_T,
                        )

                        main_loss = task.compute_loss(out.logits, batch_on_device)
                        loss = main_loss * float(sch.supervision_weight)

                        # optional halting head supervision (BCE on correctness)
                        # Use logits + BCEWithLogitsLoss so AMP/autocast is safe (plain BCE on probs is not).
                        if bool(getattr(exp.train, "use_halt_loss", True)):
                            if "halt_logit" in out.loss_parts:
                                targets = task.halt_targets(out.logits, batch_on_device)
                                if targets is not None:
                                    halt_loss = F.binary_cross_entropy_with_logits(
                                        out.loss_parts["halt_logit"], targets
                                    )
                                    loss = loss + exp.train.beta_halt * halt_loss * float(sch.supervision_weight)
                                    total_halt_f += float(halt_loss.detach().item())
                                    halt_loss_used = True
                            elif "halt_prob" in out.loss_parts:
                                targets = task.halt_targets(out.logits, batch_on_device)
                                if targets is not None:
                                    q = out.loss_parts["halt_prob"]
                                    halt_loss = F.binary_cross_entropy(q, targets)
                                    loss = loss + exp.train.beta_halt * halt_loss * float(sch.supervision_weight)
                                    total_halt_f += float(halt_loss.detach().item())
                                    halt_loss_used = True

                total_main_f += float(main_loss.detach().item())
                loss_sum_f += float(loss.detach().item())
                last_metrics = task.compute_metrics(out.logits, batch_on_device)

                if opt_main is not None:
                    if use_amp:
                        scaler_main.scale(loss).backward()
                    else:
                        loss.backward()

                prev_aux = oracle_history_tensor_from_output(model, out)
                if prev_aux is not None:
                    aux_hist.append(prev_aux)
                    if should_dump_trace:
                        schedule_rows.append(
                            {
                                "supervision_step": int(sup_step),
                                "recursion_n": int(sch.recursion_n),
                                "recursion_T": int(sch.recursion_T),
                                "supervision_weight": float(sch.supervision_weight),
                                "halt_threshold": None
                                if sch.halt_threshold is None
                                else float(sch.halt_threshold),
                            }
                        )
                        # Keep heavy tensors on CPU during the multi-step loop (oracle still uses GPU aux_hist).
                        if logits_trace is not None:
                            logits_trace.append(out.logits.detach().cpu())
                        if state_y_trace is not None and state_z_trace is not None:
                            st_pair = out.state
                            if isinstance(st_pair, tuple) and len(st_pair) == 2:
                                state_y_trace.append(st_pair[0].detach().cpu())
                                state_z_trace.append(st_pair[1].detach().cpu())
                        hp = out.loss_parts.get("halt_prob")
                        halt_trace.append(hp.detach().cpu() if isinstance(hp, torch.Tensor) else None)
                if y.dim() == 2 and out.logits.dim() == 3:
                    per_step_psloss.append(_per_sample_token_ce(out.logits.detach(), y, y_mask, y_weight))
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

            if opt_main is not None:
                if use_amp:
                    scaler_main.step(opt_main)
                    scaler_main.update()
                else:
                    opt_main.step()

            # Oracle-step training: full rollout already done, now train
            # prefix-conditioned oracle on future deltas.
            if has_oracle_head and aux_hist and per_step_psloss:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    aux_history = torch.stack(aux_hist, dim=1)
                    per_step = torch.stack(per_step_psloss, dim=0)
                    oracle_loss = model.oracle_loss_from_rollout(
                        aux_history.detach(),
                        per_step.detach(),
                    )
                    weighted_oracle = oracle_loss * float(getattr(model.cfg_oracle, "oracle_loss_weight", 1.0))

                opt_oracle.zero_grad(set_to_none=True)
                if use_amp:
                    scaler_oracle.scale(weighted_oracle).backward()
                    scaler_oracle.step(opt_oracle)
                    scaler_oracle.update()
                else:
                    weighted_oracle.backward()
                    opt_oracle.step()
                total_oracle = total_oracle + weighted_oracle.detach()

            if should_dump_trace and aux_hist and per_step_psloss:
                if len(aux_hist) != len(per_step_psloss):
                    print(
                        f"[train] oracle trace skip step={global_step}: "
                        f"len(aux)={len(aux_hist)} != len(per_step_ce)={len(per_step_psloss)}"
                    )
                elif len(schedule_rows) != len(aux_hist):
                    print(
                        f"[train] oracle trace skip step={global_step}: "
                        f"len(schedule)={len(schedule_rows)} != len(aux)={len(aux_hist)}"
                    )
                else:
                    sy_t = state_y_trace
                    sz_t = state_z_trace
                    if sy_t is not None and sz_t is not None:
                        if len(sy_t) != len(aux_hist) or len(sz_t) != len(aux_hist):
                            sy_t, sz_t = None, None
                    oc = asdict(model.cfg_oracle) if hasattr(model, "cfg_oracle") else None
                    rec = build_oracle_trace_batch_payload(
                        epoch=epoch,
                        global_step=global_step,
                        batch_idx=batch_idx,
                        x_tokens=x_tokens,
                        y=y,
                        y_mask=y_mask,
                        y_weight=y_weight,
                        aux_hist=aux_hist,
                        per_step_psloss=per_step_psloss,
                        logits_hist=logits_trace,
                        state_y_hist=sy_t,
                        state_z_hist=sz_t,
                        halt_hist=halt_trace,
                        schedule_rows=schedule_rows,
                        n_sup=n_sup,
                        used_sup=used_sup,
                        config_path=str(config_path),
                        model_name=model.__class__.__name__,
                        model_config=dict(exp.model),
                        oracle_cfg=oc,
                        fp16=bool(exp.train.dump_oracle_trace_fp16),
                        embed_shared_meta=False,
                        embed_notes=False,
                    )
                    ots = oracle_trace_state
                    buf = ots["buffer"]
                    if not isinstance(buf, list):
                        raise TypeError("oracle_trace_state['buffer'] must be a list")
                    buf.append(rec)
                    ots["batches_recorded"] = int(ots["batches_recorded"]) + 1
                    shard_sz = max(int(exp.train.dump_oracle_trace_shard_batches), 1)
                    if len(buf) >= shard_sz:
                        save_oracle_trace_shard(
                            oracle_trace_dir,
                            int(ots["shard_idx"]),
                            buf,
                            model_name=model.__class__.__name__,
                            model_config=dict(exp.model),
                            oracle_cfg=oc,
                            config_path=str(config_path),
                        )
                        ots["shard_idx"] = int(ots["shard_idx"]) + 1
                        buf.clear()

            if global_step % exp.train.log_every == 0:
                elapsed = time.time() - start_time
                rec = {
                    "kind": "train",
                    "time_s": elapsed,
                    "epoch": epoch,
                    "step": global_step,
                    "batch_idx": batch_idx,
                    "loss": float(loss_sum_f),
                    "main_loss": float(total_main_f / max(used_sup, 1)),
                    "used_sup": used_sup,
                    "lr": float(
                        opt_main.param_groups[0]["lr"]
                        if opt_main is not None
                        else opt_oracle.param_groups[0]["lr"]
                    ),
                    "lr_mult": float(lr_mult),
                    **last_metrics,
                }
                if halt_loss_used:
                    rec["halt_loss"] = float(total_halt_f / max(used_sup, 1))
                if has_oracle_head:
                    rec["oracle_loss"] = float(total_oracle.detach().item() / max(used_sup, 1))
                logger.log(rec)
                if exp.train.progress_bar:
                    postfix = {
                        "loss": f"{rec['loss']:.4f}",
                        "main": f"{rec['main_loss']:.4f}",
                        "lr": f"{rec['lr']:.2e}",
                        "sup": rec["used_sup"],
                        "step": global_step,
                    }
                    if "halt_loss" in rec:
                        postfix["halt"] = f"{rec['halt_loss']:.4f}"
                    if has_oracle_head:
                        postfix["ora"] = f"{rec['oracle_loss']:.4f}"
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
                val = _validate_loop(
                    model,
                    task,
                    val_dl,
                    device=device,
                    n_sup=n_sup,
                    scheduler=scheduler,
                    max_epochs=exp.train.epochs,
                    max_steps=max_steps,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                logger.log({"kind": "val", "epoch": epoch, "step": global_step, **val})
                last_val_metrics = dict(val)
                model.train()
            if bool(getattr(exp.train, "save_best_only", True)):
                metric_value: float | None = None
                if best_metric_name.startswith("val_"):
                    key = best_metric_name[len("val_") :]
                    if last_val_metrics is not None and key in last_val_metrics:
                        metric_value = float(last_val_metrics[key])
                elif best_metric_name in ("train_loss", "loss"):
                    metric_value = float(loss_sum_f)
                elif last_val_metrics is not None and best_metric_name in last_val_metrics:
                    metric_value = float(last_val_metrics[best_metric_name])
                if metric_value is not None:
                    improved = False
                    if best_metric_value is None:
                        improved = True
                    elif best_metric_mode == "min":
                        improved = metric_value < best_metric_value
                    else:
                        improved = metric_value > best_metric_value
                    if improved:
                        best_metric_value = metric_value
                        ckpt_path = ckpt_dir / "best.pt"
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "epoch": epoch,
                                "step": global_step,
                                "config_path": str(config_path),
                                "best_metric_name": best_metric_name,
                                "best_metric_value": best_metric_value,
                            },
                            ckpt_path,
                        )
                        print(
                            f"[train] new best checkpoint: step={global_step} "
                            f"{best_metric_name}={best_metric_value:.6f} -> {ckpt_path.name}"
                        )
            elif global_step % exp.train.ckpt_every == 0:
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

    if bool(getattr(exp.train, "save_best_only", True)) and best_metric_value is None:
        ckpt_path = ckpt_dir / "best.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch if "epoch" in locals() else 0,
                "step": global_step,
                "config_path": str(config_path),
                "best_metric_name": "fallback_final",
                "best_metric_value": None,
            },
            ckpt_path,
        )
        print(f"[train] no best metric observed; saved fallback checkpoint -> {ckpt_path.name}")

    if oracle_trace_state is not None and oracle_trace_dir is not None:
        buf_end = oracle_trace_state["buffer"]
        if isinstance(buf_end, list) and len(buf_end) > 0:
            oc_flush = asdict(model.cfg_oracle) if hasattr(model, "cfg_oracle") else None
            save_oracle_trace_shard(
                oracle_trace_dir,
                int(oracle_trace_state["shard_idx"]),
                buf_end,
                model_name=model.__class__.__name__,
                model_config=dict(exp.model),
                oracle_cfg=oc_flush,
                config_path=str(config_path),
            )
            buf_end.clear()

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
    *,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    progress_bar: bool = False,
) -> dict[str, float]:
    model.eval()
    metrics_sum: dict[str, float] = {}
    loss_sum = 0.0
    count = 0
    global_step = 0
    oracle_enabled = oracle_policy in {"greedy", "sampling"} and hasattr(model, "choose_delta")
    infer_step_cap = int(oracle_max_steps) if oracle_max_steps is not None else int(n_sup)
    infer_step_cap = max(infer_step_cap, 1)
    val_iter = val_dl
    if progress_bar:
        total = None
        try:
            total = min(len(val_dl), 50)
        except TypeError:
            total = None
        val_iter = tqdm(val_dl, total=total, desc="Validation", leave=False)

    for batch in val_iter:
        global_step += 1
        x_tokens = batch.x_tokens.to(device)
        y = batch.y.to(device)
        y_mask = batch.y_mask.to(device) if batch.y_mask is not None else None
        y_weight = batch.y_weight.to(device) if batch.y_weight is not None else None
        batch_on_device = type(batch)(x_tokens=x_tokens, y=y, y_mask=y_mask, y_weight=y_weight)

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
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
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
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(x_tokens, state=state, recursion_n=sch.recursion_n, recursion_T=sch.recursion_T)
                logits = out.logits
                prev_aux = out.aux_tensor
                state = out.state
                sup_step += 1
                h_t = oracle_history_tensor_from_output(model, out)
                if h_t is not None:
                    aux_hist.append(h_t)

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
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out2 = model(x_tokens, state=state, recursion_n=sch2.recursion_n, recursion_T=sch2.recursion_T)
                    logits = out2.logits
                    prev_aux = out2.aux_tensor
                    state = out2.state
                    sup_step += 1
                    h2 = oracle_history_tensor_from_output(model, out2)
                    if h2 is not None:
                        aux_hist.append(h2)
                    if sch2.halt_threshold is not None and "halt_prob" in out2.loss_parts:
                        q2 = out2.loss_parts["halt_prob"]
                        if bool((q2 > sch2.halt_threshold).all().item()):
                            halted = True
                            break
                if halted:
                    break

        assert logits is not None
        loss = task.compute_loss(logits, batch_on_device)
        loss_sum += float(loss.detach().item())
        m = task.compute_metrics(logits, batch_on_device)
        for k, v in m.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v)
        count += 1
        if count >= 50:
            # keep validation cheap by default
            break

    if count == 0:
        return {}
    out = {k: v / count for k, v in metrics_sum.items()}
    out["val_loss"] = loss_sum / count
    return out

