#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import torch

from diplom.runner.config import load_experiment_config
from diplom.runner.factory import build_model, build_scheduler, build_task, resolve_device
from diplom.runner.train import _per_sample_token_acc, _resolve_amp_dtype
from diplom.schedulers.base import SchedulerState

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _batch_pooled_token_acc(task, logits: torch.Tensor, batch_on_device) -> float:
    """Matches diplom-eval-stopping ``last_step_metrics['token_acc']`` aggregation within one batch."""
    m = task.compute_metrics(logits, batch_on_device)
    if "token_acc" not in m:
        raise ValueError(f"task {getattr(task, 'name', task)} has no token_acc in compute_metrics")
    return float(m["token_acc"])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot validation distribution of optimal stopping step tau*."
    )
    p.add_argument("--config", required=True, help="Path to experiment YAML config.")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    p.add_argument(
        "--out",
        required=True,
        help="Output PNG path (three panels: two tau* histograms + token_acc curves).",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Optional output JSON path with counts/probabilities.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override N_sup for rollout length.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=50,
        help="Maximum number of validation batches to evaluate.",
    )
    p.add_argument(
        "--title",
        default="Optimal stop-step distribution (validation)",
        help="Plot title.",
    )
    args = p.parse_args()

    exp = load_experiment_config(args.config)
    device = resolve_device(exp.train.device)
    model = build_model(exp.model).to(device)
    task = build_task(exp.task)
    scheduler = build_scheduler(exp.scheduler)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    _, val_dl = task.build_dataloaders(batch_size=exp.train.batch_size)
    if val_dl is None:
        raise SystemExit("Validation dataloader is required.")

    n_sup = int(args.max_steps or exp.model.get("N_sup", 16))
    n_sup = max(n_sup, 1)
    max_batches = max(int(args.max_batches), 1)

    use_amp = bool(exp.train.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(str(exp.train.amp_dtype))
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16

    counts_sample_tau = torch.zeros(n_sup, dtype=torch.long)
    counts_batch_tau = torch.zeros(n_sup, dtype=torch.long)
    total_samples = 0
    acc_sum_per_step = torch.zeros(n_sup, dtype=torch.float64)
    batch_pooled_sum_per_step = torch.zeros(n_sup, dtype=torch.float64)
    n_batches_used = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dl):
            x_tokens = batch.x_tokens.to(device)
            y = batch.y.to(device)
            y_mask = batch.y_mask.to(device) if batch.y_mask is not None else None
            y_weight = batch.y_weight.to(device) if batch.y_weight is not None else None
            batch_on_device = type(batch)(x_tokens=x_tokens, y=y, y_mask=y_mask, y_weight=y_weight)

            state = None
            prev_aux = None
            acc_hist: list[torch.Tensor] = []
            pooled_acc_hist: list[float] = []
            for sup_step in range(1, n_sup + 1):
                st = SchedulerState(
                    epoch=1,
                    max_epochs=1,
                    global_step=batch_idx + 1,
                    max_steps=max(len(val_dl), 1),
                    supervision_step=sup_step,
                    max_supervision_steps=n_sup,
                    task_name=getattr(task, "name", None),
                )
                sch = scheduler.get_schedule(st, model_aux=prev_aux)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(
                        x_tokens,
                        state=state,
                        recursion_n=sch.recursion_n,
                        recursion_T=sch.recursion_T,
                    )
                logits_det = out.logits.detach()
                acc_hist.append(_per_sample_token_acc(logits_det, y, y_mask, y_weight))
                pooled_acc_hist.append(_batch_pooled_token_acc(task, logits_det, batch_on_device))
                prev_aux = out.aux_tensor
                state = out.state

            acc_stack = torch.stack(acc_hist, dim=0)  # [T,B]
            tau_star = torch.argmax(acc_stack, dim=0) + 1  # [B], 1..T

            c = torch.bincount((tau_star - 1).to(torch.long).cpu(), minlength=n_sup)
            counts_sample_tau += c[:n_sup]
            pooled_t = torch.tensor(pooled_acc_hist, dtype=torch.float64)
            tau_batch = int(torch.argmax(pooled_t).item()) + 1
            counts_batch_tau[tau_batch - 1] += 1
            bsz = int(tau_star.numel())
            total_samples += bsz
            acc_sum_per_step += acc_stack.sum(dim=1).detach().float().cpu().double()
            batch_pooled_sum_per_step += torch.tensor(pooled_acc_hist, dtype=torch.float64)
            n_batches_used += 1

            if batch_idx + 1 >= max_batches:
                break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = torch.arange(1, n_sup + 1).tolist()
    y_sample = counts_sample_tau.tolist()
    y_batch = counts_batch_tau.tolist()
    probs_sample = (counts_sample_tau.float() / max(total_samples, 1)).tolist()
    probs_batch = (counts_batch_tau.float() / max(n_batches_used, 1)).tolist()
    denom = max(total_samples, 1)
    mean_acc_equal_sample = (acc_sum_per_step / float(denom)).tolist()
    nb = max(n_batches_used, 1)
    mean_acc_batch_pooled_macro = (batch_pooled_sum_per_step / float(nb)).tolist()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(args.title)

    ax0.bar(x, probs_sample, width=0.8, color="C0")
    ax0.set_xticks(x)
    ax0.set_xlabel(r"$\tau^{*}$")
    ax0.set_ylabel("Probability")
    ax0.set_title(r"Per-sample $\tau^{*}$\n(argmax per-sample token_acc)")
    ax0.set_ylim(bottom=0.0)

    ax1.bar(x, probs_batch, width=0.8, color="C3")
    ax1.set_xticks(x)
    ax1.set_xlabel(r"$\tau^{*}$")
    ax1.set_ylabel("Probability")
    ax1.set_title(r"Per-batch $\tau^{*}$\n(argmax batch-pooled token_acc)")
    ax1.set_ylim(bottom=0.0)

    ax2.plot(
        x,
        mean_acc_batch_pooled_macro,
        marker="o",
        color="C1",
        label="Batch pooled, macro avg batches (= stopping_eval)",
    )
    ax2.plot(
        x,
        mean_acc_equal_sample,
        linestyle="--",
        marker="x",
        color="C2",
        label="Equal weight per sample",
    )
    ax2.set_xticks(x)
    ax2.set_xlabel("Supervision step")
    ax2.set_ylabel("token_acc")
    ax2.set_title("token_acc vs step")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    plt.savefig(out_path, dpi=140)
    plt.close()

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "max_steps": n_sup,
        "max_batches": max_batches,
        "tau_star_definition_per_sample": (
            "argmax over supervision steps of per-sample masked token_acc "
            "(histogram weighted by number of examples)"
        ),
        "tau_star_definition_per_batch": (
            "argmax over supervision steps of batch-pooled token_acc (one value per batch); "
            "histogram weighted by number of batches"
        ),
        "total_samples": total_samples,
        "n_batches_used": n_batches_used,
        "counts": y_sample,
        "probabilities": probs_sample,
        "counts_tau_per_batch_pooled_trajectory": y_batch,
        "probabilities_tau_per_batch_pooled_trajectory": probs_batch,
        "mean_token_acc_per_step_batch_pooled_macro_avg": mean_acc_batch_pooled_macro,
        "mean_token_acc_per_step_equal_weight_per_sample": mean_acc_equal_sample,
        "_note": (
            "stopping_eval last_token_acc equals batch-pooled token_acc macro-averaged over batches; "
            "last step matches last element of mean_token_acc_per_step_batch_pooled_macro_avg "
            "when max_batches matches eval."
        ),
    }
    if args.out_json:
        json_path = Path(args.out_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
