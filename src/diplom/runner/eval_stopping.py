from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from diplom.models.trm_oracle import oracle_history_tensor_from_output
from diplom.runner.config import load_experiment_config
from diplom.runner.factory import build_model, build_scheduler, build_task, resolve_device
from diplom.runner.stopping import apply_stopping_strategy
from diplom.runner.train import _per_sample_token_acc, _per_sample_token_ce, _resolve_amp_dtype
from diplom.schedulers.base import SchedulerState


def _parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _parse_csv_float(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _expected_calibration_error(conf: torch.Tensor, is_correct: torch.Tensor, n_bins: int = 10) -> float:
    ece = torch.zeros((), device=conf.device)
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not bool(m.any().item()):
            continue
        acc = is_correct[m].float().mean()
        c = conf[m].mean()
        w = m.float().mean()
        ece = ece + w * torch.abs(acc - c)
    return float(ece.item())


def _record_key(r: dict) -> tuple:
    return tuple((kk, r.get(kk)) for kk in ("distribution_model", "strategy", "threshold", "budget", "selected_threshold"))


def _aggregate_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for r in records:
        k = _record_key(r)
        g = grouped.setdefault(k, {"count": 0})
        g["count"] += 1
        for kk, vv in r.items():
            if kk in ("distribution_model", "strategy", "threshold", "budget", "selected_threshold"):
                g[kk] = vv
                continue
            g[kk] = g.get(kk, 0.0) + float(vv)
    out = []
    for g in grouped.values():
        n = max(int(g.pop("count")), 1)
        agg = {}
        for kk, vv in g.items():
            if kk in ("distribution_model", "strategy", "threshold", "budget", "selected_threshold"):
                agg[kk] = vv
            else:
                agg[kk] = float(vv) / float(n)
        out.append(agg)
    return out


def _apply_budget_constraint(records: list[dict], eps: float = 1e-9) -> list[dict]:
    out: list[dict] = []
    for r in records:
        if str(r.get("strategy", "")).lower() != "budget":
            out.append(r)
            continue
        b = r.get("budget", None)
        ms = r.get("mean_steps", None)
        if b is None or ms is None:
            continue
        if float(ms) <= float(b) + float(eps):
            out.append(r)
    return out


def _pick_best_record(records: list[dict], metric: str, mode: str) -> dict | None:
    vals = []
    for r in records:
        v = r.get(metric)
        if v is None:
            continue
        vals.append((r, float(v)))
    if not vals:
        return None
    if str(mode).lower() == "min":
        return min(vals, key=lambda x: x[1])[0]
    return max(vals, key=lambda x: x[1])[0]


def _choose_answer_step(stop_step: int, pmf_at_stop: torch.Tensor, policy: str) -> int:
    s = max(int(stop_step), 1)
    pol = str(policy).strip().lower()
    if pol == "last":
        return s
    if pol in ("argmax_interval", "argmax_on_interval"):
        end = min(s, int(pmf_at_stop.numel()))
        if end <= 0:
            return s
        idx = int(torch.argmax(pmf_at_stop[:end]).item())
        return idx + 1
    raise ValueError(f"Unknown answer policy: {policy}")


def _micro_pooled_token_acc_for_answer_policy(
    *,
    stop_steps: list[int],
    pmf_per_k: list[torch.Tensor],
    logits_hist: list[torch.Tensor],
    batch_on_device,
    answer_policy: str,
) -> float:
    """
    Batch-level token_acc with the same pooling semantics as ``task.compute_metrics(..., full_batch)``:
    sum of correct masked tokens / sum of mask counts, allowing a different answer step per sample.
    Aligns strategy columns ``token_acc_*`` with ``last_step_metrics['token_acc']`` (per-batch convention).
    """
    pol = str(answer_policy).strip().lower()
    B = len(stop_steps)
    correct = 0
    denom = 0
    for b, s in enumerate(stop_steps):
        pmf_at_stop = pmf_per_k[s - 1][b]
        ans_step = _choose_answer_step(s, pmf_at_stop, pol)
        ans_step = max(1, min(ans_step, len(logits_hist)))
        logits = logits_hist[ans_step - 1][b : b + 1]
        y = batch_on_device.y[b : b + 1]
        pred = torch.argmax(logits, dim=-1)
        if batch_on_device.y_mask is not None:
            m = batch_on_device.y_mask[b : b + 1].bool()
            correct += int((pred.eq(y) & m).sum().item())
            denom += int(m.sum().item())
        else:
            correct += int(pred.eq(y).sum().item())
            denom += int(y.numel())
    return float(correct) / float(max(denom, 1))


def _metrics_for_stop_steps(
    *,
    stop_steps: list[int],
    pmf_per_k: list[torch.Tensor],
    logits_hist: list[torch.Tensor],
    batch_on_device,
    task,
    answer_policies: list[str],
) -> dict[str, float]:
    sums_by_policy: dict[str, dict[str, float]] = {p: {} for p in answer_policies}
    B = len(stop_steps)
    for b, s in enumerate(stop_steps):
        pmf_at_stop = pmf_per_k[s - 1][b]
        for pol in answer_policies:
            ans_step = _choose_answer_step(s, pmf_at_stop, pol)
            ans_step = max(1, min(ans_step, len(logits_hist)))
            m = task.compute_metrics(
                logits_hist[ans_step - 1][b : b + 1],
                type(batch_on_device)(
                    x_tokens=batch_on_device.x_tokens[b : b + 1],
                    y=batch_on_device.y[b : b + 1],
                    y_mask=None if batch_on_device.y_mask is None else batch_on_device.y_mask[b : b + 1],
                    y_weight=None if batch_on_device.y_weight is None else batch_on_device.y_weight[b : b + 1],
                ),
            )
            ps = sums_by_policy[pol]
            for mk, mv in m.items():
                if mk == "token_acc":
                    # Pooled below so ``token_acc_*`` matches ``last_token_acc`` batch semantics.
                    continue
                ps[mk] = ps.get(mk, 0.0) + float(mv)

    out: dict[str, float] = {}
    for pol in answer_policies:
        micro_tok = _micro_pooled_token_acc_for_answer_policy(
            stop_steps=stop_steps,
            pmf_per_k=pmf_per_k,
            logits_hist=logits_hist,
            batch_on_device=batch_on_device,
            answer_policy=pol,
        )
        sums_by_policy[pol]["token_acc"] = micro_tok

    for pol, ps in sums_by_policy.items():
        for mk, mv in ps.items():
            if mk == "token_acc":
                out[f"{mk}_{pol}"] = float(mv)
            else:
                out[f"{mk}_{pol}"] = mv / max(B, 1)
    if answer_policies:
        primary = answer_policies[0]
        for k, v in list(out.items()):
            suffix = f"_{primary}"
            if k.endswith(suffix):
                out[k[: -len(suffix)]] = v
    return out


def eval_stopping_from_yaml(
    config_path: str,
    checkpoint_path: str | None,
    distribution_models: str,
    strategies: str,
    threshold_grid: str,
    budget_grid: str,
    max_steps: int | None,
    out_path: str | None,
    progress_bar: bool | None = None,
    honest_split_ratio: float = 0.0,
    selection_metric: str = "token_acc",
    selection_mode: str = "max",
    answer_policies: str = "last,argmax_interval",
) -> dict:
    exp = load_experiment_config(config_path)
    device = resolve_device(exp.train.device)
    model = build_model(exp.model).to(device)
    task = build_task(exp.task)
    scheduler = build_scheduler(exp.scheduler)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)

    _, val_dl = task.build_dataloaders(batch_size=exp.train.batch_size)
    if val_dl is None:
        raise SystemExit("Validation dataloader is required for stopping evaluation.")

    n_sup = int(max_steps or exp.model.get("N_sup", 16))
    dist_models = _parse_csv_str(distribution_models)
    strategy_names = _parse_csv_str(strategies)
    answer_policy_names = _parse_csv_str(answer_policies)
    if not answer_policy_names:
        answer_policy_names = ["last"]
    threshold_vals = _parse_csv_float(threshold_grid)
    budget_vals = _parse_csv_float(budget_grid)
    use_amp = bool(exp.train.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(str(exp.train.amp_dtype))
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16

    model.eval()
    records: list[dict] = []
    records_calibration: list[dict] = []
    records_evaluation: list[dict] = []
    last_step_sum: dict[str, float] = {}
    last_step_loss_sum = 0.0
    last_step_count = 0
    last_step_regret_sum = 0.0
    per_step_token_acc_sum = [0.0] * n_sup
    split_ratio = float(max(0.0, min(1.0, honest_split_ratio)))
    split_enabled = 0.0 < split_ratio < 1.0
    use_progress = bool(exp.train.progress_bar) if progress_bar is None else bool(progress_bar)
    val_iter = enumerate(val_dl)
    split_cut = None
    if split_enabled:
        try:
            total_known = min(len(val_dl), 50)
            split_cut = max(1, min(int(total_known * split_ratio), max(total_known - 1, 1)))
        except TypeError:
            split_cut = None
    if use_progress:
        total = None
        try:
            total = min(len(val_dl), 50)
        except TypeError:
            total = None
        val_iter = enumerate(tqdm(val_dl, total=total, desc="Eval stopping", leave=False))

    for batch_idx, batch in val_iter:
        split_tag: str | None = None
        if split_enabled:
            if split_cut is not None:
                split_tag = "calibration" if batch_idx < split_cut else "evaluation"
            else:
                split_tag = "calibration" if (batch_idx % 10) < int(split_ratio * 10) else "evaluation"
        x_tokens = batch.x_tokens.to(device)
        y = batch.y.to(device)
        y_mask = batch.y_mask.to(device) if batch.y_mask is not None else None
        y_weight = batch.y_weight.to(device) if batch.y_weight is not None else None
        batch_on_device = type(batch)(x_tokens=x_tokens, y=y, y_mask=y_mask, y_weight=y_weight)

        state = None
        prev_aux = None
        aux_hist: list[torch.Tensor] = []
        logits_hist: list[torch.Tensor] = []
        ce_hist: list[torch.Tensor] = []
        acc_hist: list[torch.Tensor] = []
        with torch.no_grad():
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
                logits_hist.append(out.logits.detach())
                h_t = oracle_history_tensor_from_output(model, out)
                if h_t is not None:
                    aux_hist.append(h_t)
                ce_hist.append(_per_sample_token_ce(out.logits.detach(), y, y_mask, y_weight))
                acc_hist.append(_per_sample_token_acc(out.logits.detach(), y, y_mask, y_weight))
                prev_aux = out.aux_tensor
                state = out.state

        if not logits_hist:
            continue
        for si, logits_t in enumerate(logits_hist):
            step_metrics = task.compute_metrics(logits_t, batch_on_device)
            if "token_acc" in step_metrics:
                per_step_token_acc_sum[si] += float(step_metrics["token_acc"])
        last_logits = logits_hist[-1]
        last_loss = task.compute_loss(last_logits, batch_on_device)
        last_step_loss_sum += float(last_loss.detach().item())
        last_metrics = task.compute_metrics(last_logits, batch_on_device)
        for mk, mv in last_metrics.items():
            last_step_sum[mk] = last_step_sum.get(mk, 0.0) + float(mv)
        last_step_count += 1
        aux_stack = torch.stack(aux_hist, dim=1)  # [B,T,D]
        ce_stack = torch.stack(ce_hist, dim=0)  # [T,B]
        acc_stack = torch.stack(acc_hist, dim=0)  # [T,B]
        T = ce_stack.size(0)
        tau_star = torch.argmax(acc_stack, dim=0) + 1  # [B], 1..T — best masked token accuracy step
        tau_idx_opt = (tau_star - 1).clamp_min(0)
        ce_opt = ce_stack[tau_idx_opt, torch.arange(ce_stack.size(1), device=device)]
        reg_last = ce_stack[T - 1] - ce_opt
        last_step_regret_sum += float(reg_last.mean().item())

        for dist_model in dist_models:
            pmf_per_k: list[torch.Tensor] = []
            nll_per_k: list[torch.Tensor] = []
            brier_per_k: list[torch.Tensor] = []
            conf_per_k: list[torch.Tensor] = []
            corr_per_k: list[torch.Tensor] = []
            for k in range(1, T + 1):
                prefix = aux_stack[:, :k, :]
                pmf, _ = model.oracle_distribution(prefix, valid_horizon=T, distribution_model=dist_model)
                pmf_per_k.append(pmf.detach())
                tau_idx = (tau_star - 1).clamp_min(0)
                p_true = pmf[torch.arange(pmf.size(0), device=device), tau_idx].clamp_min(1e-8)
                nll_per_k.append(-torch.log(p_true))
                onehot = torch.zeros_like(pmf)
                onehot.scatter_(1, tau_idx[:, None], 1.0)
                brier_per_k.append(((pmf - onehot) ** 2).sum(dim=-1))
                conf, arg = torch.max(pmf, dim=-1)
                conf_per_k.append(conf)
                corr_per_k.append((arg == tau_idx).float())

            # evaluate non-budget strategies over threshold grid
            for strategy in strategy_names:
                strategy_l = strategy.lower()
                if strategy_l == "budget":
                    for budget in budget_vals:
                        for th in threshold_vals:
                            stop_steps = []
                            for b in range(x_tokens.size(0)):
                                sample_pmfs = [pmf_per_k[k][b] for k in range(T)]
                                s = apply_stopping_strategy("cumulative_probability", sample_pmfs, threshold=th)
                                stop_steps.append(s)
                            stop_tensor = torch.tensor(stop_steps, device=device)
                            idx_batch = torch.arange(stop_tensor.numel(), device=device)
                            regret = ce_stack[(stop_tensor - 1).clamp_min(0), idx_batch] - ce_opt
                            metric_acc = _metrics_for_stop_steps(
                                stop_steps=stop_steps,
                                pmf_per_k=pmf_per_k,
                                logits_hist=logits_hist,
                                batch_on_device=batch_on_device,
                                task=task,
                                answer_policies=answer_policy_names,
                            )
                            rec = {
                                "distribution_model": dist_model,
                                "strategy": "budget",
                                "budget": budget,
                                "selected_threshold": th,
                                "mean_steps": float(stop_tensor.float().mean().item()),
                                "mean_regret": float(regret.mean().item()),
                                "nll": float(torch.stack(nll_per_k).mean().item()),
                                "brier": float(torch.stack(brier_per_k).mean().item()),
                                "ece": _expected_calibration_error(torch.cat(conf_per_k), torch.cat(corr_per_k)),
                            }
                            for mk, mv in metric_acc.items():
                                rec[mk] = mv
                            records.append(rec)
                            if split_tag == "calibration":
                                records_calibration.append(rec)
                            elif split_tag == "evaluation":
                                records_evaluation.append(rec)
                    continue

                for th in threshold_vals:
                    stop_steps = []
                    for b in range(x_tokens.size(0)):
                        sample_pmfs = [pmf_per_k[k][b] for k in range(T)]
                        s = apply_stopping_strategy(strategy_l, sample_pmfs, threshold=th)
                        stop_steps.append(s)
                    metric_acc = _metrics_for_stop_steps(
                        stop_steps=stop_steps,
                        pmf_per_k=pmf_per_k,
                        logits_hist=logits_hist,
                        batch_on_device=batch_on_device,
                        task=task,
                        answer_policies=answer_policy_names,
                    )
                    stop_tensor = torch.tensor(stop_steps, device=device)
                    idx_batch = torch.arange(stop_tensor.numel(), device=device)
                    regret = ce_stack[(stop_tensor - 1).clamp_min(0), idx_batch] - ce_opt
                    rec = {
                        "distribution_model": dist_model,
                        "strategy": strategy_l,
                        "threshold": th,
                        "mean_steps": float(stop_tensor.float().mean().item()),
                        "mean_regret": float(regret.mean().item()),
                        "nll": float(torch.stack(nll_per_k).mean().item()),
                        "brier": float(torch.stack(brier_per_k).mean().item()),
                        "ece": _expected_calibration_error(torch.cat(conf_per_k), torch.cat(corr_per_k)),
                    }
                    for mk, mv in metric_acc.items():
                        rec[mk] = mv
                    records.append(rec)
                    if split_tag == "calibration":
                        records_calibration.append(rec)
                    elif split_tag == "evaluation":
                        records_evaluation.append(rec)

        if batch_idx >= 49:
            break

    if not records:
        return {"records": []}
    out = _apply_budget_constraint(_aggregate_records(records))
    honest_selection: list[dict] = []
    if split_enabled and records_calibration and records_evaluation:
        calib_agg = _apply_budget_constraint(_aggregate_records(records_calibration))
        eval_agg = _apply_budget_constraint(_aggregate_records(records_evaluation))
        eval_by_key = {_record_key(r): r for r in eval_agg}
        for dist_model in dist_models:
            calib_rows = [r for r in calib_agg if str(r.get("distribution_model", "")).lower() == dist_model.lower()]
            chosen = _pick_best_record(calib_rows, selection_metric, selection_mode)
            if chosen is None:
                continue
            eval_row = eval_by_key.get(_record_key(chosen))
            if eval_row is None:
                continue
            item = dict(eval_row)
            item["selection_metric"] = selection_metric
            item["selection_mode"] = selection_mode
            item["selection_split"] = "calibration"
            item["evaluation_split"] = "evaluation"
            item["selection_score_calibration"] = float(chosen.get(selection_metric))
            honest_selection.append(item)
    last_step_metrics: dict[str, float] = {}
    if last_step_count > 0:
        last_step_metrics = {k: v / float(last_step_count) for k, v in last_step_sum.items()}
        last_step_metrics["val_loss"] = last_step_loss_sum / float(last_step_count)
        last_step_metrics["mean_steps"] = float(n_sup)
        last_step_metrics["mean_regret"] = last_step_regret_sum / float(last_step_count)
    per_step_token_acc: list[float] = []
    if last_step_count > 0:
        per_step_token_acc = [per_step_token_acc_sum[i] / float(last_step_count) for i in range(n_sup)]
    result = {
        "records": out,
        "last_step_metrics": last_step_metrics,
        "per_step_token_acc": per_step_token_acc,
        "honest_split_ratio": split_ratio,
        "honest_selection": honest_selection,
    }
    if out_path:
        p = Path(out_path)
    else:
        p = Path(exp.train.run_dir) / "stopping_eval.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
