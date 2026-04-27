from __future__ import annotations

import json
from pathlib import Path

import torch

from diplom.runner.config import load_experiment_config
from diplom.runner.factory import build_model, build_scheduler, build_task, resolve_device
from diplom.runner.stopping import apply_stopping_strategy
from diplom.runner.train import _per_sample_token_ce, _resolve_amp_dtype
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


def eval_stopping_from_yaml(
    config_path: str,
    checkpoint_path: str | None,
    distribution_models: str,
    strategies: str,
    threshold_grid: str,
    budget_grid: str,
    max_steps: int | None,
    out_path: str | None,
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
    threshold_vals = _parse_csv_float(threshold_grid)
    budget_vals = _parse_csv_float(budget_grid)
    use_amp = bool(exp.train.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(str(exp.train.amp_dtype))
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16

    model.eval()
    records: list[dict] = []
    for batch_idx, batch in enumerate(val_dl):
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
                aux_hist.append(out.aux_tensor.detach())
                ce_hist.append(_per_sample_token_ce(out.logits.detach(), y, y_mask, y_weight))
                prev_aux = out.aux_tensor
                state = out.state

        if not logits_hist:
            continue
        aux_stack = torch.stack(aux_hist, dim=1)  # [B,T,D]
        ce_stack = torch.stack(ce_hist, dim=0)  # [T,B]
        T = ce_stack.size(0)
        idx = torch.arange(1, T + 1, device=device, dtype=ce_stack.dtype)[:, None]
        costs = ce_stack + float(getattr(model, "cfg_oracle", getattr(model, "cfg", None)).oracle_distribution_lambda if hasattr(getattr(model, "cfg_oracle", None), "oracle_distribution_lambda") else 0.0) * idx
        tau_star = torch.argmin(costs, dim=0) + 1  # [B], 1..T

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
                        best_th: float | None = None
                        best_stop = None
                        best_quality = -1e18
                        for th in threshold_vals:
                            stop_steps = []
                            for b in range(x_tokens.size(0)):
                                sample_pmfs = [pmf_per_k[k][b] for k in range(T)]
                                s = apply_stopping_strategy("cumulative_probability", sample_pmfs, threshold=th)
                                stop_steps.append(s)
                            mean_steps = float(torch.tensor(stop_steps, dtype=torch.float32).mean().item())
                            if mean_steps > budget:
                                continue
                            quality = 0.0
                            for b, s in enumerate(stop_steps):
                                m = task.compute_metrics(logits_hist[s - 1][b : b + 1], type(batch_on_device)(
                                    x_tokens=batch_on_device.x_tokens[b : b + 1],
                                    y=batch_on_device.y[b : b + 1],
                                    y_mask=None if batch_on_device.y_mask is None else batch_on_device.y_mask[b : b + 1],
                                    y_weight=None if batch_on_device.y_weight is None else batch_on_device.y_weight[b : b + 1],
                                ))
                                quality += float(sum(m.values()) / max(len(m), 1))
                            if quality > best_quality:
                                best_quality = quality
                                best_th = th
                                best_stop = stop_steps
                        if best_stop is None:
                            continue
                        stop_tensor = torch.tensor(best_stop, device=device)
                        regret = costs[stop_tensor - 1, torch.arange(stop_tensor.numel(), device=device)] - costs.min(dim=0).values
                        rec = {
                            "distribution_model": dist_model,
                            "strategy": "budget",
                            "budget": budget,
                            "selected_threshold": best_th,
                            "mean_steps": float(stop_tensor.float().mean().item()),
                            "mean_regret": float(regret.mean().item()),
                            "nll": float(torch.stack(nll_per_k).mean().item()),
                            "brier": float(torch.stack(brier_per_k).mean().item()),
                            "ece": _expected_calibration_error(torch.cat(conf_per_k), torch.cat(corr_per_k)),
                        }
                        records.append(rec)
                    continue

                for th in threshold_vals:
                    stop_steps = []
                    metric_acc: dict[str, float] = {}
                    for b in range(x_tokens.size(0)):
                        sample_pmfs = [pmf_per_k[k][b] for k in range(T)]
                        s = apply_stopping_strategy(strategy_l, sample_pmfs, threshold=th)
                        stop_steps.append(s)
                        m = task.compute_metrics(logits_hist[s - 1][b : b + 1], type(batch_on_device)(
                            x_tokens=batch_on_device.x_tokens[b : b + 1],
                            y=batch_on_device.y[b : b + 1],
                            y_mask=None if batch_on_device.y_mask is None else batch_on_device.y_mask[b : b + 1],
                            y_weight=None if batch_on_device.y_weight is None else batch_on_device.y_weight[b : b + 1],
                        ))
                        for mk, mv in m.items():
                            metric_acc[mk] = metric_acc.get(mk, 0.0) + float(mv)
                    stop_tensor = torch.tensor(stop_steps, device=device)
                    regret = costs[stop_tensor - 1, torch.arange(stop_tensor.numel(), device=device)] - costs.min(dim=0).values
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
                        rec[mk] = mv / max(len(stop_steps), 1)
                    records.append(rec)

        if batch_idx >= 49:
            break

    if not records:
        return {"records": []}
    # aggregate by keys
    grouped: dict[tuple, dict] = {}
    for r in records:
        k = tuple((kk, r.get(kk)) for kk in ("distribution_model", "strategy", "threshold", "budget", "selected_threshold"))
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
    result = {"records": out}
    if out_path:
        p = Path(out_path)
    else:
        p = Path(exp.train.run_dir) / "stopping_eval.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
