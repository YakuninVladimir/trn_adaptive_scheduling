#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


WIKITEXT_CFGS = [
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_finite_discrete.yaml",
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_smoothed_loss.yaml",
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_mixture_geometric.yaml",
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_mixture_exponential.yaml",
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_power.yaml",
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_negative_binomial.yaml",
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_lognormal.yaml",
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_hybrid.yaml",
]
ARC_CFGS = [
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_finite_discrete.yaml",
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_smoothed_loss.yaml",
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_mixture_geometric.yaml",
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_mixture_exponential.yaml",
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_power.yaml",
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_negative_binomial.yaml",
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_lognormal.yaml",
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_hybrid.yaml",
]


def _cfgs_for_group(group: str) -> list[str]:
    if group == "wikitext":
        return list(WIKITEXT_CFGS)
    if group == "arc":
        return list(ARC_CFGS)
    if group == "all":
        return list(WIKITEXT_CFGS) + list(ARC_CFGS)
    raise ValueError(f"Unknown group: {group}")


def _default_out_prefix(group: str) -> str:
    if group == "arc":
        return "oracle_sweep_arc_agi"
    if group == "wikitext":
        return "oracle_sweep_wikitext"
    return "oracle_sweep"


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v: float | None, digits: int = 6) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def _group_from_cfg(cfg_path: str) -> str:
    return "arc" if "oracle_sweep_arc_agi" in cfg_path else "wikitext"


def _best_record(records: list[dict[str, Any]], metric: str, mode: str) -> dict[str, Any] | None:
    vals = [r for r in records if _to_float(r.get(metric)) is not None]
    if not vals:
        return None
    # Tie-breaks:
    # 1) better target metric according to mode
    # 2) fewer mean_steps
    # 3) lower mean_regret
    def key_max(r: dict[str, Any]) -> tuple[float, float, float]:
        m = _to_float(r.get(metric))
        s = _to_float(r.get("mean_steps"))
        reg = _to_float(r.get("mean_regret"))
        return (
            float(m) if m is not None else float("-inf"),
            -(float(s) if s is not None else float("inf")),
            -(float(reg) if reg is not None else float("inf")),
        )

    def key_min(r: dict[str, Any]) -> tuple[float, float, float]:
        m = _to_float(r.get(metric))
        s = _to_float(r.get("mean_steps"))
        reg = _to_float(r.get("mean_regret"))
        return (
            -(float(m) if m is not None else float("inf")),
            -(float(s) if s is not None else float("inf")),
            -(float(reg) if reg is not None else float("inf")),
        )

    if mode == "max":
        return max(vals, key=key_max)
    return max(vals, key=key_min)


def main() -> None:
    p = argparse.ArgumentParser(description="Build summary table from oracle stopping_eval.json files.")
    p.add_argument("--group", choices=["wikitext", "arc", "all"], default="all")
    p.add_argument("--metric", default="token_acc", help="Metric used to select best strategy row.")
    p.add_argument("--mode", choices=["min", "max"], default="max", help="Optimization mode for --metric.")
    p.add_argument(
        "--use-honest-selection",
        action="store_true",
        help="Use per-distribution rows from stopping_eval.honest_selection when available.",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: runs/oracle_sweep_<group>/stopping_summary.csv, or runs/oracle_sweep_summary.csv for all).",
    )
    p.add_argument(
        "--out-md",
        default=None,
        help="Output Markdown table path (default: runs/oracle_sweep_<group>/stopping_summary.md, or runs/oracle_sweep_summary.md for all).",
    )
    args = p.parse_args()

    cfgs = _cfgs_for_group(args.group)
    rows: list[dict[str, Any]] = []
    for cfg in cfgs:
        cfg_data = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
        model = cfg_data.get("model", {}) or {}
        train = cfg_data.get("train", {}) or {}
        run_dir = str(train.get("run_dir", "")).strip()
        if not run_dir:
            continue
        dist = str(model.get("oracle_distribution_model", "finite_discrete")).strip() or "finite_discrete"
        out_path = Path(run_dir) / "stopping_eval.json"
        if not out_path.exists():
            continue
        data = json.loads(out_path.read_text(encoding="utf-8"))
        records = data.get("records", []) or []
        honest = data.get("honest_selection", []) or []
        filtered = [r for r in records if str(r.get("distribution_model", "")).strip() == dist]
        if args.use_honest_selection:
            honest_filtered = [r for r in honest if str(r.get("distribution_model", "")).strip() == dist]
            best = _best_record(honest_filtered, args.metric, args.mode)
        else:
            best = _best_record(filtered, args.metric, args.mode)
        last = data.get("last_step_metrics", {}) or {}

        row: dict[str, Any] = {
            "group": _group_from_cfg(cfg),
            "config": cfg,
            "run_dir": run_dir,
            "selection_metric": args.metric,
            "selection_mode": args.mode,
            "trained_distribution_model": dist,
            "n_records_all": len(records),
            "n_records_filtered": len(filtered),
            "n_honest_records": len(honest),
            "last_step_val_loss": _to_float(last.get("val_loss")),
            "last_step_token_acc": _to_float(last.get("token_acc")),
            "last_step_exact_acc": _to_float(last.get("exact_acc")),
            "last_step_mean_steps": _to_float(last.get("mean_steps")),
            "best_strategy": "" if best is None else str(best.get("strategy", "")),
            "best_threshold": "" if best is None else best.get("threshold", ""),
            "best_budget": "" if best is None else best.get("budget", ""),
            "best_mean_steps": None if best is None else _to_float(best.get("mean_steps")),
            "best_mean_regret": None if best is None else _to_float(best.get("mean_regret")),
            "best_token_acc": None if best is None else _to_float(best.get("token_acc")),
            "best_exact_acc": None if best is None else _to_float(best.get("exact_acc")),
            "best_nll": None if best is None else _to_float(best.get("nll")),
            "best_brier": None if best is None else _to_float(best.get("brier")),
            "best_ece": None if best is None else _to_float(best.get("ece")),
        }
        rows.append(row)

    if args.out_csv is None:
        out_csv = (
            Path("runs/oracle_sweep_summary.csv")
            if args.group == "all"
            else Path(f"runs/{_default_out_prefix(args.group)}/stopping_summary.csv")
        )
    else:
        out_csv = Path(args.out_csv)
    if args.out_md is None:
        out_md = (
            Path("runs/oracle_sweep_summary.md")
            if args.group == "all"
            else Path(f"runs/{_default_out_prefix(args.group)}/stopping_summary.md")
        )
    else:
        out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "group",
        "selection_metric",
        "selection_mode",
        "trained_distribution_model",
        "run_dir",
        "best_strategy",
        "best_threshold",
        "best_budget",
        "best_mean_steps",
        "best_mean_regret",
        "best_token_acc",
        "best_exact_acc",
        "best_nll",
        "best_brier",
        "best_ece",
        "last_step_mean_steps",
        "last_step_val_loss",
        "last_step_token_acc",
        "last_step_exact_acc",
        "n_records_all",
        "n_records_filtered",
        "n_honest_records",
        "config",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = dict(r)
            for k in (
                "best_mean_steps",
                "best_mean_regret",
                "best_token_acc",
                "best_exact_acc",
                "best_nll",
                "best_brier",
                "best_ece",
                "last_step_mean_steps",
                "last_step_val_loss",
                "last_step_token_acc",
                "last_step_exact_acc",
            ):
                row[k] = _fmt(_to_float(row.get(k)))
            w.writerow(row)

    md_lines = [
        f"<!-- selection_metric={args.metric}, selection_mode={args.mode} -->",
        "| group | dist | run_dir | best_strategy | threshold | budget | best_steps | best_regret | best_token_acc | best_exact_acc | last_steps | last_val_loss | last_token_acc | last_exact_acc |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(r["group"]),
                    str(r["trained_distribution_model"]),
                    str(r["run_dir"]),
                    str(r["best_strategy"]),
                    str(r["best_threshold"]),
                    str(r["best_budget"]),
                    _fmt(_to_float(r["best_mean_steps"]), 4),
                    _fmt(_to_float(r["best_mean_regret"]), 6),
                    _fmt(_to_float(r["best_token_acc"]), 6),
                    _fmt(_to_float(r["best_exact_acc"]), 6),
                    _fmt(_to_float(r["last_step_mean_steps"]), 4),
                    _fmt(_to_float(r["last_step_val_loss"]), 6),
                    _fmt(_to_float(r["last_step_token_acc"]), 6),
                    _fmt(_to_float(r["last_step_exact_acc"]), 6),
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[summary] rows={len(rows)}")
    print(f"[summary] csv={out_csv}")
    print(f"[summary] md={out_md}")


if __name__ == "__main__":
    main()
