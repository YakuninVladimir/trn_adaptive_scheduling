#!/usr/bin/env python3
"""Build tex/tables/arc_stopping_base.csv and arc_stopping_finetune_compare.csv from summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
import importlib.util

_spec = importlib.util.spec_from_file_location("summarize_stopping_eval", _SCRIPTS / "summarize_stopping_eval.py")
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_augment_policy_metrics = _mod._augment_policy_metrics
_best_record = _mod._best_record


def _strategy_short(s: Any) -> str:
    k = str(s).strip()
    return {
        "cumulative_probability": "cum.",
        "future_improvement": "fut.",
        "hazard": "haz.",
        "quantile": "qtl.",
        "budget": "bud.",
    }.get(k, k)


def _policy_label(strategy: str, threshold: str, budget: str) -> str:
    s = (strategy or "").strip() or "?"
    th = str(threshold).strip()
    b = str(budget).strip()
    short = _strategy_short(s)
    if b and b not in ("", "nan", "None"):
        return f"{short}/{b}"
    if th and th not in ("", "nan", "None"):
        return f"{short}/{th}"
    return short


def _best_from_eval_json(path: Path, dist: str, metric: str, mode: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", []) or []
    filtered = [_augment_policy_metrics(r) for r in records if str(r.get("distribution_model", "")).strip() == dist]
    return _best_record(filtered, metric, mode)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-summary",
        type=Path,
        default=Path("runs/oracle_sweep_arc_agi/stopping_summary.csv"),
    )
    p.add_argument(
        "--finetune-summary",
        type=Path,
        default=Path("runs/oracle_sweep_arc_agi/oracle_finetune/stopping_summary.csv"),
    )
    p.add_argument("--tables-dir", type=Path, default=Path("tex/tables"))
    p.add_argument(
        "--legacy-ft-csv",
        type=Path,
        default=Path("tex/tables/arc_stopping_finetune_compare.csv"),
        help="Fallback for cells when JSON is missing (partial update).",
    )
    args = p.parse_args()

    base = pd.read_csv(args.base_summary)
    ft = pd.read_csv(args.finetune_summary)
    tok_col = "best_token_acc_best_of_policies"
    if tok_col not in ft.columns:
        tok_col = "best_token_acc"
    base_tok_col = tok_col if tok_col in base.columns else "best_token_acc"
    metric = "token_acc_best_of_policies"

    args.tables_dir.mkdir(parents=True, exist_ok=True)

    base_rows = pd.DataFrame(
        {
            "family": base["trained_distribution_model"],
            "strategy": base["best_strategy"].map(_strategy_short),
            "threshold": pd.to_numeric(base["best_threshold"], errors="coerce"),
            "meantau": pd.to_numeric(base["best_mean_steps"], errors="coerce"),
            "tokbest": pd.to_numeric(base[base_tok_col], errors="coerce"),
            "nlltau": pd.to_numeric(base["best_nll"], errors="coerce"),
        }
    )

    out_base = args.tables_dir / "arc_stopping_base.csv"
    base_rows.to_csv(out_base, index=False)
    print(f"[export-arc] wrote {out_base}")

    legacy: dict[str, dict[str, str]] = {}
    if args.legacy_ft_csv.is_file():
        with args.legacy_ft_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                legacy[row["family"]] = row

    ft_rows: list[dict[str, Any]] = []
    for _, r in ft.iterrows():
        fam = str(r["trained_distribution_model"])
        run_dir = Path(str(r["run_dir"]))
        leg = legacy.get(fam, {})

        base_eval = run_dir / "stopping_eval.json"
        best_b = _best_from_eval_json(base_eval, fam, metric, "max")
        if best_b is None:
            policy_b = "—"
            lsm = pd.to_numeric(r.get("last_step_mean_steps"), errors="coerce")
            tau_b = f"{float(lsm):.2f}" if pd.notna(lsm) and float(lsm) > 0 else (leg.get("taubase", "") or "")
            btl = pd.to_numeric(r.get("best_token_acc_last"), errors="coerce")
            tok_b = f"{float(btl):.3f}" if pd.notna(btl) else leg.get("tokbase", "")
        else:
            policy_b = _policy_label(
                str(best_b.get("strategy", "")),
                str(best_b.get("threshold", "")),
                str(best_b.get("budget", "")),
            )
            tok_b = f'{float(best_b["token_acc_best_of_policies"]):.3f}'
            tau_b = f'{float(best_b["mean_steps"]):.2f}'

        policy_ft = _policy_label(
            str(r.get("best_strategy", "")),
            str(r.get("best_threshold", "")),
            str(r.get("best_budget", "")),
        )
        tok_ft = f"{float(r[tok_col]):.3f}"
        tau_ft = f"{float(r['best_mean_steps']):.2f}"

        ft_rows.append(
            {
                "family": fam,
                "policybase": policy_b,
                "policyft": policy_ft,
                "tokbase": tok_b,
                "tokft": tok_ft,
                "taubase": tau_b,
                "tauft": tau_ft,
            }
        )

    out_ft = args.tables_dir / "arc_stopping_finetune_compare.csv"
    with out_ft.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["family", "policybase", "policyft", "tokbase", "tokft", "taubase", "tauft"],
        )
        w.writeheader()
        w.writerows(ft_rows)
    print(f"[export-arc] wrote {out_ft}")


if __name__ == "__main__":
    main()
