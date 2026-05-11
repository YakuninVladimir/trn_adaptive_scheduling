#!/usr/bin/env python3
"""Build tex/tables/wikitext_lm_base.csv and wikitext_lm_ft_compare.csv from stopping summaries."""

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


def _policy_label(strategy: str, threshold: str, budget: str) -> str:
    s = (strategy or "").strip() or "?"
    th = str(threshold).strip()
    b = str(budget).strip()
    short = {
        "cumulative_probability": "cum.",
        "future_improvement": "fut.",
        "hazard": "haz.",
        "quantile": "qtl.",
        "budget": "bud.",
    }.get(s, s[:4] + ".")
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--finetune-summary",
        type=Path,
        default=Path("runs/oracle_sweep_wikitext_falcon/oracle_finetune/stopping_summary.csv"),
    )
    p.add_argument("--tables-dir", type=Path, default=Path("tex/tables"))
    p.add_argument(
        "--legacy-ft-csv",
        type=Path,
        default=Path("tex/tables/wikitext_lm_ft_compare.csv"),
        help="Previous FT table: base columns kept when run_dir/stopping_eval.json is missing.",
    )
    args = p.parse_args()

    src = pd.read_csv(args.finetune_summary)
    tok_col = "best_token_acc_best_of_policies"
    if tok_col not in src.columns:
        tok_col = "best_token_acc"
    top5_col = "best_top5_acc_best_of_policies"
    if top5_col not in src.columns:
        top5_col = "best_top5_acc"

    base_rows = pd.DataFrame(
        {
            "family": src["trained_distribution_model"],
            "strategy": src["best_strategy"],
            "threshold": src["best_threshold"],
            "meantau": pd.to_numeric(src["best_mean_steps"], errors="coerce"),
            "tokbest": pd.to_numeric(src[tok_col], errors="coerce"),
            "top5best": pd.to_numeric(src.get(top5_col), errors="coerce"),
            "nlltau": pd.to_numeric(src["best_nll"], errors="coerce"),
        }
    )
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    out_base = args.tables_dir / "wikitext_lm_base.csv"
    base_rows.to_csv(out_base, index=False)
    print(f"[export] wrote {out_base}")

    legacy: dict[str, dict[str, str]] = {}
    if args.legacy_ft_csv.is_file():
        with args.legacy_ft_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                legacy[row["family"]] = row

    metric = str(src.get("selection_metric", pd.Series(["token_acc_best_of_policies"])).iloc[0])
    if metric not in ("token_acc_best_of_policies", "token_acc"):
        metric = "token_acc_best_of_policies"

    ft_rows: list[dict[str, Any]] = []
    for _, r in src.iterrows():
        fam = str(r["trained_distribution_model"])
        run_dir = Path(str(r["run_dir"]))
        leg = legacy.get(fam, {})
        base_eval = run_dir / "stopping_eval.json"
        best_b = _best_from_eval_json(base_eval, fam, metric, "max")
        if best_b is None:
            policy_b = leg.get("policybase", "")
            tok_b = leg.get("tokbase", "")
            tau_b = leg.get("taubase", "")
            top5_b = leg.get("top5base", "") or "---"
        else:
            policy_b = _policy_label(
                str(best_b.get("strategy", "")),
                str(best_b.get("threshold", "")),
                str(best_b.get("budget", "")),
            )
            tok_b = f'{float(best_b["token_acc_best_of_policies"]):.3f}'
            tau_b = f'{float(best_b["mean_steps"]):.2f}'
            q5 = best_b.get("top5_acc_best_of_policies")
            top5_b = f"{float(q5):.3f}" if q5 is not None else "---"
        if not str(top5_b).strip() or top5_b.strip() in ("\u2014", "—"):
            top5_b = "---"

        policy_ft = _policy_label(
            str(r.get("best_strategy", "")),
            str(r.get("best_threshold", "")),
            str(r.get("best_budget", "")),
        )
        tok_ft = float(r[tok_col])
        tau_ft = float(r["best_mean_steps"])
        q5_ft = r.get(top5_col)
        top5_ft = float(q5_ft) if pd.notna(q5_ft) else float("nan")
        top5_ft_s = f"{top5_ft:.3f}" if top5_ft == top5_ft else "---"

        ft_rows.append(
            {
                "family": fam,
                "policybase": policy_b,
                "policyft": policy_ft,
                "tokbase": tok_b,
                "tokft": f"{tok_ft:.3f}",
                "top5base": top5_b,
                "top5ft": top5_ft_s,
                "taubase": tau_b,
                "tauft": f"{tau_ft:.2f}",
            }
        )

    out_ft = args.tables_dir / "wikitext_lm_ft_compare.csv"
    with out_ft.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["family", "policybase", "policyft", "tokbase", "tokft", "top5base", "top5ft", "taubase", "tauft"],
        )
        w.writeheader()
        w.writerows(ft_rows)
    print(f"[export] wrote {out_ft}")


if __name__ == "__main__":
    main()
