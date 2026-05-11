#!/usr/bin/env python3
"""Графики для главы про ARC-AGI: seaborn, отдельный файл на центральную панель (per-batch τ*).

По умолчанию использует зашитые вероятности (эмиссия как на эталонном графике per-batch).
При передаче JSON от ``plot_optimal_stop_distribution.py`` (ключ
``probabilities_tau_per_batch_pooled_trajectory``) подставляет данные из него.

Пример:

  uv run python scripts/plot_arc_arcagi_figures.py \\
    --out tex/img/arc_tau_star_hist_val.png \\
    --write-hist-csv tex/tables/tau_star_hist_finite_discrete_val.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Эталонное распределение middle panel (per-batch, argmax batch-pooled token_acc), нормировано.
_DEFAULT_BATCH_PROBS_1_TO_8 = np.array(
    [0.50, 0.12, 0.20, 0.10, 0.04, 0.01, 0.02, 0.02], dtype=np.float64
)


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    s = float(np.sum(p))
    if s <= 0:
        raise SystemExit("histogram probabilities must sum to a positive value")
    return p / s


def _counts_from_probs(probs: np.ndarray, n_total: int) -> np.ndarray:
    """Hamilton / largest remainder: целые счётчики с суммой ровно n_total."""
    probs = _normalize(probs)
    exact = probs * float(n_total)
    counts = np.floor(exact).astype(int)
    rem = int(n_total - counts.sum())
    frac = exact - counts
    order = np.argsort(-frac)
    for i in range(rem):
        counts[int(order[i % len(order)])] += 1
    return counts


def _load_probs_from_json(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    key = "probabilities_tau_per_batch_pooled_trajectory"
    if key not in obj:
        raise SystemExit(f"JSON missing key {key!r}")
    probs = np.asarray(obj[key], dtype=np.float64).ravel()
    if probs.size < 1:
        raise SystemExit("empty probability vector in JSON")
    return probs


def plot_middle_panel_only(
    probs: np.ndarray,
    out: Path,
    *,
    dpi: int,
    title: str,
    ylabel: str,
) -> None:
    probs = _normalize(probs)
    k = int(probs.size)
    steps = np.arange(1, k + 1)
    df = pd.DataFrame({"step": steps, "probability": probs})

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sns.barplot(
        data=df,
        x="step",
        y="probability",
        color="#cb4b4f",
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel(r"$\tau^\star$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(range(k)))
    ax.set_xticklabels([str(int(s)) for s in steps])
    ax.set_ylim(0.0, min(1.05, float(np.max(probs)) * 1.35 + 1e-6))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_hist_csv(probs: np.ndarray, path: Path, n_total: int) -> None:
    probs = _normalize(probs)
    counts = _counts_from_probs(probs, n_total)
    lines = ["step,count,probability"]
    for i in range(int(probs.size)):
        lines.append(f"{i + 1:d},{int(counts[i])},{float(probs[i]):.10f}".rstrip("0").rstrip("."))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_token_acc_curve(json_path: Path | None, out: Path, dpi: int) -> None:
    """Дополнительный график: token_acc по шагу (macro по батчам), если в JSON есть поля."""
    if json_path is None or not json_path.exists():
        return
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    key = "mean_token_acc_per_step_batch_pooled_macro_avg"
    if key not in obj:
        return
    ys = np.asarray(obj[key], dtype=np.float64).ravel()
    x = np.arange(1, ys.size + 1)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    sns.lineplot(x=x, y=ys, marker="o", ax=ax, color="darkorange")
    ax.set_xlabel("Шаг надзора")
    ax.set_ylabel("Токеновая точность (пакет)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(r"$\mathrm{token\_acc}$ по шагу (среднее по батчам)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="JSON с полем probabilities_tau_per_batch_pooled_trajectory (выход --out-json у plot_optimal_stop_distribution).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tex/img/arc_tau_star_hist_val.png"),
        help="Выход PNG: одна панель, per-batch распределение τ*.",
    )
    ap.add_argument("--dpi", type=int, default=160, help="DPI сохранённого PNG.")
    ap.add_argument(
        "--write-hist-csv",
        type=Path,
        default=None,
        help="При указании записать tex/tables-совместимый CSV для таблицы гистограммы.",
    )
    ap.add_argument(
        "--hist-weight",
        type=int,
        default=400,
        help="Сумма счётчиков в CSV (число батчей или задач; по умолчанию 400).",
    )
    ap.add_argument(
        "--extra-token-acc",
        type=Path,
        default=None,
        help="Если задан путь PNG, попытаться построить кривую token_acc из того же --from-json.",
    )
    ap.add_argument(
        "--title",
        default=r"Per-batch $\tau^\star$ (argmax batch-pooled token_acc)",
        help="Заголовок рисунка (на английском по умолчанию, как на эталоне).",
    )
    ap.add_argument(
        "--ylabel",
        default="Probability",
        help="Подпись оси Y.",
    )
    args = ap.parse_args()

    if args.from_json is not None:
        probs = _load_probs_from_json(args.from_json)
    else:
        probs = _DEFAULT_BATCH_PROBS_1_TO_8.copy()

    plot_middle_panel_only(
        probs,
        args.out,
        dpi=args.dpi,
        title=args.title,
        ylabel=args.ylabel,
    )
    print(f"wrote {args.out.resolve()}")

    if args.write_hist_csv is not None:
        write_hist_csv(probs, args.write_hist_csv, args.hist_weight)
        print(f"wrote {args.write_hist_csv.resolve()}")

    if args.extra_token_acc is not None and args.from_json is not None:
        plot_token_acc_curve(args.from_json, args.extra_token_acc, args.dpi)
        print(f"wrote {args.extra_token_acc.resolve()}")


if __name__ == "__main__":
    main()
