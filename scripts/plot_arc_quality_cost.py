#!/usr/bin/env python3
"""Кривая «качество — среднее число шагов» для ARC-AGI (лучшая строка на семейство).

Читает таблицу в формате как `tex/tables/arc_stopping_base.csv` и строит диаграмму
рассеяния средствами seaborn (русские подписи). Данные должны совпадать с табл.~\\ref{tab:arc_base}.

Пример:

    uv run python scripts/plot_arc_quality_cost.py \\
      --from-csv tex/tables/arc_stopping_base.csv \\
      --out tex/img/arc_quality_cost_best.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

_FAMILY_RU = {
    "finite_discrete": "конечное дискретное",
    "smoothed_loss": "сглаженная разметка",
    "mixture_geometric": "смесь геометрических",
    "mixture_exponential": "смесь экспоненциальных",
    "power": "степенной спад",
    "negative_binomial": "отрицательное биномиальное",
}


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"family", "meantau", "tokbest"}
    if not need <= set(df.columns):
        raise SystemExit(f"CSV must contain columns {sorted(need)}; got {list(df.columns)}")
    df = df[list(need)].copy()
    df["family_ru"] = df["family"].map(lambda x: _FAMILY_RU.get(str(x), str(x)))
    return df


def plot_quality_cost(df: pd.DataFrame, out: Path, *, dpi: int, title: str) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.08)
    fig, ax = plt.subplots(figsize=(5.8, 3.85))
    pal = sns.color_palette("tab10", n_colors=len(df))
    sns.scatterplot(
        data=df.sort_values("meantau"),
        x="meantau",
        y="tokbest",
        hue="family_ru",
        palette=pal[: len(df)],
        s=90,
        ax=ax,
        legend=True,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.legend(
        title="семейство",
        fontsize=8,
        title_fontsize=9,
        loc="best",
        frameon=True,
        framealpha=0.95,
    )
    ax.set_xlabel("Среднее число внутренних шагов $\\bar{\\tau}$")
    ax.set_ylabel("Токеновая точность (лучший вариант ответа)")
    ax.set_title(title)
    ax.set_xlim(float(df["meantau"].min()) - 0.35, float(df["meantau"].max()) + 0.35)
    ymin, ymax = float(df["tokbest"].min()), float(df["tokbest"].max())
    ax.set_ylim(ymin - 0.012, ymax + 0.012)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--from-csv",
        type=Path,
        default=Path("tex/tables/arc_stopping_base.csv"),
        help="Столбцы: family, meantau, tokbest (как экспорт таблицы итогов ARC).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tex/img/arc_quality_cost_best.png"),
        help="Выход PNG.",
    )
    ap.add_argument("--dpi", type=int, default=160, help="DPI.")
    ap.add_argument(
        "--title",
        default="Точность",
        help="Заголовок рисунка.",
    )
    args = ap.parse_args()
    df = _load(args.from_csv)
    plot_quality_cost(df, args.out, dpi=args.dpi, title=args.title)
    print(f"wrote {args.out.resolve()} ({len(df)} точек)")


if __name__ == "__main__":
    main()
