#!/usr/bin/env python3
"""Generate tex/presentation/presentation_data.tex from CSV tables (WikiText + ARC)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FAMILY_RU = {
    "finite_discrete": "Конечное дискретное",
    "smoothed_loss": "Сглаженная разметка",
    "mixture_geometric": "Смесь геометрических",
    "mixture_exponential": "Смесь экспоненциальных",
    "power": "Степенной спад",
    "negative_binomial": "Отрицательное биномиальное",
}

ARC_ORDER = [
    "finite_discrete",
    "smoothed_loss",
    "mixture_geometric",
    "mixture_exponential",
    "power",
    "negative_binomial",
]


def _comma(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _wiki_order(df: pd.DataFrame) -> list[str]:
    keys = [str(x) for x in df["family"].tolist()]
    return [k for k in ARC_ORDER if k in keys]


def _pgf_xy(x: float, y: float) -> str:
    return f"({x:.5g},{y:.6f})"


def _axis_limits(
    xs: list[float], ys: list[float], *, xpad_ratio: float = 0.08, ypad_ratio: float = 0.35
) -> tuple[float, float, float, float]:
    if not xs or not ys:
        return 0.0, 1.0, 0.0, 1.0
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-3)
    return (
        xmin - xpad_ratio * xspan,
        xmax + xpad_ratio * xspan,
        ymin - ypad_ratio * yspan,
        ymax + ypad_ratio * yspan,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wikitext-base",
        type=Path,
        default=Path("tex/tables/wikitext_lm_base.csv"),
        help="Строки после oracle_finetune (как в главе NLP).",
    )
    p.add_argument(
        "--arc-base",
        type=Path,
        default=Path("tex/tables/arc_stopping_base.csv"),
        help="ARC best-of-grid (как tex/tables/arc_stopping_base.csv).",
    )
    p.add_argument(
        "--wikitext-ft",
        type=Path,
        default=Path("tex/tables/wikitext_lm_ft_compare.csv"),
    )
    p.add_argument(
        "--arc-ft",
        type=Path,
        default=Path("tex/tables/arc_stopping_finetune_compare.csv"),
    )
    p.add_argument("-o", "--out", type=Path, default=Path("tex/presentation/presentation_data.tex"))
    args = p.parse_args()

    wiki = pd.read_csv(args.wikitext_base)
    arc_b = pd.read_csv(args.arc_base)
    wiki_ft = pd.read_csv(args.wikitext_ft)
    arc_ft = pd.read_csv(args.arc_ft)

    wiki_map = wiki.set_index("family").to_dict("index")
    arc_map = arc_b.set_index("family").to_dict("index")
    wiki_ft_map = wiki_ft.set_index("family").to_dict("index")
    arc_ft_map = arc_ft.set_index("family").to_dict("index")

    def wiki_row(k: str) -> str:
        r = wiki_map[k]
        thr = float(r["threshold"])
        mt = float(r["meantau"])
        tok = float(r["tokbest"])
        nll = float(r["nlltau"])
        return (
            rf"{FAMILY_RU[k]} & {_comma(thr, 2)} & {_comma(mt, 2)} & "
            rf"{_comma(tok, 3)} & {_comma(nll, 3)} \\"
        )

    def arc_row(k: str) -> str:
        r = arc_map[k]
        thr = float(r["threshold"])
        mt = float(r["meantau"])
        tok = float(r["tokbest"])
        nll = float(r["nlltau"])
        return (
            rf"{FAMILY_RU[k]} & {_comma(thr, 2)} & {_comma(mt, 2)} & "
            rf"{_comma(tok, 3)} & {_comma(nll, 3)} \\"
        )

    wiki_keys = _wiki_order(wiki)
    if not wiki_keys:
        raise SystemExit("wikitext_lm_base.csv: пусто или нет известных ключей family.")

    wiki_table_body = "\n".join(wiki_row(k) for k in wiki_keys)
    arc_table_body = "\n".join(arc_row(k) for k in ARC_ORDER if k in arc_map)

    # Качество–стоимость: легенда pgfplots (значения координат — с точкой, не запятая)
    wx = [float(wiki_map[k]["meantau"]) for k in wiki_keys]
    wy = [float(wiki_map[k]["tokbest"]) for k in wiki_keys]
    ax = [float(arc_map[k]["meantau"]) for k in ARC_ORDER if k in arc_map]
    ay = [float(arc_map[k]["tokbest"]) for k in ARC_ORDER if k in arc_map]

    wx0, wx1, wy0, wy1 = _axis_limits(wx, wy)
    ax0, ax1, ay0, ay1 = _axis_limits(ax, ay)

    def wiki_scatter_lines() -> list[str]:
        lines: list[str] = []
        for k in wiki_keys:
            row = wiki_map[k]
            x, y = float(row["meantau"]), float(row["tokbest"])
            lines.append(rf"\addplot coordinates {{{_pgf_xy(x, y)}}};")
            lines.append(rf"\addlegendentry{{{FAMILY_RU[k]}}}")
        return lines

    def arc_scatter_lines() -> list[str]:
        lines: list[str] = []
        for k in ARC_ORDER:
            if k not in arc_map:
                continue
            row = arc_map[k]
            x, y = float(row["meantau"]), float(row["tokbest"])
            lines.append(rf"\addplot coordinates {{{_pgf_xy(x, y)}}};")
            lines.append(rf"\addlegendentry{{{FAMILY_RU[k]}}}")
        return lines

    # Таблица «до/после» по FT: строки по порядку ARC_ORDER; Wiki без семьи — прочерки
    def fmt_tau_cell(raw: str | float) -> str:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return "---"
        s = str(raw).strip()
        if s in ("", "—", "---", "nan", "NaN"):
            return "---"
        try:
            return _comma(float(s.replace(",", ".")), 2)
        except ValueError:
            return s.replace(".", ",")

    ft_rows: list[str] = []
    for k in ARC_ORDER:
        if k not in arc_ft_map:
            continue
        ar = arc_ft_map[k]
        label = FAMILY_RU[k]
        if k in wiki_ft_map:
            wr = wiki_ft_map[k]
            w0 = fmt_tau_cell(wr["taubase"])
            w1 = fmt_tau_cell(wr["tauft"])
        else:
            w0, w1 = "---", "---"
        a0 = fmt_tau_cell(ar["taubase"])
        a1 = fmt_tau_cell(ar["tauft"])
        ft_rows.append(rf"{label} & {w0} & {w1} & {a0} & {a1} \\")

    ft_table_body = "\n".join(ft_rows)

    # Итоговые выводы для слайдов с таблицами (кратко по max tok)
    w_best = max(wiki_keys, key=lambda kk: float(wiki_map[kk]["tokbest"]))
    wiki_best_lab = FAMILY_RU[w_best]
    wiki_best_val = _comma(float(wiki_map[w_best]["tokbest"]), 3)

    arc_keys_avail = [k for k in ARC_ORDER if k in arc_map]
    a_best = max(arc_keys_avail, key=lambda kk: float(arc_map[kk]["tokbest"]))
    arc_best_lab = FAMILY_RU[a_best]
    arc_best_val = _comma(float(arc_map[a_best]["tokbest"]), 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% AUTO-GENERATED — do not edit by hand.",
        "% Source: tex/tables/wikitext_lm_base.csv, arc_stopping_base.csv,",
        "%          wikitext_lm_ft_compare.csv, arc_stopping_finetune_compare.csv",
        "% Regenerate: uv run python scripts/render_presentation_data.py",
        "",
        r"\providecommand{\PresWikiTabBody}{%",
        wiki_table_body,
        r"}",
        "",
        r"\providecommand{\PresArcTabBody}{%",
        arc_table_body,
        r"}",
        "",
        rf"\providecommand{{\PresWikiQcXmin}}{{{wx0:.5g}}}",
        rf"\providecommand{{\PresWikiQcXmax}}{{{wx1:.5g}}}",
        rf"\providecommand{{\PresWikiQcYmin}}{{{wy0:.6f}}}",
        rf"\providecommand{{\PresWikiQcYmax}}{{{wy1:.6f}}}",
        "",
        rf"\providecommand{{\PresArcQcXmin}}{{{ax0:.5g}}}",
        rf"\providecommand{{\PresArcQcXmax}}{{{ax1:.5g}}}",
        rf"\providecommand{{\PresArcQcYmin}}{{{ay0:.6f}}}",
        rf"\providecommand{{\PresArcQcYmax}}{{{ay1:.6f}}}",
        "",
        r"\providecommand{\PresWikiScatter}{%",
        "\n".join(wiki_scatter_lines()),
        r"}",
        "",
        r"\providecommand{\PresArcScatter}{%",
        "\n".join(arc_scatter_lines()),
        r"}",
        "",
        r"\providecommand{\PresFTtauBody}{%",
        ft_table_body,
        r"}",
        "",
        rf"\providecommand{{\PresWikiTabConclusion}}{{Максимальный топ-1 на контроле после дополнительного обучения головы оракула: \textbf{{{wiki_best_lab}}} ({wiki_best_val}). Различия между семьями малы; подробнее --- раздел WikiText в диссертации.}}",
        "",
        rf"\providecommand{{\PresArcTabConclusion}}{{Максимальная токеновая точность (ARC): \textbf{{{arc_best_lab}}} ({arc_best_val}). Смесь экспоненциальных даёт сопоставимое качество при меньшем $\tau$.}}",
        "",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[render_presentation_data] wrote {out}")


if __name__ == "__main__":
    main()
