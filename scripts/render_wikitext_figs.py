#!/usr/bin/env python3
"""Render WikiText LM result figures into tex/img/."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bar_tau_hist(csv_path: Path, out_pdf: Path, out_png: Path, dpi: int) -> None:
    steps, probs = [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            probs.append(float(row["probability"]))

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.bar(steps, probs, color="#4a72b0", edgecolor="#2b3f5c", linewidth=0.6)
    ax.set_xlabel("step index $\\tau^{\\star}$")
    ax.set_ylabel("empirical proportion")
    ax.set_xticks(steps)
    ax.set_ylim(0, max(probs) * 1.15)
    ax.grid(axis="y", linestyle=(0, (3, 3)), alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def scatter_quality_cost(csv_path: Path, out_pdf: Path, out_png: Path, dpi: int) -> None:
    xs, ys, labels = [], [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            labels.append(row["family"])
            xs.append(float(row["meantau"]))
            ys.append(float(row["tokbest"]))

    fig, ax = plt.subplots(figsize=(7.1, 4.05))
    ax.scatter(xs, ys, s=72, color="#c44e52", edgecolor="#6e2a2c", zorder=3)
    for x, y, lab in zip(xs, ys, labels, strict=True):
        ax.annotate(lab.replace("_", "\n"), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7.8)
    ax.set_xlabel(r"mean stopping depth $\bar{\tau}$")
    ax.set_ylabel(r"top-1 token accuracy $\mathrm{tok}_{\mathrm{best}}$")
    ax.grid(True, linestyle=(0, (4, 3)), alpha=0.42)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", type=Path, default=Path("tex/img"))
    p.add_argument("--tables-dir", type=Path, default=Path("tex/tables"))
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    img = args.img_dir
    tbl = args.tables_dir
    img.mkdir(parents=True, exist_ok=True)

    bar_tau_hist(
        tbl / "wikitext_tau_hist_val.csv",
        img / "wikitext_tau_star_hist_val.pdf",
        img / "wikitext_tau_star_hist_val.png",
        args.dpi,
    )

    scatter_quality_cost(
        tbl / "wikitext_lm_base.csv",
        img / "wikitext_quality_cost_best.pdf",
        img / "wikitext_quality_cost_best.png",
        args.dpi,
    )


if __name__ == "__main__":
    main()
