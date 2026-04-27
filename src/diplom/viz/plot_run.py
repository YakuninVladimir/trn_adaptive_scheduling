from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

# Force non-interactive backend for headless/WSL environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float_array(values: list[Any]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _ema_1d(y: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=np.float64)
    ema = np.nan
    a = float(alpha)
    for i in range(len(y)):
        v = y[i]
        if not np.isfinite(v):
            out[i] = ema if np.isfinite(ema) else np.nan
            continue
        if not np.isfinite(ema):
            ema = v
        else:
            ema = a * v + (1.0 - a) * ema
        out[i] = ema
    return out


def _rolling_std_1d(y: np.ndarray, window: int) -> np.ndarray:
    w = max(2, int(window))
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = y[lo : i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 2:
            out[i] = float(np.std(seg, ddof=0))
        elif seg.size == 1:
            out[i] = 0.0
    return out


def _plot_series_ema_band(
    ax,
    x: list[int] | np.ndarray,
    values: list[Any],
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    raw_alpha: float = 0.35,
    ema_alpha: float = 0.06,
    std_window: int = 25,
    std_sigma: float = 1.0,
    enabled: bool = True,
) -> None:
    if not enabled:
        yy = [float("nan") if v is None else float(v) for v in values]
        ax.plot(x, yy, color=color, linestyle=linestyle, label=label)
        return
    y = _to_float_array(values)
    if not np.any(np.isfinite(y)):
        return
    xv = np.asarray(x, dtype=np.float64)
    ax.plot(xv, y, color=color, linestyle=linestyle, alpha=raw_alpha, linewidth=1.0, label=label)
    ema = _ema_1d(y, ema_alpha)
    rstd = _rolling_std_1d(y, std_window)
    ax.plot(xv, ema, color=color, linestyle=linestyle, linewidth=2.0, label=f"{label} EMA")
    lo = ema - std_sigma * rstd
    hi = ema + std_sigma * rstd
    ax.fill_between(xv, lo, hi, color=color, alpha=0.18, linewidth=0)


def _dedupe_last_by_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep the latest occurrence for each step (helps when logs from multiple runs were appended).
    by_step: dict[int, dict[str, Any]] = {}
    for r in rows:
        step = int(r.get("step", -1))
        by_step[step] = r
    return [by_step[s] for s in sorted(by_step.keys()) if s >= 0]


def plot_run(
    run_dir: str,
    out_path: str | None = None,
    *,
    ema_band: bool = True,
    ema_alpha: float = 0.06,
    # Validation is logged sparsely; a larger alpha makes EMA track fresh val points instead of old history.
    val_ema_alpha: float = 0.35,
    std_window: int = 25,
    std_sigma: float = 1.0,
) -> None:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found: {metrics_path}")

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    with metrics_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "train":
                train_rows.append(rec)
            elif rec.get("kind") == "val":
                val_rows.append(rec)

    if not train_rows and not val_rows:
        raise RuntimeError(f"No records found in {metrics_path}")

    train_rows = _dedupe_last_by_step(train_rows)
    val_rows = _dedupe_last_by_step(val_rows)

    has_val = bool(val_rows)
    nrows = 4 if has_val else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    if train_rows:
        steps = [r["step"] for r in train_rows]
        total_loss = [r.get("loss") for r in train_rows]
        main_loss = [r.get("main_loss") for r in train_rows]
        halt_loss = [r.get("halt_loss") for r in train_rows]
        oracle_loss = [r.get("oracle_loss") for r in train_rows]
        used_sup = [r.get("used_sup") for r in train_rows]

        # 1) Main model loss (and optional total loss context).
        _plot_series_ema_band(
            axes[0],
            steps,
            main_loss,
            color="tab:blue",
            label="train_main_loss",
            ema_alpha=ema_alpha,
            std_window=std_window,
            std_sigma=std_sigma,
            enabled=ema_band,
        )
        if any(v is not None for v in total_loss):
            if ema_band:
                _plot_series_ema_band(
                    axes[0],
                    steps,
                    total_loss,
                    color="tab:gray",
                    label="train_loss_total",
                    linestyle="-",
                    raw_alpha=0.25,
                    ema_alpha=ema_alpha,
                    std_window=std_window,
                    std_sigma=std_sigma,
                    enabled=True,
                )
            else:
                axes[0].plot(steps, total_loss, label="train_loss_total", color="tab:gray", alpha=0.5)
        axes[0].set_ylabel("main loss")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(str(run_path))

        # 2) Halt BCE vs oracle CE. Older logs had no halt_loss key: oracle_loss mixed both.
        has_halt_key = any("halt_loss" in r for r in train_rows)
        has_oracle_key = any("oracle_loss" in r for r in train_rows)
        if has_halt_key:
            hl = [0.0 if v is None else v for v in halt_loss]
            _plot_series_ema_band(
                axes[1],
                steps,
                hl,
                color="tab:purple",
                label="train_halt_loss",
                ema_alpha=ema_alpha,
                std_window=std_window,
                std_sigma=std_sigma,
                enabled=ema_band,
            )
        if has_oracle_key:
            ora_label = (
                "train_oracle_loss"
                if has_halt_key
                else "train_aux_loss (legacy: halt was logged as oracle_loss)"
            )
            ol = [0.0 if v is None else v for v in oracle_loss]
            _plot_series_ema_band(
                axes[1],
                steps,
                ol,
                color="tab:green",
                label=ora_label,
                ema_alpha=ema_alpha,
                std_window=std_window,
                std_sigma=std_sigma,
                enabled=ema_band,
            )
        axes[1].set_ylabel("halt / oracle loss")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # 3) used_sup only (skip EMA band if nearly constant).
        us_arr = _to_float_array(used_sup)
        flat_sup = bool(
            ema_band
            and np.any(np.isfinite(us_arr))
            and float(np.nanmax(us_arr) - np.nanmin(us_arr)) < 1e-3
        )
        if flat_sup:
            axes[2].plot(steps, used_sup, label="used_sup", color="tab:orange")
        else:
            _plot_series_ema_band(
                axes[2],
                steps,
                used_sup,
                color="tab:orange",
                label="used_sup",
                ema_alpha=ema_alpha,
                std_window=std_window,
                std_sigma=std_sigma,
                enabled=ema_band,
            )
        axes[2].set_ylabel("used_sup")
        lrs = [r.get("lr") for r in train_rows]
        h2_left, lab2_left = axes[2].get_legend_handles_labels()
        if any(v is not None for v in lrs):
            ax2r = axes[2].twinx()
            ax2r.plot(
                steps,
                [float(v) if v is not None else float("nan") for v in lrs],
                color="tab:red",
                alpha=0.55,
                label="lr",
            )
            ax2r.set_ylabel("learning rate", color="tab:red")
            ax2r.tick_params(axis="y", labelcolor="tab:red")
            h2_right, lab2_right = ax2r.get_legend_handles_labels()
            axes[2].legend(h2_left + h2_right, lab2_left + lab2_right, loc="upper right")
        else:
            axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

    if has_val:
        val_ax = axes[3]
        v_steps = [r["step"] for r in val_rows]
        keys = sorted({k for r in val_rows for k in r.keys()} - {"kind", "epoch", "step"})
        val_win = max(3, min(std_window, max(2, len(v_steps) // 3)))
        for i, k in enumerate(keys):
            vals = [r.get(k) for r in val_rows]
            _plot_series_ema_band(
                val_ax,
                v_steps,
                vals,
                color=f"C{i % 10}",
                label=f"val_{k}",
                ema_alpha=val_ema_alpha,
                std_window=val_win,
                std_sigma=std_sigma,
                enabled=ema_band,
            )
        # Overlay train accuracies for direct train-vs-val comparison.
        if train_rows:
            t_steps = [r["step"] for r in train_rows]
            train_acc_keys = sorted(
                kk for kk in {k2 for r in train_rows for k2 in r.keys()} if kk.endswith("_acc")
            )
            for i, k in enumerate(train_acc_keys):
                vals = [r.get(k) for r in train_rows]
                _plot_series_ema_band(
                    val_ax,
                    t_steps,
                    vals,
                    color=f"C{(i + 4) % 10}",
                    label=f"train_{k}",
                    linestyle="--",
                    ema_alpha=ema_alpha,
                    std_window=std_window,
                    std_sigma=std_sigma,
                    enabled=ema_band,
                )
        val_ax.legend(loc="upper left")
        val_ax.set_ylabel("val metrics")
        val_ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("global step")
    fig.tight_layout()

    if out_path is None:
        out_path = str(run_path / "plots.png")
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_p, dpi=150)
        print(f"[plot] saved -> {out_p}")
    finally:
        # Avoid accumulating figures when plot_run is called often (e.g. train --live-plots).
        plt.close(fig)

