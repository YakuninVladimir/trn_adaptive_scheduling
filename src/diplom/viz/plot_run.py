from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

# Force non-interactive backend for headless/WSL environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _dedupe_last_by_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep the latest occurrence for each step (helps when logs from multiple runs were appended).
    by_step: dict[int, dict[str, Any]] = {}
    for r in rows:
        step = int(r.get("step", -1))
        by_step[step] = r
    return [by_step[s] for s in sorted(by_step.keys()) if s >= 0]


def plot_run(run_dir: str, out_path: str | None = None) -> None:
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
        oracle_loss = [r.get("oracle_loss") for r in train_rows]
        used_sup = [r.get("used_sup") for r in train_rows]

        # 1) Main model loss (and optional total loss context).
        axes[0].plot(steps, main_loss, label="train_main_loss", color="tab:blue")
        if any(v is not None for v in total_loss):
            axes[0].plot(steps, total_loss, label="train_loss_total", color="tab:gray", alpha=0.5)
        axes[0].set_ylabel("main loss")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(str(run_path))

        # 2) Oracle loss only.
        axes[1].plot(steps, oracle_loss, label="train_oracle_loss", color="tab:green")
        axes[1].set_ylabel("oracle loss")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # 3) used_sup only.
        axes[2].plot(steps, used_sup, label="used_sup", color="tab:orange")
        axes[2].set_ylabel("used_sup")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

    if has_val:
        val_ax = axes[3]
        v_steps = [r["step"] for r in val_rows]
        keys = sorted({k for r in val_rows for k in r.keys()} - {"kind", "epoch", "step"})
        for k in keys:
            val_ax.plot(v_steps, [r.get(k) for r in val_rows], label=f"val_{k}")
        # Overlay train accuracies for direct train-vs-val comparison.
        if train_rows:
            t_steps = [r["step"] for r in train_rows]
            train_acc_keys = sorted(
                k for k in {kk for r in train_rows for kk in r.keys()} if k.endswith("_acc")
            )
            for k in train_acc_keys:
                val_ax.plot(
                    t_steps,
                    [r.get(k) for r in train_rows],
                    label=f"train_{k}",
                    linestyle="--",
                    alpha=0.8,
                )
        val_ax.legend()
        val_ax.set_ylabel("val metrics")
        val_ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("global step")
    fig.tight_layout()

    if out_path is None:
        out_path = str(run_path / "plots.png")
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=150)
    print(f"[plot] saved -> {out_p}")

