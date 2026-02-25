from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if train_rows:
        steps = [r["step"] for r in train_rows]
        loss = [r.get("loss") for r in train_rows]
        axes[0].plot(steps, loss, label="train_loss")
        used_sup = [r.get("used_sup") for r in train_rows]
        if any(u is not None for u in used_sup):
            ax2 = axes[0].twinx()
            ax2.plot(steps, used_sup, color="tab:orange", alpha=0.5, label="used_sup")
            ax2.set_ylabel("used_sup")
        axes[0].set_ylabel("loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(str(run_path))

    if val_rows:
        v_steps = [r["step"] for r in val_rows]
        keys = sorted({k for r in val_rows for k in r.keys()} - {"kind", "epoch", "step"})
        for k in keys:
            axes[1].plot(v_steps, [r.get(k) for r in val_rows], label=f"val_{k}")
        axes[1].legend()
        axes[1].set_ylabel("val metrics")
        axes[1].grid(True, alpha=0.3)

    axes[1].set_xlabel("global step")
    fig.tight_layout()

    if out_path is None:
        out_path = str(run_path / "plots.png")
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=150)
    print(f"[plot] saved -> {out_p}")

