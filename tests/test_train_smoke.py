from __future__ import annotations

import csv

from diplom.runner.train import train_from_yaml


def _rand_sudoku_str(i: int, allow_zero: bool) -> str:
    # Deterministic pseudo-random digits from index
    digits = []
    x = i * 2654435761 % (2**32)
    for _ in range(81):
        x = (1103515245 * x + 12345) % (2**31)
        d = (x % 9) + 1
        if allow_zero and (x % 7 == 0):
            d = 0
        digits.append(str(d))
    return "".join(digits)


def test_train_from_yaml_smoke(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    with train_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["puzzle", "solution"])
        w.writeheader()
        for i in range(8):
            w.writerow({"puzzle": _rand_sudoku_str(i, allow_zero=True), "solution": _rand_sudoku_str(i, allow_zero=False)})

    with val_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["puzzle", "solution"])
        w.writeheader()
        for i in range(4):
            w.writerow({"puzzle": _rand_sudoku_str(100 + i, allow_zero=True), "solution": _rand_sudoku_str(100 + i, allow_zero=False)})

    run_dir = tmp_path / "run"
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "task:",
                "  name: sudoku",
                f"  train_path: {train_csv.as_posix()}",
                f"  val_path: {val_csv.as_posix()}",
                "model:",
                "  name: trm",
                "  vocab_size: 11",
                "  seq_len: 81",
                "  d_model: 32",
                "  n_heads: 4",
                "  n_layers: 1",
                "  d_ff: 64",
                "  use_attention: true",
                "  H_cycles: 1",
                "  L_cycles: 1",
                "  N_sup: 2",
                "scheduler:",
                "  name: fixed",
                "  recursion_n: 1",
                "  recursion_T: 1",
                "train:",
                "  seed: 42",
                "  device: cpu",
                "  epochs: 1",
                "  batch_size: 2",
                "  lr: 1.0e-3",
                "  log_every: 1",
                "  eval_every: 2",
                "  ckpt_every: 1000",
                f"  run_dir: {run_dir.as_posix()}",
            ]
        )
    )

    train_from_yaml(str(cfg_path))
    assert (run_dir / "metrics.jsonl").exists()

