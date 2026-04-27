from __future__ import annotations

import torch

from diplom.runner.tasks.sudoku_task import (
    SudokuTask,
    SudokuTaskConfig,
    _candidate_count_for_hole,
    _hole_mask_and_weights,
)


def test_reweight_returns_when_no_holes():
    """Reweight branch must always return; previously missing return caused None."""
    x = [5] * 81
    hole, w = _hole_mask_and_weights(
        x,
        loss_on_empty_only=True,
        reweight=True,
        difficulty_power=1.0,
    )
    assert hole is not None and w is not None
    assert not hole.any()


def test_hole_mask_only_empty():
    # Minimal puzzle string: one row of givens + dots (invalid as sudoku but tests masking)
    x = [5] * 81  # no holes
    hole, w = _hole_mask_and_weights(
        x,
        loss_on_empty_only=True,
        reweight=False,
        difficulty_power=1.0,
    )
    assert not hole.any()

    x2 = [0 if i % 2 == 0 else 1 for i in range(81)]  # many holes (digit 1 illegal but ok for mask)
    hole2, w2 = _hole_mask_and_weights(
        x2,
        loss_on_empty_only=True,
        reweight=False,
        difficulty_power=1.0,
    )
    assert hole2.sum() > 0
    assert (w2[hole2] == 1.0).all()


def test_candidate_count_basic():
    # Empty grid: every hole has 9 candidates
    grid = [[0] * 9 for _ in range(9)]
    assert _candidate_count_for_hole(grid, 0, 0) == 9
    # One given in row blocks that digit
    grid[0][1] = 3
    assert _candidate_count_for_hole(grid, 0, 0) == 8


def test_sudoku_task_collate_weighted_loss(tmp_path):
    p = tmp_path / "t.csv"
    # Valid-length strings; puzzle has empties ('.')
    puzzle = "53..7...." + "." * 72  # shortened header row pattern + pad
    assert len(puzzle) == 81
    sol = "5" * 81
    p.write_text("puzzle,solution\n" + f"{puzzle},{sol}\n" + f"{puzzle},{sol}\n")
    cfg = SudokuTaskConfig(
        train_path=str(p),
        val_path=None,
        loss_on_empty_cells_only=True,
        hole_difficulty_reweight=True,
    )
    task = SudokuTask(cfg)
    dl, _ = task.build_dataloaders(batch_size=2)
    batch = next(iter(dl))
    assert batch.y_mask is not None
    assert batch.y_weight is not None
    assert batch.y_mask.shape == batch.y_weight.shape
    logits = torch.randn(2, 81, 11)
    loss = task.compute_loss(logits, batch)
    assert loss.dim() == 0 and torch.isfinite(loss)
