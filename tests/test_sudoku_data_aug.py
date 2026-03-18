from __future__ import annotations

import random

from diplom.data.sudoku_extreme import _augment_grid, _grid_from_str, _str_from_grid


def test_sudoku_grid_roundtrip_keeps_blanks():
    puzzle = (
        "53..7...."
        "6..195..."
        ".98....6."
        "8...6...3"
        "4..8.3..1"
        "7...2...6"
        ".6....28."
        "...419..5"
        "....8..79"
    )
    g = _grid_from_str(puzzle)
    s = _str_from_grid(g)
    assert s == puzzle


def test_augment_preserves_blank_count():
    puzzle = (
        "53..7...."
        "6..195..."
        ".98....6."
        "8...6...3"
        "4..8.3..1"
        "7...2...6"
        ".6....28."
        "...419..5"
        "....8..79"
    )
    g = _grid_from_str(puzzle)
    blank_count = int((g == 0).sum())
    aug = _augment_grid(g, random.Random(123))
    assert int((aug == 0).sum()) == blank_count
