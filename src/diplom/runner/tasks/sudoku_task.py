from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diplom.runner.tasks.base import TaskBatch, collate_task_batches


@dataclass(frozen=True)
class SudokuTaskConfig:
    train_path: str
    val_path: str | None = None
    seq_len: int = 81
    # CE only on empty puzzle cells (token 0 / '.').
    loss_on_empty_cells_only: bool = True
    # Per-hole weights ~ (candidate_count ** power), normalized to mean 1 over holes.
    hole_difficulty_reweight: bool = True
    difficulty_power: float = 1.0


def _candidate_count_for_hole(grid: list[list[int]], r: int, c: int) -> int:
    """
    How many digits from 1..9 are still feasible for an empty cell given only
    puzzle givens (no constraint propagation). Cheap local difficulty proxy:
    more remaining candidates => locally more ambiguous => stronger loss weight.
    """
    used: set[int] = set()
    for j in range(9):
        v = grid[r][j]
        if v != 0:
            used.add(v)
        v = grid[j][c]
        if v != 0:
            used.add(v)
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            v = grid[i][j]
            if v != 0:
                used.add(v)
    return sum(1 for d in range(1, 10) if d not in used)


def _hole_mask_and_weights(
    x_tokens: list[int],
    *,
    loss_on_empty_only: bool,
    reweight: bool,
    difficulty_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      hole_mask [L] bool — True where loss applies
      weights [L] float — positive on masked positions, mean 1 over masked when reweight
    """
    seq_len = len(x_tokens)
    grid = [x_tokens[i * 9 : (i + 1) * 9] for i in range(9)]

    if not loss_on_empty_only:
        mask = torch.ones(seq_len, dtype=torch.bool)
        w = torch.ones(seq_len, dtype=torch.float32)
        return mask, w

    hole = torch.tensor([t == 0 for t in x_tokens], dtype=torch.bool)
    w = torch.zeros(seq_len, dtype=torch.float32)
    if not bool(reweight):
        w = torch.where(hole, torch.ones(seq_len, dtype=torch.float32), w)
        return hole, w

    for idx in range(seq_len):
        if not hole[idx]:
            continue
        r, c = idx // 9, idx % 9
        nc = _candidate_count_for_hole(grid, r, c)
        if nc <= 0:
            nc = 1
        w[idx] = float(nc) ** float(difficulty_power)

    wh = w[hole]
    if wh.numel() > 0:
        w = torch.where(hole, w / wh.mean().clamp_min(1e-8), w)
    return hole, w


class _SudokuCsvDataset(Dataset[TaskBatch]):
    def __init__(self, path: str, cfg: SudokuTaskConfig) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Sudoku CSV not found: {path}")
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.df = pd.read_csv(self.path)
        cols = set(self.df.columns)
        if {"puzzle", "solution"}.issubset(cols):
            self.puzzle_col = "puzzle"
            self.solution_col = "solution"
        elif {"question", "answer"}.issubset(cols):
            self.puzzle_col = "question"
            self.solution_col = "answer"
        else:
            raise ValueError(
                f"Expected columns puzzle/solution or question/answer in {path}, got {list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> TaskBatch:
        row = self.df.iloc[idx]
        p = str(row[self.puzzle_col]).strip()
        s = str(row[self.solution_col]).strip()
        if len(p) != self.seq_len or len(s) != self.seq_len:
            raise ValueError(f"Bad sudoku length at idx={idx}: puzzle={len(p)} solution={len(s)}")
        x_list = [self._to_token(c) for c in p]
        x = torch.tensor(x_list, dtype=torch.long)
        y = torch.tensor([self._to_token(c) for c in s], dtype=torch.long)
        mask, w = _hole_mask_and_weights(
            x_list,
            loss_on_empty_only=self.cfg.loss_on_empty_cells_only,
            reweight=self.cfg.hole_difficulty_reweight,
            difficulty_power=self.cfg.difficulty_power,
        )
        return TaskBatch(x_tokens=x, y=y, y_mask=mask, y_weight=w)

    @staticmethod
    def _to_token(c: str) -> int:
        if c == ".":
            return 0
        if c.isdigit():
            v = int(c)
            if 0 <= v <= 9:
                return v
        raise ValueError(f"Unsupported sudoku token character: {c!r}")


class SudokuTask:
    name = "sudoku"

    def __init__(self, cfg: SudokuTaskConfig) -> None:
        self.cfg = cfg

    def build_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader | None]:
        train_ds = _SudokuCsvDataset(self.cfg.train_path, cfg=self.cfg)
        val_ds = _SudokuCsvDataset(self.cfg.val_path, cfg=self.cfg) if self.cfg.val_path else None
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_task_batches)
        val_dl = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_task_batches)
            if val_ds
            else None
        )
        return train_dl, val_dl

    @staticmethod
    def _masked_weighted_ce_mean(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        # logits [B,L,V], y [B,L], mask [B,L] bool, weight [B,L] float
        B, L, V = logits.shape
        ce = F.cross_entropy(logits.view(B * L, V), y.view(B * L), reduction="none").view(B, L)
        m = mask.float()
        w = weight if weight is not None else torch.ones_like(m)
        numer = (ce * m * w).sum()
        denom = (m * w).sum().clamp_min(1e-8)
        return numer / denom

    def compute_loss(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor:
        y = batch.y.to(logits.device)
        if batch.y_mask is None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        mask = batch.y_mask.to(logits.device)
        # Degenerate row: no empty cells — fall back to CE over full grid so we still get gradients.
        if not mask.any():
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        w = batch.y_weight.to(logits.device) if batch.y_weight is not None else None
        return self._masked_weighted_ce_mean(logits, y, mask, w)

    def compute_metrics(self, logits: torch.Tensor, batch: TaskBatch) -> dict[str, float]:
        y = batch.y.to(logits.device)
        pred = torch.argmax(logits, dim=-1)
        correct = pred == y
        # Token accuracy: only empty cells (where we supervise), if mask is hole-only.
        if batch.y_mask is not None:
            mask = batch.y_mask.to(logits.device)
            if mask.any():
                token_acc = (correct & mask).sum().float() / mask.sum().float().clamp_min(1.0)
            else:
                token_acc = correct.float().mean()
        else:
            token_acc = correct.float().mean()
        # Exact: entire grid must match the solution (including givens).
        exact = correct.all(dim=1).float().mean()
        return {"token_acc": float(token_acc.item()), "exact_acc": float(exact.item())}

    def halt_targets(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor:
        y = batch.y.to(logits.device)
        pred = torch.argmax(logits, dim=-1)
        return (pred == y).all(dim=1).float()
