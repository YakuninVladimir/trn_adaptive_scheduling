from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diplom.runner.tasks.base import TaskBatch, collate_task_batches


class _SudokuCsvDataset(Dataset[TaskBatch]):
    def __init__(self, path: str, seq_len: int = 81) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Sudoku CSV not found: {path}")
        self.seq_len = seq_len
        self.df = pd.read_csv(self.path)
        if "puzzle" not in self.df.columns or "solution" not in self.df.columns:
            raise ValueError(f"Expected columns puzzle/solution in {path}, got {list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> TaskBatch:
        row = self.df.iloc[idx]
        p = str(row["puzzle"]).strip()
        s = str(row["solution"]).strip()
        if len(p) != self.seq_len or len(s) != self.seq_len:
            raise ValueError(f"Bad sudoku length at idx={idx}: puzzle={len(p)} solution={len(s)}")
        x = torch.tensor([ord(c) - ord("0") for c in p], dtype=torch.long)
        y = torch.tensor([ord(c) - ord("0") for c in s], dtype=torch.long)
        # all positions contribute
        mask = torch.ones(self.seq_len, dtype=torch.bool)
        return TaskBatch(x_tokens=x, y=y, y_mask=mask)


@dataclass(frozen=True)
class SudokuTaskConfig:
    train_path: str
    val_path: str | None = None
    seq_len: int = 81


class SudokuTask:
    name = "sudoku"

    def __init__(self, cfg: SudokuTaskConfig) -> None:
        self.cfg = cfg

    def build_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader | None]:
        train_ds = _SudokuCsvDataset(self.cfg.train_path, seq_len=self.cfg.seq_len)
        val_ds = _SudokuCsvDataset(self.cfg.val_path, seq_len=self.cfg.seq_len) if self.cfg.val_path else None
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_task_batches)
        val_dl = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_task_batches)
            if val_ds
            else None
        )
        return train_dl, val_dl

    def compute_loss(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor:
        # logits: [B, L, V]
        y = batch.y.to(logits.device)
        if batch.y_mask is None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        mask = batch.y_mask.to(logits.device).view(-1)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
        loss = loss[mask].mean()
        return loss

    def compute_metrics(self, logits: torch.Tensor, batch: TaskBatch) -> dict[str, float]:
        y = batch.y.to(logits.device)
        pred = torch.argmax(logits, dim=-1)
        correct = (pred == y)
        if batch.y_mask is not None:
            mask = batch.y_mask.to(logits.device)
            token_acc = (correct & mask).sum().float() / mask.sum().float().clamp_min(1.0)
            exact = ((correct | ~mask).all(dim=1)).float().mean()
        else:
            token_acc = correct.float().mean()
            exact = correct.all(dim=1).float().mean()
        return {"token_acc": float(token_acc.item()), "exact_acc": float(exact.item())}

    def halt_targets(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor:
        y = batch.y.to(logits.device)
        pred = torch.argmax(logits, dim=-1)
        correct = (pred == y).all(dim=1).float()  # [B]
        return correct

