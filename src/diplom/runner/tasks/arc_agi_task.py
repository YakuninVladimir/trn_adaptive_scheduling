from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diplom.runner.tasks.base import TaskBatch, collate_task_batches


@dataclass(frozen=True)
class ArcAgiTaskConfig:
    train_path: str
    val_path: str | None = None
    max_height: int = 30
    max_width: int = 30
    max_color: int = 9


def _load_json_records(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ARC-AGI file not found: {path}")
    if p.suffix.lower() == ".jsonl":
        out: list[dict] = []
        for line in p.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
    raw = json.loads(p.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "tasks" in raw and isinstance(raw["tasks"], list):
            return list(raw["tasks"])
        return [raw]
    raise ValueError(f"Unsupported ARC file structure in: {path}")


def _extract_pairs(record: dict) -> list[tuple[list[list[int]], list[list[int]]]]:
    pairs: list[tuple[list[list[int]], list[list[int]]]] = []
    if "input" in record and ("output" in record or "target" in record):
        out_key = "output" if "output" in record else "target"
        pairs.append((record["input"], record[out_key]))
        return pairs
    for key in ("train", "test"):
        items = record.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if "input" in item and ("output" in item or "target" in item):
                out_key = "output" if "output" in item else "target"
                pairs.append((item["input"], item[out_key]))
    return pairs


class _ArcDataset(Dataset[TaskBatch]):
    def __init__(self, path: str, cfg: ArcAgiTaskConfig) -> None:
        self.cfg = cfg
        pairs: list[tuple[list[list[int]], list[list[int]]]] = []
        for rec in _load_json_records(path):
            pairs.extend(_extract_pairs(rec))
        if not pairs:
            raise ValueError(f"No ARC input/output pairs found in {path}")
        self.pairs = pairs
        self.seq_len = int(cfg.max_height) * int(cfg.max_width)

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode_grid(self, grid: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        h = min(len(grid), self.cfg.max_height)
        w = min(len(grid[0]) if grid else 0, self.cfg.max_width)
        tokens = torch.zeros(self.seq_len, dtype=torch.long)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        for i in range(h):
            row = grid[i]
            for j in range(min(len(row), w)):
                idx = i * self.cfg.max_width + j
                color = int(row[j])
                if color < 0 or color > self.cfg.max_color:
                    raise ValueError(f"ARC color out of range [0,{self.cfg.max_color}]: {color}")
                tokens[idx] = color + 1  # reserve 0 for padding
                mask[idx] = True
        return tokens, mask

    def __getitem__(self, idx: int) -> TaskBatch:
        inp, out = self.pairs[int(idx)]
        x_tokens, _ = self._encode_grid(inp)
        y_tokens, y_mask = self._encode_grid(out)
        return TaskBatch(x_tokens=x_tokens, y=y_tokens, y_mask=y_mask)


class ArcAgiTask:
    name = "arc_agi"

    def __init__(self, cfg: ArcAgiTaskConfig) -> None:
        self.cfg = cfg

    def build_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader | None]:
        train_ds = _ArcDataset(self.cfg.train_path, self.cfg)
        val_ds = _ArcDataset(self.cfg.val_path, self.cfg) if self.cfg.val_path else None
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_task_batches)
        val_dl = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_task_batches)
            if val_ds
            else None
        )
        return train_dl, val_dl

    def compute_loss(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor:
        y = batch.y.to(logits.device)
        if batch.y_mask is None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        mask = batch.y_mask.to(logits.device)
        if not bool(mask.any().item()):
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
        ce = ce.view_as(y)
        return ce[mask].mean()

    def compute_metrics(self, logits: torch.Tensor, batch: TaskBatch) -> dict[str, float]:
        y = batch.y.to(logits.device)
        pred = torch.argmax(logits, dim=-1)
        if batch.y_mask is not None:
            mask = batch.y_mask.to(logits.device)
            token_acc = (pred[mask] == y[mask]).float().mean()
            per_sample_exact = []
            for i in range(y.size(0)):
                mi = mask[i]
                per_sample_exact.append(float((pred[i][mi] == y[i][mi]).all().item()) if mi.any() else 0.0)
            exact = torch.tensor(per_sample_exact, device=logits.device).mean()
        else:
            token_acc = (pred == y).float().mean()
            exact = (pred == y).all(dim=1).float().mean()
        return {"token_acc": float(token_acc.item()), "exact_acc": float(exact.item())}

    def halt_targets(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor:
        y = batch.y.to(logits.device)
        pred = torch.argmax(logits, dim=-1)
        if batch.y_mask is None:
            return (pred == y).all(dim=1).float()
        mask = batch.y_mask.to(logits.device)
        out = []
        for i in range(y.size(0)):
            mi = mask[i]
            out.append(float((pred[i][mi] == y[i][mi]).all().item()) if mi.any() else 0.0)
        return torch.tensor(out, dtype=torch.float32, device=logits.device)
