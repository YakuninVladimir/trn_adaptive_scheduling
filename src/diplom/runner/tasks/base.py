from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class TaskBatch:
    x_tokens: torch.Tensor  # [B, L] int64
    y: torch.Tensor  # task-specific (often [B, L] int64)
    y_mask: torch.Tensor | None = None  # [B, L] bool, True where y contributes to loss/metrics
    # Optional per-position loss weights [B, L] float (only positions with y_mask True are used).
    y_weight: torch.Tensor | None = None


def collate_task_batches(items: list[TaskBatch]) -> TaskBatch:
    x = torch.stack([it.x_tokens for it in items], dim=0)
    # y can be vector or scalar
    y0 = items[0].y
    if y0.dim() == 0:
        y = torch.stack([it.y for it in items], dim=0)
    else:
        y = torch.stack([it.y for it in items], dim=0)
    if items[0].y_mask is None:
        mask = None
    else:
        mask = torch.stack([it.y_mask for it in items], dim=0)
    if items[0].y_weight is None:
        w = None
    else:
        w = torch.stack([it.y_weight for it in items], dim=0)
    return TaskBatch(x_tokens=x, y=y, y_mask=mask, y_weight=w)


class Task(Protocol):
    name: str

    def build_dataloaders(self, batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
        ...

    def compute_loss(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor:
        ...

    def compute_metrics(self, logits: torch.Tensor, batch: TaskBatch) -> dict[str, float]:
        ...

    def halt_targets(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor | None:
        """
        Optional: returns [B] float targets in {0,1} for halting head BCE supervision.
        """
        return None

