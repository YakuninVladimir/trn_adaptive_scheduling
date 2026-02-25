from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux_tensor: torch.Tensor | None = None
    state: Any = None
    loss_parts: dict[str, torch.Tensor] = field(default_factory=dict)

