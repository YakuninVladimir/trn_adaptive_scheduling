from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from diplom.schedulers.base import RecursionScheduler, Schedule, SchedulerState


@dataclass(frozen=True)
class LearnedSchedulerConfig:
    n_choices: tuple[int, ...] = (2, 4, 6)
    T_choices: tuple[int, ...] = (1, 2, 3)
    init_temperature: float = 1.0


class LearnedScheduler(nn.Module, RecursionScheduler):
    """
    Minimal learnable scheduler skeleton.

    Not wired into training by default; provided for future research where the scheduler depends on `model_aux`.
    """

    def __init__(self, cfg: LearnedSchedulerConfig, aux_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_head = nn.Linear(aux_dim, len(cfg.n_choices))
        self.T_head = nn.Linear(aux_dim, len(cfg.T_choices))
        self.temperature = nn.Parameter(torch.tensor(float(cfg.init_temperature)))

    def get_schedule(self, state: SchedulerState, model_aux: torch.Tensor | None = None) -> Schedule:
        if model_aux is None:
            # fallback to middle choice
            n = self.cfg.n_choices[len(self.cfg.n_choices) // 2]
            T = self.cfg.T_choices[len(self.cfg.T_choices) // 2]
        else:
            # model_aux: [B, D] or [D]
            if model_aux.dim() == 2:
                aux = model_aux.mean(dim=0)
            else:
                aux = model_aux
            tau = torch.clamp(self.temperature, 0.1, 10.0)
            n_logits = self.n_head(aux) / tau
            T_logits = self.T_head(aux) / tau
            n = self.cfg.n_choices[int(torch.argmax(n_logits).item())]
            T = self.cfg.T_choices[int(torch.argmax(T_logits).item())]

        w = 1.0 / max(state.max_supervision_steps, 1)
        return Schedule(recursion_n=int(n), recursion_T=int(T), supervision_weight=float(w), halt_threshold=None)

