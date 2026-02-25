from __future__ import annotations

from dataclasses import dataclass

import torch

from diplom.schedulers.base import RecursionScheduler, Schedule, SchedulerState


@dataclass(frozen=True)
class FixedSchedulerConfig:
    recursion_n: int = 2
    recursion_T: int = 2
    uniform_weighting: bool = True


class FixedScheduler(RecursionScheduler):
    def __init__(self, cfg: FixedSchedulerConfig) -> None:
        self.cfg = cfg

    def get_schedule(self, state: SchedulerState, model_aux: torch.Tensor | None = None) -> Schedule:
        if self.cfg.uniform_weighting:
            w = 1.0 / max(state.max_supervision_steps, 1)
        else:
            w = 1.0
        return Schedule(
            recursion_n=int(self.cfg.recursion_n),
            recursion_T=int(self.cfg.recursion_T),
            supervision_weight=float(w),
            halt_threshold=None,
        )

