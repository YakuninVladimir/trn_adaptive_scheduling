from __future__ import annotations

from dataclasses import dataclass

import torch

from diplom.schedulers.base import RecursionScheduler, Schedule, SchedulerState


@dataclass(frozen=True)
class CGARSchedulerConfig:
    # Progressive Depth Curriculum
    pdc_thresholds: tuple[float, float] = (0.3, 0.6)
    pdc_stages: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((2, 1), (4, 2), (6, 3))

    # Hierarchical Supervision Weighting
    lambda_decay: float = 0.7


class CGARScheduler(RecursionScheduler):
    """
    CGAR scheduler:
    - PDC: schedule (n, T) by training progress
    - HSW: exponential supervision weighting by supervision step
    """

    def __init__(self, cfg: CGARSchedulerConfig) -> None:
        self.cfg = cfg

    def _pdc(self, rho: float) -> tuple[int, int]:
        t1, t2 = self.cfg.pdc_thresholds
        if rho < t1:
            return self.cfg.pdc_stages[0]
        if rho < t2:
            return self.cfg.pdc_stages[1]
        return self.cfg.pdc_stages[2]

    def _hsw_weight(self, step: int, max_steps: int) -> float:
        lam = float(self.cfg.lambda_decay)
        if not (0.0 < lam < 1.0):
            return 1.0 / max(max_steps, 1)
        # step is 1..Nsup
        z = (1.0 - lam**max_steps) / (1.0 - lam)
        return (lam ** (step - 1)) / z

    def get_schedule(self, state: SchedulerState, model_aux: torch.Tensor | None = None) -> Schedule:
        n, T = self._pdc(state.progress)
        w = self._hsw_weight(state.supervision_step, state.max_supervision_steps)
        return Schedule(recursion_n=int(n), recursion_T=int(T), supervision_weight=float(w), halt_threshold=0.5)

