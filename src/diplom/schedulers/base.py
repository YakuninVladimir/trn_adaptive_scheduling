from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SchedulerState:
    epoch: int
    max_epochs: int
    global_step: int
    max_steps: int
    supervision_step: int  # 1..N_sup
    max_supervision_steps: int
    task_name: str | None = None

    @property
    def progress(self) -> float:
        # Prefer step-based progress because curriculum scheduling is tied to
        # optimization updates, not to epoch counters (which can be very long).
        if self.max_steps > 0:
            return min(max(self.global_step / self.max_steps, 0.0), 1.0)
        if self.max_epochs > 0:
            return min(max(self.epoch / self.max_epochs, 0.0), 1.0)
        return 0.0


@dataclass(frozen=True)
class Schedule:
    recursion_n: int
    recursion_T: int
    supervision_weight: float
    halt_threshold: float | None = None


class RecursionScheduler(ABC):
    @abstractmethod
    def get_schedule(self, state: SchedulerState, model_aux: torch.Tensor | None = None) -> Schedule:
        """
        Compute recursion parameters and per-supervision-step weighting.

        `model_aux` is an optional tensor exported by the model (e.g., pooled hidden state),
        enabling future learnable / model-dependent schedulers.
        """

