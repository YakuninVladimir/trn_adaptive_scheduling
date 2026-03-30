from __future__ import annotations

from diplom.schedulers.base import SchedulerState
from diplom.schedulers.cgar import CGARScheduler, CGARSchedulerConfig


def test_cgar_pdc_stages_and_hsw_weights():
    sch = CGARScheduler(
        CGARSchedulerConfig(
            pdc_thresholds=(0.3, 0.6),
            pdc_stages=((2, 1), (4, 2), (6, 3)),
            lambda_decay=0.7,
        )
    )

    s1 = SchedulerState(
        epoch=1,
        max_epochs=10,
        global_step=1,
        max_steps=100,
        supervision_step=1,
        max_supervision_steps=16,
        task_name="sudoku",
    )
    a1 = sch.get_schedule(s1, model_aux=None)
    assert (a1.recursion_n, a1.recursion_T) == (2, 1)
    assert 0.0 < a1.supervision_weight <= 1.0

    s2 = SchedulerState(**{**s1.__dict__, "global_step": 40})
    a2 = sch.get_schedule(s2, model_aux=None)
    assert (a2.recursion_n, a2.recursion_T) == (4, 2)

    s3 = SchedulerState(**{**s1.__dict__, "global_step": 90})
    a3 = sch.get_schedule(s3, model_aux=None)
    assert (a3.recursion_n, a3.recursion_T) == (6, 3)

    # HSW should decay with step (step 1 > step 2)
    s_step2 = SchedulerState(**{**s1.__dict__, "supervision_step": 2})
    a_step2 = sch.get_schedule(s_step2, model_aux=None)
    assert a1.supervision_weight > a_step2.supervision_weight

