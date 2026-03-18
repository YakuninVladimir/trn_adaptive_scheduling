from __future__ import annotations

from dataclasses import fields
from typing import Any, TypeVar

import torch

from diplom.models.hrm import HRM, HRMConfig
from diplom.models.trm import TRM, TRMConfig
from diplom.models.trm_oracle import TRMOracle, TRMOracleConfig
from diplom.runner.tasks.sudoku_task import SudokuTask, SudokuTaskConfig
from diplom.runner.tasks.text_lm_task import TextLMTask, TextLMTaskConfig
from diplom.runner.tasks.timeseries_task import TimeSeriesTask, TimeSeriesTaskConfig
from diplom.schedulers.cgar import CGARScheduler, CGARSchedulerConfig
from diplom.schedulers.fixed import FixedScheduler, FixedSchedulerConfig

T = TypeVar("T")


def _filter_kwargs(dc_type: type[T], raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(dc_type)}
    return {k: v for k, v in raw.items() if k in allowed}


def build_task(task_raw: dict[str, Any]):
    name = (task_raw.get("name") or "").lower()
    if name == "sudoku":
        cfg = SudokuTaskConfig(**_filter_kwargs(SudokuTaskConfig, task_raw))
        return SudokuTask(cfg)
    if name in ("text_lm", "wikitext_lm"):
        # accept either dataset_name or name
        if "dataset_name" not in task_raw and "name" in task_raw:
            task_raw = dict(task_raw)
            task_raw["dataset_name"] = task_raw["name"]
        cfg = TextLMTaskConfig(**_filter_kwargs(TextLMTaskConfig, task_raw))
        return TextLMTask(cfg)
    if name.startswith("timeseries"):
        mode = "tokenized"
        if name.endswith("regression"):
            mode = "regression"
        cfg_dict = dict(task_raw)
        cfg_dict.setdefault("mode", mode)
        cfg = TimeSeriesTaskConfig(**_filter_kwargs(TimeSeriesTaskConfig, cfg_dict))
        return TimeSeriesTask(cfg)
    raise ValueError(f"Unknown task name: {name!r}")


def build_model(model_raw: dict[str, Any]) -> torch.nn.Module:
    name = (model_raw.get("name") or "").lower()
    if name == "hrm":
        cfg = HRMConfig(**_filter_kwargs(HRMConfig, model_raw))
        return HRM(cfg)
    if name == "trm":
        cfg = TRMConfig(**_filter_kwargs(TRMConfig, model_raw))
        return TRM(cfg)
    if name in ("trm_oracle", "trm-lookahead"):
        cfg = TRMOracleConfig(**_filter_kwargs(TRMOracleConfig, model_raw))
        return TRMOracle(cfg)
    raise ValueError(f"Unknown model name: {name!r}")


def build_scheduler(sched_raw: dict[str, Any]):
    name = (sched_raw.get("name") or "").lower()
    if name == "fixed":
        cfg = FixedSchedulerConfig(**_filter_kwargs(FixedSchedulerConfig, sched_raw))
        return FixedScheduler(cfg)
    if name == "cgar":
        cfg_dict = dict(sched_raw)
        if "pdc_thresholds" in cfg_dict and isinstance(cfg_dict["pdc_thresholds"], list):
            cfg_dict["pdc_thresholds"] = tuple(float(x) for x in cfg_dict["pdc_thresholds"])
        if "pdc_stages" in cfg_dict and isinstance(cfg_dict["pdc_stages"], list):
            cfg_dict["pdc_stages"] = tuple(tuple(int(a) for a in pair) for pair in cfg_dict["pdc_stages"])
        cfg = CGARSchedulerConfig(**_filter_kwargs(CGARSchedulerConfig, cfg_dict))
        return CGARScheduler(cfg)
    raise ValueError(f"Unknown scheduler name: {name!r}")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

