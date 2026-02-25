from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diplom.runner.tasks.base import TaskBatch, collate_task_batches


class _UniformQuantizer:
    def __init__(self, vmin: float, vmax: float, n_bins: int) -> None:
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.n_bins = int(n_bins)
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if not (self.vmax > self.vmin):
            self.vmax = self.vmin + 1e-6
        self.edges = np.linspace(self.vmin, self.vmax, self.n_bins + 1, dtype=np.float64)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2.0

    def encode(self, x: np.ndarray) -> np.ndarray:
        idx = np.digitize(x, self.edges[1:-1], right=False)
        return idx.astype(np.int64)


class _TimeSeriesWindowDataset(Dataset[TaskBatch]):
    def __init__(
        self,
        series: np.ndarray,
        window: int,
        horizon: int,
        quantizer: _UniformQuantizer,
        mode: str,
    ) -> None:
        self.series = series.astype(np.float32)
        self.window = int(window)
        self.horizon = int(horizon)
        self.quantizer = quantizer
        self.mode = mode
        self.max_i = len(self.series) - (self.window + self.horizon) + 1
        if self.max_i <= 0:
            raise ValueError("Series too short for window+horizon")

    def __len__(self) -> int:
        return self.max_i

    def __getitem__(self, idx: int) -> TaskBatch:
        i = int(idx)
        x_win = self.series[i : i + self.window]
        y_val = self.series[i + self.window + self.horizon - 1]

        x_tok = torch.tensor(self.quantizer.encode(x_win), dtype=torch.long)
        if self.mode == "tokenized":
            y_tok = torch.tensor(int(self.quantizer.encode(np.array([y_val]))[0]), dtype=torch.long)
            return TaskBatch(x_tokens=x_tok, y=y_tok)
        if self.mode == "regression":
            return TaskBatch(x_tokens=x_tok, y=torch.tensor(float(y_val), dtype=torch.float32))
        raise ValueError(f"Unknown mode: {self.mode}")


@dataclass(frozen=True)
class TimeSeriesTaskConfig:
    source: str  # stocks|csv
    tickers: list[str] | None = None
    csv_path: str | None = None
    feature: str = "Close"
    window: int = 64
    horizon: int = 1
    n_bins: int = 256
    mode: str = "tokenized"  # tokenized|regression


class TimeSeriesTask:
    name = "timeseries"

    def __init__(self, cfg: TimeSeriesTaskConfig) -> None:
        self.cfg = cfg
        self._centers_t: torch.Tensor | None = None

    def _load_series(self) -> np.ndarray:
        if self.cfg.source == "stocks":
            if not self.cfg.tickers:
                raise ValueError("tickers must be provided for stocks source")
            # Use first ticker for now; multi-series batching can be added later.
            t = self.cfg.tickers[0]
            path = Path("data/timeseries/stocks") / f"{t}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing stock CSV for {t}: {path}. Run diplom-data timeseries-stocks ...")
            df = pd.read_csv(path)
        elif self.cfg.source == "csv":
            if not self.cfg.csv_path:
                raise ValueError("csv_path must be provided for csv source")
            df = pd.read_csv(self.cfg.csv_path)
        else:
            raise ValueError(f"Unknown timeseries source: {self.cfg.source}")

        if self.cfg.feature not in df.columns:
            raise ValueError(f"Feature {self.cfg.feature} not in columns: {list(df.columns)}")
        series = df[self.cfg.feature].to_numpy(dtype=np.float32)
        series = series[~np.isnan(series)]
        return series

    def build_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader | None]:
        series = self._load_series()
        # split last 20% as val
        n = len(series)
        n_train = int(n * 0.8)
        train_series = series[:n_train]
        val_series = series[n_train - self.cfg.window - self.cfg.horizon :]

        q = _UniformQuantizer(vmin=float(np.min(train_series)), vmax=float(np.max(train_series)), n_bins=self.cfg.n_bins)
        self._centers_t = torch.tensor(q.centers.astype(np.float32))
        train_ds = _TimeSeriesWindowDataset(
            train_series, window=self.cfg.window, horizon=self.cfg.horizon, quantizer=q, mode=self.cfg.mode
        )
        val_ds = _TimeSeriesWindowDataset(
            val_series, window=self.cfg.window, horizon=self.cfg.horizon, quantizer=q, mode=self.cfg.mode
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_task_batches),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_task_batches),
        )

    def compute_loss(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor:
        if self.cfg.mode == "tokenized":
            # Use last position prediction for next-step bin
            y = batch.y.to(logits.device).long()  # [B]
            last = logits[:, -1, :]  # [B, V]
            return F.cross_entropy(last, y)
        if self.cfg.mode == "regression":
            y = batch.y.to(logits.device).float()  # [B]
            last = logits[:, -1, :]  # [B, V]
            if self._centers_t is None:
                raise RuntimeError("TimeSeriesTask not initialized (centers missing). Call build_dataloaders first.")
            centers = self._centers_t.to(logits.device)  # [V]
            prob = torch.softmax(last, dim=-1)
            pred = torch.sum(prob * centers[None, :], dim=-1)  # [B]
            return F.mse_loss(pred, y)
        raise ValueError(f"Unknown mode: {self.cfg.mode}")

    def compute_metrics(self, logits: torch.Tensor, batch: TaskBatch) -> dict[str, float]:
        if self.cfg.mode == "tokenized":
            y = batch.y.to(logits.device).long()
            pred = torch.argmax(logits[:, -1, :], dim=-1)
            acc = (pred == y).float().mean()
            return {"nextbin_acc": float(acc.item())}
        if self.cfg.mode == "regression":
            y = batch.y.to(logits.device).float()
            last = logits[:, -1, :]
            if self._centers_t is None:
                raise RuntimeError("TimeSeriesTask not initialized (centers missing). Call build_dataloaders first.")
            centers = self._centers_t.to(logits.device)
            prob = torch.softmax(last, dim=-1)
            pred = torch.sum(prob * centers[None, :], dim=-1)
            mae = torch.mean(torch.abs(pred - y))
            return {"mae": float(mae.item())}
        return {}

