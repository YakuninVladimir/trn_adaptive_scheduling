from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from diplom.runner.tasks.base import TaskBatch, collate_task_batches


class _TextLMDataset(Dataset[TaskBatch]):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer_name: str,
        seq_len: int,
        dataset_config: str | None = None,
        text_field: str | None = None,
        sample_fraction: float | None = None,
        max_samples: int | None = None,
    ) -> None:
        if dataset_config:
            self.ds = load_dataset(dataset_name, dataset_config, split=split)
        else:
            self.ds = load_dataset(dataset_name, split=split)
        if sample_fraction is not None:
            frac = float(sample_fraction)
            if not (0.0 < frac <= 1.0):
                raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")
            keep = max(1, int(len(self.ds) * frac))
            self.ds = self.ds.select(range(keep))
        if max_samples is not None:
            self.ds = self.ds.select(range(min(int(max_samples), len(self.ds))))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            # fallback for GPT-like tokenizers
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self.text_field = text_field

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> TaskBatch:
        ex = self.ds[int(idx)]
        text = ex.get(self.text_field) if self.text_field else None
        if text is None:
            text = ex.get("text")
            if text is None:
                # try common alternatives
                text = ex.get("content") or ex.get("sentence") or ex.get("review") or ""
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.seq_len + 1,
            return_tensors=None,
        )
        ids = torch.tensor(enc["input_ids"], dtype=torch.long)  # [seq_len+1]
        x = ids[: self.seq_len]
        y = ids[1 : self.seq_len + 1]
        mask = y != self.tokenizer.pad_token_id
        return TaskBatch(x_tokens=x, y=y, y_mask=mask)


@dataclass(frozen=True)
class TextLMTaskConfig:
    dataset_name: str
    dataset_config: str | None = None
    split_train: str = "train"
    split_val: str | None = None
    tokenizer: str = "distilbert-base-uncased"
    seq_len: int = 256
    text_field: str | None = None
    train_fraction: float | None = None
    val_fraction: float | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None


class TextLMTask:
    name = "text_lm"

    def __init__(self, cfg: TextLMTaskConfig) -> None:
        self.cfg = cfg

    def build_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader | None]:
        train_ds = _TextLMDataset(
            dataset_name=self.cfg.dataset_name,
            dataset_config=self.cfg.dataset_config,
            split=self.cfg.split_train,
            tokenizer_name=self.cfg.tokenizer,
            seq_len=self.cfg.seq_len,
            text_field=self.cfg.text_field,
            sample_fraction=self.cfg.train_fraction,
            max_samples=self.cfg.max_train_samples,
        )
        val_ds = (
            _TextLMDataset(
                dataset_name=self.cfg.dataset_name,
                dataset_config=self.cfg.dataset_config,
                split=self.cfg.split_val,
                tokenizer_name=self.cfg.tokenizer,
                seq_len=self.cfg.seq_len,
                text_field=self.cfg.text_field,
                sample_fraction=self.cfg.val_fraction,
                max_samples=self.cfg.max_val_samples,
            )
            if self.cfg.split_val
            else None
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_task_batches),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_task_batches)
            if val_ds
            else None,
        )

    def compute_loss(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor:
        y = batch.y.to(logits.device)
        mask = batch.y_mask.to(logits.device) if batch.y_mask is not None else None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
        if mask is None:
            return loss.mean()
        loss = loss[mask.view(-1)].mean()
        return loss

    def compute_metrics(self, logits: torch.Tensor, batch: TaskBatch) -> dict[str, float]:
        y = batch.y.to(logits.device)
        pred = torch.argmax(logits, dim=-1)
        correct = (pred == y)
        topk = min(5, logits.size(-1))
        topk_idx = torch.topk(logits, k=topk, dim=-1).indices
        top5_correct = (topk_idx == y.unsqueeze(-1)).any(dim=-1)
        if batch.y_mask is not None:
            mask = batch.y_mask.to(logits.device)
            acc = (correct & mask).sum().float() / mask.sum().float().clamp_min(1.0)
            top5_acc = (top5_correct & mask).sum().float() / mask.sum().float().clamp_min(1.0)
        else:
            acc = correct.float().mean()
            top5_acc = top5_correct.float().mean()
        return {"token_acc": float(acc.item()), "top5_acc": float(top5_acc.item())}

    def halt_targets(self, logits: torch.Tensor, batch: TaskBatch) -> torch.Tensor | None:
        _ = logits, batch
        # Text LM task currently does not define a halting supervision signal.
        return None

