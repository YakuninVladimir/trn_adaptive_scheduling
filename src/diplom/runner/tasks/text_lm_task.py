from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from diplom.runner.tasks.base import TaskBatch, collate_task_batches


class _TextLMDataset(Dataset[TaskBatch]):
    def __init__(self, dataset_name: str, split: str, tokenizer_name: str, seq_len: int) -> None:
        self.ds = load_dataset(dataset_name, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            # fallback for GPT-like tokenizers
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> TaskBatch:
        ex = self.ds[int(idx)]
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
    split_train: str = "train"
    split_val: str | None = None
    tokenizer: str = "distilbert-base-uncased"
    seq_len: int = 256


class TextLMTask:
    name = "text_lm"

    def __init__(self, cfg: TextLMTaskConfig) -> None:
        self.cfg = cfg

    def build_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader | None]:
        train_ds = _TextLMDataset(
            dataset_name=self.cfg.dataset_name,
            split=self.cfg.split_train,
            tokenizer_name=self.cfg.tokenizer,
            seq_len=self.cfg.seq_len,
        )
        val_ds = (
            _TextLMDataset(
                dataset_name=self.cfg.dataset_name,
                split=self.cfg.split_val,
                tokenizer_name=self.cfg.tokenizer,
                seq_len=self.cfg.seq_len,
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
        if batch.y_mask is not None:
            mask = batch.y_mask.to(logits.device)
            acc = (correct & mask).sum().float() / mask.sum().float().clamp_min(1.0)
        else:
            acc = correct.float().mean()
        return {"token_acc": float(acc.item())}

