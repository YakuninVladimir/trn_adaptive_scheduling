from __future__ import annotations

from unittest.mock import patch

import torch

from diplom.runner.tasks.base import TaskBatch
from diplom.runner.tasks.text_lm_task import TextLMTask, TextLMTaskConfig


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, text, truncation, padding, max_length, return_tensors):
        _ = text, truncation, padding, return_tensors
        return {"input_ids": [1] * max_length}


def test_text_task_passes_dataset_config():
    calls = []

    class _FakeDs:
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            _ = idx
            return {"text": "hello world"}

        def select(self, rng):
            _ = rng
            return self

    def _fake_load_dataset(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return _FakeDs()

    cfg = TextLMTaskConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        split_train="train",
        split_val="validation",
        tokenizer="distilbert-base-uncased",
        seq_len=8,
        max_train_samples=2,
        max_val_samples=2,
    )
    with patch("diplom.runner.tasks.text_lm_task.load_dataset", side_effect=_fake_load_dataset):
        with patch("diplom.runner.tasks.text_lm_task.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()):
            task = TextLMTask(cfg)
            train_dl, val_dl = task.build_dataloaders(batch_size=2)
            b = next(iter(train_dl))
            assert isinstance(b.x_tokens, torch.Tensor)
            assert val_dl is not None
    assert calls, "load_dataset should be called"
    assert calls[0][0] == "wikitext"
    assert calls[0][1][0] == "wikitext-103-raw-v1"


def test_text_task_fraction_selects_subset():
    class _FakeDs:
        def __init__(self, n: int = 10):
            self._n = n
            self._rows = [{"text": "hello"} for _ in range(n)]

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, idx):
            return self._rows[idx]

        def select(self, rng):
            idxs = list(rng)
            out = _FakeDs(0)
            out._rows = [self._rows[i] for i in idxs]
            return out

    cfg = TextLMTaskConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        split_train="train",
        split_val="validation",
        tokenizer="distilbert-base-uncased",
        seq_len=8,
        train_fraction=0.2,
        val_fraction=0.3,
    )
    with patch("diplom.runner.tasks.text_lm_task.load_dataset", return_value=_FakeDs(10)):
        with patch("diplom.runner.tasks.text_lm_task.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()):
            task = TextLMTask(cfg)
            train_dl, val_dl = task.build_dataloaders(batch_size=2)
    assert len(train_dl.dataset) == 2
    assert val_dl is not None
    assert len(val_dl.dataset) == 3


def test_text_metrics_include_top5_accuracy():
    cfg = TextLMTaskConfig(
        dataset_name="wikitext",
        tokenizer="distilbert-base-uncased",
        seq_len=4,
    )
    task = TextLMTask(cfg)
    logits = torch.full((2, 4, 6), -100.0)
    y = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 1, 0, 2],
        ],
        dtype=torch.long,
    )
    # Keep true class in top-5, while argmax is wrong for most positions.
    logits[..., 0] = 5.0  # wrong top-1 baseline
    logits.scatter_(-1, y.unsqueeze(-1), 4.0)  # true class is second-best => in top-5
    logits[0, 0, 1] = 6.0  # one correct top-1 token
    mask = torch.ones_like(y, dtype=torch.bool)
    batch = TaskBatch(x_tokens=torch.zeros_like(y), y=y, y_mask=mask)
    m = task.compute_metrics(logits, batch)
    assert "token_acc" in m
    assert "top5_acc" in m
    assert m["top5_acc"] >= m["token_acc"]
