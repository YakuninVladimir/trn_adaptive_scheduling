from __future__ import annotations

from unittest.mock import patch

import torch

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
