from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def make_norm(kind: str, d_model: int) -> nn.Module:
    kind = kind.lower()
    if kind == "layernorm":
        return nn.LayerNorm(d_model)
    if kind == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"Unknown norm kind: {kind}")


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        norm: str = "rmsnorm",
    ) -> None:
        super().__init__()
        self.norm1 = make_norm(norm, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = make_norm(norm, d_model)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(h)
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout(h)
        return x


@dataclass(frozen=True)
class MlpMixerConfig:
    seq_len: int
    d_model: int
    d_token_mlp: int
    d_channel_mlp: int
    dropout: float = 0.0
    norm: str = "rmsnorm"


class MlpMixerBlock(nn.Module):
    """
    Minimal MLP-Mixer style block for fixed-length sequences.
    """

    def __init__(self, cfg: MlpMixerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm1 = make_norm(cfg.norm, cfg.d_model)
        self.token_mlp_1 = nn.Linear(cfg.seq_len, cfg.d_token_mlp)
        self.token_mlp_2 = nn.Linear(cfg.d_token_mlp, cfg.seq_len)
        self.norm2 = make_norm(cfg.norm, cfg.d_model)
        self.channel_mlp = FeedForward(cfg.d_model, d_ff=cfg.d_channel_mlp, dropout=cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # token mixing
        h = self.norm1(x)
        h = h.transpose(1, 2)  # [B, D, L]
        h = self.token_mlp_1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.token_mlp_2(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)  # [B, L, D]
        x = x + h

        # channel mixing
        h = self.norm2(x)
        h = self.channel_mlp(h)
        h = self.dropout(h)
        x = x + h
        return x


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.pos = nn.Embedding(seq_len, d_model)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.size(1)}")
        idx = torch.arange(self.seq_len, device=x.device)
        return x + self.pos(idx)[None, :, :]

