from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from diplom.models.base import ModelOutput
from diplom.models.layers import LearnedPositionalEmbedding, TransformerBlock


@dataclass(frozen=True)
class HRMConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers_L: int = 4
    n_layers_H: int = 4
    dropout: float = 0.0
    norm: str = "rmsnorm"
    pos_encoding: str = "learned"  # learned|none

    # recursion (paper naming often uses (n, T))
    L_cycles: int = 2  # n: low-level steps per high-level update
    H_cycles: int = 2  # T: number of high-level updates

    # deep supervision (handled in runner; kept here for config completeness)
    N_sup: int = 16

    # gradient approximation
    one_step_grad_approx: bool = True

    # optional halting head (for ACT-like training)
    halt_head: bool = False


class _RecurrentModule(nn.Module):
    def __init__(self, cfg: HRMConfig, n_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    norm=cfg.norm,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class HRM(nn.Module):
    def __init__(self, cfg: HRMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = LearnedPositionalEmbedding(cfg.seq_len, cfg.d_model) if cfg.pos_encoding == "learned" else None

        self.f_L = _RecurrentModule(cfg, n_layers=cfg.n_layers_L)
        self.f_H = _RecurrentModule(cfg, n_layers=cfg.n_layers_H)

        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.zH0 = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.zL0 = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))

        self.halt_proj = nn.Linear(cfg.d_model, 1) if cfg.halt_head else None

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        zH = self.zH0.expand(batch_size, -1, -1).to(device)
        zL = self.zL0.expand(batch_size, -1, -1).to(device)
        return zH, zL

    def _embed_input(self, x_tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(x_tokens)
        if self.pos is not None:
            x = self.pos(x)
        return x

    def forward(
        self,
        x_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        recursion_n: int | None = None,
        recursion_T: int | None = None,
    ) -> ModelOutput:
        """
        One HRM forward-pass producing logits and the updated state (zH, zL).

        `recursion_n` and `recursion_T` can be overridden by a scheduler.
        """
        B, L = x_tokens.shape
        if L != self.cfg.seq_len:
            raise ValueError(f"HRM expects seq_len={self.cfg.seq_len}, got {L}")

        n = int(recursion_n) if recursion_n is not None else self.cfg.L_cycles
        T = int(recursion_T) if recursion_T is not None else self.cfg.H_cycles

        device = x_tokens.device
        x = self._embed_input(x_tokens)
        if state is None:
            zH, zL = self.init_state(B, device=device)
        else:
            zH, zL = state

        total_steps = T * n

        if self.cfg.one_step_grad_approx and total_steps >= 2:
            with torch.no_grad():
                for i in range(total_steps - 1):
                    # low step
                    zL = self.f_L(x + zH + zL)
                    if (i + 1) % n == 0:
                        zH = self.f_H(zH + zL)
            # 1-step grad
            zL = self.f_L(x + zH + zL)
            zH = self.f_H(zH + zL)
        else:
            for i in range(total_steps):
                zL = self.f_L(x + zH + zL)
                if (i + 1) % n == 0:
                    zH = self.f_H(zH + zL)

        logits = self.out_proj(zH)
        aux = zH.mean(dim=1)

        loss_parts: dict[str, torch.Tensor] = {}
        if self.halt_proj is not None:
            q = torch.sigmoid(self.halt_proj(aux)).squeeze(-1)  # [B]
            loss_parts["halt_prob"] = q

        return ModelOutput(logits=logits, aux_tensor=aux, state=(zH, zL), loss_parts=loss_parts)

