from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diplom.models.base import ModelOutput
from diplom.models.layers import LearnedPositionalEmbedding, TransformerBlock
from diplom.models.trm import TRM, TRMConfig


@dataclass(frozen=True)
class TRMOracleConfig(TRMConfig):
    # Maximum prefix length passed to oracle transformer.
    oracle_max_steps: int = 16
    # Predictive horizon (how many steps ahead to score from current prefix).
    oracle_horizon: int = 8
    oracle_n_heads: int = 4
    oracle_n_layers: int = 2
    oracle_d_ff: int = 1024
    oracle_dropout: float = 0.0
    oracle_loss_weight: float = 1.0


class _OracleLookaheadHead(nn.Module):
    """
    Predicts distribution over which delta step is optimal.

    Input: aux history tensor [B, T, D]
    Output: logits over deltas [B, H], where index 0 means "stop now".
    """

    def __init__(
        self,
        d_model: int,
        history_max_steps: int,
        horizon: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.history_max_steps = history_max_steps
        self.horizon = horizon
        self.pos = LearnedPositionalEmbedding(history_max_steps, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    norm="rmsnorm",
                )
                for _ in range(n_layers)
            ]
        )
        self.pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, horizon),
        )

    def forward(self, aux_history: torch.Tensor) -> torch.Tensor:
        B, T, D = aux_history.shape
        if T > self.history_max_steps:
            raise ValueError(f"aux_history length {T} exceeds oracle_max_steps={self.history_max_steps}")
        if T < self.history_max_steps:
            pad = torch.zeros(B, self.history_max_steps - T, D, device=aux_history.device, dtype=aux_history.dtype)
            x = torch.cat([aux_history, pad], dim=1)
        else:
            x = aux_history
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)

        # attention pooling over time
        score = torch.einsum("btd,d->bt", x, self.pool_query)
        attn = torch.softmax(score, dim=1)
        pooled = torch.einsum("bt,btd->bd", attn, x)
        logits = self.mlp(pooled)
        return logits  # [B, horizon]


class TRMOracle(TRM):
    """
    TRM + lookahead oracle head.

    Training intent:
    1) Run full rollout for T_max supervision steps.
    2) Use collected aux history to train oracle head to predict best delta in
       a local window, including delta=0 ("stop now").
    """

    requires_full_rollout: bool = True

    def __init__(self, cfg: TRMOracleConfig) -> None:
        super().__init__(cfg)
        self.cfg_oracle = cfg
        self.oracle_head = _OracleLookaheadHead(
            d_model=cfg.d_model,
            history_max_steps=cfg.oracle_max_steps,
            horizon=cfg.oracle_horizon,
            n_heads=cfg.oracle_n_heads,
            n_layers=cfg.oracle_n_layers,
            d_ff=cfg.oracle_d_ff,
            dropout=cfg.oracle_dropout,
        )

    def oracle_parameters(self):
        return self.oracle_head.parameters()

    def backbone_parameters(self):
        oracle_ids = {id(p) for p in self.oracle_head.parameters()}
        for p in self.parameters():
            if id(p) not in oracle_ids:
                yield p

    def oracle_logits(self, aux_history: torch.Tensor) -> torch.Tensor:
        return self.oracle_head(aux_history)

    def oracle_loss(self, aux_history: torch.Tensor, target_delta_idx: torch.Tensor, valid_horizon: int | None = None) -> torch.Tensor:
        # target_delta_idx: [B], values in [0, H-1], with idx=0 => stop now.
        logits = self.oracle_logits(aux_history)
        if valid_horizon is not None:
            logits = logits[:, :valid_horizon]
        return F.cross_entropy(logits, target_delta_idx.long())

    def oracle_loss_from_rollout(self, aux_history_full: torch.Tensor, per_step_psloss: torch.Tensor) -> torch.Tensor:
        """
        Train oracle on prefix -> best future delta mapping.

        aux_history_full: [B, T, D]
        per_step_psloss:  [T, B] (lower is better)
        """
        B, T, _ = aux_history_full.shape
        H = int(self.cfg_oracle.oracle_horizon)
        losses: list[torch.Tensor] = []
        # Current step s predicts best delta in [0..H-1] including "stop now".
        for s in range(T):
            valid_h = min(H, T - s)
            if valid_h <= 0:
                continue
            prefix = aux_history_full[:, : s + 1, :]
            future = per_step_psloss[s : s + valid_h, :]  # [valid_h, B], includes current
            target_delta_idx = torch.argmin(future, dim=0)  # [B], 0..valid_h-1
            l = self.oracle_loss(prefix, target_delta_idx, valid_horizon=valid_h)
            losses.append(l)
        if not losses:
            return torch.zeros((), device=aux_history_full.device)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def choose_delta(
        self,
        aux_history: torch.Tensor,
        *,
        valid_horizon: int,
        policy: str,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> int:
        """
        Select one global delta for the whole batch from oracle distribution.

        policy:
          - greedy: argmax over mean batch logits.
          - sampling: sample from softmax(mean_logits / temperature).
        """
        logits = self.oracle_logits(aux_history)[:, :valid_horizon]  # [B, valid_horizon]
        pooled = logits.mean(dim=0)  # [valid_horizon], batch-level decision
        if policy == "greedy":
            return int(torch.argmax(pooled).item())
        if policy == "sampling":
            t = max(float(temperature), 1e-6)
            probs = torch.softmax(pooled / t, dim=0)
            idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator)
            return int(idx.item())
        raise ValueError(f"Unknown oracle policy: {policy}")

    def forward(
        self,
        x_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        recursion_n: int | None = None,
        recursion_T: int | None = None,
    ) -> ModelOutput:
        out = super().forward(
            x_tokens=x_tokens,
            state=state,
            recursion_n=recursion_n,
            recursion_T=recursion_T,
        )
        return out

