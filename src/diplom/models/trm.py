from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from diplom.models.base import ModelOutput
from diplom.models.layers import LearnedPositionalEmbedding, MlpMixerBlock, MlpMixerConfig, TransformerBlock


@dataclass(frozen=True)
class TRMConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 2048
    dropout: float = 0.0
    norm: str = "rmsnorm"
    pos_encoding: str = "learned"  # learned|none

    # TRM core architecture choice
    use_attention: bool = True  # if false -> MLP-Mixer style token mixing
    mlp_t: bool = False  # alias for use_attention=False on fixed-size grids
    mixer_d_token: int = 256
    mixer_d_channel: int = 1024

    # recursion params (n, T)
    L_cycles: int = 6  # n
    H_cycles: int = 3  # T

    # deep supervision (handled in runner; kept here for config completeness)
    N_sup: int = 16

    # optional halting head (TRM paper uses BCE halt head)
    halt_head: bool = True


class _TRMCore(nn.Module):
    def __init__(self, cfg: TRMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(3 * cfg.d_model, cfg.d_model, bias=False)

        use_attention = bool(cfg.use_attention) and not bool(cfg.mlp_t)
        if use_attention:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=cfg.d_model,
                        n_heads=cfg.n_heads,
                        d_ff=cfg.d_ff,
                        dropout=cfg.dropout,
                        norm=cfg.norm,
                    )
                    for _ in range(cfg.n_layers)
                ]
            )
            self.mixer_blocks = None
        else:
            mixer_cfg = MlpMixerConfig(
                seq_len=cfg.seq_len,
                d_model=cfg.d_model,
                d_token_mlp=cfg.mixer_d_token,
                d_channel_mlp=cfg.mixer_d_channel,
                dropout=cfg.dropout,
                norm=cfg.norm,
            )
            self.mixer_blocks = nn.ModuleList([MlpMixerBlock(mixer_cfg) for _ in range(cfg.n_layers)])
            self.blocks = None

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x_cat)
        if self.blocks is not None:
            for b in self.blocks:
                x = b(x)
        else:
            assert self.mixer_blocks is not None
            for b in self.mixer_blocks:
                x = b(x)
        return x


class TRM(nn.Module):
    def __init__(self, cfg: TRMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = LearnedPositionalEmbedding(cfg.seq_len, cfg.d_model) if cfg.pos_encoding == "learned" else None

        self.core = _TRMCore(cfg)
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.halt_proj = nn.Linear(cfg.d_model, 1) if cfg.halt_head else None

        self.y0 = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.z0 = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.y0.expand(batch_size, -1, -1).to(device)
        z = self.z0.expand(batch_size, -1, -1).to(device)
        return y, z

    def _embed_input(self, x_tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(x_tokens)
        if self.pos is not None:
            x = self.pos(x)
        return x

    def _latent_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Update z n times given (x, y, z)
        for _ in range(n):
            z = self.core(torch.cat([x, y, z], dim=-1))
        # Update y given (y, z). Feed y twice to match input dim without introducing a second network.
        y = self.core(torch.cat([y, z, y], dim=-1))
        return y, z

    def _deep_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        # T-1 recursion processes without gradients, then 1 with gradients
        if T <= 0:
            return y, z
        if T > 1:
            with torch.no_grad():
                for _ in range(T - 1):
                    y, z = self._latent_recursion(x, y, z, n=n)
        y, z = self._latent_recursion(x, y, z, n=n)
        return y, z

    def forward(
        self,
        x_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        recursion_n: int | None = None,
        recursion_T: int | None = None,
    ) -> ModelOutput:
        """
        One TRM refinement pass (one deep-recursion) producing logits and updated state (y, z).

        Deep supervision loop (N_sup) is implemented in the training runner.
        """
        B, L = x_tokens.shape
        if L != self.cfg.seq_len:
            raise ValueError(f"TRM expects seq_len={self.cfg.seq_len}, got {L}")

        n = int(recursion_n) if recursion_n is not None else self.cfg.L_cycles
        T = int(recursion_T) if recursion_T is not None else self.cfg.H_cycles

        device = x_tokens.device
        x = self._embed_input(x_tokens)
        if state is None:
            y, z = self.init_state(B, device=device)
        else:
            y, z = state

        y, z = self._deep_recursion(x, y, z, n=n, T=T)

        logits = self.out_proj(y)
        aux = y.mean(dim=1)

        loss_parts: dict[str, torch.Tensor] = {}
        if self.halt_proj is not None:
            q = torch.sigmoid(self.halt_proj(aux)).squeeze(-1)  # [B]
            loss_parts["halt_prob"] = q

        return ModelOutput(logits=logits, aux_tensor=aux, state=(y, z), loss_parts=loss_parts)

