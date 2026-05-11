from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM

from diplom.models.base import ModelOutput
from diplom.models.trm import TRMConfig, _TRMCore
from diplom.models.trm_oracle import TRMOracle, TRMOracleConfig


@dataclass(frozen=True)
class FrozenLLMTRMOracleConfig(TRMOracleConfig):
    """TRMOracle whose recurrence input ``x`` is a linear map of frozen LM hidden states (e.g. Falcon)."""

    base_model_name: str = ""
    freeze_backbone: bool = True


class FrozenLLMTRMOracle(TRMOracle):
    """
    Same training/eval contract as ``TRMOracle``, but ``x`` at each supervision step comes from
    ``Linear(backbone(last_hidden_state))`` instead of token embeddings + positional encoding.
    """

    requires_full_rollout: bool = True

    def __init__(self, cfg: FrozenLLMTRMOracleConfig) -> None:
        if not str(cfg.base_model_name).strip():
            raise ValueError("frozen_llm_trm_oracle requires non-empty model.base_model_name (HF model id)")
        super().__init__(cfg)
        self._llm_oracle_cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.base_model_name)
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        hidden_size = int(self.backbone.config.hidden_size)
        self.backbone_in_proj = nn.Linear(hidden_size, cfg.d_model)
        for p in self.embed.parameters():
            p.requires_grad = False
        if self.pos is not None:
            for p in self.pos.parameters():
                p.requires_grad = False

    def _input_embeddings(self, x_tokens: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self._llm_oracle_cfg.freeze_backbone else torch.enable_grad()
        with ctx:
            out = self.backbone(input_ids=x_tokens, output_hidden_states=False, return_dict=True)
            h = out.last_hidden_state
        return self.backbone_in_proj(h)

    def forward(
        self,
        x_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        recursion_n: int | None = None,
        recursion_T: int | None = None,
    ) -> ModelOutput:
        B, L = x_tokens.shape
        if L != self.cfg.seq_len:
            raise ValueError(f"TRM expects seq_len={self.cfg.seq_len}, got {L}")

        n = int(recursion_n) if recursion_n is not None else self.cfg.L_cycles
        T = int(recursion_T) if recursion_T is not None else self.cfg.H_cycles

        device = x_tokens.device
        x = self._input_embeddings(x_tokens)
        if state is None:
            y, z = self.init_state(B, device=device)
        else:
            y, z = state

        y, z = self._deep_recursion(x, y, z, n=n, T=T)

        logits = self.out_proj(y)
        aux = y.mean(dim=1)

        loss_parts: dict[str, torch.Tensor] = {}
        if self.halt_proj is not None:
            halt_logit = self.halt_proj(aux).squeeze(-1)
            loss_parts["halt_logit"] = halt_logit
            loss_parts["halt_prob"] = torch.sigmoid(halt_logit)

        return ModelOutput(logits=logits, aux_tensor=aux, state=(y, z), loss_parts=loss_parts)

    def backbone_parameters(self):
        oracle_ids = {id(p) for p in self.oracle_parameters()}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in oracle_ids:
                continue
            if name.startswith("embed.") or name.startswith("pos."):
                continue
            yield p


@dataclass(frozen=True)
class FrozenLLMTRMConfig:
    base_model_name: str
    seq_len: int
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 2048
    dropout: float = 0.0
    norm: str = "rmsnorm"
    use_attention: bool = True
    mlp_t: bool = False
    mixer_d_token: int = 256
    mixer_d_channel: int = 1024
    halt_head: bool = False
    freeze_backbone: bool = True
    correction_iterations_mode: str = "fixed"  # fixed|predicted
    correction_fixed_steps: int = 2
    correction_max_predicted_steps: int = 4
    L_cycles: int = 2
    H_cycles: int = 1
    N_sup: int = 1


class FrozenLLMTRM(nn.Module):
    def __init__(self, cfg: FrozenLLMTRMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.base_model_name)
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        hidden_size = int(self.backbone.config.hidden_size)
        self.in_proj = nn.Linear(hidden_size, cfg.d_model)
        trm_cfg = TRMConfig(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            norm=cfg.norm,
            use_attention=cfg.use_attention,
            mlp_t=cfg.mlp_t,
            mixer_d_token=cfg.mixer_d_token,
            mixer_d_channel=cfg.mixer_d_channel,
            halt_head=cfg.halt_head,
            L_cycles=cfg.L_cycles,
            H_cycles=cfg.H_cycles,
            N_sup=cfg.N_sup,
        )
        self.core = _TRMCore(trm_cfg)
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.halt_proj = nn.Linear(cfg.d_model, 1) if cfg.halt_head else None
        self.iter_proj = nn.Linear(cfg.d_model, cfg.correction_max_predicted_steps)
        self.y0 = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.z0 = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.y0.expand(batch_size, -1, -1).to(device)
        z = self.z0.expand(batch_size, -1, -1).to(device)
        return y, z

    def _latent_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(n):
            z = self.core(torch.cat([x, y, z], dim=-1))
        y = self.core(torch.cat([y, z, y], dim=-1))
        return y, z

    def _deep_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int, t_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        if t_steps <= 0:
            return y, z
        if t_steps > 1:
            with torch.no_grad():
                for _ in range(t_steps - 1):
                    y, z = self._latent_recursion(x, y, z, n=n)
        y, z = self._latent_recursion(x, y, z, n=n)
        return y, z

    def _resolve_correction_steps(self, x_embed: torch.Tensor) -> tuple[int, torch.Tensor | None]:
        mode = (self.cfg.correction_iterations_mode or "fixed").lower()
        if mode == "fixed":
            return max(int(self.cfg.correction_fixed_steps), 1), None
        pooled = x_embed.mean(dim=1)
        logits = self.iter_proj(pooled)
        pred = int(torch.argmax(logits.mean(dim=0)).item()) + 1
        pred = max(1, min(pred, int(self.cfg.correction_max_predicted_steps)))
        return pred, logits

    def forward(
        self,
        x_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        recursion_n: int | None = None,
        recursion_T: int | None = None,
    ) -> ModelOutput:
        if x_tokens.size(1) != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {x_tokens.size(1)}")
        with torch.no_grad() if self.cfg.freeze_backbone else torch.enable_grad():
            out = self.backbone(input_ids=x_tokens, output_hidden_states=False, return_dict=True)
            x = out.last_hidden_state
        x = self.in_proj(x)
        if state is None:
            y, z = self.init_state(x_tokens.size(0), device=x_tokens.device)
        else:
            y, z = state
        n = int(recursion_n) if recursion_n is not None else int(self.cfg.L_cycles)
        t_steps = int(recursion_T) if recursion_T is not None else int(self.cfg.H_cycles)
        corr_steps, iter_logits = self._resolve_correction_steps(x)
        for _ in range(corr_steps):
            y, z = self._deep_recursion(x, y, z, n=n, t_steps=t_steps)
        logits = self.out_proj(y)
        aux = y.mean(dim=1)
        loss_parts: dict[str, torch.Tensor] = {"correction_steps": torch.tensor(float(corr_steps), device=logits.device)}
        if iter_logits is not None:
            loss_parts["correction_iteration_logits"] = iter_logits
        if self.halt_proj is not None:
            halt_logit = self.halt_proj(aux).squeeze(-1)
            loss_parts["halt_logit"] = halt_logit
            loss_parts["halt_prob"] = torch.sigmoid(halt_logit)
        return ModelOutput(logits=logits, aux_tensor=aux, state=(y, z), loss_parts=loss_parts)


@dataclass(frozen=True)
class LoRATextLMConfig:
    base_model_name: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    N_sup: int = 1


class LoRATextLM(nn.Module):
    def __init__(self, cfg: LoRATextLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(cfg.lora_r),
            lora_alpha=int(cfg.lora_alpha),
            lora_dropout=float(cfg.lora_dropout),
            target_modules=list(cfg.lora_target_modules),
        )
        self.model = get_peft_model(base, lora_cfg)

    def forward(
        self,
        x_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        recursion_n: int | None = None,
        recursion_T: int | None = None,
    ) -> ModelOutput:
        _ = state, recursion_n, recursion_T
        out = self.model(input_ids=x_tokens, output_hidden_states=True, return_dict=True, use_cache=False)
        logits = out.logits
        hs = out.hidden_states[-1]
        return ModelOutput(logits=logits, aux_tensor=hs.mean(dim=1), state=None, loss_parts={})
