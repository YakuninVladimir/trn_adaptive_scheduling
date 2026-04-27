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
    oracle_target_mode: str = "delta"  # delta|distribution
    oracle_distribution_model: str = "finite_discrete"
    oracle_distribution_components: int = 2
    oracle_distribution_lambda: float = 0.0
    oracle_distribution_beta: float = 0.5
    oracle_distribution_epsilon: float = 1e-6


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

    def encode(self, aux_history: torch.Tensor) -> torch.Tensor:
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
        score = torch.einsum("btd,d->bt", x, self.pool_query)
        attn = torch.softmax(score, dim=1)
        pooled = torch.einsum("bt,btd->bd", attn, x)
        return pooled

    def forward(self, aux_history: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(aux_history)
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
        horizon = int(cfg.oracle_horizon)
        m = int(max(cfg.oracle_distribution_components, 1))
        d = int(cfg.d_model)
        self.dist_finite = nn.Linear(d, horizon)
        self.dist_mix_geom_pi = nn.Linear(d, m)
        self.dist_mix_geom_rho = nn.Linear(d, m)
        self.dist_mix_exp_pi = nn.Linear(d, m)
        self.dist_mix_exp_alpha = nn.Linear(d, m)
        self.dist_power = nn.Linear(d, 2)  # a, c
        self.dist_nb = nn.Linear(d, 2)  # r, rho
        self.dist_logn = nn.Linear(d, 2)  # mu, sigma
        self.dist_hybrid_alpha = nn.Linear(d, 1)

    def oracle_parameters(self):
        for p in self.oracle_head.parameters():
            yield p
        for name in (
            "dist_finite",
            "dist_mix_geom_pi",
            "dist_mix_geom_rho",
            "dist_mix_exp_pi",
            "dist_mix_exp_alpha",
            "dist_power",
            "dist_nb",
            "dist_logn",
            "dist_hybrid_alpha",
        ):
            mod = getattr(self, name)
            yield from mod.parameters()

    def backbone_parameters(self):
        oracle_ids = {id(p) for p in self.oracle_head.parameters()}
        for p in self.parameters():
            if id(p) not in oracle_ids:
                yield p

    def oracle_logits(self, aux_history: torch.Tensor) -> torch.Tensor:
        return self.oracle_head(aux_history)

    def _oracle_features(self, aux_history: torch.Tensor) -> torch.Tensor:
        return self.oracle_head.encode(aux_history)

    def _renorm(self, pmf: torch.Tensor, eps: float) -> torch.Tensor:
        pmf = pmf.clamp_min(eps)
        return pmf / pmf.sum(dim=-1, keepdim=True).clamp_min(eps)

    def _finite_discrete(self, feat: torch.Tensor, K: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.dist_finite(feat)[:, :K]
        pmf = torch.softmax(logits, dim=-1)
        return pmf, {"logits": logits}

    def _smoothed_finite(self, feat: torch.Tensor, K: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._finite_discrete(feat, K)

    def _mix_geometric(self, feat: torch.Tensor, K: int, eps: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        M = int(max(self.cfg_oracle.oracle_distribution_components, 1))
        pi_logits = self.dist_mix_geom_pi(feat)[:, :M]
        rho_logits = self.dist_mix_geom_rho(feat)[:, :M]
        pi = torch.softmax(pi_logits, dim=-1)
        rho = torch.sigmoid(rho_logits).clamp(eps, 1.0 - eps)
        j = torch.arange(1, K + 1, device=feat.device, dtype=feat.dtype)[None, None, :]
        geom = ((1.0 - rho[:, :, None]) ** (j - 1.0)) * rho[:, :, None]
        pmf = torch.sum(pi[:, :, None] * geom, dim=1)
        pmf = self._renorm(pmf, eps)
        return pmf, {"pi": pi, "rho": rho}

    def _mix_exponential(self, feat: torch.Tensor, K: int, eps: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        M = int(max(self.cfg_oracle.oracle_distribution_components, 1))
        pi_logits = self.dist_mix_exp_pi(feat)[:, :M]
        alpha_raw = self.dist_mix_exp_alpha(feat)[:, :M]
        pi = torch.softmax(pi_logits, dim=-1)
        alpha = F.softplus(alpha_raw) + eps
        j = torch.arange(1, K + 1, device=feat.device, dtype=feat.dtype)[None, None, :]
        part = torch.exp(-alpha[:, :, None] * (j - 1.0)) - torch.exp(-alpha[:, :, None] * j)
        pmf = torch.sum(pi[:, :, None] * part, dim=1)
        pmf = self._renorm(pmf, eps)
        return pmf, {"pi": pi, "alpha": alpha}

    def _power_tail(self, feat: torch.Tensor, K: int, eps: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.dist_power(feat)
        a = F.softplus(raw[:, 0]) + eps
        c = F.softplus(raw[:, 1])
        j = torch.arange(1, K + 1, device=feat.device, dtype=feat.dtype)[None, :]
        pmf = (j + c[:, None]).pow(-a[:, None])
        pmf = self._renorm(pmf, eps)
        return pmf, {"a": a, "c": c}

    def _negative_binomial(self, feat: torch.Tensor, K: int, eps: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.dist_nb(feat)
        r = F.softplus(raw[:, 0]) + 1.0
        rho = torch.sigmoid(raw[:, 1]).clamp(eps, 1.0 - eps)
        j = torch.arange(1, K + 1, device=feat.device, dtype=feat.dtype)[None, :]
        # log comb via lgamma for continuous r
        lg = torch.lgamma(j + r[:, None] - 1.0) - torch.lgamma(j) - torch.lgamma(r[:, None])
        log_p = lg + (j - 1.0) * torch.log1p(-rho[:, None]) + r[:, None] * torch.log(rho[:, None])
        pmf = torch.exp(log_p)
        pmf = self._renorm(pmf, eps)
        return pmf, {"r": r, "rho": rho}

    def _lognormal_discrete(self, feat: torch.Tensor, K: int, eps: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.dist_logn(feat)
        mu = raw[:, 0]
        sigma = F.softplus(raw[:, 1]) + eps
        j = torch.arange(1, K + 1, device=feat.device, dtype=feat.dtype)[None, :]
        z_hi = (torch.log(j + 1.0) - mu[:, None]) / sigma[:, None]
        z_lo = (torch.log(j) - mu[:, None]) / sigma[:, None]
        cdf_hi = 0.5 * (1.0 + torch.erf(z_hi / 1.4142135623730951))
        cdf_lo = 0.5 * (1.0 + torch.erf(z_lo / 1.4142135623730951))
        pmf = (cdf_hi - cdf_lo).clamp_min(eps)
        pmf = self._renorm(pmf, eps)
        return pmf, {"mu": mu, "sigma": sigma}

    def _hybrid_finite_tail(self, feat: torch.Tensor, K: int, eps: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        finite, finite_params = self._finite_discrete(feat, K)
        tail, tail_params = self._power_tail(feat, K, eps)
        alpha = torch.sigmoid(self.dist_hybrid_alpha(feat))  # [B,1]
        pmf = alpha * finite + (1.0 - alpha) * tail
        pmf = self._renorm(pmf, eps)
        params = {"alpha": alpha.squeeze(-1), **finite_params, **tail_params}
        return pmf, params

    def oracle_distribution(
        self,
        aux_history: torch.Tensor,
        *,
        valid_horizon: int | None = None,
        distribution_model: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        K = int(valid_horizon) if valid_horizon is not None else int(self.cfg_oracle.oracle_horizon)
        K = max(1, min(K, int(self.cfg_oracle.oracle_horizon)))
        dist = (distribution_model or self.cfg_oracle.oracle_distribution_model or "finite_discrete").lower()
        feat = self._oracle_features(aux_history)
        eps = float(self.cfg_oracle.oracle_distribution_epsilon)
        if dist in ("finite_discrete", "discrete"):
            return self._finite_discrete(feat, K)
        if dist in ("smoothed_loss", "smoothed"):
            return self._smoothed_finite(feat, K)
        if dist in ("mixture_geometric", "mix_geometric"):
            return self._mix_geometric(feat, K, eps)
        if dist in ("mixture_exponential", "mix_exponential", "discrete_exponential"):
            return self._mix_exponential(feat, K, eps)
        if dist in ("power", "power_law"):
            return self._power_tail(feat, K, eps)
        if dist in ("negative_binomial", "nbinom"):
            return self._negative_binomial(feat, K, eps)
        if dist in ("lognormal", "discrete_lognormal"):
            return self._lognormal_discrete(feat, K, eps)
        if dist in ("hybrid", "hybrid_finite_tail"):
            return self._hybrid_finite_tail(feat, K, eps)
        raise ValueError(f"Unknown oracle distribution model: {dist}")

    def oracle_loss(self, aux_history: torch.Tensor, target_delta_idx: torch.Tensor, valid_horizon: int | None = None) -> torch.Tensor:
        # target_delta_idx: [B], values in [0, H-1], with idx=0 => stop now.
        logits = self.oracle_logits(aux_history)
        if valid_horizon is not None:
            logits = logits[:, :valid_horizon]
        return F.cross_entropy(logits, target_delta_idx.long())

    def oracle_distribution_loss(
        self,
        aux_history: torch.Tensor,
        target_pmf: torch.Tensor,
        valid_horizon: int | None = None,
        distribution_model: str | None = None,
    ) -> torch.Tensor:
        pred, _ = self.oracle_distribution(
            aux_history,
            valid_horizon=valid_horizon,
            distribution_model=distribution_model,
        )
        target = target_pmf[:, : pred.size(1)]
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ce = -(target * torch.log(pred.clamp_min(1e-8))).sum(dim=-1)
        return ce.mean()

    def oracle_loss_from_rollout(self, aux_history_full: torch.Tensor, per_step_psloss: torch.Tensor) -> torch.Tensor:
        """
        Train oracle on prefix -> best future delta mapping.

        aux_history_full: [B, T, D]
        per_step_psloss:  [T, B] (lower is better)
        """
        mode = (self.cfg_oracle.oracle_target_mode or "delta").lower()
        if mode == "delta":
            return self._oracle_delta_loss_from_rollout(aux_history_full, per_step_psloss)
        return self.oracle_distribution_loss_from_rollout(aux_history_full, per_step_psloss)

    def _oracle_delta_loss_from_rollout(self, aux_history_full: torch.Tensor, per_step_psloss: torch.Tensor) -> torch.Tensor:
        B, T, _ = aux_history_full.shape
        H = int(self.cfg_oracle.oracle_horizon)
        losses: list[torch.Tensor] = []
        for s in range(T):
            valid_h = min(H, T - s)
            if valid_h <= 0:
                continue
            prefix = aux_history_full[:, : s + 1, :]
            future = per_step_psloss[s : s + valid_h, :]
            target_delta_idx = torch.argmin(future, dim=0)
            losses.append(self.oracle_loss(prefix, target_delta_idx, valid_horizon=valid_h))
        if not losses:
            return torch.zeros((), device=aux_history_full.device)
        return torch.stack(losses).mean()

    def oracle_distribution_loss_from_rollout(self, aux_history_full: torch.Tensor, per_step_psloss: torch.Tensor) -> torch.Tensor:
        B, T, _ = aux_history_full.shape
        H = min(int(self.cfg_oracle.oracle_horizon), T)
        device = aux_history_full.device
        losses: list[torch.Tensor] = []
        step_cost = torch.arange(1, T + 1, device=device, dtype=per_step_psloss.dtype)[:, None]
        costs = per_step_psloss + float(self.cfg_oracle.oracle_distribution_lambda) * step_cost
        best_idx = torch.argmin(costs, dim=0)  # [B], absolute best 0..T-1
        beta = max(float(self.cfg_oracle.oracle_distribution_beta), 1e-6)
        dist_model = (self.cfg_oracle.oracle_distribution_model or "finite_discrete").lower()
        for s in range(T):
            prefix = aux_history_full[:, : s + 1, :]
            target = torch.zeros(B, H, device=device, dtype=per_step_psloss.dtype)
            if dist_model in ("smoothed_loss", "smoothed"):
                smoothed = torch.softmax(-costs[:H, :].transpose(0, 1) / beta, dim=-1)
                target = smoothed
            else:
                clamped = best_idx.clamp_max(H - 1)
                target.scatter_(1, clamped[:, None], 1.0)
            l = self.oracle_distribution_loss(prefix, target, valid_horizon=H, distribution_model=dist_model)
            losses.append(l)
        if not losses:
            return torch.zeros((), device=device)
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

