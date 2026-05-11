# Oracle Distributions and Losses

This note documents the **current implementation** of TRM + oracle distribution modeling in this repository.

Relevant code:
- `src/diplom/models/trm.py`
- `src/diplom/models/trm_oracle.py`
- `src/diplom/runner/train.py`

### Oracle input (trajectory of `y`)

By default (`oracle_use_full_y: true` in `TRMOracleConfig`), the oracle sees a prefix of **full** latent `y` tensors shaped `[B, T, L, d_model]` (one `[B, L, d_model]` per supervision step, stacked over `T`). A small spatial encoder (Transformer layers over position `L`, then mean pool) maps each step to `d_model` before the existing temporal oracle Transformer over `T`. Set `oracle_use_full_y: false` to use mean-pooled `aux_tensor` only, shape `[B, T, d_model]` (legacy). Config: `oracle_spatial_n_layers`, `oracle_spatial_n_heads`, `oracle_spatial_d_ff` (defaults follow the main oracle head).

## 1) TRM Backbone and Training Loop

### TRM pass
`TRM.forward()` produces one refinement pass:
- token embedding (+ optional positional embedding),
- deep recursion through `_TRMCore`,
- output `ModelOutput(logits, aux_tensor, state, loss_parts)`.

`_TRMCore` is either:
- Transformer blocks (`use_attention=true`), or
- MLP-Mixer style blocks (`use_attention=false` / `mlp_t=true`).

### Multi-step supervision
The outer multi-step loop is in `train_from_yaml()`:
- for `sup_step in 1..N_sup`:
  - scheduler gives `recursion_n`, `recursion_T`, `supervision_weight`,
  - model forward is called,
  - task loss is computed and backpropagated.

So TRM itself is one-step; recursion across supervision steps is runner-driven.

## 2) Losses Used During Training

Per supervision step (in `src/diplom/runner/train.py`):

1. **Main task loss**
\[
\mathcal{L}_{\text{main}}^{(k)} = w_k \cdot \text{TaskLoss}(\text{logits}_k, y)
\]
where \(w_k=\text{supervision\_weight}\) from scheduler.

2. **Optional halt loss** (if halt head exists and task provides halt targets)
\[
\mathcal{L}_{\text{halt}}^{(k)} = \beta_{\text{halt}} \cdot w_k \cdot \text{BCE}(\hat h_k, h_k^\*)
\]

After full rollout, for oracle-capable model:

3. **Oracle loss** (separate optimizer over oracle parameters)
\[
\mathcal{L}_{\text{oracle}} = \text{oracle\_loss\_weight} \cdot \mathcal{L}_{\text{oracle,raw}}
\]

Important:
- for `TRMOracle`, `requires_full_rollout=True`, so training performs full rollout (no early halt break before oracle supervision is built).

## 3) Oracle Modes

`TRMOracleConfig.oracle_target_mode`:
- `delta`: legacy lookahead-delta classifier
- `distribution`: predicts full PMF over stopping time indices \(j\in\{1,\dots,K\}\)

`K` is `oracle_horizon` (possibly clipped by `valid_horizon`).

## 4) Distribution Families (Implemented)

All families produce \(\hat p(j)\), then code clamps and renormalizes where needed.

### 4.1 finite_discrete
\[
z_j = W_j h + b_j,\quad
\hat p(j)=\frac{e^{z_j}}{\sum_{r=1}^{K} e^{z_r}}
\]

### 4.2 smoothed_loss
Model form is same as `finite_discrete`; difference is **target** (soft labels, see Section 5).

### 4.3 mixture_geometric
\[
\hat p(j)=\sum_{m=1}^{M} \pi_m (1-\rho_m)^{j-1}\rho_m
\]
\[
\pi=\text{softmax}(\cdot),\quad \rho=\sigma(\cdot)\in(0,1)
\]

### 4.4 mixture_exponential
\[
\hat p(j)=\sum_{m=1}^{M}\pi_m\left(e^{-\alpha_m(j-1)}-e^{-\alpha_m j}\right)
\]
\[
\pi=\text{softmax}(\cdot),\quad \alpha=\text{softplus}(\cdot)>0
\]

### 4.5 power
\[
\tilde p(j)=(j+c)^{-a},\quad
\hat p(j)=\frac{\tilde p(j)}{\sum_{r=1}^{K}\tilde p(r)}
\]
\[
a=\text{softplus}(\cdot)>0,\quad c=\text{softplus}(\cdot)\ge 0
\]

### 4.6 negative_binomial
\[
\hat p(j)\propto \binom{j+r-2}{j-1}(1-\rho)^{j-1}\rho^r,\quad j=1..K
\]
Implemented in log-space via `lgamma`, then normalized.

### 4.7 lognormal
\[
\hat p(j)\propto
\Phi\!\left(\frac{\log(j+1)-\mu}{\sigma}\right)-
\Phi\!\left(\frac{\log j-\mu}{\sigma}\right)
\]
\[
\sigma=\text{softplus}(\cdot)>0
\]

### 4.8 hybrid
\[
\hat p(j)=\alpha\,p_{\text{finite}}(j)+(1-\alpha)\,p_{\text{power}}(j)
\]
\[
\alpha=\sigma(\cdot)\in[0,1]
\]

## 5) Target Construction and Distribution Loss

Per rollout, per sample, per step-cost coefficient \(\lambda\):
\[
C_j = L_j + \lambda j
\]
where \(L_j\) is per-step task loss from rollout.

### Target in `distribution` mode
- For most families: one-hot at
\[
j^\*=\arg\min_j C_j
\]
- For `smoothed_loss`: soft target
\[
y_j=\frac{e^{-C_j/\beta}}{\sum_{r=1}^K e^{-C_r/\beta}}
\]
\(\beta=\) `oracle_distribution_beta`.

### Distribution loss
\[
\mathcal{L}_{\text{dist}} = -\sum_{j=1}^K y_j\log \hat p(j)
\]
averaged over batch and rollout prefixes.

## 6) Legacy Delta Oracle Loss

In `delta` mode:
- Oracle predicts logits over \(\delta\in\{0,\dots,H-1\}\), where \(\delta=0\) means "stop now".
- Target is best local delta index by minimal future loss in the valid horizon window.

Loss:
\[
\mathcal{L}_{\delta}=\text{CE}(\text{logits}_\delta,\delta^\*)
\]

## 7) Practical Note About "All Families in One Pass"

Current training uses exactly **one** selected family per run:
- `oracle_distribution_model` picks one head/loss path.
- Other heads are present in the class but not jointly optimized unless explicitly selected in another run.

So fair model comparison currently means **separate train runs** per family (configs in `configs/oracle_sweep_wikitext/` — WikiText использует **frozen Falcon + `frozen_llm_trm_oracle`**, `task.train_fraction: 0.1`, см. `runs/oracle_sweep_wikitext_falcon/`, — и `configs/oracle_sweep_arc_agi/`).
