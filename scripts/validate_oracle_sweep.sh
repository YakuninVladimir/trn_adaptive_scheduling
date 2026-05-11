#!/usr/bin/env bash
set -euo pipefail

# Validate trained oracle sweep runs and compute stopping metrics.
# Usage:
#   ./scripts/validate_oracle_sweep.sh [group] [--dry-run] [--continue-on-error] [--all-distributions] [--honest-split-ratio=<0..1>] [--batch-multiplier=<float>] [--val-reduce-factor=<float>] [--checkpoint-subdir=<path>]
# groups:
#   wikitext | arc | all (default: all)

GROUP="${1:-all}"
if [[ "${GROUP}" == "--dry-run" || "${GROUP}" == "--continue-on-error" ]]; then
  GROUP="all"
fi

DRY_RUN=0
CONTINUE_ON_ERROR=0
ALL_DISTRIBUTIONS=0
HONEST_SPLIT_RATIO="0.0"
BATCH_MULTIPLIER="2.0"
VAL_REDUCE_FACTOR="5.0"
CHECKPOINT_SUBDIR=""
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --continue-on-error) CONTINUE_ON_ERROR=1 ;;
    --all-distributions) ALL_DISTRIBUTIONS=1 ;;
    --honest-split-ratio=*) HONEST_SPLIT_RATIO="${arg#*=}" ;;
    --batch-multiplier=*) BATCH_MULTIPLIER="${arg#*=}" ;;
    --val-reduce-factor=*) VAL_REDUCE_FACTOR="${arg#*=}" ;;
    --checkpoint-subdir=*) CHECKPOINT_SUBDIR="${arg#*=}" ;;
    wikitext|arc|all) ;;
    *)
      if [[ "${arg}" != "${GROUP}" ]]; then
        echo "Unknown arg: ${arg}" >&2
        echo "Usage: ./scripts/validate_oracle_sweep.sh [wikitext|arc|all] [--dry-run] [--continue-on-error] [--all-distributions] [--honest-split-ratio=<0..1>] [--batch-multiplier=<float>] [--val-reduce-factor=<float>] [--checkpoint-subdir=<path>]" >&2
        exit 2
      fi
      ;;
  esac
done

declare -a CFGS=()
if [[ "${GROUP}" == "all" || "${GROUP}" == "wikitext" ]]; then
  CFGS+=(
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_finite_discrete.yaml"
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_smoothed_loss.yaml"
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_mixture_geometric.yaml"
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_mixture_exponential.yaml"
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_power.yaml"
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_negative_binomial.yaml"
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_lognormal.yaml"
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_hybrid.yaml"
  )
fi
if [[ "${GROUP}" == "all" || "${GROUP}" == "arc" ]]; then
  CFGS+=(
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_finite_discrete.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_smoothed_loss.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_mixture_geometric.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_mixture_exponential.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_power.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_negative_binomial.yaml"
    # "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_lognormal.yaml"
    # "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_hybrid.yaml"
  )
fi

if [[ "${#CFGS[@]}" -eq 0 ]]; then
  echo "No configs selected." >&2
  exit 2
fi

echo "[validate-sweep] group=${GROUP} dry_run=${DRY_RUN} continue_on_error=${CONTINUE_ON_ERROR} all_distributions=${ALL_DISTRIBUTIONS} honest_split_ratio=${HONEST_SPLIT_RATIO} batch_multiplier=${BATCH_MULTIPLIER} val_reduce_factor=${VAL_REDUCE_FACTOR} checkpoint_subdir=${CHECKPOINT_SUBDIR:-<none>}"

for cfg in "${CFGS[@]}"; do
  run_dir="$(uv run python - <<PY
import yaml
with open('${cfg}', 'r', encoding='utf-8') as f:
    cfg=yaml.safe_load(f)
print(cfg.get('train',{}).get('run_dir',''))
PY
)"
  if [[ -z "${run_dir}" ]]; then
    echo "[validate-sweep] missing run_dir in ${cfg}" >&2
    if [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]; then
      continue
    fi
    exit 1
  fi
  base_dir="${run_dir}"
  if [[ -n "${CHECKPOINT_SUBDIR}" ]]; then
    base_dir="${run_dir}/${CHECKPOINT_SUBDIR}"
  fi
  ckpt="${base_dir}/checkpoints/best.pt"
  out_json="${base_dir}/stopping_eval.json"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[validate-sweep] checkpoint not found: ${ckpt}" >&2
    if [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]; then
      continue
    fi
    exit 1
  fi

  cfg_for_eval="${cfg}"
  # Build a temporary compatible config if checkpoint/model-schema drift is detected
  # and apply eval-time overrides (batch_size, val split reduction).
  compat_cfg="$(mktemp)"
  uv run python - <<PY
import torch, yaml, pathlib
cfg_path = pathlib.Path("${cfg}")
ckpt_path = pathlib.Path("${ckpt}")
out_path = pathlib.Path("${compat_cfg}")
cfg = yaml.safe_load(cfg_path.read_text())
state = torch.load(ckpt_path, map_location="cpu")["model"]
keys = list(state.keys())
model = cfg.setdefault("model", {})
task = cfg.setdefault("task", {})
train = cfg.setdefault("train", {})
changed = False

# If checkpoint lacks spatial oracle params, force legacy pooled-history mode.
if not any(k.startswith("oracle_head.spatial.") for k in keys):
    if model.get("oracle_use_full_y", None) is not False:
        model["oracle_use_full_y"] = False
        changed = True

# If checkpoint contains halt head params, enable halt_head for strict load.
if any(k.startswith("halt_proj.") for k in keys):
    if model.get("halt_head", None) is not True:
        model["halt_head"] = True
        changed = True

# Eval-time override: larger batch size for faster validation.
try:
    mult = float("${BATCH_MULTIPLIER}")
except Exception:
    mult = 1.0
if mult > 0:
    old_bs = int(train.get("batch_size", 1))
    new_bs = max(1, int(round(old_bs * mult)))
    if new_bs != old_bs:
        train["batch_size"] = new_bs
        changed = True

# Eval-time override: reduce validation subset size.
try:
    reduce = float("${VAL_REDUCE_FACTOR}")
except Exception:
    reduce = 1.0
if reduce > 0:
    if task.get("val_fraction", None) is None:
        new_vf = 1.0 / reduce
        if new_vf < 1.0:
            task["val_fraction"] = new_vf
            changed = True
    else:
        old_vf = float(task.get("val_fraction"))
        new_vf = max(1e-6, min(1.0, old_vf / reduce))
        if abs(new_vf - old_vf) > 1e-12:
            task["val_fraction"] = new_vf
            changed = True
    if task.get("max_val_samples", None) is not None:
        old_mvs = int(task.get("max_val_samples"))
        new_mvs = max(1, int(round(old_mvs / reduce)))
        if new_mvs != old_mvs:
            task["max_val_samples"] = new_mvs
            changed = True

if changed:
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
else:
    # keep file empty as a marker: no compat rewrite needed
    out_path.write_text("")
PY
  if [[ -s "${compat_cfg}" ]]; then
    cfg_for_eval="${compat_cfg}"
    echo "[validate-sweep] using compatibility config for ${cfg}"
  fi

  eval_distributions="$(uv run python - <<PY
import yaml
cfg_path = "${cfg_for_eval}"
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
model = cfg.get("model", {}) or {}
target_mode = str(model.get("oracle_target_mode", "delta")).strip().lower()
dist = str(model.get("oracle_distribution_model", "finite_discrete")).strip()
if target_mode == "delta":
    print("finite_discrete")
else:
    print(dist if dist else "finite_discrete")
PY
)"
  if [[ "${ALL_DISTRIBUTIONS}" -eq 1 ]]; then
    eval_distributions="finite_discrete,smoothed_loss,mixture_geometric,mixture_exponential,power,negative_binomial,lognormal,hybrid"
  fi

  validate_cmd=(uv run diplom-validate --config "${cfg_for_eval}" --checkpoint "${ckpt}" --oracle-policy greedy --oracle-max-steps 8 --progress-bar true)
  eval_cmd=(
    uv run diplom-eval-stopping
    --config "${cfg_for_eval}"
    --checkpoint "${ckpt}"
    --distribution-models "${eval_distributions}"
    --strategies cumulative_probability,future_improvement,hazard,quantile,budget
    --threshold-grid 0.5,0.6,0.7,0.8,0.9
    --budget-grid 2,4,6,8
    --out "${out_json}"
    --honest-split-ratio "${HONEST_SPLIT_RATIO}"
    --selection-metric token_acc
    --selection-mode max
    --answer-policies last,argmax_interval
    --progress-bar true
  )

  echo "[validate-sweep] validate -> ${cfg}"
  echo "  checkpoint: ${ckpt}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  CMD: '; printf '%q ' "${validate_cmd[@]}"; echo
    printf '  CMD: '; printf '%q ' "${eval_cmd[@]}"; echo
    rm -f "${compat_cfg}"
    continue
  fi

  if [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]; then
    "${validate_cmd[@]}" || { echo "[validate-sweep] validate failed: ${cfg}" ; rm -f "${compat_cfg}"; continue; }
    "${eval_cmd[@]}" || echo "[validate-sweep] eval-stopping failed: ${cfg}"
  else
    "${validate_cmd[@]}"
    "${eval_cmd[@]}"
  fi
  rm -f "${compat_cfg}"
done

echo "[validate-sweep] done"
