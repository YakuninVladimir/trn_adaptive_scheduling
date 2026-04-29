#!/usr/bin/env bash
set -euo pipefail

# Oracle-only fine-tuning sweep from existing best checkpoints.
# Usage:
#   ./scripts/finetune_oracle_sweep.sh [group] [--dry-run] [--continue-on-error]
# groups:
#   wikitext | arc | all (default: all)

GROUP="${1:-all}"
if [[ "${GROUP}" == "--dry-run" || "${GROUP}" == "--continue-on-error" ]]; then
  GROUP="all"
fi

DRY_RUN=0
CONTINUE_ON_ERROR=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --continue-on-error) CONTINUE_ON_ERROR=1 ;;
    wikitext|arc|all) ;;
    *)
      if [[ "${arg}" != "${GROUP}" ]]; then
        echo "Unknown arg: ${arg}" >&2
        echo "Usage: ./scripts/finetune_oracle_sweep.sh [wikitext|arc|all] [--dry-run] [--continue-on-error]" >&2
        exit 2
      fi
      ;;
  esac
done

declare -a CFGS=()
if [[ "${GROUP}" == "all" || "${GROUP}" == "wikitext" ]]; then
  CFGS+=(
    "configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_delta.yaml"
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
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_delta.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_finite_discrete.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_smoothed_loss.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_mixture_geometric.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_mixture_exponential.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_power.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_negative_binomial.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_lognormal.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_hybrid.yaml"
  )
fi

if [[ "${#CFGS[@]}" -eq 0 ]]; then
  echo "No configs selected." >&2
  exit 2
fi

echo "[oracle-ft] group=${GROUP} dry_run=${DRY_RUN} continue_on_error=${CONTINUE_ON_ERROR}"
for cfg in "${CFGS[@]}"; do
  run_dir="$(uv run python - <<PY
import yaml
with open('${cfg}', 'r', encoding='utf-8') as f:
    cfg=yaml.safe_load(f)
print(cfg.get('train',{}).get('run_dir',''))
PY
)"
  if [[ -z "${run_dir}" ]]; then
    echo "[oracle-ft] missing run_dir in ${cfg}" >&2
    if [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]; then
      continue
    fi
    exit 1
  fi

  base_ckpt="${run_dir}/checkpoints/best.pt"
  if [[ ! -f "${base_ckpt}" ]]; then
    echo "[oracle-ft] checkpoint not found: ${base_ckpt}" >&2
    if [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]; then
      continue
    fi
    exit 1
  fi

  tmp_cfg="$(mktemp)"
  uv run python - <<PY
import pathlib, yaml
cfg_path = pathlib.Path("${cfg}")
out_path = pathlib.Path("${tmp_cfg}")
cfg = yaml.safe_load(cfg_path.read_text())
cfg.setdefault("train", {})
cfg["train"]["run_dir"] = str(pathlib.Path("${run_dir}") / "oracle_finetune")
out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY

  train_cmd=(uv run diplom-train --config "${tmp_cfg}" --init-checkpoint "${base_ckpt}" --oracle-only)
  echo "[oracle-ft] train -> ${cfg}"
  echo "  checkpoint: ${base_ckpt}"
  echo "  finetune run_dir: ${run_dir}/oracle_finetune"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  CMD: '; printf '%q ' "${train_cmd[@]}"; echo
    rm -f "${tmp_cfg}"
    continue
  fi

  if [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]; then
    "${train_cmd[@]}" || echo "[oracle-ft] failed: ${cfg}"
  else
    "${train_cmd[@]}"
  fi
  rm -f "${tmp_cfg}"
done

echo "[oracle-ft] done"
