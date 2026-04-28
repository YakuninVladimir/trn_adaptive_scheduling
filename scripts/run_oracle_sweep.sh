#!/usr/bin/env bash
set -euo pipefail

# Run oracle sweep configs in one go.
# Usage:
#   ./scripts/run_oracle_sweep.sh [group] [--dry-run] [--continue-on-error]
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
        echo "Usage: ./scripts/run_oracle_sweep.sh [wikitext|arc|all] [--dry-run] [--continue-on-error]" >&2
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
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_lognormal.yaml"
    "configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_hybrid.yaml"
  )
fi

if [[ "${#CFGS[@]}" -eq 0 ]]; then
  echo "No configs selected." >&2
  exit 2
fi

echo "[sweep] group=${GROUP} dry_run=${DRY_RUN} continue_on_error=${CONTINUE_ON_ERROR}"
for cfg in "${CFGS[@]}"; do
  echo "[sweep] train -> ${cfg}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi
  if [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]; then
    uv run diplom-train --config "${cfg}" || echo "[sweep] failed: ${cfg}"
  else
    uv run diplom-train --config "${cfg}"
  fi
done

echo "[sweep] done"
