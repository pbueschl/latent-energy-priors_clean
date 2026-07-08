#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

FINAL_CSV="${1:-}"
SPLIT_FILE="${SPLIT_FILE:-data/splits/medshape_diverse.json}"
OUT_DIR="${OUT_DIR:-outputs/paper/medshape_diverse/final}"
OBS_LIST="${OBS_LIST:-16 32 64 128 256}"
L2_VARIANT="${L2_VARIANT:-l2_lam_1e_3}"
CONDITIONAL_VARIANT="${CONDITIONAL_VARIANT:-conditional_lam_1e_5}"
L2_CONDITIONAL_VARIANT="${L2_CONDITIONAL_VARIANT:-l2_conditional_l2_1e_3_cond_3e_5}"
INCLUDE_SHUFFLED_CONTEXT="${INCLUDE_SHUFFLED_CONTEXT:-0}"
CONDITIONAL_SHUFFLED_VARIANT="${CONDITIONAL_SHUFFLED_VARIANT:-conditional_shuffled_context_lam_1e_5}"
L2_CONDITIONAL_SHUFFLED_VARIANT="${L2_CONDITIONAL_SHUFFLED_VARIANT:-l2_conditional_shuffled_context_l2_1e_3_cond_3e_5}"
USER_VARIANTS="${VARIANTS:-}"
VARIANTS="${VARIANTS:-${L2_VARIANT},${CONDITIONAL_VARIANT},${L2_CONDITIONAL_VARIANT}}"
TUNED_L2_VARIANTS="${TUNED_L2_VARIANTS:-${L2_CONDITIONAL_VARIANT}}"
PLOT_METRICS="${PLOT_METRICS:-eval_l1,grid_mean_abs_sdf_error,eval_dice,eval_iou,latent_norm,conditional_energy}"
BOOTSTRAP_METRICS="${BOOTSTRAP_METRICS:-eval_l1 grid_mean_abs_sdf_error eval_dice eval_iou}"
HIGHER_IS_BETTER_METRICS="${HIGHER_IS_BETTER_METRICS:-eval_dice eval_iou eval_accuracy}"
NUM_BOOTSTRAP="${NUM_BOOTSTRAP:-10000}"

if [ -z "${USER_VARIANTS}" ] && [ "${INCLUDE_SHUFFLED_CONTEXT}" = "1" ]; then
  VARIANTS="${VARIANTS},${CONDITIONAL_SHUFFLED_VARIANT},${L2_CONDITIONAL_SHUFFLED_VARIANT}"
  TUNED_L2_VARIANTS="${TUNED_L2_VARIANTS},${L2_CONDITIONAL_SHUFFLED_VARIANT}"
fi

process_csv() {
  local csv_path="$1"
  local obs_out_dir="$2"

  python3 scripts/aggregate_sparse_by_class.py "${csv_path}" \
    --split-file "${SPLIT_FILE}" \
    --out-class "${obs_out_dir}/sparse_sweep_by_class.csv" \
    --out-overall "${obs_out_dir}/sparse_sweep_overall.csv" \
    --summary-json "${obs_out_dir}/sparse_sweep_by_class_summary.json"

  python3 scripts/plot_results.py "${csv_path}" \
    --metrics "${PLOT_METRICS}" \
    --baseline-variant no_prior \
    --out-dir "${obs_out_dir}/plots"

  for METRIC in ${BOOTSTRAP_METRICS}; do
    DIRECTION_ARG=()
    for HIGHER_METRIC in ${HIGHER_IS_BETTER_METRICS}; do
      if [ "${METRIC}" = "${HIGHER_METRIC}" ]; then
        DIRECTION_ARG=(--higher-is-better)
        break
      fi
    done

    python3 scripts/bootstrap_results.py "${csv_path}" \
      --baseline-variant no_prior \
      --variants "${VARIANTS}" \
      --metric "${METRIC}" \
      "${DIRECTION_ARG[@]}" \
      --num-bootstrap "${NUM_BOOTSTRAP}" \
      --out-json "${obs_out_dir}/bootstrap_${METRIC}_vs_no_prior.json" \
      --out-csv "${obs_out_dir}/bootstrap_${METRIC}_vs_no_prior.csv"

    python3 scripts/bootstrap_results.py "${csv_path}" \
      --baseline-variant "${L2_VARIANT}" \
      --variants "${TUNED_L2_VARIANTS}" \
      --metric "${METRIC}" \
      "${DIRECTION_ARG[@]}" \
      --num-bootstrap "${NUM_BOOTSTRAP}" \
      --out-json "${obs_out_dir}/bootstrap_${METRIC}_vs_tuned_l2.json" \
      --out-csv "${obs_out_dir}/bootstrap_${METRIC}_vs_tuned_l2.csv"
  done
}

if [ -n "${FINAL_CSV}" ]; then
  process_csv "${FINAL_CSV}" "${OUT_DIR}"
else
  for OBS in ${OBS_LIST}; do
    process_csv "${OUT_DIR}/obs_${OBS}/sparse_sweep.csv" "${OUT_DIR}/obs_${OBS}"
  done
fi
