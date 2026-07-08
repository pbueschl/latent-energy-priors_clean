#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/paper/medshape_diverse_final_eval.yaml}"
SDF_ROOT="${SDF_ROOT:-data/sdf/medshape_diverse}"
SPLIT_FILE="${SPLIT_FILE:-data/splits/medshape_diverse.json}"
DEEPSDF_CHECKPOINT="${DEEPSDF_CHECKPOINT:-outputs/paper/medshape_diverse/deepsdf/deepsdf_final.pt}"
CONDITIONAL_ENERGY_CHECKPOINT="${CONDITIONAL_ENERGY_CHECKPOINT:-outputs/paper/medshape_diverse/conditional_energy/conditional_energy_prior_best_loss.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/paper/medshape_diverse/final}"
TMP_ROOT="${TMP_ROOT:-outputs/paper/medshape_diverse/tmp}"
OBS_LIST="${OBS_LIST:-16 32 64 128 256}"
L2_LAMBDA="${L2_LAMBDA:-0.001}"
L2_VARIANT="${L2_VARIANT:-l2_lam_1e_3}"
CONDITIONAL_LAMBDA="${CONDITIONAL_LAMBDA:-0.00001}"
CONDITIONAL_VARIANT="${CONDITIONAL_VARIANT:-conditional_lam_1e_5}"
L2_CONDITIONAL_LAMBDA="${L2_CONDITIONAL_LAMBDA:-0.00003}"
L2_CONDITIONAL_VARIANT="${L2_CONDITIONAL_VARIANT:-l2_conditional_l2_1e_3_cond_3e_5}"
INCLUDE_SHUFFLED_CONTEXT="${INCLUDE_SHUFFLED_CONTEXT:-0}"
CONDITIONAL_SHUFFLED_VARIANT="${CONDITIONAL_SHUFFLED_VARIANT:-conditional_shuffled_context_lam_1e_5}"
L2_CONDITIONAL_SHUFFLED_VARIANT="${L2_CONDITIONAL_SHUFFLED_VARIANT:-l2_conditional_shuffled_context_l2_1e_3_cond_3e_5}"

DEVICE_ARG=()
if [ -n "${DEVICE:-}" ]; then
  DEVICE_ARG=(--device "${DEVICE}")
fi

mkdir -p "${TMP_ROOT}"

for OBS in ${OBS_LIST}; do
  OBS_OUT_DIR="${OUT_ROOT}/obs_${OBS}"
  TMP_CONFIG="${TMP_ROOT}/final_eval_obs_${OBS}.yaml"
  mkdir -p "${OBS_OUT_DIR}"
  python3 - "${CONFIG}" "${TMP_CONFIG}" "${OBS}" \
    "${L2_LAMBDA}" "${L2_VARIANT}" \
    "${CONDITIONAL_LAMBDA}" "${CONDITIONAL_VARIANT}" \
    "${L2_CONDITIONAL_LAMBDA}" \
    "${L2_CONDITIONAL_VARIANT}" \
    "${INCLUDE_SHUFFLED_CONTEXT}" \
    "${CONDITIONAL_SHUFFLED_VARIANT}" \
    "${L2_CONDITIONAL_SHUFFLED_VARIANT}" <<'PY'
import sys
from pathlib import Path

import yaml

src = sys.argv[1]
dst = sys.argv[2]
obs = int(sys.argv[3])
l2_lambda = float(sys.argv[4])
l2_variant = sys.argv[5]
conditional_lambda = float(sys.argv[6])
conditional_variant = sys.argv[7]
l2_conditional_lambda = float(sys.argv[8])
l2_conditional_variant = sys.argv[9]
include_shuffled_context = sys.argv[10].strip().lower() in {"1", "true", "yes", "on"}
conditional_shuffled_variant = sys.argv[11]
l2_conditional_shuffled_variant = sys.argv[12]
with Path(src).open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
sweep = cfg.setdefault("sparse_sweep", {})
sweep["observation_points"] = obs
sweep["lambda_l2"] = l2_lambda
sweep["lambda_conditional_energy"] = l2_conditional_lambda
no_prior = next(
    (
        dict(variant)
        for variant in sweep.get("variants", [])
        if variant.get("method") == "no_prior" or variant.get("name") == "no_prior"
    ),
    {"name": "no_prior", "method": "no_prior"},
)
no_prior.update({"name": "no_prior", "method": "no_prior", "lambda_l2": 0.0, "lambda_conditional_energy": 0.0})
sweep["variants"] = [
    no_prior,
    {
        "name": l2_variant,
        "method": "l2",
        "lambda_l2": l2_lambda,
        "lambda_conditional_energy": 0.0,
    },
    {
        "name": conditional_variant,
        "method": "conditional_energy",
        "lambda_l2": 0.0,
        "lambda_conditional_energy": conditional_lambda,
    },
    {
        "name": l2_conditional_variant,
        "method": "l2_conditional_energy",
        "lambda_l2": l2_lambda,
        "lambda_conditional_energy": l2_conditional_lambda,
    },
]
if include_shuffled_context:
    sweep["variants"].extend(
        [
            {
                "name": conditional_shuffled_variant,
                "method": "conditional_energy_shuffled_context",
                "lambda_l2": 0.0,
                "lambda_conditional_energy": conditional_lambda,
            },
            {
                "name": l2_conditional_shuffled_variant,
                "method": "l2_conditional_energy_shuffled_context",
                "lambda_l2": l2_lambda,
                "lambda_conditional_energy": l2_conditional_lambda,
            },
        ]
    )
Path(dst).parent.mkdir(parents=True, exist_ok=True)
with Path(dst).open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  python3 scripts/run_sparse_sweep.py \
    --config "${TMP_CONFIG}" \
    --sdf-root "${SDF_ROOT}" \
    --split-file "${SPLIT_FILE}" \
    --checkpoint "${DEEPSDF_CHECKPOINT}" \
    --conditional-energy-checkpoint "${CONDITIONAL_ENERGY_CHECKPOINT}" \
    --out "${OBS_OUT_DIR}/sparse_sweep.csv" \
    "${DEVICE_ARG[@]}"
done
