#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/paper/medshape_diverse_conditional_validation_grid.yaml}"
SDF_ROOT="${SDF_ROOT:-data/sdf/medshape_diverse}"
SPLIT_FILE="${SPLIT_FILE:-data/splits/medshape_diverse.json}"
DEEPSDF_CHECKPOINT="${DEEPSDF_CHECKPOINT:-outputs/paper/medshape_diverse/deepsdf/deepsdf_final.pt}"
CONDITIONAL_ENERGY_CHECKPOINT="${CONDITIONAL_ENERGY_CHECKPOINT:-outputs/paper/medshape_diverse/conditional_energy/conditional_energy_prior_best_loss.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/paper/medshape_diverse/conditional_val}"
TMP_ROOT="${TMP_ROOT:-outputs/paper/medshape_diverse/tmp}"
OBS_LIST="${OBS_LIST:-16 32 64 128}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
L2_LAMBDA="${L2_LAMBDA:-0.0001}"
L2_VARIANT="${L2_VARIANT:-l2_lam_1e_4}"
CONDITIONAL_GRID="${CONDITIONAL_GRID:-1e_5:0.00001 3e_5:0.00003 1e_4:0.0001}"

DEVICE_ARG=()
if [ -n "${DEVICE:-}" ]; then
  DEVICE_ARG=(--device "${DEVICE}")
fi

mkdir -p "${TMP_ROOT}"

run_observation() {
  local OBS="$1"
  local OBS_OUT_DIR="${OUT_ROOT}/obs_${OBS}"
  local TMP_CONFIG="${TMP_ROOT}/conditional_validation_obs_${OBS}.yaml"
  mkdir -p "${OBS_OUT_DIR}"
  python3 - "${CONFIG}" "${TMP_CONFIG}" "${OBS}" \
    "${L2_LAMBDA}" "${L2_VARIANT}" "${CONDITIONAL_GRID}" <<'PY'
import sys
from pathlib import Path

import yaml

src = sys.argv[1]
dst = sys.argv[2]
obs = int(sys.argv[3])
l2_lambda = float(sys.argv[4])
l2_variant = sys.argv[5]
conditional_grid = sys.argv[6].split()

with Path(src).open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

l2_suffix = l2_variant.removeprefix("l2_lam_")
pairs = []
for item in conditional_grid:
    suffix, value = item.split(":", 1)
    pairs.append((suffix, float(value)))

sweep = cfg.setdefault("sparse_sweep", {})
sweep["observation_points"] = obs
sweep["lambda_l2"] = l2_lambda
sweep["variants"] = [
    {
        "name": "no_prior",
        "method": "no_prior",
        "lambda_l2": 0.0,
        "lambda_conditional_energy": 0.0,
    },
    {
        "name": l2_variant,
        "method": "l2",
        "lambda_l2": l2_lambda,
        "lambda_conditional_energy": 0.0,
    },
]
for suffix, value in pairs:
    sweep["variants"].append(
        {
            "name": f"conditional_lam_{suffix}",
            "method": "conditional_energy",
            "lambda_l2": 0.0,
            "lambda_conditional_energy": value,
        }
    )
for suffix, value in pairs:
    sweep["variants"].append(
        {
            "name": f"l2_conditional_l2_{l2_suffix}_cond_{suffix}",
            "method": "l2_conditional_energy",
            "lambda_l2": l2_lambda,
            "lambda_conditional_energy": value,
        }
    )

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with Path(dst).open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  local OUT_CSV="${OBS_OUT_DIR}/sparse_sweep.csv"
  python3 scripts/run_sparse_sweep.py \
    --config "${TMP_CONFIG}" \
    --sdf-root "${SDF_ROOT}" \
    --split-file "${SPLIT_FILE}" \
    --checkpoint "${DEEPSDF_CHECKPOINT}" \
    --conditional-energy-checkpoint "${CONDITIONAL_ENERGY_CHECKPOINT}" \
    --out "${OUT_CSV}" \
    "${DEVICE_ARG[@]}"

  python3 scripts/select_variants.py "${OUT_CSV}" \
    --metric "${SELECTION_METRIC:-eval_l1}" \
    --out-json "${OBS_OUT_DIR}/selection.json" \
    --out-csv "${OBS_OUT_DIR}/ranking.csv"
}

for OBS in ${OBS_LIST}; do
  while [ "$(jobs -pr | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    wait -n
  done
  run_observation "${OBS}" &
done

wait
