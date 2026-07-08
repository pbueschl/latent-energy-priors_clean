#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/paper/medshape_diverse_l2_validation_grid.yaml}"
SDF_ROOT="${SDF_ROOT:-data/sdf/medshape_diverse}"
SPLIT_FILE="${SPLIT_FILE:-data/splits/medshape_diverse.json}"
DEEPSDF_CHECKPOINT="${DEEPSDF_CHECKPOINT:-outputs/paper/medshape_diverse/deepsdf/deepsdf_final.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/paper/medshape_diverse/l2_val}"
TMP_ROOT="${TMP_ROOT:-outputs/paper/medshape_diverse/tmp}"
OBS_LIST="${OBS_LIST:-16 32 64 128}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

DEVICE_ARG=()
if [ -n "${DEVICE:-}" ]; then
  DEVICE_ARG=(--device "${DEVICE}")
fi

mkdir -p "${TMP_ROOT}"

run_observation() {
  local OBS="$1"
  OBS_OUT_DIR="${OUT_ROOT}/obs_${OBS}"
  TMP_CONFIG="${TMP_ROOT}/l2_validation_obs_${OBS}.yaml"
  mkdir -p "${OBS_OUT_DIR}"
  python3 - "${CONFIG}" "${TMP_CONFIG}" "${OBS}" <<'PY'
import sys
from pathlib import Path

import yaml

src, dst, obs = sys.argv[1], sys.argv[2], int(sys.argv[3])
with Path(src).open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("sparse_sweep", {})["observation_points"] = obs
Path(dst).parent.mkdir(parents=True, exist_ok=True)
with Path(dst).open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  OUT_CSV="${OBS_OUT_DIR}/sparse_sweep.csv"
  python3 scripts/run_sparse_sweep.py \
    --config "${TMP_CONFIG}" \
    --sdf-root "${SDF_ROOT}" \
    --split-file "${SPLIT_FILE}" \
    --checkpoint "${DEEPSDF_CHECKPOINT}" \
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
