#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/paper/medshape_diverse_conditional_energy.yaml}"
SDF_ROOT="${SDF_ROOT:-data/sdf/medshape_diverse}"
DEEPSDF_CHECKPOINT="${DEEPSDF_CHECKPOINT:-outputs/paper/medshape_diverse/deepsdf/deepsdf_final.pt}"
OUT_DIR="${OUT_DIR:-outputs/paper/medshape_diverse/conditional_energy}"

DEVICE_ARG=()
if [ -n "${DEVICE:-}" ]; then
  DEVICE_ARG=(--device "${DEVICE}")
fi

EPOCH_ARG=()
if [ -n "${CONDITIONAL_ENERGY_EPOCHS:-}" ]; then
  EPOCH_ARG=(--epochs "${CONDITIONAL_ENERGY_EPOCHS}")
fi

python3 scripts/train_conditional_energy.py \
  --config "${CONFIG}" \
  --checkpoint "${DEEPSDF_CHECKPOINT}" \
  --sdf-root "${SDF_ROOT}" \
  --out "${OUT_DIR}" \
  "${DEVICE_ARG[@]}" \
  "${EPOCH_ARG[@]}"
