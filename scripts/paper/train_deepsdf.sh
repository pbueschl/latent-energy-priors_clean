#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/paper/medshape_diverse_deepsdf.yaml}"
SDF_ROOT="${SDF_ROOT:-data/sdf/medshape_diverse}"
SPLIT_FILE="${SPLIT_FILE:-data/splits/medshape_diverse.json}"
OUT_DIR="${OUT_DIR:-outputs/paper/medshape_diverse/deepsdf}"

DEVICE_ARG=()
if [ -n "${DEVICE:-}" ]; then
  DEVICE_ARG=(--device "${DEVICE}")
fi

EPOCH_ARG=()
if [ -n "${DEEPSDF_EPOCHS:-}" ]; then
  EPOCH_ARG=(--epochs "${DEEPSDF_EPOCHS}")
fi

python3 scripts/train_deepsdf.py \
  --config "${CONFIG}" \
  --sdf-root "${SDF_ROOT}" \
  --split-file "${SPLIT_FILE}" \
  --out "${OUT_DIR}" \
  "${DEVICE_ARG[@]}" \
  "${EPOCH_ARG[@]}"
