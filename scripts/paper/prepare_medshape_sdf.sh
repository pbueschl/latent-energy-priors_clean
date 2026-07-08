#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/paper/prepare_medshape_sdf.sh data/manifests/medshape_diverse.csv data/raw/medshape_diverse"
  echo "Optional env: SPLIT_FILE, SDF_ROOT, PER_CLASS_TARGET, MESH_LAYOUT, PREPROCESS_WORKERS, SDF_SAMPLES, SURFACE_SAMPLES"
  exit 2
fi

MANIFEST_CSV="$1"
MESH_ROOT="$2"
SPLIT_FILE="${SPLIT_FILE:-data/splits/medshape_diverse.json}"
SDF_ROOT="${SDF_ROOT:-data/sdf/medshape_diverse}"
PER_CLASS_TARGET="${PER_CLASS_TARGET:-250}"
MESH_LAYOUT="${MESH_LAYOUT:-class_subdirs}"

python3 scripts/make_medshape_split.py \
  --manifest-csv "${MANIFEST_CSV}" \
  --mesh-root "${MESH_ROOT}" \
  --mesh-layout "${MESH_LAYOUT}" \
  --per-class-target "${PER_CLASS_TARGET}" \
  --out "${SPLIT_FILE}"

python3 scripts/preprocess_shapenet.py \
  --mesh-root "${MESH_ROOT}" \
  --mesh-layout "${MESH_LAYOUT}" \
  --split-file "${SPLIT_FILE}" \
  --out "${SDF_ROOT}" \
  --samples "${SDF_SAMPLES:-50000}" \
  --surface-samples "${SURFACE_SAMPLES:-50000}" \
  --workers "${PREPROCESS_WORKERS:-4}" \
  --continue-on-error
