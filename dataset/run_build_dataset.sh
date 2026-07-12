#!/usr/bin/env bash
# Run dataset.build_dataset (via the collider.sif Apptainer image) for one or
# more barrier trajectories under FEM_DIR.
#
# Usage:
#   bash dataset/run_build_dataset.sh <name> [<name> ...]
#   bash dataset/run_build_dataset.sh --all      # every dir under FEM_DIR
#
# Each <name> is a subdirectory of FEM_DIR containing car_and_barriers.k plus
# the d3plot state files. Output lands at OUT_DIR/<name>.h5 (+ .gif).
# Already-converted datasets (output .h5 exists) are skipped; set
# FORCE=1 to reconvert.
# Frame stride defaults to 10; set FRAME_STRIDE=1 to keep every frame
# (frame-stride must be >=1 — build_dataset.py does unique[::stride], which
# raises ValueError for 0).
# bash dataset/run_build_dataset.sh T_lok_F_shape_barrier_9_3_rubber_concrete_15_Modified_DIF_60km T_lok_F_shape_barrier_9_3_rubber_concrete_30_Modified_DIF_60km

set -euo pipefail

FEM_DIR=/raid/proj_iim1/xrkong/fem
OUT_DIR=/raid/proj_iim1/xrkong/h5_fps_2ms_no_wheel
SIF=/staging/proj_iim1/xrkong/container/collider.sif
EXCLUDE_PARTS_CONFIG=configs/data/exclude_parts_tires.yaml
FORCE="${FORCE:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-10}"

if [ "${FRAME_STRIDE}" -lt 1 ]; then
  echo "FRAME_STRIDE must be >=1 (got ${FRAME_STRIDE}); use FRAME_STRIDE=1 to keep every frame." >&2
  exit 1
fi

cd /home/xangruik/collider

if [ "${1:-}" = "--all" ]; then
  NAMES=()
  for d in "${FEM_DIR}"/*/; do
    name="$(basename "${d}")"
    [ -f "${d}/car_and_barriers.k" ] && NAMES+=("${name}")
  done
else
  if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <name> [<name> ...]   OR   $0 --all" >&2
    exit 1
  fi
  NAMES=("$@")
fi

for name in "${NAMES[@]}"; do
  kfile="${FEM_DIR}/${name}/car_and_barriers.k"
  src="${FEM_DIR}/${name}"
  out="${OUT_DIR}/${name}.h5"

  echo "=== ${name} ==="

  if [ ! -f "${kfile}" ]; then
    echo "Skipping: no car_and_barriers.k in ${src}"
    continue
  fi

  if [ -f "${out}" ] && [ "${FORCE}" != "1" ]; then
    echo "Skipping: ${out} already exists (set FORCE=1 to reconvert)"
    continue
  fi

  mkdir -p "$(dirname "${out}")"

  apptainer exec --bind /raid "${SIF}" \
    python -m dataset.build_dataset \
      --kfile  "${kfile}" \
      --src    "${src}" \
      --out    "${out}" \
      --method fps \
      --seed 42 \
      --exclude-parts-config "${EXCLUDE_PARTS_CONFIG}" \
      --frame-stride "${FRAME_STRIDE}" --n-jobs 8 \
      --gif

  echo "=== Done ${name} ==="
done

echo "All conversions complete."
