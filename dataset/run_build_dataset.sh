#!/usr/bin/env bash
# Run dataset.build_dataset (via the collider.sif Apptainer image) for one or
# more barrier trajectories under FEM_DIR.
#
# Usage:
#   bash dataset/run_build_dataset.sh <name> [<name> ...]
#   bash dataset/run_build_dataset.sh --all      # every dir under FEM_DIR
#
# Each <name> is a subdirectory of FEM_DIR containing a k-file (either
# car_and_barriers.k or car_and_new_barrier.k — see KFILE_CANDIDATES below)
# plus the d3plot state files. Output lands at OUT_DIR/<name>.h5, with the
# GIF + GC/barrier CSV/plots under OUT_DIR/<name>_analysis/.
# Already-converted datasets (output .h5 exists) are skipped; set
# FORCE=1 to reconvert.
# Frame stride defaults to 10; set FRAME_STRIDE=1 to keep every frame
# (frame-stride must be >=1 — build_dataset.py does unique[::stride], which
# raises ValueError for 0).
# Set FRAME_LIMIT=<n> to cap the total frames kept (applied after stride) —
# e.g. FRAME_STRIDE=4 FRAME_LIMIT=50 on a 5ms-native-dt run gives exactly
# 50 frames at 20ms/frame. Unset (default) keeps every strided frame.
# NOTE: this only affects the main HDF5 — the GC/barrier CSV is always full
# native resolution regardless of FRAME_STRIDE/FRAME_LIMIT.
# GC/barrier ORA_x/ORA_y/ASI plots default to a 50ms moving-average filter
# (EN 1317); set GC_BARRIER_WINDOW_MS=0 for raw, unfiltered plots instead
# (the CSV itself is always raw regardless of this setting).
# Set FULL_RES=1 to skip node sampling entirely (raw FEM mesh, every node) —
# pair with a distinct OUT_DIR (e.g. h5_full_res) so it doesn't clobber the
# downsampled dataset. Temporal resolution is independent — FRAME_STRIDE
# still applies on top of FULL_RES.
# Set BUDGET_SCALE=<factor> to raise the ~100k-node region budget without
# going all the way to FULL_RES — e.g. BUDGET_SCALE=5 for a ~500k-node
# dataset (SPEC §4.4 defaults: barrier_fine=40000, barrier_coarse=20000,
# veh_contact=10000, veh_near=18000, veh_far=12000, scaled proportionally).
# Ignored if FULL_RES=1. Larger budgets make --method fps (the default)
# slower — it's O(n * region_size) — so a big BUDGET_SCALE may need more
# --time in the slurm job or METHOD=stride/random for a faster pass.
# bash dataset/run_build_dataset.sh T_lok_F_shape_barrier_9_3_rubber_concrete_15_Modified_DIF_60km T_lok_F_shape_barrier_9_3_rubber_concrete_30_Modified_DIF_60km

set -euo pipefail

FEM_DIR=/raid/proj_iim1/xrkong/fem
OUT_DIR="${OUT_DIR:-/raid/proj_iim1/xrkong/h5_fps_no_wheel}"
FULL_RES="${FULL_RES:-0}"
BUDGET_SCALE="${BUDGET_SCALE:-1}"
METHOD="${METHOD:-fps}"
# Scratch space for d3plot state-file copies (build_dataset.py's --tmp).
# Must NOT be /tmp (build_dataset.py's own default) — that's on a
# quota-limited filesystem and copying a single d3plot state (can be
# multi-GB) blows the quota. /raid has ~1.2T free. Per-name subdirectory so
# concurrent sbatch jobs for different <name>s don't race on
# build_dataset.py's startup cleanup (it rmtree's stray scan_*/extract_*
# dirs under --tmp).
TMP_DIR="${TMP_DIR:-/raid/proj_iim1/xrkong/tmp}"
SIF=/raid/proj_iim1/xrkong/container/collider.sif
EXCLUDE_PARTS_CONFIG=configs/data/exclude_parts_tires.yaml
FORCE="${FORCE:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-10}"
# k-file basename varies by dataset (some FEM_DIR subdirs were renamed
# upstream) — first match wins per directory.
KFILE_CANDIDATES=(car_and_barriers.k car_and_new_barrier.k)
FRAME_LIMIT="${FRAME_LIMIT:-}"
# ORA_x/ORA_y/ASI moving-average window (ms) in the GC/barrier plots (the CSV
# itself is always raw/unfiltered). Set to 0 for raw, unfiltered plots too.
GC_BARRIER_WINDOW_MS="${GC_BARRIER_WINDOW_MS:-50}"

if [ "${FRAME_STRIDE}" -lt 1 ]; then
  echo "FRAME_STRIDE must be >=1 (got ${FRAME_STRIDE}); use FRAME_STRIDE=1 to keep every frame." >&2
  exit 1
fi

cd /home/xangruik/collider

# Print the first existing KFILE_CANDIDATES path under dir $1, or nothing.
find_kfile() {
  local dir="$1" candidate
  for candidate in "${KFILE_CANDIDATES[@]}"; do
    if [ -f "${dir}/${candidate}" ]; then
      echo "${dir}/${candidate}"
      return
    fi
  done
}

if [ "${1:-}" = "--all" ]; then
  NAMES=()
  for d in "${FEM_DIR}"/*/; do
    name="$(basename "${d}")"
    [ -n "$(find_kfile "${d%/}")" ] && NAMES+=("${name}")
  done
else
  if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <name> [<name> ...]   OR   $0 --all" >&2
    exit 1
  fi
  NAMES=("$@")
fi

for name in "${NAMES[@]}"; do
  src="${FEM_DIR}/${name}"
  kfile="$(find_kfile "${src}")"
  out="${OUT_DIR}/${name}.h5"

  echo "=== ${name} ==="

  if [ -z "${kfile}" ]; then
    echo "Skipping: none of [${KFILE_CANDIDATES[*]}] found in ${src}"
    continue
  fi

  if [ -f "${out}" ] && [ "${FORCE}" != "1" ]; then
    echo "Skipping: ${out} already exists (set FORCE=1 to reconvert)"
    continue
  fi

  mkdir -p "$(dirname "${out}")"

  tmp_dir="${TMP_DIR}/${name}"
  mkdir -p "${tmp_dir}"

  extra_args=()
  if [ -n "${FRAME_LIMIT}" ]; then
    extra_args+=(--frame-limit "${FRAME_LIMIT}")
  fi
  if [ "${FULL_RES}" = "1" ]; then
    extra_args+=(--full-res)
  fi

  apptainer exec --bind /raid "${SIF}" \
    python -m dataset.build_dataset \
      --kfile  "${kfile}" \
      --src    "${src}" \
      --tmp    "${tmp_dir}" \
      --out    "${out}" \
      --method "${METHOD}" \
      --budget-scale "${BUDGET_SCALE}" \
      --seed 42 \
      --exclude-parts-config "${EXCLUDE_PARTS_CONFIG}" \
      --frame-stride "${FRAME_STRIDE}" --n-jobs 8 \
      --gc-barrier-window-ms "${GC_BARRIER_WINDOW_MS}" \
      --gif \
      "${extra_args[@]}"

  echo "=== Done ${name} ==="
done

echo "All conversions complete."
