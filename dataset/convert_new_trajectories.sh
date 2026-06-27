#!/usr/bin/env bash
# Unzip and convert the 6 new barrier trajectories (plus400kg/plus800kg variants
# of the 60/80/100km cases) to h5 format, following the same recipe used for
# the original T_lok_F_shape_barrier_9_3_60km run.
#
# Usage:
#   bash dataset/convert_new_trajectories.sh
#
# Each trajectory is unzipped, converted, then the extracted source files are
# removed to keep disk usage bounded (zips total ~270GB, ~525GB free on disk).
# Set KEEP_EXTRACTED=1 to skip the cleanup step.

set -euo pipefail

FEM_DIR=/home/kong/datasets/barrier/fem
TMP_DIR=/home/kong/datasets/barrier/tmp
OUT_DIR=/home/kong/datasets/barrier/h5dt_50ns_10fs_mat
SAMPLING_CONFIG=configs/data/sampling_config.yaml
KEEP_EXTRACTED="${KEEP_EXTRACTED:-0}"

cd /home/kong/xrkong/collider

# name1: zip file name (without .zip), this becomes the extracted folder name
# name2: normalized name used for the output folder
TRAJS=(
  "T_lok_F_shape_barrier_9_3_60km_Plus800kg:T_lok_F_shape_barrier_9_3_60km_plus800kg"
  "T_lok_F_shape_barrier_9_3_80km_plus400kg:T_lok_F_shape_barrier_9_3_80km_plus400kg"
  "T_lok_F_shape_barrier_9_3_80km_plus800kg:T_lok_F_shape_barrier_9_3_80km_plus800kg"
  "T_lok_F_shape_barrier_9_3_100km_plus400kg:T_lok_F_shape_barrier_9_3_100km_plus400kg"
  "T_lok_F_shape_barrier_9_3_100km_plus800kg:T_lok_F_shape_barrier_9_3_100km_plus800kg"
)

UNZIP_TRAJS=(
  # "T_lok_F_shape_barrier_9_3_60km_plus400kg:T_lok_F_shape_barrier_9_3_60km_plus400kg"
  # "T_lok_F_shape_barrier_9_3_60km:T_lok_F_shape_barrier_9_3_60km"
  "T_lok_F_shape_barrier_9_3_80km:T_lok_F_shape_barrier_9_3_80km"
  "T_lok_F_shape_barrier_9_3_100km:T_lok_F_shape_barrier_9_3_100km"
)

# python dataset/d3plot_to_h5_dt.py \
#     --src /home/kong/datasets/barrier/fem/T_lok_F_shape_barrier_9_3_60km \
#     --tmp /home/kong/datasets/barrier/tmp1 \
#     --out /home/kong/datasets/barrier/h5dt_50ns_10fs_mat_woSusp/T_lok_F_shape_barrier_9_3_60km/output.h5 \
#     --sampling-config configs/data/sampling_config.yaml \
#     --node-stride 50 \
#     --frame-stride 10

for entry in "${TRAJS[@]}"; do
  zip_name="${entry%%:*}"
  out_name="${entry##*:}"

  zip_path="${FEM_DIR}/${zip_name}.zip"
  src_dir="${FEM_DIR}/${zip_name}"
  out_path="${OUT_DIR}/${out_name}/output.h5"

  echo "=== ${zip_name} ==="

  if [ -d "${src_dir}" ]; then
    echo "Source already extracted at ${src_dir}, skipping unzip."
  else
    echo "Unzipping ${zip_path} ..."
    unzip -qo "${zip_path}" -d "${FEM_DIR}"
  fi

  mkdir -p "$(dirname "${out_path}")"

  echo "Converting ${src_dir} -> ${out_path}"
  python dataset/d3plot_to_h5_dt.py \
    --src "${src_dir}" \
    --tmp "${TMP_DIR}" \
    --out "${out_path}" \
    --sampling-config "${SAMPLING_CONFIG}" \
    --node-stride 50 \
    --frame-stride 10

  if [ "${KEEP_EXTRACTED}" != "1" ]; then
    echo "Removing extracted source ${src_dir} to free disk space ..."
    # zip stores the dir as read-only (dr-xr-xr-x), need write bit to delete contents
    chmod -R u+w "${src_dir}"
    rm -rf "${src_dir}"
  fi

  echo "=== Done ${zip_name} ==="
done

for entry in "${UNZIP_TRAJS[@]}"; do
  src_name="${entry%%:*}"
  out_name="${entry##*:}"

  src_dir="${FEM_DIR}/${src_name}"
  out_path="${OUT_DIR}/${out_name}/output.h5"

  echo "=== ${src_name} (pre-extracted) ==="

  if [ ! -d "${src_dir}" ]; then
    echo "ERROR: Expected source directory not found: ${src_dir}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${out_path}")"

  echo "Converting ${src_dir} -> ${out_path}"
  python dataset/d3plot_to_h5_dt.py \
    --src "${src_dir}" \
    --tmp "${TMP_DIR}" \
    --out "${out_path}" \
    --sampling-config "${SAMPLING_CONFIG}" \
    --node-stride 50 \
    --frame-stride 10

  echo "=== Done ${src_name} ==="
done

echo "All conversions complete."
