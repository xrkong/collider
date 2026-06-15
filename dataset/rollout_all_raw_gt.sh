#!/usr/bin/env bash
# Run raw_gt rollouts (with GIF) for every converted h5 trajectory under
# /home/kong/datasets/barrier/h5dt_50ns_5fs_mat, for visual sanity checking.
#
# Usage:
#   bash dataset/rollout_all_raw_gt.sh
#
# Each output goes to outputs/rollouts/<traj_name>/traj_9_3_gt.gif
# (out_dir defaults to the h5's parent folder name).

set -euo pipefail

H5_ROOT=/home/kong/datasets/barrier/h5dt_50ns_5fs_mat

cd /home/kong/xrkong/collider

for h5 in "${H5_ROOT}"/*/output.h5; do
  echo "=== ${h5} ==="
  python src/rollout.py \
    --raw-h5 "${h5}" \
    --mode raw_gt \
    --gif --gif-fps 10 --gif-name traj_9_3_gt
done

echo "All rollouts complete."
