#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-build-dataset-all
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Batch-converts EVERY trajectory under FEM_DIR (dataset/run_build_dataset.sh's
# default: /raid/proj_iim1/xrkong/fem) from raw k-file + d3plot into one HDF5
# each — i.e. `dataset/run_build_dataset.sh --all` as its own dedicated job,
# so you don't have to remember the --all argument each time. Same underlying
# script as build_dataset_weitj.slurm (see that file for the single/multi
# -name form) — this one is just pinned to "convert everything".
#
# Already-converted datasets (output .h5 exists in OUT_DIR) are skipped, so
# this is safe to re-submit if the job times out partway through — it picks
# up where it left off. Set FORCE=1 to reconvert everything from scratch.
#
# --time=48:00:00 is a rough upper bound for ~11 trajectories run serially
# (each trajectory's own conversion is budgeted ~4h in build_dataset_weitj.slurm) —
# raise it with `sbatch --time=...` below if you have more trajectories or a
# larger --budget-scale/--full-res (both make node sampling slower).
#
# Usage (all env-var overrides from run_build_dataset.sh apply — pass them
# through with --export so they reach the job):
#
#   # Default: same ~100k-node FPS-sampled dataset as always, into
#   # h5_fps_no_wheel/ (matches OUT_DIR's default in run_build_dataset.sh).
#   sbatch configs/hpc/build_dataset_all.slurm
#
#   # Higher point-cloud density (e.g. ~500k nodes/region-budget * 5),
#   # written to its own OUT_DIR so it doesn't clobber the existing dataset:
#   sbatch --export=ALL,BUDGET_SCALE=5,OUT_DIR=/raid/proj_iim1/xrkong/h5_budget5x \
#       configs/hpc/build_dataset_all.slurm
#
#   # True raw/full-resolution mesh (every node, no sampling — see
#   # dataset/build_dataset.py's --full-res for caveats: ~1.8M-3.3M
#   # nodes/trajectory, much more host RAM/GPU memory at train time):
#   sbatch --export=ALL,FULL_RES=1,OUT_DIR=/raid/proj_iim1/xrkong/h5_full_res \
#       configs/hpc/build_dataset_all.slurm
#
#   # Faster (less even) sampling if fps is too slow at a high BUDGET_SCALE:
#   sbatch --export=ALL,BUDGET_SCALE=5,METHOD=stride,OUT_DIR=/raid/proj_iim1/xrkong/h5_budget5x \
#       configs/hpc/build_dataset_all.slurm

set -euo pipefail

REPO=/home/xangruik/collider
cd "${REPO}"
mkdir -p logs

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID:-none}"

bash dataset/run_build_dataset.sh --all

echo "Done."
