#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-build-dataset
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs dataset/run_build_dataset.sh (k-file + d3plot -> HDF5, plus the
# GC/barrier CSV+plots) as a SLURM batch job. No GPU requested — d3plot
# parsing/extraction is pure CPU + joblib work (no torch/CUDA involved).
# --cpus-per-task=8 matches run_build_dataset.sh's --n-jobs 8.
#
# Usage:
#   sbatch configs/experiments/build_dataset_weitj.sh <name> [<name> ...]
#   sbatch configs/experiments/build_dataset_weitj.sh --all
#
# <name> is a subdirectory of FEM_DIR (see dataset/run_build_dataset.sh),
# e.g.:
#   sbatch configs/experiments/build_dataset_weitj.sh T_lok_F_shape_barrier_9_3_100km
#
# All of run_build_dataset.sh's env-var overrides (FRAME_STRIDE, FRAME_LIMIT,
# GC_BARRIER_WINDOW_MS, FORCE, TMP_DIR) work the same way here — pass them
# through with --export so they reach the job regardless of the cluster's
# default export policy, e.g.:
#   sbatch --export=ALL,FRAME_STRIDE=4,FRAME_LIMIT=50 \
#       configs/experiments/build_dataset_weitj.sh T_lok_F_shape_barrier_9_3_100km

set -euo pipefail

REPO=/home/xangruik/collider
cd "${REPO}"
mkdir -p logs

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID:-none}"

bash dataset/run_build_dataset.sh "$@"

echo "Done."
