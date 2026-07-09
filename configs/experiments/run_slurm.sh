#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-train
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:2
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Usage: sbatch configs/experiments/run_slurm.sh configs/experiments/dg024.yaml

set -euo pipefail

EXPERIMENT=${1:?"Usage: sbatch run_slurm.sh <path/to/experiment.yaml>"}
SIF=/staging/proj_iim1/xrkong/container/collider.sif
REPO=/home/xangruik/collider

# ~/.profile forces CUDA_VISIBLE_DEVICES=-1 to stop GPU use outside SLURM.
# sbatch inherits that, so it must be cleared here or torch will see 0 GPUs
# even though SLURM has allocated real ones.
unset CUDA_VISIBLE_DEVICES

cd "${REPO}"
mkdir -p logs

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID}"
nvidia-smi -L

NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
echo "Launching with ${NUM_GPUS} GPU(s) on experiment ${EXPERIMENT}"

apptainer exec --nv "${SIF}" \
    accelerate launch \
        --num_processes="${NUM_GPUS}" \
        --mixed_precision=bf16 \
        train.py --experiment "${EXPERIMENT}" --skip-git-check

echo "Done."
