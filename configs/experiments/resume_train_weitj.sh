#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-resume-train
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Usage:
#   sbatch configs/experiments/resume_train_weitj.sh <experiment.yaml> <resume-source>
#
# <resume-source> is either:
#   - a local checkpoint file, e.g. outputs/checkpoints/wj01/checkpoint-best.safetensors
#   - a W&B artifact ref,      e.g. checkpoint-wj01:best
# Detected by extension (.safetensors/.pt -> local file, else -> W&B artifact).

set -euo pipefail

EXPERIMENT=${1:?"Usage: sbatch resume_train_weitj.sh <path/to/experiment.yaml> <resume-source>"}
RESUME=${2:?"Usage: sbatch resume_train_weitj.sh <path/to/experiment.yaml> <resume-source>"}
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

case "${RESUME}" in
  *.safetensors|*.pt)
    RESUME_FLAG="--resume-checkpoint"
    ;;
  *)
    RESUME_FLAG="--resume-artifact"
    ;;
esac

echo "Launching with ${NUM_GPUS} GPU(s) on experiment ${EXPERIMENT}"
echo "Resuming weights from ${RESUME} (${RESUME_FLAG})"

apptainer exec --nv --bind /raid "${SIF}" \
    accelerate launch \
        --num_processes="${NUM_GPUS}" \
        --mixed_precision=bf16 \
        train.py --experiment "${EXPERIMENT}" --skip-git-check \
            "${RESUME_FLAG}" "${RESUME}"

echo "Done."
