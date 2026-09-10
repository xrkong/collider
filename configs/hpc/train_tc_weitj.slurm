#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=dot-train-tc
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Time-conditioned (TC) counterpart to train_weitj.slurm — trains via
# train_tc.py instead of train.py, using the same `accelerate launch`
# multi-GPU pattern (bf16, DDP across every GPU in gres=gpu:N). train_tc.py
# also auto-probes each GPU's max batch size at startup (doubling until OOM,
# then binary search) instead of using a fixed train.batch_size — see its
# "Auto batch-size probe" log line. Set train.auto_batch_size: false in the
# experiment yaml to disable this and use a fixed train.batch_size instead.
#
# Usage:
#   sbatch configs/hpc/train_tc_weitj.slurm <experiment.yaml> [resume-source]
#
# [resume-source] is optional — omit it to train from scratch. If given, it's
# either:
#   - a local checkpoint file, e.g. outputs/checkpoints/wj11_tc/checkpoint-best.safetensors
#   - a W&B artifact ref,      e.g. checkpoint-wj11_tc:best
# Detected by extension (.safetensors/.pt -> local file, else -> W&B artifact).
#
# On success, this job self-chains configs/hpc/rollout_tc_weitj.slurm
# (TC per-frame rollout against data.val_dirs[0]) via
# `sbatch --dependency=afterok:$SLURM_JOB_ID`, submitted from inside this job
# right before it exits — Slurm holds the rollout job pending until this one
# finishes, then only releases it if this one exited 0. `set -euo pipefail`
# means a failed run exits the script before reaching that line, so a failed
# training run never queues a rollout against a stale/absent checkpoint.
# Set AUTO_ROLLOUT=0 to skip this and only train:
#   sbatch --export=ALL,AUTO_ROLLOUT=0 configs/hpc/train_tc_weitj.slurm configs/experiments/wj11_tc.yaml

set -euo pipefail

EXPERIMENT=${1:?"Usage: sbatch train_tc_weitj.slurm <path/to/experiment.yaml> [resume-source]"}
RESUME=${2:-}
SIF=/staging/proj_iim1/xrkong/container/collider.sif
REPO=/home/xangruik/collider
AUTO_ROLLOUT="${AUTO_ROLLOUT:-1}"   # 0 = train only, don't queue the rollout job

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

RESUME_ARGS=()
if [ -n "${RESUME}" ]; then
  case "${RESUME}" in
    *.safetensors|*.pt) RESUME_FLAG="--resume-checkpoint" ;;
    *)                  RESUME_FLAG="--resume-artifact" ;;
  esac
  RESUME_ARGS=("${RESUME_FLAG}" "${RESUME}")
  echo "Resuming weights from ${RESUME} (${RESUME_FLAG})"
fi

apptainer exec --nv --bind /raid "${SIF}" \
    accelerate launch \
        --num_processes="${NUM_GPUS}" \
        --mixed_precision=bf16 \
        train_tc.py --experiment "${EXPERIMENT}" --skip-git-check \
            "${RESUME_ARGS[@]}"

echo "Done."

if [ "${AUTO_ROLLOUT}" = "1" ]; then
  echo "Queuing rollout_tc_weitj (TC per-frame rollout, val_dirs[0]) — "
  echo "will start once job ${SLURM_JOB_ID} is confirmed successful."
  ROLLOUT_JOBID=$(sbatch --parsable --dependency=afterok:"${SLURM_JOB_ID}" \
      "${REPO}/configs/hpc/rollout_tc_weitj.slurm" "${EXPERIMENT}")
  echo "Queued rollout_tc_weitj job ${ROLLOUT_JOBID} (dependency: afterok:${SLURM_JOB_ID})"
else
  echo "AUTO_ROLLOUT=0 — skipping automatic rollout queuing."
fi
