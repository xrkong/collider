#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

SIF=/staging/proj_iim1/xrkong/container/collider.sif

mkdir -p logs
unset CUDA_VISIBLE_DEVICES

echo "hello from slurm, job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi -L

apptainer exec --nv "${SIF}" python -c "
import torch, accelerate
print('torch', torch.__version__)
print('accelerate', accelerate.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
"
