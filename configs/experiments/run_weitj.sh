#!/bin/bash 

#SBATCH --account=project_iim1

#SBATCH --gres=gpu:2 

#SBATCH --gpu-bind=closest 

#SBATCH --time=00:20:00  

 

 

echo "hello from slurm" 

echo `nvidia-smi -L` 

echo "--!--" 

echo "Slurm uid" ${SLURM_JOB_UID} 

echo "Slurm Job id", ${SLURM_JOB_ID} 

echo "cat /proc/self/cgroup", `cat /proc/self/cgroup` 

echo "Slurm localid", ${SLURM_LOCALID} 

 

 

echo "running nvidia-smi" 

echo `nvidia-smi -L` 