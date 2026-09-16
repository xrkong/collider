#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-build-sif
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Rebuilds the collider.sif Apptainer image from source (collider.def) and
# writes it straight to /raid (NOT /staging — that's currently inaccessible,
# see collider-build-dataset-all-990.err), so it's a backup you fully
# control and can push to HuggingFace afterwards.
#
# Needs --fakeroot (no sudo required, just /etc/subuid+subgid entries,
# already present for this user) since it's building a fresh Ubuntu 22.04
# rootfs (apt-get install, conda, pip) rather than exec-ing an existing one.
#
# Usage:
#   sbatch configs/hpc/build_sif.slurm
#
# Output: /raid/proj_iim1/xrkong/container/collider.sif
# To point existing scripts (run_build_dataset.sh, train_weitj.slurm, etc.)
# at this copy instead of the /staging one, update their SIF=... line.

set -euo pipefail

REPO=/home/xangruik/collider
cd "${REPO}"
mkdir -p logs

OUT_DIR=/raid/proj_iim1/xrkong/container
OUT_SIF="${OUT_DIR}/collider.sif"
mkdir -p "${OUT_DIR}"

# apptainer build's scratch/layer-cache writes can be large (base image +
# every %post download) — keep them off the quota-limited default (~/.apptainer
# under $HOME) and on /raid instead, same reasoning as run_build_dataset.sh's
# TMP_DIR.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/raid/proj_iim1/xrkong/tmp/apptainer_build_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/raid/proj_iim1/xrkong/tmp/apptainer_cache}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

# $HOME (/dev/md0) has a hard 40GB user quota and is already ~91% full
# (miniconda3/, voxel/, caches, ...) — a fakeroot build touches $HOME for a
# few things beyond just APPTAINER_TMPDIR/CACHEDIR (fakeroot overlay state,
# XDG cache/data dirs, etc.) and that's enough to hit EDQUOT partway through
# the torch install even with the two vars above set. Override HOME itself
# for this job so nothing can land on the quota-limited filesystem at all.
export HOME="${APPTAINER_TMPDIR%/*}/build_home"
export XDG_CACHE_HOME="${HOME}/.cache"
export XDG_DATA_HOME="${HOME}/.local/share"
export XDG_CONFIG_HOME="${HOME}/.config"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_DATA_HOME}" "${XDG_CONFIG_HOME}"
echo "HOME overridden to ${HOME} for this build (real \$HOME is out of quota)"

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID:-none}"
echo "Building ${OUT_SIF} from collider.def ..."

apptainer build --fakeroot "${OUT_SIF}" collider.def

echo "Done. Built: ${OUT_SIF}"
ls -lh "${OUT_SIF}"

# Sanity check — same as check_sif.slurm's torch/cuda probe.
apptainer exec --nv "${OUT_SIF}" python -c "
import torch
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
"
