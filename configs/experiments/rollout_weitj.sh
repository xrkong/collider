#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-rollout
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs src/rollout.py (one-step + autoregressive, GIF + RMSE plots) for each
# experiment's checkpoint-best.safetensors against its own val trajectory,
# then overlays all of them on one multi-experiment RMSE comparison plot.
#
# Checkpoints aren't kept on local disk — rollout.py pulls checkpoint-<name>:best
# (and its global_stats.json) straight from the W&B artifact given just
# --experiment, so no local outputs/checkpoints/<name>/ is required.
#
# Usage: sbatch configs/experiments/rollout_weitj.sh [name ...]
#        (defaults to wj01 wj02 wj03)

set -euo pipefail

SIF=/staging/proj_iim1/xrkong/container/collider.sif
REPO=/home/xangruik/collider

# ~/.profile forces CUDA_VISIBLE_DEVICES=-1 to stop GPU use outside SLURM.
unset CUDA_VISIBLE_DEVICES

cd "${REPO}"
mkdir -p logs

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID:-none}"
nvidia-smi -L

# Extract data.val_dirs[0] straight from the experiment yaml (via train.py's
# own load_config + parse_dir_entry — the same parser train.py/rollout.py
# already use), instead of hand-duplicating each path here. Strips the
# ":<angle>" suffix (rollout.py re-derives the angle from the h5 name itself).
get_val_h5() {
  apptainer exec --bind /raid "${SIF}" python -c "
from train import load_config, parse_dir_entry
cfg = load_config('configs/experiments/${1}.yaml')
path, _ = parse_dir_entry(cfg['data']['val_dirs'][0])
print(path)
"
}

if [ "$#" -eq 0 ]; then
  NAMES=(wj01 wj02 wj03)
else
  NAMES=("$@")
fi

COMPARE_DIRS=()
for name in "${NAMES[@]}"; do
  COMPARE_DIRS+=("${name}:outputs/rollouts/${name}")
done

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  experiment="configs/experiments/${name}.yaml"

  echo "=== ${name} ==="
  raw_h5="$(get_val_h5 "${name}")"
  echo "val h5 (from ${experiment}): ${raw_h5}"

  extra_args=()
  if [ "$((i + 1))" -eq "${#NAMES[@]}" ]; then
    # Last experiment: also emit the multi-experiment comparison plot, using
    # the pkls this loop just wrote to outputs/rollouts/<name>/.
    extra_args=(--compare-dirs "${COMPARE_DIRS[@]}")
  fi

  # Older checkpoint-<name>:best artifacts (logged before global_stats.json
  # started being bundled into them, or by a still-running job that loaded
  # train.py before that change) won't carry norm stats. Fall back to the
  # local copy left behind by training, if one is still on disk.
  local_stats="outputs/checkpoints/${name}/global_stats.json"
  if [ -f "${local_stats}" ]; then
    extra_args+=(--stats-path "${local_stats}")
  fi

  apptainer exec --nv --bind /raid "${SIF}" \
    python src/rollout.py \
      --experiment "${experiment}" \
      --raw-h5 "${raw_h5}" \
      --mode both \
      --gif --plot \
      --output-dir "outputs/rollouts/${name}" \
      "${extra_args[@]}"

  echo "=== Done ${name} ==="
done

echo "All rollouts complete. Comparison plot: outputs/rollouts/${NAMES[-1]}/multi_experiment_rmse.png"
