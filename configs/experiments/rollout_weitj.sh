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
# Usage: sbatch configs/experiments/rollout_weitj.sh <experiment.yaml> [more.yaml ...]
#   sbatch configs/experiments/rollout_weitj.sh configs/experiments/wj06_0.yaml
#   sbatch configs/experiments/rollout_weitj.sh configs/experiments/wj06_0.yaml configs/experiments/wj06_1.yaml
#
# Each arg must be an explicit path to a yaml file — no bare-name resolution
# and no default experiment list; at least one path is required.
#
# The checkpoint/output directory name always comes from the yaml's own
# top-level `name:` field (cfg["name"], same as train.py uses for its W&B
# artifact and outputs/checkpoints/<name>/), NOT from the filename — a yaml
# file's stem doesn't have to match the `name:` it declares inside.

set -euo pipefail

SIF=/staging/proj_iim1/xrkong/container/collider.sif
REPO=/home/xangruik/collider

# ~/.profile forces CUDA_VISIBLE_DEVICES=-1 to stop GPU use outside SLURM.
unset CUDA_VISIBLE_DEVICES

cd "${REPO}"
mkdir -p logs

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID:-none}"
nvidia-smi -L

# Extract cfg["name"] and data.val_dirs[0] straight from the experiment yaml
# (via train.py's own load_config + parse_dir_entry — the same parser
# train.py/rollout.py already use), instead of hand-duplicating either here.
# val_h5 has its ":<angle>" suffix stripped (rollout.py re-derives the angle
# from the h5 name itself).
get_exp_info() {
  apptainer exec --bind /raid "${SIF}" python -c "
from train import load_config, parse_dir_entry
cfg = load_config('${1}')
path, _ = parse_dir_entry(cfg['data']['val_dirs'][0])
print(cfg['name'])
print(path)
"
}

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch configs/experiments/rollout_weitj.sh <path/to/experiment.yaml> [more.yaml ...]" >&2
  exit 1
fi

EXPERIMENTS=("$@")
for experiment in "${EXPERIMENTS[@]}"; do
  case "${experiment}" in
    */*|*.yaml|*.yml) ;;
    *)
      echo "Error: '${experiment}' is not a yaml path (expected e.g. configs/experiments/wj06_0.yaml)." >&2
      exit 1
      ;;
  esac
done

NAMES=()
VAL_H5S=()
for experiment in "${EXPERIMENTS[@]}"; do
  info="$(get_exp_info "${experiment}")"
  NAMES+=("$(echo "${info}" | sed -n 1p)")
  VAL_H5S+=("$(echo "${info}" | sed -n 2p)")
done

COMPARE_DIRS=()
for name in "${NAMES[@]}"; do
  COMPARE_DIRS+=("${name}:outputs/rollouts/${name}")
done

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  experiment="${EXPERIMENTS[$i]}"

  echo "=== ${name} ==="
  raw_h5="${VAL_H5S[$i]}"
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

  # GC / barrier kinematics plots (ORA_x, ORA_y, ASI, displacement) from the
  # gc_barrier_onestep.csv / gc_barrier_autoregressive.csv rollout.py just
  # wrote — see src/gc_barrier.py + src/plot_gc_barrier.py. Only needs
  # numpy/matplotlib, so no --nv.
  apptainer exec --bind /raid "${SIF}" \
    python src/plot_gc_barrier.py "outputs/rollouts/${name}"

  echo "=== Done ${name} ==="
done

echo "All rollouts complete. Comparison plot: outputs/rollouts/${NAMES[-1]}/multi_experiment_rmse.png"
