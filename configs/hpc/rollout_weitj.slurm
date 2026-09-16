#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-rollout
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs src/rollout.py (one-step + autoregressive, GIF + RMSE plots) for each
# experiment's checkpoint-best.safetensors against one of its trajectories,
# then overlays all of them on one multi-experiment RMSE comparison plot.
#
# Checkpoints aren't kept on local disk — rollout.py pulls checkpoint-<name>:best
# (and its global_stats.json) straight from the W&B artifact given just
# --experiment, so no local outputs/checkpoints/<name>/ is required.
#
# Usage: sbatch configs/hpc/rollout_weitj.slurm <experiment.yaml> [more.yaml ...]
#   sbatch configs/hpc/rollout_weitj.slurm configs/experiments/wj06_0.yaml
#   sbatch configs/hpc/rollout_weitj.slurm configs/experiments/wj06_0.yaml configs/experiments/wj06_1.yaml
#
# Each arg must be an explicit path to a yaml file — no bare-name resolution
# and no default experiment list; at least one path is required.
#
# By default this rolls out on data.val_dirs[0], same as before. To pick a
# different entry (e.g. a specific train_dirs trajectory), set DIR_SOURCE
# (val|train) and/or DIR_INDEX:
#   sbatch --export=ALL,DIR_SOURCE=train,DIR_INDEX=1 \
#       configs/hpc/rollout_weitj.slurm configs/experiments/wj09_1st_run.yaml
# Non-default selections get their own output dir
# (outputs/rollouts/<name>_<source><index>) so they never collide with the
# default val_dirs[0] rollout for the same experiment.
#
# The checkpoint/output directory name always comes from the yaml's own
# top-level `name:` field (cfg["name"], same as train.py uses for its W&B
# artifact and outputs/checkpoints/<name>/), NOT from the filename — a yaml
# file's stem doesn't have to match the `name:` it declares inside.

set -euo pipefail

SIF=/raid/proj_iim1/xrkong/container/collider.sif
REPO=/home/xangruik/collider

DIR_SOURCE="${DIR_SOURCE:-val}"     # "val" or "train" — which data.*_dirs list to pull from
DIR_INDEX="${DIR_INDEX:-0}"         # index into that list

case "${DIR_SOURCE}" in
  val|train) ;;
  *) echo "DIR_SOURCE must be 'val' or 'train' (got '${DIR_SOURCE}')." >&2; exit 1 ;;
esac
case "${DIR_INDEX}" in
  ''|*[!0-9]*) echo "DIR_INDEX must be a non-negative integer (got '${DIR_INDEX}')." >&2; exit 1 ;;
esac

# Default selection (val, index 0) keeps the original output path unchanged;
# any other selection gets its own suffixed dir so it can't collide with it.
if [ "${DIR_SOURCE}" = "val" ] && [ "${DIR_INDEX}" = "0" ]; then
  OUT_SUFFIX=""
else
  OUT_SUFFIX="_${DIR_SOURCE}${DIR_INDEX}"
fi

# ~/.profile forces CUDA_VISIBLE_DEVICES=-1 to stop GPU use outside SLURM.
unset CUDA_VISIBLE_DEVICES

cd "${REPO}"
mkdir -p logs

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID:-none}"
nvidia-smi -L

# Extract cfg["name"] and data.<DIR_SOURCE>_dirs[DIR_INDEX] straight from the
# experiment yaml (via train.py's own load_config + parse_dir_entry — the same
# parser train.py/rollout.py already use), instead of hand-duplicating either
# here. h5 path has any ":<angle>" suffix stripped (rollout.py re-derives the
# angle from the h5 name itself).
get_exp_info() {
  apptainer exec --bind /raid "${SIF}" python -c "
from train import load_config, parse_dir_entry
cfg = load_config('${1}')
dirs = cfg['data']['${DIR_SOURCE}_dirs']
if ${DIR_INDEX} >= len(dirs):
    raise IndexError(f'DIR_INDEX=${DIR_INDEX} out of range for data.${DIR_SOURCE}_dirs '
                      f'(only {len(dirs)} entries) in ${1}')
entry = parse_dir_entry(dirs[${DIR_INDEX}])
print(cfg['name'])
print(entry['path'])
"
}

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch configs/hpc/rollout_weitj.slurm <path/to/experiment.yaml> [more.yaml ...]" >&2
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
  COMPARE_DIRS+=("${name}${OUT_SUFFIX}:outputs/rollouts/${name}${OUT_SUFFIX}")
done

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  experiment="${EXPERIMENTS[$i]}"

  echo "=== ${name} (data.${DIR_SOURCE}_dirs[${DIR_INDEX}]) ==="
  raw_h5="${VAL_H5S[$i]}"
  echo "h5 (from ${experiment}): ${raw_h5}"

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
      --output-dir "outputs/rollouts/${name}${OUT_SUFFIX}" \
      "${extra_args[@]}"

  # GC / barrier kinematics plots (ORA_x, ORA_y, ASI, displacement) from the
  # gc_barrier_onestep.csv / gc_barrier_autoregressive.csv rollout.py just
  # wrote — see src/gc_barrier.py + src/plot_gc_barrier.py. Only needs
  # numpy/matplotlib, so no --nv.
  apptainer exec --bind /raid "${SIF}" \
    python src/plot_gc_barrier.py "outputs/rollouts/${name}${OUT_SUFFIX}"

  echo "=== Done ${name} ==="
done

echo "All rollouts complete. Comparison plot: outputs/rollouts/${NAMES[-1]}${OUT_SUFFIX}/multi_experiment_rmse.png"
