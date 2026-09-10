#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-rollout-tc
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Time-conditioned (TC) counterpart to rollout_weitj.slurm — runs
# src/rollout_tc.py (independent per-frame query, no rollout state) against
# data.val_dirs[0], with a pred-vs-gt GIF (reuses src/rollout.py's render_vis).
#
# Unlike the AR rollout jobs, this reads the checkpoint + both stats jsons
# straight from local disk (outputs/checkpoints/<name>/) — train_tc.py always
# writes there, and rollout_tc.py has no W&B-artifact-pull path (unlike
# src/rollout.py's load_model, which can fall back to one).
#
# Usage:
#   sbatch configs/hpc/rollout_tc_weitj.slurm <experiment.yaml> [more.yaml ...]
#
# Each arg must be an explicit path to a yaml file.

set -euo pipefail

SIF=/staging/proj_iim1/xrkong/container/collider.sif
REPO=/home/xangruik/collider

unset CUDA_VISIBLE_DEVICES

cd "${REPO}"
mkdir -p logs

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID:-none}"
nvidia-smi -L

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch configs/hpc/rollout_tc_weitj.slurm <path/to/experiment.yaml> [more.yaml ...]" >&2
  exit 1
fi

for experiment in "$@"; do
  case "${experiment}" in
    */*|*.yaml|*.yml) ;;
    *)
      echo "Error: '${experiment}' is not a yaml path (expected e.g. configs/experiments/wj11_tc.yaml)." >&2
      exit 1
      ;;
  esac
done

# Extract cfg["name"] and data.val_dirs[0] (path + condition fields, defaults
# already applied by parse_dir_entry) straight from the experiment yaml.
get_exp_info() {
  apptainer exec --bind /raid "${SIF}" python -c "
from train import load_config, parse_dir_entry
cfg = load_config('${1}')
dirs = cfg['data']['val_dirs']
if not dirs:
    raise IndexError('data.val_dirs is empty in ${1}')
entry = parse_dir_entry(dirs[0])
print(cfg['name'])
print(entry['path'])
print(entry['barrier_label'])
print(entry['layers'])
print(entry['kirigami_thickness'])
print(entry['inter_layer_plate_thickness'] if entry['inter_layer_plate_thickness'] is not None else '')
print(entry['w_beam_thickness'] if entry['w_beam_thickness'] is not None else '')
print(entry['speed'] if entry['speed'] is not None else '')
print(entry['angle'])
"
}

for experiment in "$@"; do
  info="$(get_exp_info "${experiment}")"
  name="$(echo "${info}" | sed -n 1p)"
  raw_h5="$(echo "${info}" | sed -n 2p)"
  barrier_label="$(echo "${info}" | sed -n 3p)"
  layers="$(echo "${info}" | sed -n 4p)"
  kirigami_thickness="$(echo "${info}" | sed -n 5p)"
  inter_layer_plate_thickness="$(echo "${info}" | sed -n 6p)"
  w_beam_thickness="$(echo "${info}" | sed -n 7p)"
  speed="$(echo "${info}" | sed -n 8p)"
  angle="$(echo "${info}" | sed -n 9p)"

  echo ""
  echo "=== ${name} (data.val_dirs[0]) ==="
  echo "h5=${raw_h5} barrier_label=${barrier_label} layers=${layers} kirigami_thickness=${kirigami_thickness}"

  ckpt_dir="outputs/checkpoints/${name}"
  checkpoint="${ckpt_dir}/checkpoint-best.safetensors"
  if [ ! -f "${checkpoint}" ]; then
    echo "Error: checkpoint not found: ${checkpoint}" >&2
    exit 1
  fi

  extra_args=()
  [ -n "${kirigami_thickness}" ] && extra_args+=(--kirigami-thickness "${kirigami_thickness}")
  [ -n "${inter_layer_plate_thickness}" ] && extra_args+=(--inter-layer-plate-thickness "${inter_layer_plate_thickness}")
  [ -n "${w_beam_thickness}" ] && extra_args+=(--w-beam-thickness "${w_beam_thickness}")
  [ -n "${speed}" ] && extra_args+=(--speed "${speed}")

  out_dir="outputs/rollouts_tc/${name}"
  mkdir -p "${out_dir}"

  apptainer exec --nv --bind /raid "${SIF}" \
    python src/rollout_tc.py \
      --checkpoint "${checkpoint}" \
      --experiment "${experiment}" \
      --raw-h5 "${raw_h5}" \
      --barrier-label "${barrier_label}" \
      --layers "${layers}" \
      --angle "${angle}" \
      --output-dir "${out_dir}" \
      --gif \
      "${extra_args[@]}" \
    | tee "${out_dir}/rmse.log"

  echo "=== Done ${name} — log: ${out_dir}/rmse.log, GIF: ${out_dir}/*.gif ==="
done
