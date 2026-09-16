#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-rollout-all-tc
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Time-conditioned (TC) counterpart to rollout_all_weitj.slurm — runs
# src/rollout_tc.py (independent per-frame query, no rollout state) against
# EVERY trajectory in an experiment's data.train_dirs AND data.val_dirs, then
# writes one summary.csv with each case's overall RMSE.
#
# For each case this produces:
#   outputs/rollouts_all_tc/<name>/<label>/rmse.log   (full per-frame + overall RMSE)
#   outputs/rollouts_all_tc/<name>/<label>/*.gif       (pred-vs-gt animation, see GIF=0 below)
# and after every case:
#   outputs/rollouts_all_tc/<name>/summary.csv        (label, overall_rmse_mm)
#
# GIF rendering (via src/rollout.py's render_vis) is on by default — set
# GIF=0 to skip it and only compute RMSE, e.g. for a quick numeric sweep:
#   sbatch --export=ALL,GIF=0 configs/hpc/rollout_all_tc_weitj.slurm configs/experiments/wj11_tc.yaml
#
# Reads the checkpoint + both stats jsons straight from local disk
# (outputs/checkpoints/<name>/) — train_tc.py always writes there, and
# rollout_tc.py has no W&B-artifact-pull path (unlike src/rollout.py's
# load_model, which can fall back to one).
#
# Each case's barrier_label/layers/kirigami_thickness/inter_layer_plate_thickness/
# w_beam_thickness/speed/angle (needed for the model's condition input) is
# read straight from that case's own data.train_dirs/val_dirs entry — see
# train.py's parse_dir_entry — so mixed-condition experiments are rolled out
# with the correct condition per case.
#
# Usage: sbatch configs/hpc/rollout_all_tc_weitj.slurm <experiment.yaml>
#   sbatch configs/hpc/rollout_all_tc_weitj.slurm configs/experiments/wj11_tc.yaml
#
# Exactly one experiment yaml per submission (unlike rollout_tc_weitj.slurm) —
# "all cases" here already means every trajectory *within* that one
# experiment's config.

set -euo pipefail

SIF=/raid/proj_iim1/xrkong/container/collider.sif
REPO=/home/xangruik/collider
GIF="${GIF:-1}"   # 0 = skip GIF rendering, RMSE-only sweep

unset CUDA_VISIBLE_DEVICES

cd "${REPO}"
mkdir -p logs

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID:-none}"
nvidia-smi -L

EXPERIMENT=${1:?"Usage: sbatch configs/hpc/rollout_all_tc_weitj.slurm <path/to/experiment.yaml>"}
case "${EXPERIMENT}" in
  */*|*.yaml|*.yml) ;;
  *)
    echo "Error: '${EXPERIMENT}' is not a yaml path (expected e.g. configs/experiments/wj11_tc.yaml)." >&2
    exit 1
    ;;
esac

# Extract cfg["name"] and every data.train_dirs/val_dirs entry (path +
# condition fields, defaults already applied by parse_dir_entry) straight
# from the experiment yaml, one '|'-separated line per case:
#   <label>|<path>|<barrier_label>|<layers>|<kirigami_thickness>|
#     <inter_layer_plate_thickness>|<w_beam_thickness>|<speed>|<angle>
# label = "<source><index>_<traj_name>", e.g. "train0_New_Road_Barrier_..." /
# "val0_T_lok_F_shape_barrier_9_3_100km" — unique even if two entries reuse
# the same h5 path with different overrides.
#
# '|' (not tab) on purpose: T_lok/baseline rows leave inter_layer_plate_thickness
# and w_beam_thickness unset (None -> empty field), and bash's `read` treats
# tab as "IFS whitespace" no matter what IFS is set to, which collapses runs
# of consecutive empty tab-delimited fields and silently shifts every field
# after them. '|' has no such special-casing, so empty fields stay empty.
get_cases() {
  apptainer exec --bind /raid "${SIF}" python -c "
from train import load_config, parse_dir_entry
from src.dataset import traj_name_from_h5
cfg = load_config('${1}')
print(cfg['name'])
for source in ('train', 'val'):
    dirs = cfg['data'].get(f'{source}_dirs', [])
    for i, raw_entry in enumerate(dirs):
        e = parse_dir_entry(raw_entry)
        ts_name = traj_name_from_h5(e['path'])
        label = f'{source}{i}_{ts_name}'
        ilpt  = e['inter_layer_plate_thickness'] if e['inter_layer_plate_thickness'] is not None else ''
        wbt   = e['w_beam_thickness'] if e['w_beam_thickness'] is not None else ''
        speed = e['speed'] if e['speed'] is not None else ''
        print(f\"{label}|{e['path']}|{e['barrier_label']}|{e['layers']}|{e['kirigami_thickness']}|{ilpt}|{wbt}|{speed}|{e['angle']}\")
"
}

echo "=== Reading cases from ${EXPERIMENT} ==="
CASES_RAW="$(get_cases "${EXPERIMENT}")"
NAME="$(echo "${CASES_RAW}" | sed -n 1p)"
CASE_LINES="$(echo "${CASES_RAW}" | tail -n +2)"

if [ -z "${CASE_LINES}" ]; then
  echo "Error: no data.train_dirs/val_dirs entries found in ${EXPERIMENT}." >&2
  exit 1
fi

N_CASES="$(echo "${CASE_LINES}" | wc -l)"
echo "Experiment: ${NAME} — ${N_CASES} case(s) (train_dirs + val_dirs)"

checkpoint="outputs/checkpoints/${NAME}/checkpoint-best.safetensors"
if [ ! -f "${checkpoint}" ]; then
  echo "Error: checkpoint not found: ${checkpoint}" >&2
  exit 1
fi

OUT_ROOT="outputs/rollouts_all_tc/${NAME}"
mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/summary.csv"
echo "label,overall_rmse_mm" > "${SUMMARY}"

while IFS='|' read -r label h5_path barrier_label layers kirigami_thickness inter_layer_plate_thickness w_beam_thickness speed angle; do
  [ -z "${label}" ] && continue

  case_dir="${OUT_ROOT}/${label}"
  mkdir -p "${case_dir}"
  echo ""
  echo "=== ${NAME} :: ${label} ==="
  echo "h5=${h5_path} barrier_label=${barrier_label} layers=${layers} kirigami_thickness=${kirigami_thickness}"

  extra_args=()
  [ -n "${kirigami_thickness}" ] && extra_args+=(--kirigami-thickness "${kirigami_thickness}")
  [ -n "${inter_layer_plate_thickness}" ] && extra_args+=(--inter-layer-plate-thickness "${inter_layer_plate_thickness}")
  [ -n "${w_beam_thickness}" ] && extra_args+=(--w-beam-thickness "${w_beam_thickness}")
  [ -n "${speed}" ] && extra_args+=(--speed "${speed}")
  [ "${GIF}" = "1" ] && extra_args+=(--output-dir "${case_dir}" --gif)

  apptainer exec --nv --bind /raid "${SIF}" \
    python src/rollout_tc.py \
      --checkpoint "${checkpoint}" \
      --experiment "${EXPERIMENT}" \
      --raw-h5 "${h5_path}" \
      --barrier-label "${barrier_label}" \
      --layers "${layers}" \
      --angle "${angle}" \
      "${extra_args[@]}" \
    | tee "${case_dir}/rmse.log"

  overall="$(grep -oP 'Overall RMSE: \K[0-9.]+' "${case_dir}/rmse.log" || echo "")"
  echo "${label},${overall}" >> "${SUMMARY}"

  echo "=== Done ${label} — overall RMSE: ${overall} mm ==="
done <<< "${CASE_LINES}"

echo ""
echo "=== All ${N_CASES} case(s) complete ==="
echo "Per-case logs: ${OUT_ROOT}/<case>/rmse.log"
echo "Summary:       ${SUMMARY}"
column -s, -t "${SUMMARY}"
