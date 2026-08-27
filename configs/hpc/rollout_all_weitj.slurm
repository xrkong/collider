#!/bin/bash

#SBATCH --account=proj_iim1
#SBATCH --job-name=collider-rollout-all
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs src/rollout.py in AUTOREGRESSIVE-ONLY mode (no one-step) against every
# trajectory in an experiment's data.train_dirs AND data.val_dirs — i.e. all
# train/val cases, not just val_dirs[0] like rollout_weitj.slurm.
#
# For each case this produces its own output folder
# (outputs/rollouts_all/<name>/<source><index>_<traj_name>/) containing:
#   - autoregressive.pkl, GIF, rmse plots      (src/rollout.py)
#   - gc_barrier_autoregressive.csv            (src/gc_barrier.py)
#   - gc_barrier_ora_x.png / ora_y.png / asi.png / displacement.png,
#     GT vs autoregressive only (no one-step curve)   (src/plot_gc_barrier.py)
#
# After every case is done, it also writes THREE aggregate plots overlaying
# every case's autoregressive curve on one figure each (legend = case label):
#   outputs/rollouts_all/<name>/gc_barrier_ora_x_all_cases.png
#   outputs/rollouts_all/<name>/gc_barrier_ora_y_all_cases.png
#   outputs/rollouts_all/<name>/gc_barrier_displacement_all_cases.png
#                                               (src/plot_gc_barrier_multi.py)
#
# Each case's barrier_label/layers/thickness (needed for the model's
# condition input whenever condition.use_material is set) is read straight
# from that case's own data.train_dirs/val_dirs entry — see
# train.py's parse_dir_entry — so mixed-condition experiments (e.g.
# wj09_round_3.yaml, where different trajectories use different barrier
# designs) are rolled out with the correct condition per case.
#
# Checkpoints aren't kept on local disk — rollout.py pulls checkpoint-<name>:best
# (and its global_stats.json) straight from the W&B artifact given just
# --experiment, so no local outputs/checkpoints/<name>/ is required.
#
# Usage: sbatch configs/hpc/rollout_all_weitj.slurm <experiment.yaml>
#   sbatch configs/hpc/rollout_all_weitj.slurm configs/experiments/wj09_round_3.yaml
#
# Exactly one experiment yaml per submission (unlike rollout_weitj.slurm) —
# the "all cases" here already means every trajectory *within* that one
# experiment's config, so multiple experiments would just make the aggregate
# plots ambiguous about which model produced which curve.

set -euo pipefail

SIF=/staging/proj_iim1/xrkong/container/collider.sif
REPO=/home/xangruik/collider

# ~/.profile forces CUDA_VISIBLE_DEVICES=-1 to stop GPU use outside SLURM.
unset CUDA_VISIBLE_DEVICES

cd "${REPO}"
mkdir -p logs

echo "Running on $(hostname), Job ID ${SLURM_JOB_ID:-none}"
nvidia-smi -L

EXPERIMENT=${1:?"Usage: sbatch configs/hpc/rollout_all_weitj.slurm <path/to/experiment.yaml>"}
case "${EXPERIMENT}" in
  */*|*.yaml|*.yml) ;;
  *)
    echo "Error: '${EXPERIMENT}' is not a yaml path (expected e.g. configs/experiments/wj09_round_3.yaml)." >&2
    exit 1
    ;;
esac

# Extract cfg["name"] and every data.train_dirs/val_dirs entry (path +
# barrier_label/layers/thickness, defaults already applied by parse_dir_entry)
# straight from the experiment yaml, one tab-separated line per case:
#   <label>\t<path>\t<barrier_label>\t<layers>\t<thickness>
# label = "<source><index>_<traj_name>", e.g. "train0_New_Road_Barrier_..." /
# "val0_T_lok_F_shape_barrier_9_3_100km" — unique even if two entries reuse
# the same h5 path with different barrier overrides.
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
        print(f\"{label}\t{e['path']}\t{e['barrier_label']}\t{e['layers']}\t{e['thickness']}\")
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

OUT_ROOT="outputs/rollouts_all/${NAME}"
mkdir -p "${OUT_ROOT}"

COMPARE_ARGS=()

while IFS=$'\t' read -r label h5_path barrier_label layers thickness; do
  [ -z "${label}" ] && continue

  case_dir="${OUT_ROOT}/${label}"
  echo ""
  echo "=== ${NAME} :: ${label} ==="
  echo "h5=${h5_path} barrier_label=${barrier_label} layers=${layers} thickness=${thickness}"

  # Older checkpoint-<name>:best artifacts (logged before global_stats.json
  # started being bundled into them) won't carry norm stats. Fall back to the
  # local copy left behind by training, if one is still on disk.
  extra_args=()
  local_stats="outputs/checkpoints/${NAME}/global_stats.json"
  if [ -f "${local_stats}" ]; then
    extra_args+=(--stats-path "${local_stats}")
  fi

  apptainer exec --nv --bind /raid "${SIF}" \
    python src/rollout.py \
      --experiment "${EXPERIMENT}" \
      --raw-h5 "${h5_path}" \
      --barrier-label "${barrier_label}" \
      --layers "${layers}" \
      --thickness "${thickness}" \
      --mode autoregressive \
      --gif --plot \
      --output-dir "${case_dir}" \
      "${extra_args[@]}"

  # GC / barrier kinematics plots (ORA_x, ORA_y, ASI, displacement): GT vs
  # autoregressive only — no one-step CSV exists in this mode, see
  # src/plot_gc_barrier.py's optional-onestep handling.
  apptainer exec --bind /raid "${SIF}" \
    python src/plot_gc_barrier.py "${case_dir}"

  COMPARE_ARGS+=("${label}:${case_dir}")

  echo "=== Done ${label} ==="
done <<< "${CASE_LINES}"

echo ""
echo "=== Aggregate AR-only comparison across all ${N_CASES} case(s) ==="
apptainer exec --bind /raid "${SIF}" \
  python src/plot_gc_barrier_multi.py \
    --cases "${COMPARE_ARGS[@]}" \
    --out-dir "${OUT_ROOT}"

echo ""
echo "All cases complete. Per-case outputs under ${OUT_ROOT}/<case>/, aggregate plots:"
echo "  ${OUT_ROOT}/gc_barrier_ora_x_all_cases.png"
echo "  ${OUT_ROOT}/gc_barrier_ora_y_all_cases.png"
echo "  ${OUT_ROOT}/gc_barrier_displacement_all_cases.png"
