"""Plot vehicle rigid-body rotation (roll/pitch/yaw): ground truth vs
one-step vs autoregressive.

Rotation isn't one of the columns in gc_barrier_onestep.csv/
gc_barrier_autoregressive.csv (those track two single points' translational
kinematics) — it's fit from all 8 nodes of the vehicle-CG rigid hex (PID
9000100, see src/gc_barrier.py) via the Kabsch algorithm, using the full
(T, N, 15) position arrays rollout.py already pickles to
{stem}onestep.pkl / {stem}autoregressive.pkl. This script reads those plus
the original --raw-h5 (for node_part_id and the frame-0 reference shape) —
no changes to rollout.py or its CSV/pkl output.

Usage:
    python src/plot_rotation.py outputs/rollouts/wj06 --raw-h5 /path/to/T_lok_..._60km.h5
    python src/plot_rotation.py outputs/rollouts/wj06 --raw-h5 ... --stem T_lok_F_shape_barrier_9_3_60km
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gc_barrier import locate_cg_cluster, compute_rigid_rotation_series  # noqa: E402
from src.plot_gc_barrier import _RCPARAMS, STYLE_GT, STYLE_OS, STYLE_AR, make_plot  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROTATION_YLIM = (-50.0, 20.0)


def load_positions_and_pids(h5_path: str) -> dict:
    """Minimal h5 read — just node_part_id, times, and frame-0 positions
    (this script never needs velocity/acceleration/full trajectory, so it
    avoids src.rollout.load_raw_h5's much heavier import chain, which pulls
    in the whole model registry (torch/timm/torchvision) just to read an
    h5 file)."""
    with h5py.File(h5_path, "r") as f:
        return {
            "node_part_id": f["metadata/node_part_id"][:],
            "positions_frame0": f["states/positions"][0].astype(np.float32),
            "times": f["states/times"][:],
        }


def find_pkls(rollout_dir: Path, stem: str | None) -> tuple[Path | None, Path | None]:
    prefix = f"{stem}_" if stem else "*"
    onestep_matches = sorted(rollout_dir.glob(f"{prefix}onestep.pkl"))
    autoreg_matches = sorted(rollout_dir.glob(f"{prefix}autoregressive.pkl"))
    if stem:
        onestep_matches = [p for p in onestep_matches if not p.name.startswith("gc_barrier")]
        autoreg_matches = [p for p in autoreg_matches if not p.name.startswith("gc_barrier")]

    onestep_path = onestep_matches[0] if len(onestep_matches) == 1 else None
    autoreg_path = autoreg_matches[0] if len(autoreg_matches) == 1 else None
    if onestep_path is None and len(onestep_matches) > 1:
        print(f"[Rotation] WARNING: multiple onestep.pkl matches in {rollout_dir}, "
              f"skipping ({[p.name for p in onestep_matches]}); pass --stem to disambiguate.")
    if autoreg_path is None and len(autoreg_matches) > 1:
        print(f"[Rotation] WARNING: multiple autoregressive.pkl matches in {rollout_dir}, "
              f"skipping ({[p.name for p in autoreg_matches]}); pass --stem to disambiguate.")
    if onestep_path is None and autoreg_path is None:
        raise FileNotFoundError(
            f"No onestep.pkl or autoregressive.pkl found in {rollout_dir}. "
            f"Run rollout.py with --mode onestep/autoregressive/both first."
        )
    return onestep_path, autoreg_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot vehicle rigid-body rotation: GT vs one-step vs autoregressive")
    parser.add_argument("rollout_dir", type=Path,
                        help="Folder containing {stem}onestep.pkl / {stem}autoregressive.pkl "
                             "(e.g. outputs/rollouts/wj06)")
    parser.add_argument("--raw-h5", required=True,
                        help="Original raw h5 passed to rollout.py for this rollout — needed "
                             "for node_part_id and the frame-0 reference cluster shape.")
    parser.add_argument("--stem", default=None,
                        help="Test-set filename prefix, for multi-test-set folders "
                             "(e.g. 'T_lok_F_shape_barrier_9_3_60km')")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to save the PNGs (defaults to rollout_dir)")
    args = parser.parse_args()

    plt.rcParams.update(_RCPARAMS)

    onestep_path, autoreg_path = find_pkls(args.rollout_dir, args.stem)

    raw_data = load_positions_and_pids(args.raw_h5)
    cg_cluster_idx = locate_cg_cluster(raw_data)
    ref_points = raw_data["positions_frame0"][cg_cluster_idx]   # (8, 3) frame-0 reference shape
    print(f"[Rotation] vehicle-CG rigid cluster: {len(cg_cluster_idx)} nodes "
          f"(PID 9000100)")

    T_full = len(raw_data["times"])

    gt_angles = None
    os_angles = None
    ar_angles = None
    times = None

    for label, path in (("onestep", onestep_path), ("autoregressive", autoreg_path)):
        if path is None:
            continue
        with open(path, "rb") as f:
            result = pickle.load(f)
        pred_frames = result["pred_frames"]   # (T_steps, N, 15), position in [..., 0:3]
        gt_frames = result["gt_frames"]
        T_steps = pred_frames.shape[0]
        input_frames = T_full - 2 - T_steps   # inverse of T_eval = T-2, T_steps = T_eval - INPUT_FRAMES
        if times is None:
            times = raw_data["times"][input_frames: input_frames + T_steps]

        pred_cluster = pred_frames[:, cg_cluster_idx, 0:3]   # (T_steps, 8, 3)
        gt_cluster = gt_frames[:, cg_cluster_idx, 0:3]

        pred_angles = compute_rigid_rotation_series(ref_points, pred_cluster)
        if gt_angles is None:
            gt_angles = compute_rigid_rotation_series(ref_points, gt_cluster)

        if label == "onestep":
            os_angles = pred_angles
            print(f"[Rotation] one-step: {T_steps} steps, input_frames={input_frames}")
        else:
            ar_angles = pred_angles
            print(f"[Rotation] autoregressive: {T_steps} steps, input_frames={input_frames}")

    out_dir = args.out_dir or args.rollout_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name_suffix = f" — {args.stem or args.rollout_dir.name}"

    axis_names = ["roll", "pitch", "yaw"]
    for i, axis_name in enumerate(axis_names):
        curves = []
        if gt_angles is not None:
            curves.append((times, gt_angles[:, i], STYLE_GT))
        if os_angles is not None:
            curves.append((times, os_angles[:, i], STYLE_OS))
        if ar_angles is not None:
            curves.append((times, ar_angles[:, i], STYLE_AR))
        make_plot(
            "Time (s)", f"{axis_name.capitalize()} (deg)",
            f"Vehicle {axis_name} rotation{name_suffix}",
            curves, out_dir / f"rotation_{axis_name}.png",
            ylim=ROTATION_YLIM,
        )


if __name__ == "__main__":
    main()
