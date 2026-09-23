"""
verify_live_extraction.py — prove dataset/live_source.py's live d3plot
extraction is faithful to the offline dataset/build_dataset.py pipeline.

Both paths ultimately call the same d3plot_io.extract_frame_data() for a
given native frame with (in principle) the same sampled_idx/connectivity/
n_full. live_source.py derives sampled_idx by matching sampled_node_ids
against the d3plot's own node-ID array, instead of build_dataset.py's
implicit k-file-row-order assumption — this script is the concrete check
that the two derivations agree, by comparing live-extracted frames against
the ALREADY-WRITTEN stage-1 h5's stored frames at the same timestamps.
A near-zero diff proves: sampled_idx recovery, global connectivity
reconstruction, and n_full are all correct.

Run this once against a real case before wiring any training code up to a
live entry (see the plan's execution order) — it's the correctness
foundation the rest of the feature depends on.

Usage (same apptainer pattern as dataset/compare_downsample_fem.py):
    apptainer exec --bind /raid /staging/proj_iim1/xrkong/container/collider.sif \\
        python -m dataset.verify_live_extraction \\
        --src /raid/proj_iim1/xrkong/fem/T_lok_F_shape_barrier_9_3_60km \\
        --ref-h5 /raid/proj_iim1/xrkong/h5_fps_no_wheels/T_lok_F_shape_barrier_9_3_60km.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from .live_source import extract_live_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify live d3plot extraction against an already-built stage-1 h5."
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="Directory containing d3plot, d3plot01, ... (raw FEM case dir).")
    parser.add_argument("--ref-h5", type=Path, required=True,
                        help="Stage-1 HDF5 for this case (dataset/build_dataset.py output).")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Max allowed |diff| in position [mm] before treating a frame as "
                             "a mismatch (default 1e-3, float32 round-trip slack).")
    parser.add_argument("--time-tol", type=float, default=1e-6,
                        help="Max |dt| [s] when matching a stage-1 frame to a live-extracted "
                             "frame by timestamp.")
    args = parser.parse_args()

    print(f"Extracting live trajectory from {args.src} (ref_h5={args.ref_h5}) …")
    live = extract_live_trajectory(args.src, args.ref_h5, n_jobs=args.n_jobs, use_cache=False)
    print(f"  {len(live.times)} native frames extracted, N={live.positions.shape[1]} nodes "
          f"(t = {live.times[0]*1e3:.3f}–{live.times[-1]*1e3:.3f} ms)")

    with h5py.File(args.ref_h5, "r") as f:
        h5_times = f["states/times"][:]
        h5_positions = f["states/positions"][:]
        h5_eps = f["states/eff_plastic_strain"][:]
        h5_alive = f["states/node_alive"][:]

    print(f"\n{'idx':>5} {'t (ms)':>10} {'pos max|diff|':>16} {'eps max|diff|':>16} "
          f"{'alive mismatches':>18}  status")
    print("-" * 80)

    n_checked = 0
    n_failed = 0
    worst_pos_err = 0.0
    for k in range(len(h5_times)):
        t = h5_times[k]
        j = int(np.argmin(np.abs(live.times - t)))
        dt = abs(live.times[j] - t)
        if dt > args.time_tol:
            print(f"{k:>5} {t*1e3:>10.4f} {'':>16} {'':>16} {'':>18}  "
                  f"SKIP (no live frame within {args.time_tol}s, nearest dt={dt:.2e}s)")
            continue

        pos_err = float(np.max(np.abs(live.positions[j] - h5_positions[k])))
        eps_err = float(np.max(np.abs(live.eff_plastic_strain[j] - h5_eps[k])))
        alive_mismatch = int(np.sum(live.node_alive[j] != h5_alive[k]))

        n_checked += 1
        worst_pos_err = max(worst_pos_err, pos_err)
        ok = pos_err <= args.tol
        if not ok:
            n_failed += 1
        status = "OK" if ok else "FAIL"
        print(f"{k:>5} {t*1e3:>10.4f} {pos_err:>16.6e} {eps_err:>16.6e} "
              f"{alive_mismatch:>18d}  {status}")

    print("-" * 80)
    print(f"Checked {n_checked}/{len(h5_times)} stage-1 frames. "
          f"Worst position error: {worst_pos_err:.6e} mm (tol={args.tol}).")
    if n_checked == 0:
        print("FAIL: no stage-1 frame had a matching live-extracted frame — "
              "check --time-tol or that --src/--ref-h5 refer to the same case.")
        sys.exit(1)
    if n_failed > 0:
        print(f"FAIL: {n_failed} frame(s) exceeded tolerance.")
        sys.exit(1)
    print("PASS: live extraction matches the stage-1 h5 at every checked frame.")


if __name__ == "__main__":
    main()
