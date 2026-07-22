"""Vehicle-GC / barrier-displacement kinematics export.

Locates two fixed physical points in the mesh and writes their per-timestep
kinematics (position / velocity / acceleration, plus barrier displacement
relative to frame 0) to CSV — for comparing ground truth against one-step
and autoregressive rollout predictions.

Point 1 — vehicle GC: k-file car_and_barriers.k has a
*DATABASE_HISTORY_NODE_ID card that literally labels node 9000100 as
"VEHICLE_CG_Global". That node's PID (9000100) is one of the 7 PIDs in
dataset/constants.py's FORCE_KEEP_PIDS, so it's always retained by
sampling.

Point 2 — barrier reference: the barrier fine-mesh (FINE_PIDS) node
nearest the vehicle GC's frame-0 position, used as a proxy for barrier
deflection at the point of impact.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.constants import FORCE_KEEP_PIDS, FINE_PIDS  # noqa: E402

# k-file car_and_barriers.k, *DATABASE_HISTORY_NODE_ID: node 9000100 = "VEHICLE_CG_Global"
VEHICLE_CG_NODE_ID = 9000100
VEHICLE_CG_PID = 9000100


def locate_gc_barrier(raw_data: dict) -> tuple[int, int]:
    """Return (gc_idx, barrier_idx) — node indices into the (N,) node axis."""
    gc_idx, method = _locate_gc(raw_data)
    barrier_idx = _locate_barrier(raw_data, gc_idx)
    print(f"[GC/Barrier] vehicle GC → node index {gc_idx} (method={method}), "
          f"barrier ref → node index {barrier_idx} (nearest fine-mesh node "
          f"to GC at frame 0)")
    return gc_idx, barrier_idx


def _locate_gc(raw_data: dict) -> tuple[int, str]:
    sampled_node_ids = raw_data.get("sampled_node_ids")
    if sampled_node_ids is not None:
        matches = np.where(sampled_node_ids == VEHICLE_CG_NODE_ID)[0]
        if len(matches) > 0:
            return int(matches[0]), "exact node id 9000100"

    node_part_id = raw_data["node_part_id"]
    matches = np.where(node_part_id == VEHICLE_CG_PID)[0]
    if len(matches) > 0:
        return int(matches[0]), "PID 9000100 match"

    force_keep_mask = np.isin(node_part_id, FORCE_KEEP_PIDS)
    force_keep_idx = np.where(force_keep_mask)[0]
    if len(force_keep_idx) == 0:
        raise ValueError(
            "Could not locate vehicle GC: no node with sampled_node_id "
            f"{VEHICLE_CG_NODE_ID}, no node with PID {VEHICLE_CG_PID}, and "
            "no nodes with PID in FORCE_KEEP_PIDS."
        )
    frame0_x = raw_data["positions"][0, force_keep_idx, 0]
    rightmost = force_keep_idx[int(np.argmax(frame0_x))]
    return int(rightmost), "right-most FORCE_KEEP_PIDS node (fallback)"


def _locate_barrier(raw_data: dict, gc_idx: int) -> int:
    node_part_id = raw_data["node_part_id"]
    fine_mask = np.isin(node_part_id, list(FINE_PIDS))
    fine_idx = np.where(fine_mask)[0]
    if len(fine_idx) == 0:
        raise ValueError("Could not locate barrier reference point: no "
                          "nodes with PID in FINE_PIDS.")

    fine_pos0 = raw_data["positions"][0, fine_idx]
    gc_pos0 = raw_data["positions"][0, gc_idx]
    tree = cKDTree(fine_pos0)
    _, nearest = tree.query(gc_pos0)
    return int(fine_idx[nearest])


def export_gt_kinematics_csv(raw_data: dict, gc_idx: int, barrier_idx: int,
                              out_path: Path | str) -> None:
    """Ground-truth-only kinematics for both points, full trajectory."""
    positions = raw_data["positions"]
    velocity = raw_data["velocity"]
    acceleration = raw_data["acceleration"]
    times = raw_data["times"]
    T = positions.shape[0]

    barrier_pos0 = positions[0, barrier_idx]

    header = ["frame", "time",
              "gc_pos_x", "gc_pos_y", "gc_pos_z",
              "gc_vel_x", "gc_vel_y", "gc_vel_z",
              "gc_acc_x", "gc_acc_y", "gc_acc_z",
              "barrier_pos_x", "barrier_pos_y", "barrier_pos_z",
              "barrier_vel_x", "barrier_vel_y", "barrier_vel_z",
              "barrier_acc_x", "barrier_acc_y", "barrier_acc_z",
              "barrier_disp_x", "barrier_disp_y", "barrier_disp_z"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(T):
            disp = positions[t, barrier_idx] - barrier_pos0
            writer.writerow([
                t, float(times[t]),
                *positions[t, gc_idx], *velocity[t, gc_idx], *acceleration[t, gc_idx],
                *positions[t, barrier_idx], *velocity[t, barrier_idx], *acceleration[t, barrier_idx],
                *disp,
            ])
    print(f"[GC/Barrier] Ground-truth kinematics CSV saved → {out_path}")


def export_pred_gt_kinematics_csv(result: dict, raw_data: dict, gc_idx: int,
                                   barrier_idx: int, input_frames: int,
                                   out_path: Path | str) -> None:
    """Predicted-vs-ground-truth kinematics for both points, eval-step range."""
    if "track_pred_pos" not in result:
        raise ValueError(
            "result has no track_* keys — call run_onestep/run_autoregressive "
            "with gc_idx and barrier_idx set to populate them."
        )

    times = raw_data["times"]
    barrier_pos0 = raw_data["positions"][0, barrier_idx]

    pred_pos, pred_vel, pred_acc = (result["track_pred_pos"], result["track_pred_vel"],
                                     result["track_pred_acc"])
    gt_pos, gt_vel, gt_acc = (result["track_gt_pos"], result["track_gt_vel"],
                               result["track_gt_acc"])
    T_steps = pred_pos.shape[0]

    header = ["frame", "time",
              "pred_gc_pos_x", "pred_gc_pos_y", "pred_gc_pos_z",
              "pred_gc_vel_x", "pred_gc_vel_y", "pred_gc_vel_z",
              "pred_gc_acc_x", "pred_gc_acc_y", "pred_gc_acc_z",
              "gt_gc_pos_x", "gt_gc_pos_y", "gt_gc_pos_z",
              "gt_gc_vel_x", "gt_gc_vel_y", "gt_gc_vel_z",
              "gt_gc_acc_x", "gt_gc_acc_y", "gt_gc_acc_z",
              "pred_barrier_pos_x", "pred_barrier_pos_y", "pred_barrier_pos_z",
              "pred_barrier_vel_x", "pred_barrier_vel_y", "pred_barrier_vel_z",
              "pred_barrier_acc_x", "pred_barrier_acc_y", "pred_barrier_acc_z",
              "gt_barrier_pos_x", "gt_barrier_pos_y", "gt_barrier_pos_z",
              "gt_barrier_vel_x", "gt_barrier_vel_y", "gt_barrier_vel_z",
              "gt_barrier_acc_x", "gt_barrier_acc_y", "gt_barrier_acc_z",
              "pred_barrier_disp_x", "pred_barrier_disp_y", "pred_barrier_disp_z",
              "gt_barrier_disp_x", "gt_barrier_disp_y", "gt_barrier_disp_z"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for s in range(T_steps):
            t = input_frames + s
            pred_disp = pred_pos[s, 1] - barrier_pos0
            gt_disp = gt_pos[s, 1] - barrier_pos0
            writer.writerow([
                t, float(times[t]),
                *pred_pos[s, 0], *pred_vel[s, 0], *pred_acc[s, 0],
                *gt_pos[s, 0], *gt_vel[s, 0], *gt_acc[s, 0],
                *pred_pos[s, 1], *pred_vel[s, 1], *pred_acc[s, 1],
                *gt_pos[s, 1], *gt_vel[s, 1], *gt_acc[s, 1],
                *pred_disp, *gt_disp,
            ])
    print(f"[GC/Barrier] Pred-vs-GT kinematics CSV saved → {out_path}")
