"""
Region partitioning — SPEC §4. Computed once on reference (t=0) geometry,
never recomputed (no element erosion ⇒ region membership never changes).
"""

from __future__ import annotations

import numpy as np

from .constants import CENTERLINE, CENTERLINE_DENOM, FORCE_KEEP_PIDS, VEHICLE_PID_RANGE
from .kfile_parser import MeshData


def centerline_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Perpendicular distance from the barrier centerline, in the xy-plane (mm).

    Validated in SPEC §4.3 against true 3D nearest-distance-to-barrier
    (corr 0.991) — this analytic metric is the approved proxy.
    """
    a, b, c = CENTERLINE
    return np.abs(a * x + b * y + c) / CENTERLINE_DENOM


def build_region_masks(mesh: MeshData) -> dict[str, np.ndarray]:
    """Boolean masks (len = mesh.n_nodes) for each of the six sampling regions."""
    pid = mesh.node_pid
    coords = mesh.coords

    from .constants import FINE_PIDS, COARSE_PIDS
    mask_fine = np.isin(pid, list(FINE_PIDS))
    mask_coarse = np.isin(pid, list(COARSE_PIDS))
    mask_force_keep = np.isin(pid, FORCE_KEEP_PIDS)

    vlo, vhi = VEHICLE_PID_RANGE
    mask_vehicle_all = (pid >= vlo) & (pid <= vhi)

    d = centerline_distance(coords[:, 0], coords[:, 1])
    mask_veh_contact = mask_vehicle_all & (d < 500.0)
    mask_veh_near    = mask_vehicle_all & (d >= 500.0) & (d < 1000.0)
    mask_veh_far     = mask_vehicle_all & (d >= 1000.0)

    return {
        "barrier_fine":   mask_fine,
        "barrier_coarse": mask_coarse,
        "force_keep":     mask_force_keep,
        "veh_contact":    mask_veh_contact,
        "veh_near":       mask_veh_near,
        "veh_far":        mask_veh_far,
    }
