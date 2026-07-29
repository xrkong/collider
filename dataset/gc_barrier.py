"""Vehicle-GC / barrier-displacement kinematics CSV + plots, run as a side
artifact of dataset/build_dataset.py, in two resolutions:

  run_gc_barrier_full_res     full native temporal resolution, extracted
                               directly from raw d3plot state files (no
                               --frame-stride downsampling, no node
                               sampling) — "ground truth", written to
                               <out_stem>_analysis/fem/.
  run_gc_barrier_downsampled  read back from the finished HDF5's
                               /states/positions — i.e. exactly the
                               sampled/strided data a model actually
                               trains on — written to
                               <out_stem>_analysis/downsampled/.

Companion to src/gc_barrier.py, which does a similar job but at rollout-eval
time (ground truth vs. model prediction curves), reading from whatever h5 a
run was evaluated against. This module's two functions instead run once,
right when a dataset is built, so a stride/sampling problem (e.g. too coarse
for the ORA/ASI moving-average filter — 20ms/frame at --frame-stride 4 on a
5ms-native d3plot output) or a mismatch between the two resolutions is
visible immediately, no rollout needed.

Point 1 — vehicle GC: k-file car_and_barriers.k has a
*DATABASE_HISTORY_NODE_ID card that literally labels node 9000100 as
"VEHICLE_CG_Global". That node's PID (9000100) is one of the 7 PIDs in
dataset/constants.py's FORCE_KEEP_PIDS.

Point 2 — barrier reference: the barrier fine-mesh (FINE_PIDS) node nearest
the vehicle GC's frame-0 position — found here directly in the FULL
(unsampled) mesh, so it isn't limited to whichever fine-mesh nodes the
region sampler happened to keep.

Vehicle rigid-body rotation (roll/pitch/yaw, see src/plot_rotation.py): all
8 nodes of the PID-9000100 hex (not just one) form a small rigid sensor
cube attached at the vehicle CG — fitting a rotation matrix between this
cluster's frame-0 shape and its shape at time t (Kabsch algorithm) isolates
pure rigid-body rotation, since the cluster is far too small (~15mm) to be
meaningfully deformed by the crash itself. Axis convention (SAE): X
longitudinal -> roll, Y lateral -> pitch, Z vertical -> yaw. Both
orchestrators below extract the cluster in the same pass as GC/barrier (one
combined row_idx) and write a companion <out_stem>_rotation.csv +
<out_stem>_rotation_plots/ next to the gc_barrier CSV/plots.

dataset/ must not import from src/ (src/ imports from dataset/, not the
reverse — see src/gc_barrier.py, src/plot_gc_barrier.py, src/plot_rotation.py),
so the ORA/ASI/rotation math here is a small, deliberately self-contained
duplicate of those modules' — this is the ground-truth-only, single-curve
case (no model predictions exist at dataset-build time).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from lasso.dyna import ArrayType
from scipy.spatial import cKDTree

from .constants import FORCE_KEEP_PIDS, FINE_PIDS, G_IN_MM_S2
from .d3plot_io import extract_node_array, select_frames
from .kfile_parser import MeshData

# k-file car_and_barriers.k, *DATABASE_HISTORY_NODE_ID: node 9000100 = "VEHICLE_CG_Global"
VEHICLE_CG_NODE_ID = 9000100
VEHICLE_CG_PID = 9000100

# EN 1317 Acceleration Severity Index: ASI(t) = sqrt((ax/12)^2+(ay/9)^2+(az/10)^2),
# with ax/ay/az the 50ms moving-average-filtered accelerations in g.
ASI_LIMIT_X_G = 12.0
ASI_LIMIT_Y_G = 9.0
ASI_LIMIT_Z_G = 10.0
ASI_FILTER_WINDOW_MS = 50.0

_RCPARAMS = {
    "font.family":     "DejaVu Serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


# ── Node location (full, unsampled mesh) ──────────────────────────────────────

def locate_gc_barrier_in_mesh(mesh: MeshData, log_prefix: str = "[GC/Barrier full-res]") -> tuple[int, int]:
    """Return (gc_idx, barrier_idx) — indices into mesh.node_ids/coords/node_pid.

    Works on any MeshData, not just the full mesh — pass one built from the
    sampled node set (see run_gc_barrier_downsampled) to locate within that
    subset instead; log_prefix should match so log lines aren't misleading.
    """
    gc_idx, method = _locate_gc(mesh)
    barrier_idx = _locate_barrier(mesh, gc_idx)
    print(f"{log_prefix} vehicle GC -> node id {mesh.node_ids[gc_idx]} "
          f"(mesh index {gc_idx}, method={method}); barrier ref -> node id "
          f"{mesh.node_ids[barrier_idx]} (mesh index {barrier_idx})")
    return gc_idx, barrier_idx


def _locate_gc(mesh: MeshData) -> tuple[int, str]:
    matches = np.where(mesh.node_ids == VEHICLE_CG_NODE_ID)[0]
    if len(matches) > 0:
        return int(matches[0]), "exact node id 9000100"

    matches = np.where(mesh.node_pid == VEHICLE_CG_PID)[0]
    if len(matches) > 0:
        return int(matches[0]), "PID 9000100 match"

    force_keep_idx = np.where(np.isin(mesh.node_pid, FORCE_KEEP_PIDS))[0]
    if len(force_keep_idx) == 0:
        raise ValueError(
            "Could not locate vehicle GC: no node id 9000100, no PID 9000100, "
            "and no nodes with PID in FORCE_KEEP_PIDS."
        )
    rightmost = force_keep_idx[int(np.argmax(mesh.coords[force_keep_idx, 0]))]
    return int(rightmost), "right-most FORCE_KEEP_PIDS node (fallback)"


def _locate_barrier(mesh: MeshData, gc_idx: int) -> int:
    fine_idx = np.where(np.isin(mesh.node_pid, list(FINE_PIDS)))[0]
    if len(fine_idx) == 0:
        raise ValueError("Could not locate barrier reference point: no nodes "
                          "with PID in FINE_PIDS.")
    tree = cKDTree(mesh.coords[fine_idx])
    _, nearest = tree.query(mesh.coords[gc_idx])
    return int(fine_idx[nearest])


# ── Rigid-body rotation (roll/pitch/yaw) via the vehicle-CG node cluster ─────

def locate_cg_cluster_in_mesh(mesh: MeshData) -> np.ndarray:
    """All node indices belonging to the vehicle-CG rigid hex (PID
    VEHICLE_CG_PID) — used for rotation fitting (needs >=3 points), not
    just the single GC point locate_gc_barrier_in_mesh returns."""
    idx = np.where(mesh.node_pid == VEHICLE_CG_PID)[0]
    if len(idx) < 3:
        raise ValueError(
            f"Need >=3 nodes to fit a rotation; found {len(idx)} with PID "
            f"{VEHICLE_CG_PID} (the VEHICLE_CG_Global/Local hex)."
        )
    return idx


def kabsch_rotation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Best-fit rotation matrix R (3,3) mapping point cluster `p` onto `q`
    (both (M,3), same M points, same order, measured at two different
    times) — minimizes sum_i ||R @ p_i - q_i||^2. Translation-invariant
    (both clusters are centered internally)."""
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r


def rotation_matrix_to_euler_zyx_deg(r: np.ndarray) -> tuple[float, float, float]:
    """Decompose R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (SAE roll-pitch-yaw,
    X=longitudinal/roll, Y=lateral/pitch, Z=vertical/yaw). Returns degrees."""
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    roll = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def compute_rigid_rotation_series(ref_points: np.ndarray, points_over_time: np.ndarray) -> np.ndarray:
    """ref_points: (M,3) frame-0 cluster shape. points_over_time: (T,M,3).
    Returns (T,3) [roll, pitch, yaw] in degrees, relative to ref_points."""
    T = points_over_time.shape[0]
    angles = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        r = kabsch_rotation(ref_points, points_over_time[t])
        angles[t] = rotation_matrix_to_euler_zyx_deg(r)
    return angles


# ── Full-resolution extraction ────────────────────────────────────────────────

def extract_full_res_positions(
    state_files: list[Path],
    entries: list[tuple[float, str, int]],
    row_idx: np.ndarray,
    tmp: Path,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract positions for `row_idx` mesh nodes at every state in
    `entries` — no stride. `row_idx` is arbitrary (e.g. [gc_idx,
    barrier_idx] plus the 8-node CG rotation cluster, extracted together in
    one pass rather than re-reading every d3plot state twice).

    `entries` is the already-scanned (time, filename, state_idx) list
    (dataset/build_dataset.py's `all_entries`, from scan_times) — reused here
    rather than re-scanning, `select_frames(..., stride=1)` only dedupes/sorts.

    Returns (times (T,) float64, positions (T, len(row_idx), 3) float32)
    sorted by time.
    """
    unstrided = select_frames(entries, stride=1, limit=None)
    state_file_map = {p.name: p for p in state_files}
    row_idx = np.asarray(row_idx)

    print(f"[GC/Barrier full-res] extracting {len(unstrided)} frames "
          f"(native resolution, no stride) for {len(row_idx)} nodes ...")
    if n_jobs == 1:
        results = [
            extract_node_array(state_file_map[fname], sidx, ArrayType.node_displacement,
                                row_idx, tmp)
            for _, fname, sidx in unstrided
        ]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(extract_node_array)(state_file_map[fname], sidx,
                                         ArrayType.node_displacement, row_idx, tmp)
            for _, fname, sidx in unstrided
        )

    times = np.array([t for t, _, _ in unstrided], dtype=np.float64)
    positions = np.stack(results).astype(np.float32)   # (T, len(row_idx), 3)
    times, positions = _drop_near_duplicate_frames(times, positions)
    return times, positions


def _drop_near_duplicate_frames(
    times: np.ndarray, positions: np.ndarray, rel_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop samples whose time gap to the previous KEPT sample is a tiny
    fraction of the sequence's typical (median) gap.

    LS-DYNA's final d3plot state can land a near-duplicate of the previous
    one (e.g. a restart/termination dump written ~1us after a regular 5ms
    write) — `select_frames`'s dedup only catches EXACT duplicates
    (`abs(dt) > 1e-9`), not this near-duplicate case. Left in, a naive
    forward-difference derivative divides by an almost-zero dt right there,
    producing a spurious huge velocity/acceleration spike at that single
    frame (and, because _forward_diff_padded tail-pads, at every frame after
    it too) — this is a data artifact, not a plotting bug; the fix belongs
    here, before any derivative is computed. Always keeps the first sample.
    """
    if len(times) < 3:
        return times, positions
    diffs = np.diff(times)
    median_dt = np.median(diffs)
    keep = [0]
    for i in range(1, len(times)):
        if times[i] - times[keep[-1]] >= rel_threshold * median_dt:
            keep.append(i)
    if len(keep) < len(times):
        print(f"[GC/Barrier full-res] dropped {len(times) - len(keep)} near-duplicate "
              f"frame(s) (gap < {rel_threshold:.0%} of median dt={median_dt * 1000:.3f} ms) "
              f"— likely a terminal/restart-dump artifact, not real motion")
    keep = np.array(keep)
    return times[keep], positions[keep]


def _forward_diff_padded(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """vel[i]=pos[i+1]-pos[i], acc[i]=vel[i+1]-vel[i], tail-padded — per
    output-frame (not yet divided by dt), same convention as
    src/rollout.py's _derive_padded_kinematics."""
    T = pos.shape[0]
    pos64 = pos.astype(np.float64)
    vel = np.zeros_like(pos64)
    acc = np.zeros_like(pos64)
    n_vel_valid = max(T - 1, 0)
    n_acc_valid = max(T - 2, 0)
    if n_vel_valid > 0:
        vel[:n_vel_valid] = np.diff(pos64, axis=0)
        vel[n_vel_valid:] = vel[n_vel_valid - 1]
    if n_acc_valid > 0:
        acc[:n_acc_valid] = np.diff(vel[:n_vel_valid], axis=0)
        acc[n_acc_valid:] = acc[n_acc_valid - 1]
    return vel.astype(np.float32), acc.astype(np.float32)


# ── CSV export ────────────────────────────────────────────────────────────────

def export_gc_barrier_csv(times: np.ndarray, positions: np.ndarray, out_path: Path,
                           log_prefix: str = "[GC/Barrier full-res]") -> None:
    """positions: (T, 2, 3) -- [:, 0] = GC, [:, 1] = barrier. Ground-truth
    only (no model at dataset-build time). Temporal resolution depends on
    caller: full native (run_gc_barrier_full_res) or sampled/strided
    (run_gc_barrier_downsampled). Same column scheme as src/gc_barrier.py's
    export_gt_kinematics_csv."""
    gc_pos, barrier_pos = positions[:, 0], positions[:, 1]
    gc_vel, gc_acc = _forward_diff_padded(gc_pos)
    barrier_vel, barrier_acc = _forward_diff_padded(barrier_pos)
    barrier_pos0 = barrier_pos[0]

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
        for t in range(len(times)):
            disp = barrier_pos[t] - barrier_pos0
            writer.writerow([
                t, float(times[t]),
                *gc_pos[t], *gc_vel[t], *gc_acc[t],
                *barrier_pos[t], *barrier_vel[t], *barrier_acc[t],
                *disp,
            ])
    print(f"{log_prefix} CSV saved -> {out_path}")


# ── Plotting: ORA_x, ORA_y, ASI, displacement (single ground-truth curve) ────

def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(round(window)))
    if window <= 1:
        return x.astype(float)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    xpad = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xpad, kernel, mode="valid")


def _accel_to_g(accel_per_frame2: np.ndarray, dt_seconds: float) -> np.ndarray:
    return accel_per_frame2 / dt_seconds ** 2 / G_IN_MM_S2


def _compute_asi(ax_g: np.ndarray, ay_g: np.ndarray, az_g: np.ndarray) -> np.ndarray:
    return np.sqrt((ax_g / ASI_LIMIT_X_G) ** 2 + (ay_g / ASI_LIMIT_Y_G) ** 2 +
                   (az_g / ASI_LIMIT_Z_G) ** 2)


def plot_gc_barrier_full_res(
    times: np.ndarray, positions: np.ndarray, out_dir: Path,
    window_ms: float = ASI_FILTER_WINDOW_MS, title_suffix: str = "",
    log_prefix: str = "[GC/Barrier full-res]", series_label: str = "Ground truth (full res)",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_RCPARAMS)

    gc_pos, barrier_pos = positions[:, 0], positions[:, 1]
    _, gc_acc = _forward_diff_padded(gc_pos)
    dt_seconds = float(np.median(np.diff(times)))
    window_frames = max(1, round((window_ms / 1000.0) / dt_seconds)) if window_ms > 0 else 1
    if window_frames <= 1:
        print(f"{log_prefix} dt = {dt_seconds * 1000:.2f} ms, no smoothing "
              f"(raw per-frame values)")
    else:
        print(f"{log_prefix} dt = {dt_seconds * 1000:.2f} ms, moving-average "
              f"window = {window_frames} frame(s) (~{window_frames * dt_seconds * 1000:.1f} ms)")

    ax_g = _moving_average(_accel_to_g(gc_acc[:, 0], dt_seconds), window_frames)
    ay_g = _moving_average(_accel_to_g(gc_acc[:, 1], dt_seconds), window_frames)
    az_g = _moving_average(_accel_to_g(gc_acc[:, 2], dt_seconds), window_frames)
    asi = _compute_asi(ax_g, ay_g, az_g)
    disp_mag = np.sqrt(np.sum((barrier_pos - barrier_pos[0]) ** 2, axis=-1))

    style = dict(color="#2a78d6", linestyle="-", linewidth=1.8, label=series_label)

    def _make_plot(x_label, y_label, title, y, out_path, hlines=None):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(times, y, **style)
        for hy, hl in (hlines or []):
            ax.axhline(hy, color="0.5", linestyle=":", linewidth=1.0, label=hl)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"{log_prefix} plot saved -> {out_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_plot("Time (s)", "ORA$_x$ (g)", f"Longitudinal acceleration (ORA$_x$){title_suffix}",
               ax_g, out_dir / "gc_barrier_ora_x.png")
    _make_plot("Time (s)", "ORA$_y$ (g)", f"Lateral acceleration (ORA$_y$){title_suffix}",
               ay_g, out_dir / "gc_barrier_ora_y.png")
    _make_plot("Time (s)", "ASI (-)", f"Acceleration Severity Index (EN 1317){title_suffix}",
               asi, out_dir / "gc_barrier_asi.png",
               hlines=[(1.0, "ASI = 1 (Class A limit)")])
    _make_plot("Time (s)", "Displacement (mm)", f"Barrier resultant displacement{title_suffix}",
               disp_mag, out_dir / "gc_barrier_displacement.png")


# ── Rotation CSV export + plotting ────────────────────────────────────────────

def export_rotation_csv(times: np.ndarray, angles: np.ndarray, out_path: Path,
                         log_prefix: str = "[GC/Barrier full-res]") -> None:
    """angles: (T,3) [roll, pitch, yaw] degrees, relative to frame 0 (see
    compute_rigid_rotation_series)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "roll_deg", "pitch_deg", "yaw_deg"])
        for t in range(len(times)):
            writer.writerow([t, float(times[t]), *angles[t]])
    print(f"{log_prefix} rotation CSV saved -> {out_path}")


def plot_rotation(
    times: np.ndarray, angles: np.ndarray, out_dir: Path, title_suffix: str = "",
    log_prefix: str = "[GC/Barrier full-res]", series_label: str = "Ground truth (full res)",
) -> None:
    """angles: (T,3) [roll, pitch, yaw] degrees -> rotation_roll.png /
    rotation_pitch.png / rotation_yaw.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_RCPARAMS)
    style = dict(color="#2a78d6", linestyle="-", linewidth=1.8, label=series_label)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, axis_name in enumerate(("roll", "pitch", "yaw")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(times, angles[:, i], **style)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{axis_name.capitalize()} (deg)")
        ax.set_title(f"Vehicle {axis_name} rotation{title_suffix}")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        out_path = out_dir / f"rotation_{axis_name}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"{log_prefix} plot saved -> {out_path}")


# ── Downsampled extraction (reads back the already-written HDF5) ─────────────

def extract_downsampled_positions(
    h5_path: Path, row_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read positions for `row_idx` nodes straight out of the finished
    HDF5's /states/positions — i.e. exactly the sampled/strided data a
    model trains on, not a fresh d3plot extraction. `row_idx` are row
    indices into the SAMPLED node set (see locate_gc_barrier_in_mesh /
    locate_cg_cluster_in_mesh, called on a MeshData built from
    /metadata/sampled_node_ids etc. — not the full mesh).

    Read one node at a time rather than h5py fancy-indexing `row_idx` in
    one call — row_idx may be unsorted or contain duplicates (the GC node
    is also one of the 8 rotation-cluster nodes), which h5py's fancy
    indexing doesn't reliably support.

    Returns (times (T,) float64, positions (T, len(row_idx), 3) float32).
    """
    import h5py

    row_idx = np.asarray(row_idx)
    with h5py.File(h5_path, "r") as h5f:
        times = h5f["states/times"][:]
        cols = [h5f["states/positions"][:, int(i), :] for i in row_idx]
    positions = np.stack(cols, axis=1).astype(np.float32)   # (T, len(row_idx), 3)
    return times, positions


def run_gc_barrier_downsampled(
    h5_path: Path,
    out_csv: Path,
    out_plot_dir: Path,
    out_rotation_csv: Path,
    out_rotation_plot_dir: Path,
    title_suffix: str = "",
    window_ms: float = ASI_FILTER_WINDOW_MS,
) -> None:
    """Same CSV + 4 plots as run_gc_barrier_full_res, plus the rotation
    CSV + 3 plots, but sourced from the already-written, sampled/strided
    HDF5 (--frame-stride / --frame-limit / node sampling all apply) rather
    than a fresh full-res d3plot pass — this is "what the model actually
    sees", the companion to the fem/ full-res group which is "ground truth
    at native resolution".

    GC/barrier/rotation-cluster nodes are re-located within the SAMPLED
    node set (not reused from the full-mesh full-res pass) since the exact
    barrier reference node chosen there may not have survived sampling —
    the rotation cluster itself is force-kept (PID in FORCE_KEEP_PIDS) so
    it's unaffected, but re-locating keeps this function self-contained.
    """
    import h5py

    with h5py.File(h5_path, "r") as h5f:
        sampled_node_ids = h5f["metadata/sampled_node_ids"][:]
        node_part_id = h5f["metadata/node_part_id"][:]
        ref_positions = h5f["metadata/ref_positions"][:].astype(np.float64)

    log_prefix = "[GC/Barrier downsampled]"
    series_label = "Ground truth (downsampled)"
    sampled_mesh = MeshData(node_ids=sampled_node_ids, coords=ref_positions, node_pid=node_part_id)
    gc_idx, barrier_idx = locate_gc_barrier_in_mesh(sampled_mesh, log_prefix=log_prefix)
    cluster_idx = locate_cg_cluster_in_mesh(sampled_mesh)

    row_idx = np.concatenate([[gc_idx, barrier_idx], cluster_idx])
    times, positions = extract_downsampled_positions(h5_path, row_idx)
    gc_barrier_pos, cluster_pos = positions[:, :2], positions[:, 2:]

    export_gc_barrier_csv(times, gc_barrier_pos, out_csv, log_prefix=log_prefix)
    plot_gc_barrier_full_res(times, gc_barrier_pos, out_plot_dir,
                              window_ms=window_ms, title_suffix=title_suffix,
                              log_prefix=log_prefix, series_label=series_label)

    angles = compute_rigid_rotation_series(cluster_pos[0], cluster_pos)
    export_rotation_csv(times, angles, out_rotation_csv, log_prefix=log_prefix)
    plot_rotation(times, angles, out_rotation_plot_dir, title_suffix=title_suffix,
                  log_prefix=log_prefix, series_label=series_label)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_gc_barrier_full_res(
    mesh: MeshData,
    state_files: list[Path],
    entries: list[tuple[float, str, int]],
    tmp: Path,
    n_jobs: int,
    out_csv: Path,
    out_plot_dir: Path,
    out_rotation_csv: Path,
    out_rotation_plot_dir: Path,
    title_suffix: str = "",
    window_ms: float = ASI_FILTER_WINDOW_MS,
) -> None:
    """Locate GC/barrier/rotation-cluster nodes in the full mesh, extract
    their positions at every available d3plot state (native dt, no
    --frame-stride) in ONE combined pass, and write the gc_barrier CSV + 4
    plots plus the rotation CSV + 3 plots. Independent of the main
    sampled/strided HDF5 output.

    window_ms: moving-average window (ms) for the ORA_x/ORA_y/ASI plots
    only — the CSV is always the raw, unfiltered per-frame data regardless.
    Pass 0 to plot raw (unfiltered) values too."""
    gc_idx, barrier_idx = locate_gc_barrier_in_mesh(mesh)
    cluster_idx = locate_cg_cluster_in_mesh(mesh)

    row_idx = np.concatenate([[gc_idx, barrier_idx], cluster_idx])
    times, positions = extract_full_res_positions(state_files, entries, row_idx, tmp, n_jobs=n_jobs)
    gc_barrier_pos, cluster_pos = positions[:, :2], positions[:, 2:]

    export_gc_barrier_csv(times, gc_barrier_pos, out_csv)
    plot_gc_barrier_full_res(times, gc_barrier_pos, out_plot_dir,
                              window_ms=window_ms, title_suffix=title_suffix)

    angles = compute_rigid_rotation_series(cluster_pos[0], cluster_pos)
    export_rotation_csv(times, angles, out_rotation_csv)
    plot_rotation(times, angles, out_rotation_plot_dir, title_suffix=title_suffix)


# ── Re-plot from an existing CSV (no d3plot re-extraction needed) ────────────

def replot_from_csv(csv_path: Path, out_dir: Path,
                     window_ms: float = ASI_FILTER_WINDOW_MS) -> None:
    """Re-render the 4 plots from an already-exported <out_stem>_gc_barrier.csv
    — e.g. one produced before the near-duplicate-frame fix in
    extract_full_res_positions — without re-running the expensive d3plot
    extraction. Recomputes velocity/acceleration fresh from the CSV's raw
    gc_pos_x/y/z + time columns (does NOT trust the CSV's own gc_acc_x/y/z
    columns, since a pre-fix CSV has the artifact baked into exactly those
    columns) and re-applies the near-duplicate-frame drop before doing so.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")

    times = np.array([float(r["time"]) for r in rows], dtype=np.float64)
    gc_pos = np.array([[float(r["gc_pos_x"]), float(r["gc_pos_y"]), float(r["gc_pos_z"])]
                        for r in rows], dtype=np.float32)
    barrier_pos = np.array([[float(r["barrier_pos_x"]), float(r["barrier_pos_y"]),
                              float(r["barrier_pos_z"])] for r in rows], dtype=np.float32)
    positions = np.stack([gc_pos, barrier_pos], axis=1)   # (T, 2, 3)

    times, positions = _drop_near_duplicate_frames(times, positions)
    plot_gc_barrier_full_res(times, positions, out_dir, window_ms=window_ms,
                              title_suffix=f" — {csv_path.stem}")


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description="Re-plot GC/barrier ORA_x/ORA_y/ASI/displacement from an "
                    "existing <out_stem>_gc_barrier.csv (no d3plot re-extraction).")
    _parser.add_argument("csv", type=Path, help="Path to <out_stem>_gc_barrier.csv")
    _parser.add_argument("out_dir", type=Path, help="Where to save the 4 PNGs")
    _parser.add_argument("--window-ms", type=float, default=ASI_FILTER_WINDOW_MS,
                         help="Moving-average window in ms (default 50, per EN 1317; "
                              "0 = raw unfiltered)")
    _args = _parser.parse_args()
    replot_from_csv(_args.csv, _args.out_dir, window_ms=_args.window_ms)
