"""
d3plot_to_h5.py  –  Compressed HDF5 exporter for LS-DYNA d3plot sequences.

Pipeline
--------
  Step 1a – Part filtering
      Keep only the parts listed / matched in required_parts.config.
      All nodes not belonging to any selected part are discarded.

  Step 1b – Spatial decimation  (--node-stride N)
      From the selected nodes (ordered by global node index), keep
      every N-th entry.  N=1 keeps all nodes.

  Step 2  – Temporal decimation  (--frame-stride N)
      Sort all states across every d3plot file by physical time,
      deduplicate, then keep every N-th frame.  N=2 keeps frames
      0, 2, 4, ...

  Step 3  – Per-frame extraction  (direct from d3plot, no manual math)
      For each kept frame:
        • position      (xyz)         ref_coords + node_displacement
        • velocity      (xyz)         finite difference of positions (dt=1 frame)
        • acceleration  (xyz)         finite difference of velocity   (dt=1 frame)
        • stress        (6-component) element solid/shell stress averaged to
                                      nodes:  sxx, syy, szz, sxy, syz, sxz
        • node_part_id  stored once in /metadata

python dataset/d3plot_to_h5_dt.py \
    --src /home/kong/datasets/barrier/fem/T_lok_F_shape_barrier_9_3_100km \
    --tmp /home/kong/datasets/barrier/tmp \
    --out /home/kong/datasets/barrier/h5/T_lok_F_shape_barrier_9_3_100km_50_1_100_dt/output.h5 \
    --required-config configs/data/required_parts.config \
    --node-stride 50 \
    --frame-stride 1 \
    --frame-limit 100


HDF5 layout
-----------
  /metadata/
      ref_positions       (N, 3)   float32   undeformed coords [mm]
      node_global_idx     (N,)     int64     global node indices
      node_part_label     (N,)     int64     compact label in [0, P-1]
      node_part_id        (N,)     int64     LS-DYNA part ID per node
      node_part_name      (N,)     bytes     UTF-8 part name per node
      part_ids            (P,)     int64
      part_names          (P,)     bytes
      part_patterns       (K,)     bytes
      attrs: node_stride, frame_stride, n_frames, n_nodes,
             stress_components = "sxx,syy,szz,sxy,syz,sxz"

  /states/
      times               (T,)        float64  [s]
      positions           (T, N, 3)   float32  [mm]   deformed xyz
      velocity            (T, N, 3)   float32  [mm/frame]   finite diff, dt=1
      acceleration        (T, N, 3)   float32  [mm/frame^2] finite diff, dt=1
      stress              (T, N, 6)   float32  [MPa]  sxx…sxz
"""

from __future__ import annotations

import argparse
import json
from fnmatch import fnmatch
from pathlib import Path
import shutil

import h5py
import numpy as np
from lasso.dyna import ArrayType, D3plot


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_SRC            = Path("/home/kong/datasets/barrier/fem/T_lok_F_shape_barrier_9_3_100km")
DEFAULT_TMP            = Path("/home/kong/datasets/barrier/tmp")
DEFAULT_OUT            = Path("/home/kong/datasets/barrier/h5/output.h5")
DEFAULT_REQUIRED_CONFIG = Path("dataset/required_parts.config")
DEFAULT_NODE_STRIDE    = 50   # 1 = keep all selected nodes
DEFAULT_FRAME_STRIDE   = 2   # 2 = keep every 2nd frame (0, 2, 4, …)


# ── Part-name helpers ──────────────────────────────────────────────────────────

def _decode_part_name(raw: object) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("ascii", errors="ignore").strip().strip("\x00")
    return str(raw).strip().strip("\x00")


def _load_patterns(path: Path) -> list[str]:
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line.lower())
    return patterns


def _pattern_matches(name_lower: str, pat: str) -> bool:
    """fnmatch for wildcard patterns; substring match otherwise."""
    if any(c in pat for c in "*?["):
        return fnmatch(name_lower, pat)
    return pat in name_lower


def _build_selected_part_mask(part_names: list[str], patterns: list[str]) -> np.ndarray:
    mask = np.zeros(len(part_names), dtype=bool)
    for i, name in enumerate(part_names):
        low = name.lower()
        mask[i] = any(_pattern_matches(low, p) for p in patterns)
    return mask


# ── Node / element selection ───────────────────────────────────────────────────

def _select_elements_and_nodes(
    part_indexes: np.ndarray | None,
    node_indexes: np.ndarray | None,
    sel_part_mask: np.ndarray,
    n_total_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    sel_elem_idx  : (E_sel,) int64  – element indices whose part is selected
    sel_node_idx  : (N_sel,) int64  – global node indices of those elements
    node_part_idx : (N_sel,) int64  – global part index for each selected node
    """
    empty = np.array([], dtype=np.int64)
    if part_indexes is None or node_indexes is None or len(part_indexes) == 0:
        return empty, empty, empty

    elem_mask = sel_part_mask[part_indexes]
    elem_idx  = np.where(elem_mask)[0].astype(np.int64)
    if elem_idx.size == 0:
        return empty, empty, empty

    # first-writer-wins assignment: each node gets the part of the first
    # element that touches it (deterministic for boundary nodes)
    node_part_full = np.full(n_total_nodes, -1, dtype=np.int64)
    for ei in elem_idx:
        pi = int(part_indexes[ei])
        for ni in node_indexes[ei]:
            ni = int(ni)
            if ni >= 0 and node_part_full[ni] == -1:
                node_part_full[ni] = pi

    sel_node_idx  = np.where(node_part_full >= 0)[0].astype(np.int64)
    node_part_idx = node_part_full[sel_node_idx].astype(np.int64)
    return elem_idx, sel_node_idx, node_part_idx


# ── Stress helpers ─────────────────────────────────────────────────────────────

def _average_stress_to_elem(stress_frame: np.ndarray) -> np.ndarray:
    """
    Collapse integration-point / layer dimensions → (n_elem, 6).

    Accepted input shapes:
      solid : (n_elem, n_ip, 6)
      shell : (n_elem, n_layer, n_ip, 6)  or  (n_elem, n_ip, 6)
    Last axis must be 6 :  sxx, syy, szz, sxy, syz, sxz
    """
    s = np.asarray(stress_frame, dtype=np.float32)
    if s.ndim > 2:
        # average over all middle dimensions (ip / layer)
        s = s.mean(axis=tuple(range(1, s.ndim - 1)))
    if s.ndim != 2 or s.shape[-1] < 6:
        raise ValueError(f"Cannot reduce stress shape {stress_frame.shape} to (n_elem, 6)")
    return s[:, :6]


def _scatter_stress_to_nodes(
    stress_elem: np.ndarray,       # (E_sel, 6) float32
    elem_node_conn: np.ndarray,    # (E_sel, nodes_per_elem) int64
    sel_node_idx: np.ndarray,      # (N_sel,) int64
    n_total_nodes: int,
) -> np.ndarray:                   # (N_sel, 6) float32
    """
    Average element stress tensors (all 6 components) onto nodes using
    vectorised np.add.at scatter – no Python element loop.
    """
    node_sum   = np.zeros((n_total_nodes, 6), dtype=np.float64)
    node_count = np.zeros(n_total_nodes,      dtype=np.int64)

    nodes_flat = elem_node_conn.ravel()                         # (E*K,)
    # repeat each element's stress row K times (one per node)
    K       = elem_node_conn.shape[1]
    s_rep   = np.repeat(stress_elem, K, axis=0).astype(np.float64)  # (E*K, 6)

    # mask out padding indices (lasso uses -1 or similar)
    valid = nodes_flat >= 0
    np.add.at(node_sum,   nodes_flat[valid], s_rep[valid])
    np.add.at(node_count, nodes_flat[valid], 1)

    out   = np.zeros((len(sel_node_idx), 6), dtype=np.float32)
    denom = node_count[sel_node_idx]
    good  = denom > 0
    out[good] = (node_sum[sel_node_idx][good] / denom[good, None]).astype(np.float32)
    return out


def _compute_node_stress(
    solid_sf: np.ndarray | None,        # (n_solid_all, ..., 6) for this state
    shell_sf: np.ndarray | None,        # (n_shell_all, ..., 6) for this state
    sel_solid_elem_idx:   np.ndarray,   # (E_solid_sel,)
    sel_shell_elem_idx:   np.ndarray,   # (E_shell_sel,)
    sel_solid_elem_nodes: np.ndarray,   # (E_solid_sel, 8)
    sel_shell_elem_nodes: np.ndarray,   # (E_shell_sel, 4)
    sel_node_idx: np.ndarray,           # (N_sel,)
    n_total_nodes: int,
) -> np.ndarray:                        # (N_sel, 6) float32
    """
    Accumulate contributions from solid and shell elements, then scatter
    the averaged 6-component stress tensor to selected nodes.
    """
    node_sum   = np.zeros((n_total_nodes, 6), dtype=np.float64)
    node_count = np.zeros(n_total_nodes,      dtype=np.int64)

    for sf, elem_idx, elem_nodes in [
        (solid_sf, sel_solid_elem_idx, sel_solid_elem_nodes),
        (shell_sf, sel_shell_elem_idx, sel_shell_elem_nodes),
    ]:
        if sf is None or len(elem_idx) == 0:
            continue
        s6         = _average_stress_to_elem(sf[elem_idx]).astype(np.float64)  # (E, 6)
        nodes_flat = elem_nodes.ravel()
        K          = elem_nodes.shape[1]
        s_rep      = np.repeat(s6, K, axis=0)                                 # (E*K, 6)
        valid      = nodes_flat >= 0
        np.add.at(node_sum,   nodes_flat[valid], s_rep[valid])
        np.add.at(node_count, nodes_flat[valid], 1)

    out   = np.zeros((len(sel_node_idx), 6), dtype=np.float32)
    denom = node_count[sel_node_idx]
    good  = denom > 0
    out[good] = (node_sum[sel_node_idx][good] / denom[good, None]).astype(np.float32)
    return out


# ── Finite-difference velocity / acceleration (dt = 1 frame) ───────────────────

def _diff_velocity_acceleration(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Forward finite-difference kinematics with dt = 1 frame.

    Matches the training loader's np.diff convention exactly:
        vel[i] = pos[i+1] - pos[i]      valid for i in [0, T-2]
        acc[i] = vel[i+1] - vel[i]      valid for i in [0, T-3]
                = pos[i+2] - 2*pos[i+1] + pos[i]

    Returned arrays are length T (for storage compatibility with positions).
    Trailing frames that have no valid forward difference are PADDED by
    repeating the last valid value; these padded frames are physically
    meaningless and must NOT be used as targets.  The caller is told how
    many leading frames are valid so statistics can exclude the padding.

    Units are mm/frame and mm/frame^2 (dt treated as 1).  The physical
    timestep is stored separately (dt_seconds) for later unit conversion.

    Returns
    -------
    vel          : (T, N, 3) float32  – forward-diff velocity, tail-padded
    acc          : (T, N, 3) float32  – forward-diff acceleration, tail-padded
    n_vel_valid  : int  – number of valid (unpadded) velocity frames  = max(T-1, 0)
    n_acc_valid  : int  – number of valid (unpadded) acceleration frames = max(T-2, 0)
    """
    T = positions.shape[0]
    pos = positions.astype(np.float64)

    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    n_vel_valid = max(T - 1, 0)
    n_acc_valid = max(T - 2, 0)

    if n_vel_valid > 0:
        vel[:n_vel_valid] = np.diff(pos, axis=0)                 # (T-1, N, 3)
        vel[n_vel_valid:] = vel[n_vel_valid - 1]                 # pad tail
    if n_acc_valid > 0:
        acc[:n_acc_valid] = np.diff(vel[:n_vel_valid], axis=0)   # (T-2, N, 3)
        acc[n_acc_valid:] = acc[n_acc_valid - 1]                 # pad tail

    return vel.astype(np.float32), acc.astype(np.float32), n_vel_valid, n_acc_valid


# ── File helpers ───────────────────────────────────────────────────────────────

def _suffix_number(path: Path) -> int:
    s = path.name.replace("d3plot", "", 1)
    if s == "":
        return -1
    try:
        return int(s)
    except ValueError:
        return 10 ** 12


def _find_state_files(src_dir: Path) -> list[Path]:
    """Return d3plot01, d3plot02, … sorted by numeric suffix."""
    files = [p for p in src_dir.glob("d3plot*") if p.is_file() and p.name != "d3plot"]
    return sorted(files, key=lambda p: (_suffix_number(p), p.name))


def _copy_to_tmp(src: Path, tmp_dir: Path) -> Path:
    """Copy the main d3plot header + one state file into tmp.

    State files (d3plot01, d3plot02, …) are not self-contained: they have
    no file header and cannot be opened directly.  Lasso needs the main
    d3plot header to know the mesh layout, then reads state data from
    d3plot01.  So we copy:
        src_dir/d3plot   -> tmp/d3plot    (header, read-only geometry)
        src_dir/d3plotXX -> tmp/d3plot01  (single state file)
    and open tmp/d3plot.  Clean up both files in the caller.
    """
    src_dir = src.parent
    hdr_dst = tmp_dir / "d3plot"
    st_dst  = tmp_dir / "d3plot01"
    hdr_dst.unlink(missing_ok=True)
    st_dst.unlink(missing_ok=True)
    shutil.copy2(src_dir / "d3plot", hdr_dst)
    shutil.copy2(src, st_dst)
    return st_dst   # caller deletes this; hdr_dst is also cleaned up by caller


# ── Time scanning & frame selection ───────────────────────────────────────────

def _scan_state_times(
    state_files: list[Path],
    tmp_dir: Path,
) -> list[tuple[float, str, int]]:
    """
    Open each state file (timestamp-only load), collect (time, filename, state_idx).
    """
    entries: list[tuple[float, str, int]] = []
    print("\nPass 1/2 – scanning state times …")
    for src in state_files:
        _copy_to_tmp(src, tmp_dir)
        try:
            d3  = D3plot(str(tmp_dir / "d3plot"), state_array_filter=[ArrayType.global_timesteps])
            t_arr = d3.arrays.get(ArrayType.global_timesteps, np.array([0.0]))
            for s, t in enumerate(t_arr):
                entries.append((float(t), src.name, int(s)))
            if len(t_arr):
                print(f"  {src.name}: {len(t_arr)} state(s), "
                      f"t = {float(t_arr[0])*1e3:.2f} – {float(t_arr[-1])*1e3:.2f} ms")
            del d3
        finally:
            (tmp_dir / "d3plot").unlink(missing_ok=True)
            (tmp_dir / "d3plot01").unlink(missing_ok=True)
    return entries


def _select_frames(
    entries: list[tuple[float, str, int]],
    frame_stride: int,
) -> list[tuple[float, str, int]]:
    """
    Sort by time → deduplicate → keep every `frame_stride`-th entry.
    """
    if not entries:
        raise RuntimeError("No states found in source d3plot files.")

    entries_sorted = sorted(entries, key=lambda x: x[0])
    unique: list[tuple[float, str, int]] = []
    last_t: float | None = None
    for e in entries_sorted:
        if last_t is None or abs(e[0] - last_t) > 1e-9:
            unique.append(e)
            last_t = e[0]

    times = np.asarray([e[0] for e in unique])
    mean_dt_ms = float(np.mean(np.diff(times)) * 1e3) if len(times) > 1 else 0.0
    print(f"\nTotal unique states : {len(unique)}"
          f"  (t = {times[0]*1e3:.2f} – {times[-1]*1e3:.2f} ms,"
          f"  mean Δt = {mean_dt_ms:.3f} ms)")

    selected = unique[::frame_stride]
    print(f"After frame-stride={frame_stride} : {len(selected)} frames kept")
    return selected


# ── HDF5 creation ──────────────────────────────────────────────────────────────

def _create_h5_datasets(h5f: h5py.File, n_frames: int, n_nodes: int) -> None:
    """Pre-allocate all /states datasets with chunked gzip compression."""
    sg = h5f.require_group("states")
    ct = min(16, n_frames)          # time chunk
    cn = min(4096, n_nodes)         # node chunk

    def ds(name: str, shape: tuple, dtype: str) -> None:
        chunks = (ct,) + shape[1:]  # all non-time dims kept whole per chunk slice
        # override node chunk only when we have a node dimension
        if len(shape) == 2 and shape[1] == n_nodes:
            chunks = (ct, cn)
        elif len(shape) == 3 and shape[1] == n_nodes:
            chunks = (ct, cn, shape[2])
        sg.create_dataset(name, shape=shape, dtype=dtype,
                          chunks=chunks, compression="gzip", compression_opts=4)

    ds("times",        (n_frames,),          "float64")
    ds("positions",    (n_frames, n_nodes, 3), "float32")
    ds("velocity",     (n_frames, n_nodes, 3), "float32")
    ds("acceleration", (n_frames, n_nodes, 3), "float32")
    ds("stress",       (n_frames, n_nodes, 6), "float32")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LS-DYNA d3plot sequence → single HDF5 training file."
    )
    parser.add_argument("--src",             type=Path,  default=DEFAULT_SRC,
                        help="Directory containing d3plot, d3plot01, d3plot02 …")
    parser.add_argument("--tmp",             type=Path,  default=DEFAULT_TMP,
                        help="Scratch directory for single-file copies.")
    parser.add_argument("--out",             type=Path,  default=DEFAULT_OUT,
                        help="Output HDF5 file path  (e.g. output.h5).")
    parser.add_argument("--required-config", type=Path,  default=DEFAULT_REQUIRED_CONFIG,
                        help="Part-name filter list (one pattern per line).")
    parser.add_argument("--node-stride",     type=int,   default=DEFAULT_NODE_STRIDE,
                        help="Spatial decimation: keep every N-th node  (1 = all).")
    parser.add_argument("--frame-stride",    type=int,   default=DEFAULT_FRAME_STRIDE,
                        help="Temporal decimation: keep every N-th frame (1 = all, 2 = default).")
    frame_limit_group = parser.add_mutually_exclusive_group()
    frame_limit_group.add_argument("--frame-limit", type=int,   default=None, metavar="K",
                        help="Keep only the first K frames after stride (exact count).")
    frame_limit_group.add_argument("--frame-scale", type=float, default=None, metavar="FRAC",
                        help="Keep only the first FRAC fraction of frames, e.g. 0.1 for 10%%.")
    args = parser.parse_args()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    for path, label in [
        (args.src,             "source folder"),
        (args.tmp,             "tmp folder"),
        (args.required_config, "required_parts config"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.node_stride < 1:
        raise ValueError("--node-stride must be >= 1")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for old in args.tmp.glob("d3plot0*"):
        old.unlink()

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1a – Part selection (from d3plot header)
    # ══════════════════════════════════════════════════════════════════════════
    # Copy ONLY the bare header file into tmp so lasso cannot follow the
    # file-chain links (d3plot -> d3plot01 -> d3plot02 -> ...) and pull all
    # state files into memory at once.  Delete the copy immediately after.
    print("Copying d3plot header to tmp ...")
    hdr_tmp = args.tmp / "d3plot"
    hdr_tmp.unlink(missing_ok=True)
    shutil.copy2(args.src / "d3plot", hdr_tmp)
    print("Loading d3plot header for part / connectivity info ...")
    try:
        d3hdr = D3plot(str(hdr_tmp), state_array_filter=[])
    finally:
        hdr_tmp.unlink(missing_ok=True)

    part_titles   = d3hdr.arrays[ArrayType.part_titles]
    part_ids_full = d3hdr.arrays[ArrayType.part_titles_ids]
    part_names    = [_decode_part_name(x) for x in part_titles]

    patterns      = _load_patterns(args.required_config)
    sel_part_mask = _build_selected_part_mask(part_names, patterns)
    sel_part_idx  = np.where(sel_part_mask)[0].astype(np.int64)

    print(f"Patterns: {len(patterns)}  →  matched {len(sel_part_idx)}/{len(part_names)} parts")
    for i in sel_part_idx:
        print(f"  part_idx={int(i):4d}  part_id={int(part_ids_full[i]):6d}"
              f"  name={part_names[i]}")
    if len(sel_part_idx) == 0:
        raise RuntimeError("No parts matched. Check required_parts.config.")

    solid_part_idx  = d3hdr.arrays.get(ArrayType.element_solid_part_indexes)
    solid_node_idx  = d3hdr.arrays.get(ArrayType.element_solid_node_indexes)
    shell_part_idx  = d3hdr.arrays.get(ArrayType.element_shell_part_indexes)
    shell_node_idx  = d3hdr.arrays.get(ArrayType.element_shell_node_indexes)
    coords_ref      = d3hdr.arrays[ArrayType.node_coordinates]
    n_total_nodes   = len(coords_ref)

    sel_solid_ei, solid_node_sel, solid_node_part = _select_elements_and_nodes(
        solid_part_idx, solid_node_idx, sel_part_mask, n_total_nodes)
    sel_shell_ei, shell_node_sel, shell_node_part = _select_elements_and_nodes(
        shell_part_idx, shell_node_idx, sel_part_mask, n_total_nodes)

    # merge solid + shell node assignments (solid wins on shared nodes)
    node_part_full = np.full(n_total_nodes, -1, dtype=np.int64)
    for ni, pi in zip(solid_node_sel, solid_node_part):
        node_part_full[int(ni)] = int(pi)
    for ni, pi in zip(shell_node_sel, shell_node_part):
        if node_part_full[int(ni)] == -1:
            node_part_full[int(ni)] = int(pi)

    all_sel_nodes  = np.where(node_part_full >= 0)[0].astype(np.int64)
    all_node_parts = node_part_full[all_sel_nodes].astype(np.int64)
    print(f"Selected  solid elems : {len(sel_solid_ei)}")
    print(f"          shell elems : {len(sel_shell_ei)}")
    print(f"          nodes       : {len(all_sel_nodes)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1b – Spatial decimation  (every node_stride-th node)
    # ══════════════════════════════════════════════════════════════════════════
    keep_mask    = np.zeros(len(all_sel_nodes), dtype=bool)
    keep_mask[::args.node_stride] = True
    sel_node_idx  = all_sel_nodes[keep_mask]
    node_part_idx = all_node_parts[keep_mask]
    n_nodes       = len(sel_node_idx)

    print(f"\nAfter node-stride={args.node_stride}: {n_nodes} nodes kept"
          f"  (from {len(all_sel_nodes)})")
    if n_nodes == 0:
        raise RuntimeError("No nodes remain after decimation. Reduce --node-stride.")

    # element connectivity sub-arrays for the kept elements only
    sel_solid_elem_nodes = (
        solid_node_idx[sel_solid_ei]
        if solid_node_idx is not None and len(sel_solid_ei) > 0
        else np.empty((0, 8), dtype=np.int64)
    )
    sel_shell_elem_nodes = (
        shell_node_idx[sel_shell_ei]
        if shell_node_idx is not None and len(sel_shell_ei) > 0
        else np.empty((0, 4), dtype=np.int64)
    )

    # reference (undeformed) positions of the final node set
    ref_pos_sel = coords_ref[sel_node_idx].astype(np.float32)

    # per-node metadata arrays
    global_to_compact = {int(pi): j for j, pi in enumerate(sel_part_idx)}
    node_part_label   = np.array([global_to_compact[int(pi)] for pi in node_part_idx], dtype=np.int64)
    node_part_id      = np.array([int(part_ids_full[int(pi)]) for pi in node_part_idx],  dtype=np.int64)
    node_part_name    = np.array([part_names[int(pi)]         for pi in node_part_idx])
    sel_part_ids      = np.array([int(part_ids_full[i])       for i in sel_part_idx],     dtype=np.int64)
    sel_part_names    = np.array([part_names[i]               for i in sel_part_idx])
    del d3hdr   # free header memory

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2 – Frame selection (every frame_stride-th state, globally sorted)
    # ══════════════════════════════════════════════════════════════════════════
    state_files = _find_state_files(args.src)
    if not state_files:
        raise FileNotFoundError(f"No d3plot state files found in {args.src}")

    all_entries      = _scan_state_times(state_files, args.tmp)
    selected_entries = _select_frames(all_entries, args.frame_stride)

    # ── Optional head-trim (applied after stride so stats match the kept data) ─
    if args.frame_limit is not None:
        k = args.frame_limit
    elif args.frame_scale is not None:
        import math
        k = max(1, math.ceil(len(selected_entries) * args.frame_scale))
    else:
        k = len(selected_entries)
    if k < len(selected_entries):
        print(f"Head-trim: keeping first {k} / {len(selected_entries)} frames "
              f"({k/len(selected_entries)*100:.1f}%)")
        selected_entries = selected_entries[:k]

    n_frames         = len(selected_entries)

    # ══════════════════════════════════════════════════════════════════════════
    # Create HDF5 and write static metadata
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nCreating {args.out}  ({n_frames} frames × {n_nodes} nodes × fields) …")
    h5f = h5py.File(args.out, "w")
    _create_h5_datasets(h5f, n_frames, n_nodes)

    mg = h5f.require_group("metadata")
    mg.create_dataset("ref_positions",   data=ref_pos_sel)
    mg.create_dataset("node_global_idx", data=sel_node_idx.astype(np.int64))
    mg.create_dataset("node_part_label", data=node_part_label)
    mg.create_dataset("node_part_id",    data=node_part_id)
    mg.create_dataset("node_part_name",  data=node_part_name.astype("S"))   # bytes
    mg.create_dataset("part_ids",        data=sel_part_ids)
    mg.create_dataset("part_names",      data=sel_part_names.astype("S"))
    mg.create_dataset("part_patterns",   data=np.asarray(patterns, dtype="S"))
    mg.attrs["node_stride"]         = args.node_stride
    mg.attrs["frame_stride"]        = args.frame_stride
    mg.attrs["n_frames"]            = n_frames
    mg.attrs["n_nodes"]             = n_nodes
    mg.attrs["n_parts"]             = len(sel_part_idx)
    mg.attrs["stress_components"]   = "sxx,syy,szz,sxy,syz,sxz"
    mg.attrs["velocity_source"]     = "pending"       # updated below
    mg.attrs["acceleration_source"] = "pending"

    # add front face / barrier part mask for distance calc
    part_names_str = [str(n).strip() for n in node_part_name]

    barrier_patterns   = ["concrete_fine_mesh"]
    car_collision_patterns = ["frontface"]

    barrier_mask   = _build_selected_part_mask(part_names_str, barrier_patterns)
    frontface_mask = _build_selected_part_mask(part_names_str, car_collision_patterns)

    barrier_idx   = np.where(barrier_mask)[0].astype(np.int64)
    frontface_idx = np.where(frontface_mask)[0].astype(np.int64)

    assert barrier_idx.size   > 0, "barrier mask is empty, check your patterns"
    assert frontface_idx.size > 0, "frontface mask is empty, check your patterns"
    assert not (barrier_mask & frontface_mask).any(), "barrier and frontface overlap, check your patterns"

    mg.create_dataset("barrier_idx",        data=barrier_idx)
    mg.create_dataset("frontface_idx",      data=frontface_idx)
    mg.create_dataset("barrier_patterns",   data=np.asarray(barrier_patterns, dtype="S"))
    mg.create_dataset("frontface_patterns", data=np.asarray(car_collision_patterns, dtype="S"))
    mg.attrs["n_barrier_nodes"]   = int(barrier_idx.size)
    mg.attrs["n_frontface_nodes"] = int(frontface_idx.size)

    print(f"[h5] barrier:   {barrier_idx.size}/{n_nodes}")
    print(f"[h5] frontface: {frontface_idx.size}/{n_nodes}")

    metadata = {
        "config": {
            "source": str(args.src.resolve()),
            "output": str(args.out.resolve()),
            "required_config": str(args.required_config.resolve()),
            "node_stride": args.node_stride,
            "frame_stride": args.frame_stride,
            "n_frames": n_frames,
            "n_nodes": n_nodes,
            "n_parts": len(sel_part_idx),
            "normalised": False,
            "value_mode": "raw_physical_units",
            "stress_components": "sxx,syy,szz,sxy,syz,sxz",
            "velocity_source": None,
            "acceleration_source": None,
            "selected_part_patterns": patterns,
        },
        "dataset_stats": {
            "num_simulations": 1,
            "train_windows": 0,
            "val_windows": 0,
            "test_trajectories": 0,
        },
        "sim_mask_info": {
            "0": {
                "barrier_idx": int(barrier_idx.size),
                "frontface_idx": int(frontface_idx.size),
            }
        },
    }

    meta_path = args.out.parent / "metadata.json"

    # ── Simple stats accumulator for field summary ─────────────────────────────
    class FieldStats:
        """Track min, max, and online Welford mean/variance for a field."""
        def __init__(self, name: str, dim: int):
            self.name = name
            self.dim = dim
            self.count = 0
            self.min_vals = np.full(dim, np.inf, dtype=np.float64)
            self.max_vals = np.full(dim, -np.inf, dtype=np.float64)
            self.mean = np.zeros(dim, dtype=np.float64)
            self.M2 = np.zeros(dim, dtype=np.float64)

        def update(self, arr: np.ndarray):
            """arr shape (N, D) or (T, N, D) – flatten to (M, D)"""
            arr = arr.astype(np.float64).reshape(-1, self.dim)
            for row in arr:
                self.count += 1
                self.min_vals = np.minimum(self.min_vals, row)
                self.max_vals = np.maximum(self.max_vals, row)
                delta = row - self.mean
                self.mean += delta / self.count
                delta2 = row - self.mean
                self.M2 += delta * delta2

        def finalize(self) -> dict:
            """Return {mean, std, min, max} as lists."""
            if self.count < 2:
                std = np.ones_like(self.mean)
            else:
                std = np.sqrt(self.M2 / self.count)
            return {
                "mean": self.mean.tolist(),
                "std": std.tolist(),
                "min": self.min_vals.tolist(),
                "max": self.max_vals.tolist(),
            }

    pos_stats = FieldStats("positions", 3)
    vel_stats = FieldStats("velocity", 3)
    acc_stats = FieldStats("acceleration", 3)
    stress_stats = FieldStats("stress", 6)

    # accumulators for median calculation (store per-frame flattened arrays)
    vel_accum: list[np.ndarray] = []
    acc_accum: list[np.ndarray] = []
    stress_accum: list[np.ndarray] = []

    # ══════════════════════════════════════════════════════════════════════════
    # Pass 2/2 – Extract fields frame by frame (globally time-ordered)
    # ══════════════════════════════════════════════════════════════════════════
    print("\nPass 2/2 – extracting frames in time order …")

    state_file_map = {p.name: p for p in state_files}
    current_name: str | None = None
    d3 = None

    # We need positions in memory to compute finite-difference vel/acc afterward.
    positions_all = np.empty((n_frames, n_nodes, 3), dtype=np.float32)
    times_all     = np.empty(n_frames,               dtype=np.float64)

    def _close() -> None:
        nonlocal d3
        if d3 is not None:
            del d3;  d3 = None
        (args.tmp / "d3plot").unlink(missing_ok=True)
        (args.tmp / "d3plot01").unlink(missing_ok=True)

    try:
        for fi, (t, fname, sidx) in enumerate(selected_entries):

            # ── open new state file when needed ───────────────────────────
            if current_name != fname:
                _close()
                src_file     = state_file_map[fname]
                _copy_to_tmp(src_file, args.tmp)
                current_name = fname
                print(f"  loading {fname}  (frame {fi+1}/{n_frames})")

                d3 = D3plot(
                    str(args.tmp / "d3plot"),
                    state_array_filter=[
                        ArrayType.global_timesteps,
                        ArrayType.node_displacement,
                        ArrayType.element_solid_stress,
                        ArrayType.element_shell_stress,
                    ],
                )

            # ── position  (ref coords + displacement from d3plot) ─────────
            disp_full = d3.arrays.get(ArrayType.node_displacement)
            if disp_full is None:
                raise RuntimeError(f"node_displacement missing in {fname}")

            disp_sel = disp_full[sidx][sel_node_idx].astype(np.float32)
            pos      = ref_pos_sel + disp_sel                    # true deformed position

            positions_all[fi] = pos
            times_all[fi]     = t
            h5f["states/times"][fi]     = t
            h5f["states/positions"][fi] = pos
            pos_stats.update(pos)

            # velocity / acceleration are computed AFTER the loop from the full
            # positions array by finite difference (dt = 1 frame) — see below.

            # ── stress (6-component tensor, element-avg → nodes) ──────────
            solid_sf = d3.arrays.get(ArrayType.element_solid_stress)
            shell_sf = d3.arrays.get(ArrayType.element_shell_stress)
            stress_node = _compute_node_stress(
                solid_sf[sidx] if solid_sf is not None else None,
                shell_sf[sidx] if shell_sf is not None else None,
                sel_solid_ei,
                sel_shell_ei,
                sel_solid_elem_nodes,
                sel_shell_elem_nodes,
                sel_node_idx,
                n_total_nodes,
            )
            h5f["states/stress"][fi] = stress_node
            stress_stats.update(stress_node)
            stress_accum.append(stress_node.reshape(-1, 6))

            if fi % 20 == 0 or fi == n_frames - 1:
                print(f"  [{fi+1:>5}/{n_frames}]  t = {t*1e3:.3f} ms")

    finally:
        _close()

    # ── Finite-difference kinematics (dt = 1 frame) ───────────────────────────
    # velocity/acceleration are ALWAYS recomputed from positions by forward
    # difference with dt treated as 1 (units: mm/frame, mm/frame^2).  The
    # physical Δt is stored separately (dt_seconds) so these can be converted
    # back to mm/s, mm/s^2 downstream if needed.
    print("\nComputing velocity/acceleration by finite difference (dt = 1 frame) …")
    vel_diff, acc_diff, n_vel_valid, n_acc_valid = _diff_velocity_acceleration(positions_all)
    h5f["states/velocity"][:]     = vel_diff
    h5f["states/acceleration"][:] = acc_diff

    # statistics over VALID frames only (exclude the padded tail so the
    # normalization stats in metadata.json match exactly what the loader sees)
    if n_vel_valid > 0:
        vel_stats.update(vel_diff[:n_vel_valid])
        vel_accum.append(vel_diff[:n_vel_valid].reshape(-1, 3))
    if n_acc_valid > 0:
        acc_stats.update(acc_diff[:n_acc_valid])
        acc_accum.append(acc_diff[:n_acc_valid].reshape(-1, 3))
    print(f"  done.  valid frames: vel={n_vel_valid}, acc={n_acc_valid} (of {n_frames})")

    # ── Physical timestep Δt (for unit conversion only; NOT used in the diff) ──
    if len(times_all) > 1:
        dts     = np.diff(times_all)
        dt_mean = float(np.mean(dts))
        dt_min  = float(np.min(dts))
        dt_max  = float(np.max(dts))
        dt_std  = float(np.std(dts))
    else:
        dt_mean = dt_min = dt_max = dt_std = 0.0
    # dt=1 finite difference assumes uniform spacing — warn if it is not
    if dt_mean > 0 and (dt_max - dt_min) / dt_mean > 1e-3:
        print(f"  WARNING: non-uniform frame spacing "
              f"(Δt min={dt_min*1e3:.4f} ms, max={dt_max*1e3:.4f} ms, "
              f"std={dt_std*1e3:.4f} ms).  dt=1 finite difference assumes uniform Δt; "
              f"per-frame ↔ physical unit conversion will be approximate.")
    print(f"  physical Δt: mean={dt_mean*1e3:.4f} ms  "
          f"(min={dt_min*1e3:.4f}, max={dt_max*1e3:.4f})")

    velocity_source     = "finite_difference_dt1"
    acceleration_source = "finite_difference_dt1"
    convention_str      = "forward_diff: vel[i]=pos[i+1]-pos[i], acc[i]=vel[i+1]-vel[i], dt=1 frame"
    h5f["metadata"].attrs["velocity_source"]       = velocity_source
    h5f["metadata"].attrs["acceleration_source"]   = acceleration_source
    h5f["metadata"].attrs["dt_seconds"]            = dt_mean
    h5f["metadata"].attrs["kinematics_convention"] = convention_str
    h5f["metadata"].attrs["n_vel_valid_frames"]    = int(n_vel_valid)
    h5f["metadata"].attrs["n_acc_valid_frames"]    = int(n_acc_valid)
    h5f.close()

    metadata["config"]["velocity_source"]     = velocity_source
    metadata["config"]["acceleration_source"] = acceleration_source
    metadata["config"]["value_mode"]          = "positions_physical_mm; vel_acc_per_frame_dt1"
    metadata["config"]["dt_seconds"]          = dt_mean
    metadata["config"]["dt_min_seconds"]      = dt_min
    metadata["config"]["dt_max_seconds"]      = dt_max
    metadata["config"]["dt_std_seconds"]      = dt_std
    metadata["config"]["dt_unit_note"]        = (
        "velocity/acceleration use dt=1 frame (units: mm/frame, mm/frame^2). "
        "Multiply velocity by 1/dt_seconds and acceleration by 1/dt_seconds^2 "
        "to recover mm/s and mm/s^2."
    )
    metadata["config"]["kinematics_convention"] = convention_str
    metadata["config"]["n_vel_valid_frames"]     = int(n_vel_valid)
    metadata["config"]["n_acc_valid_frames"]     = int(n_acc_valid)

    # Finalize field statistics
    # finalize numeric summaries
    fs_pos = pos_stats.finalize()
    fs_vel = vel_stats.finalize()
    fs_acc = acc_stats.finalize()
    fs_stress = stress_stats.finalize()

    # medians
    try:
        pos_median = np.median(positions_all.reshape(-1, 3), axis=0).tolist()
    except Exception:
        pos_median = [0.0, 0.0, 0.0]

    def _safe_median(accum_list, dim):
        if not accum_list:
            return [0.0] * dim
        arr = np.concatenate(accum_list, axis=0)
        return np.median(arr, axis=0).tolist()

    vel_median = _safe_median(vel_accum, 3)
    acc_median = _safe_median(acc_accum, 3)
    stress_median = _safe_median(stress_accum, 6)

    fs_pos["median"] = pos_median
    fs_vel["median"] = vel_median
    fs_acc["median"] = acc_median
    fs_stress["median"] = stress_median

    metadata["field_stats"] = {
        "positions": fs_pos,
        "velocity": fs_vel,
        "acceleration": fs_acc,
        "stress": fs_stress,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    size_gb = args.out.stat().st_size / 1e9
    print(f"\n{'─'*60}")
    print(f"Done.  Output  : {args.out}")
    print(f"Size           : {size_gb:.3f} GB")
    print(f"Frames (T)     : {n_frames}  (frame-stride = {args.frame_stride})")
    print(f"Nodes  (N)     : {n_nodes}  (node-stride  = {args.node_stride})")
    print(f"Parts  (P)     : {len(sel_part_idx)}")
    print(f"Velocity src   : finite difference (dt = 1 frame)")
    print(f"Accel src      : finite difference (dt = 1 frame)")
    print(f"Physical Δt    : {dt_mean*1e3:.4f} ms  (stored as dt_seconds for unit conversion)")
    print(f"HDF5 layout    : /metadata  +  /states/{{times,positions,velocity,acceleration,stress}}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()