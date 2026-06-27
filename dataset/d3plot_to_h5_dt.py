"""
d3plot_to_h5.py  –  Compressed HDF5 exporter for LS-DYNA d3plot sequences.

Pipeline
--------
  Step 1a – Part filtering (two-layer, --sampling-config YAML)
      Layer 1 (collision_zone):  all nodes from car_contact_parts and
                                 barrier_parts are kept unconditionally.
      Layer 2 (required_parts):  nodes from all other structural parts are
                                 kept every --node-stride-th entry.
                                 Nodes already in Layer 1 are deduplicated out.
      The union of both layers forms the final node set sel_node_idx.

  Step 1b – (removed; stride is now applied per-layer inside Step 1a)

  Step 2  – Temporal decimation  (--frame-stride N)
      Sort all states across every d3plot file by physical time,
      deduplicate, then keep every N-th frame.  N=2 keeps frames
      0, 2, 4, ...

  Step 3  – Per-frame extraction  (direct from d3plot, no manual math)
      For each kept frame:
        • position        (xyz)         ref_coords + node_displacement
        • velocity        (xyz)         finite difference of positions (dt=1 frame)
        • acceleration    (xyz)         finite difference of velocity   (dt=1 frame)
        • stress          (6-component) element solid/shell stress averaged to nodes
        • plastic_strain  (scalar)      element eff. plastic strain averaged to nodes
        • node_part_id    stored once in /metadata
        • material props  stored once in /metadata (from --kfile)

python dataset/d3plot_to_h5_dt.py \
    --src /home/kong/datasets/barrier/fem/T_lok_F_shape_barrier_9_3_100km_plus800kg \
    --tmp /home/kong/datasets/barrier/tmp \
    --out /home/kong/datasets/barrier/h5dt_50ns_10fs_mat/T_lok_F_shape_barrier_9_3_100km_plus800kg/output.h5 \
    --sampling-config configs/data/sampling_config.yaml \
    --node-stride 50 \
    --frame-stride 10 


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
      node_mat_type_id    (N,)     int32     material type code (see _MAT_TYPE_ID_MAP)
      node_mat_type_name  (N,)     bytes     material keyword string (e.g. PIECEWISE_LINEAR_PLASTICITY)
      node_mat_label      (N,)     bytes     precise title from *MAT_xxx_TITLE card (e.g. "Steel - 300")
                                             falls back to type_name when _TITLE absent in k-file
      node_mat_E          (N,)     float32   Young's modulus [MPa]
      node_mat_sigy       (N,)     float32   yield stress [MPa]
      node_mat_rho        (N,)     float32   density [t/mm³]
      node_mat_nu         (N,)     float32   Poisson's ratio
      attrs: node_stride, node_selection, frame_stride, n_frames, n_nodes,
             stress_components = "sxx,syy,szz,sxy,syz,sxz"

  /states/
      times               (T,)        float64  [s]
      positions           (T, N, 3)   float32  [mm]   deformed xyz
      velocity            (T, N, 3)   float32  [mm/frame]   finite diff, dt=1
      acceleration        (T, N, 3)   float32  [mm/frame^2] finite diff, dt=1
      stress              (T, N, 6)   float32  [MPa]  sxx…sxz
      plastic_strain      (T, N)      float32  [-]    eff. plastic strain at nodes
"""

from __future__ import annotations

import argparse
import json
import re
from fnmatch import fnmatch
from pathlib import Path
import shutil

import h5py
import numpy as np
import yaml
from lasso.dyna import ArrayType, D3plot


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_SRC             = Path("/home/kong/datasets/barrier/fem/T_lok_F_shape_barrier_9_3_100km")
DEFAULT_TMP             = Path("/home/kong/datasets/barrier/tmp")
DEFAULT_OUT             = Path("/home/kong/datasets/barrier/h5/output.h5")
DEFAULT_SAMPLING_CONFIG = Path("configs/data/sampling_config.yaml")
DEFAULT_NODE_STRIDE     = 50   # applies to Layer 2 (required_parts) only
DEFAULT_FRAME_STRIDE    = 2    # 2 = keep every 2nd frame (0, 2, 4, …)


# ── Part-name helpers ──────────────────────────────────────────────────────────

def _decode_part_name(raw: object) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("ascii", errors="ignore").strip().strip("\x00")
    return str(raw).strip().strip("\x00")


def _load_sampling_config(path: Path) -> tuple[list[str], list[str], dict]:
    """
    Returns:
        collision_patterns : flat list of all patterns from collision_zone.*
        required_patterns  : flat list of all patterns from required_parts.*
        collision_zone_cfg : raw collision_zone sub-dict for sub-group access
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    collision_zone_cfg: dict = cfg.get("collision_zone", {})
    collision_patterns: list[str] = []
    for group in collision_zone_cfg.values():
        if isinstance(group, list):
            collision_patterns.extend(p.lower() for p in group)

    required_patterns: list[str] = []
    for group in cfg.get("required_parts", {}).values():
        if isinstance(group, list):
            required_patterns.extend(p.lower() for p in group)

    return collision_patterns, required_patterns, collision_zone_cfg


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


# ── K-file material parser ────────────────────────────────────────────────────

_MAT_TYPE_ID_MAP: dict[str, int] = {
    "PIECEWISE_LINEAR_PLASTICITY":          0,
    "MODIFIED_PIECEWISE_LINEAR_PLASTICITY": 0,
    "RIGID":                                1,
    "ELASTIC":                              2,
    "BLATZ-KO_RUBBER":                      3,
    "CONCRETE_DAMAGE_REL3":                 4,
    "LOW_DENSITY_FOAM":                     5,
    "SPOTWELD":                             6,
    "SPRING_ELASTIC":                       7,
    "SPRING_NONLINEAR_ELASTIC":             7,
    "DAMPER_NONLINEAR_VISCOUS":             7,
}
_MAT_TYPE_UNKNOWN = 8


def _kfields(line: str) -> list[str]:
    """Parse LS-DYNA 10-char fixed-width data line into token list."""
    s = line.rstrip()
    if len(s) < 10:
        return s.split()
    chunks = [s[k:k+10].strip() for k in range(0, len(s), 10)]
    chunks = [c for c in chunks if c]
    return chunks if len(chunks) > 1 else s.split()


def _kf(tok: str) -> float:
    try:
        return float(tok)
    except (ValueError, TypeError):
        return 0.0


def _ki(tok: str) -> int:
    try:
        return int(tok.split(".")[0])
    except (ValueError, TypeError):
        return 0


def _parse_kfile_materials(path: Path) -> dict[str, dict]:
    """
    Single-pass parse of an LS-DYNA keyword file.

    Returns
    -------
    name_to_props : {part_name -> {type_name, type_id, rho, E, nu, sigy}}
        Keyed by the part title string so it matches node_part_name from d3plot
        (the d3plot part_titles_ids are sequential 1-N, not the k-file PIDs).
    """
    # Collect parts: list of {name, mid}  and  mats: {mid -> props}
    parts: list[dict] = []
    mats:  dict[int, dict] = {}

    try:
        with open(path, encoding="ascii", errors="ignore") as fh:
            lines = fh.readlines()
    except OSError as exc:
        print(f"  WARNING: cannot read k-file {path}: {exc}")
        return {}

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        kw  = raw.strip().upper()

        # ── *PART ──────────────────────────────────────────────────────────
        if kw == "*PART":
            j = i + 1
            # skip comment lines, then read the title text line
            while j < len(lines) and lines[j].startswith("$"):
                j += 1
            part_title = ""
            if j < len(lines) and not lines[j].startswith("*"):
                part_title = lines[j].strip()
                j += 1
            # skip comments before the data line
            while j < len(lines) and lines[j].startswith("$"):
                j += 1
            if j < len(lines) and not lines[j].startswith("*"):
                tok = _kfields(lines[j])
                if len(tok) >= 3:
                    parts.append({"name": part_title, "mid": _ki(tok[2])})
                j += 1
            i = j
            continue

        # ── *MAT_xxx or *MAT_xxx_TITLE ─────────────────────────────────────
        if raw.startswith("*MAT_"):
            mat_kw    = raw.strip()
            has_title = mat_kw.upper().endswith("_TITLE")
            mat_type  = re.sub(r"_TITLE$", "", mat_kw, flags=re.IGNORECASE)
            mat_type  = mat_type.upper().replace("*MAT_", "")

            j = i + 1
            mat_label = ""
            if has_title:
                while j < len(lines) and lines[j].startswith("$"):
                    j += 1
                if j < len(lines) and not lines[j].startswith("*"):
                    mat_label = lines[j].strip()   # ← capture precise title
                    j += 1

            while j < len(lines) and lines[j].startswith("$"):
                j += 1

            if j < len(lines) and not lines[j].startswith("*"):
                tok = _kfields(lines[j])
                if len(tok) >= 2:
                    mid = int(_kf(tok[0]))
                    rho = _kf(tok[1])
                    if "CONCRETE" in mat_type:
                        E, nu, sigy = 0.0, _kf(tok[2]) if len(tok) > 2 else 0.0, 0.0
                    elif "BLATZ" in mat_type or "RUBBER" in mat_type:
                        E, nu, sigy = 0.0, 0.0, 0.0
                    elif "FOAM" in mat_type:
                        E, nu, sigy = _kf(tok[2]) if len(tok) > 2 else 0.0, 0.0, 0.0
                    else:
                        E    = _kf(tok[2]) if len(tok) > 2 else 0.0
                        nu   = _kf(tok[3]) if len(tok) > 3 else 0.0
                        sigy = _kf(tok[4]) if len(tok) > 4 else 0.0
                    mats[mid] = {
                        "type_name": mat_type,
                        "type_id":   _MAT_TYPE_ID_MAP.get(mat_type, _MAT_TYPE_UNKNOWN),
                        # precise label: title string if available, else keyword name
                        "label":     mat_label if mat_label else mat_type,
                        "rho": rho, "E": E, "nu": nu, "sigy": sigy,
                    }
                j += 1
            i = j
            continue

        i += 1

    # Join parts + mats on mid, keyed by part name
    name_to_props: dict[str, dict] = {}
    for p in parts:
        props = mats.get(p["mid"])
        if props is not None:
            name_to_props[p["name"]] = props

    return name_to_props


def _build_node_material_arrays(
    node_part_name: np.ndarray,         # (N,) str – part name per node
    name_to_props:  dict[str, dict],    # part_name -> material props
) -> dict[str, np.ndarray]:
    """
    Build per-node material property arrays by looking up part name.
    The d3plot part_titles_ids are sequential (1-N), not the k-file PIDs,
    so we look up by part name which matches between both sources.

    Output keys
    -----------
    type_id    (N,) int32    coarse type code (see _MAT_TYPE_ID_MAP)
    type_name  (N,) str      MAT keyword without *MAT_ prefix
    label      (N,) str      precise title from *MAT_xxx_TITLE; falls back to
                             type_name when _TITLE was absent in the k-file
    E          (N,) float32  Young's modulus [MPa]
    sigy       (N,) float32  yield stress [MPa]
    rho        (N,) float32  density [t/mm³]
    nu         (N,) float32  Poisson's ratio
    """
    N = len(node_part_name)
    type_ids   = np.full(N, _MAT_TYPE_UNKNOWN, dtype=np.int32)
    type_names = np.full(N, "UNKNOWN",         dtype=object)
    labels     = np.full(N, "UNKNOWN",         dtype=object)
    E_arr      = np.zeros(N, dtype=np.float32)
    sigy_arr   = np.zeros(N, dtype=np.float32)
    rho_arr    = np.zeros(N, dtype=np.float32)
    nu_arr     = np.zeros(N, dtype=np.float32)

    for i, raw_name in enumerate(node_part_name):
        name  = raw_name if isinstance(raw_name, str) else raw_name.decode("ascii", errors="ignore").strip()
        props = name_to_props.get(name)
        if props is not None:
            type_ids[i]   = props["type_id"]
            type_names[i] = props["type_name"]
            labels[i]     = props.get("label", props["type_name"])
            E_arr[i]      = props["E"]
            sigy_arr[i]   = props["sigy"]
            rho_arr[i]    = props["rho"]
            nu_arr[i]     = props["nu"]

    return {
        "type_id":   type_ids,
        "type_name": type_names,
        "label":     labels,
        "E":         E_arr,
        "sigy":      sigy_arr,
        "rho":       rho_arr,
        "nu":        nu_arr,
    }


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


# ── Scalar field (e.g. effective plastic strain) element → node ───────────────

def _average_scalar_to_elem(field_frame: np.ndarray) -> np.ndarray:
    """
    Collapse integration-point / layer dimensions → (n_elem,) scalar.
    Accepted shapes: (n_elem, n_ip), (n_elem, n_ip, 1), etc.
    """
    s = np.asarray(field_frame, dtype=np.float32)
    if s.ndim > 1:
        s = s.mean(axis=tuple(range(1, s.ndim)))
    return s  # (n_elem,)


def _compute_node_plastic_strain(
    solid_eps: np.ndarray | None,       # (n_solid_all, ...) for this state
    shell_eps: np.ndarray | None,       # (n_shell_all, ...) for this state
    sel_solid_elem_idx:   np.ndarray,
    sel_shell_elem_idx:   np.ndarray,
    sel_solid_elem_nodes: np.ndarray,   # (E_solid_sel, 8)
    sel_shell_elem_nodes: np.ndarray,   # (E_shell_sel, 4)
    sel_node_idx: np.ndarray,           # (N_sel,)
    n_total_nodes: int,
) -> np.ndarray:                        # (N_sel,) float32
    """
    Average element effective plastic strain to nodes.
    Mirrors _compute_node_stress but for a single scalar component.
    """
    node_sum   = np.zeros(n_total_nodes, dtype=np.float64)
    node_count = np.zeros(n_total_nodes, dtype=np.int64)

    for eps, elem_idx, elem_nodes in [
        (solid_eps, sel_solid_elem_idx, sel_solid_elem_nodes),
        (shell_eps, sel_shell_elem_idx, sel_shell_elem_nodes),
    ]:
        if eps is None or len(elem_idx) == 0:
            continue
        e1         = _average_scalar_to_elem(eps[elem_idx]).astype(np.float64)  # (E,)
        nodes_flat = elem_nodes.ravel()
        K          = elem_nodes.shape[1]
        e_rep      = np.repeat(e1, K)                                            # (E*K,)
        valid      = nodes_flat >= 0
        np.add.at(node_sum,   nodes_flat[valid], e_rep[valid])
        np.add.at(node_count, nodes_flat[valid], 1)

    out   = np.zeros(len(sel_node_idx), dtype=np.float32)
    denom = node_count[sel_node_idx]
    good  = denom > 0
    out[good] = (node_sum[sel_node_idx][good] / denom[good]).astype(np.float32)
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
        except RuntimeError as exc:
            if "endmark" in str(exc).lower():
                print(f"  WARNING: {src.name} skipped – missing endmark ({exc})")
            else:
                raise
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

    ds("times",          (n_frames,),            "float64")
    ds("positions",      (n_frames, n_nodes, 3), "float32")
    ds("velocity",       (n_frames, n_nodes, 3), "float32")
    ds("acceleration",   (n_frames, n_nodes, 3), "float32")
    ds("stress",         (n_frames, n_nodes, 6), "float32")
    ds("plastic_strain", (n_frames, n_nodes),    "float32")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LS-DYNA d3plot sequence → single HDF5 training file."
    )
    parser.add_argument("--src",              type=Path,  default=DEFAULT_SRC,
                        help="Directory containing d3plot, d3plot01, d3plot02 …")
    parser.add_argument("--tmp",              type=Path,  default=DEFAULT_TMP,
                        help="Scratch directory for single-file copies.")
    parser.add_argument("--out",              type=Path,  default=DEFAULT_OUT,
                        help="Output HDF5 file path  (e.g. output.h5).")
    parser.add_argument("--sampling-config",  type=Path,  default=DEFAULT_SAMPLING_CONFIG,
                        help="YAML node-selection config (collision_zone + required_parts).")
    parser.add_argument("--node-stride",      type=int,   default=DEFAULT_NODE_STRIDE,
                        help="Spatial decimation for Layer 2 (required_parts): keep every N-th node (1 = all).")
    parser.add_argument("--frame-stride",     type=int,   default=DEFAULT_FRAME_STRIDE,
                        help="Temporal decimation: keep every N-th frame (1 = all, 2 = default).")
    parser.add_argument("--kfile",           type=Path,  default=None,
                        help="LS-DYNA keyword file (.k) for material metadata. "
                             "Auto-detected from --src dir if not given.")
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
        (args.sampling_config, "sampling config"),
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
    # Step 1 – Two-layer part / node selection (from d3plot header + YAML config)
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

    collision_patterns, required_patterns, collision_zone_cfg = _load_sampling_config(args.sampling_config)

    # Layer 1: collision_zone — all nodes kept, no stride
    collision_mask = _build_selected_part_mask(part_names, collision_patterns)
    # Layer 2: required_parts — stride sampled
    required_mask  = _build_selected_part_mask(part_names, required_patterns)
    # Combined mask used for element selection (stress needs all elements from both layers)
    combined_mask  = collision_mask | required_mask
    sel_part_idx   = np.where(combined_mask)[0].astype(np.int64)

    all_patterns = list(dict.fromkeys(collision_patterns + required_patterns))
    print(f"Collision patterns : {len(collision_patterns)},  Required patterns : {len(required_patterns)}")
    print(f"Matched {len(sel_part_idx)}/{len(part_names)} parts")
    for i in sel_part_idx:
        layer = "L1+L2" if collision_mask[i] and required_mask[i] else ("L1" if collision_mask[i] else "L2")
        print(f"  [{layer}] part_idx={int(i):4d}  part_id={int(part_ids_full[i]):6d}"
              f"  name={part_names[i]}")
    if len(sel_part_idx) == 0:
        raise RuntimeError("No parts matched. Check sampling_config.yaml.")

    solid_part_idx  = d3hdr.arrays.get(ArrayType.element_solid_part_indexes)
    solid_node_idx  = d3hdr.arrays.get(ArrayType.element_solid_node_indexes)
    shell_part_idx  = d3hdr.arrays.get(ArrayType.element_shell_part_indexes)
    shell_node_idx  = d3hdr.arrays.get(ArrayType.element_shell_node_indexes)
    coords_ref      = d3hdr.arrays[ArrayType.node_coordinates]
    n_total_nodes   = len(coords_ref)

    sel_solid_ei, solid_node_sel, solid_node_part = _select_elements_and_nodes(
        solid_part_idx, solid_node_idx, combined_mask, n_total_nodes)
    sel_shell_ei, shell_node_sel, shell_node_part = _select_elements_and_nodes(
        shell_part_idx, shell_node_idx, combined_mask, n_total_nodes)

    # merge solid + shell node assignments (solid wins on shared nodes)
    node_part_full = np.full(n_total_nodes, -1, dtype=np.int64)
    for ni, pi in zip(solid_node_sel, solid_node_part):
        node_part_full[int(ni)] = int(pi)
    for ni, pi in zip(shell_node_sel, shell_node_part):
        if node_part_full[int(ni)] == -1:
            node_part_full[int(ni)] = int(pi)

    all_sel_nodes  = np.where(node_part_full >= 0)[0].astype(np.int64)
    all_node_parts = node_part_full[all_sel_nodes].astype(np.int64)

    # ── Two-layer node selection ───────────────────────────────────────────────
    # Layer 1: collision_zone — ALL nodes kept unconditionally
    col_local_mask     = collision_mask[all_node_parts]
    collision_node_idx = all_sel_nodes[col_local_mask]
    collision_node_set = set(collision_node_idx.tolist())

    # Layer 2: required_parts — strided, then deduplicate collision nodes
    req_local_mask    = required_mask[all_node_parts]
    required_node_idx = all_sel_nodes[req_local_mask]
    required_strided  = required_node_idx[::args.node_stride]
    required_new      = required_strided[~np.isin(required_strided, list(collision_node_set))]

    n_col_nodes   = len(collision_node_idx)
    n_req_strided = len(required_strided)
    n_req_new     = len(required_new)
    print(f"Selected  solid elems                      : {len(sel_solid_ei)}")
    print(f"          shell elems                      : {len(sel_shell_ei)}")
    print(f"Collision-zone nodes (all)                 : {n_col_nodes}")
    print(f"Required-parts nodes (strided, stride={args.node_stride})"
          f"  : {n_req_strided}")
    print(f"  of which new (not in collision zone)     : {n_req_new}")

    # Union: collision (all) + required (strided, deduplicated)
    sel_node_idx  = np.union1d(collision_node_idx, required_new)
    node_part_idx = node_part_full[sel_node_idx].astype(np.int64)
    n_nodes       = len(sel_node_idx)
    print(f"Total nodes                                : {n_nodes}")

    if n_nodes == 0:
        raise RuntimeError("No nodes selected. Check sampling_config.yaml.")

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

    # ── Material metadata from k-file ──────────────────────────────────────────
    kfile = args.kfile
    if kfile is None:
        k_candidates = sorted(args.src.glob("*.k")) + sorted(args.src.glob("*.key"))
        kfile = k_candidates[0] if k_candidates else None
    if kfile is not None:
        print(f"Parsing k-file for material metadata: {kfile.name} …")
        name_to_props = _parse_kfile_materials(kfile)
        print(f"  Resolved material props for {len(name_to_props)} unique part names")
    else:
        print("WARNING: no .k file found – material metadata will be zero-filled")
        name_to_props = {}
    node_mat = _build_node_material_arrays(node_part_name, name_to_props)

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
    mg.create_dataset("part_patterns",   data=np.asarray(all_patterns, dtype="S"))
    mg.attrs["node_stride"]         = args.node_stride
    mg.attrs["node_selection"]      = "collision_zone:all + required_parts:strided"
    mg.attrs["frame_stride"]        = args.frame_stride
    mg.attrs["n_frames"]            = n_frames
    mg.attrs["n_nodes"]             = n_nodes
    mg.attrs["n_parts"]             = len(sel_part_idx)
    mg.attrs["stress_components"]   = "sxx,syy,szz,sxy,syz,sxz"
    mg.attrs["velocity_source"]     = "pending"       # updated below
    mg.attrs["acceleration_source"] = "pending"

    # material properties per node (static, from k-file)
    mg.create_dataset("node_mat_type_id",   data=node_mat["type_id"])
    mg.create_dataset("node_mat_type_name", data=node_mat["type_name"].astype("S"))
    mg.create_dataset("node_mat_label",     data=node_mat["label"].astype("S"))
    mg.create_dataset("node_mat_E",         data=node_mat["E"])
    mg.create_dataset("node_mat_sigy",      data=node_mat["sigy"])
    mg.create_dataset("node_mat_rho",       data=node_mat["rho"])
    mg.create_dataset("node_mat_nu",        data=node_mat["nu"])
    mg.attrs["mat_type_id_legend"] = (
        "0=plasticity 1=rigid 2=elastic 3=rubber "
        "4=concrete 5=foam 6=spotweld 7=spring/damper 8=unknown"
    )
    mg.attrs["kfile"] = str(kfile.resolve()) if kfile else "not_provided"

    # add front face / barrier part mask for distance calc
    part_names_str = [str(n).strip() for n in node_part_name]

    barrier_patterns       = [p.lower() for p in collision_zone_cfg.get("barrier_parts", ["concrete_fine_mesh"])]
    car_collision_patterns = [p.lower() for p in collision_zone_cfg.get("car_contact_parts", ["frontface"])]

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
            "sampling_config": str(args.sampling_config.resolve()),
            "node_selection": "collision_zone:all + required_parts:strided",
            "node_stride": args.node_stride,
            "collision_zone_n_nodes": n_col_nodes,
            "required_parts_n_nodes": n_req_new,
            "frame_stride": args.frame_stride,
            "n_frames": n_frames,
            "n_nodes": n_nodes,
            "n_parts": len(sel_part_idx),
            "normalised": False,
            "value_mode": "raw_physical_units",
            "stress_components": "sxx,syy,szz,sxy,syz,sxz",
            "velocity_source": None,
            "acceleration_source": None,
            "collision_patterns": collision_patterns,
            "required_patterns": required_patterns,
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
            m = len(arr)
            if m == 0:
                return
            self.min_vals = np.minimum(self.min_vals, arr.min(axis=0))
            self.max_vals = np.maximum(self.max_vals, arr.max(axis=0))
            # parallel Welford merge: combine existing (count, mean, M2) with new batch
            batch_mean = arr.mean(axis=0)
            batch_M2   = ((arr - batch_mean) ** 2).sum(axis=0)
            new_count  = self.count + m
            delta      = batch_mean - self.mean
            self.mean  = (self.count * self.mean + m * batch_mean) / new_count
            self.M2   += batch_M2 + delta ** 2 * (self.count * m / new_count)
            self.count = new_count

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

    pos_stats          = FieldStats("positions", 3)
    vel_stats          = FieldStats("velocity", 3)
    acc_stats          = FieldStats("acceleration", 3)
    stress_stats       = FieldStats("stress", 6)
    plastic_strain_stats = FieldStats("plastic_strain", 1)

    # accumulators for median calculation (store per-frame flattened arrays)
    vel_accum:            list[np.ndarray] = []
    acc_accum:            list[np.ndarray] = []
    stress_accum:         list[np.ndarray] = []
    plastic_strain_accum: list[np.ndarray] = []

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
                        ArrayType.element_solid_effective_plastic_strain,
                        ArrayType.element_shell_effective_plastic_strain,
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

            # ── effective plastic strain (scalar, element-avg → nodes) ─────
            solid_eps = d3.arrays.get(ArrayType.element_solid_effective_plastic_strain)
            shell_eps = d3.arrays.get(ArrayType.element_shell_effective_plastic_strain)
            eps_node = _compute_node_plastic_strain(
                solid_eps[sidx] if solid_eps is not None else None,
                shell_eps[sidx] if shell_eps is not None else None,
                sel_solid_ei,
                sel_shell_ei,
                sel_solid_elem_nodes,
                sel_shell_elem_nodes,
                sel_node_idx,
                n_total_nodes,
            )
            h5f["states/plastic_strain"][fi] = eps_node
            plastic_strain_stats.update(eps_node.reshape(-1, 1))
            plastic_strain_accum.append(eps_node)

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
    fs_pos           = pos_stats.finalize()
    fs_vel           = vel_stats.finalize()
    fs_acc           = acc_stats.finalize()
    fs_stress        = stress_stats.finalize()
    fs_plastic_strain = plastic_strain_stats.finalize()

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

    vel_median           = _safe_median(vel_accum, 3)
    acc_median           = _safe_median(acc_accum, 3)
    stress_median        = _safe_median(stress_accum, 6)
    plastic_strain_median = _safe_median(
        [a.reshape(-1, 1) for a in plastic_strain_accum], 1
    )

    fs_pos["median"]            = pos_median
    fs_vel["median"]            = vel_median
    fs_acc["median"]            = acc_median
    fs_stress["median"]         = stress_median
    fs_plastic_strain["median"] = plastic_strain_median

    metadata["field_stats"] = {
        "positions":      fs_pos,
        "velocity":       fs_vel,
        "acceleration":   fs_acc,
        "stress":         fs_stress,
        "plastic_strain": fs_plastic_strain,
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
    print(f"Nodes  (N)     : {n_nodes}  "
          f"(collision={n_col_nodes}, required_new={n_req_new}, stride={args.node_stride})")
    print(f"Parts  (P)     : {len(sel_part_idx)}")
    print(f"Velocity src   : finite difference (dt = 1 frame)")
    print(f"Accel src      : finite difference (dt = 1 frame)")
    print(f"Physical Δt    : {dt_mean*1e3:.4f} ms  (stored as dt_seconds for unit conversion)")
    print(f"HDF5 layout    : /metadata  +  /states/{{times,positions,velocity,acceleration,stress,plastic_strain}}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()