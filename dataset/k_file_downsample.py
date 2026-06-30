"""
Mesh downsampling and field reconstruction pipeline for a Transolver crash surrogate.
Implements §2-9 of SPEC_sampling_reconstruction.md.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Appendix A constants
# ---------------------------------------------------------------------------
TOTAL_NODES = 1_776_987
CENTERLINE = (0.474835, 1.0, -976.536)   # a, b, c
CENTERLINE_DENOM = 1.1069                # hypot(a, b)
G_IN_MM_S2 = 9806.65
FORCE_KEEP_PIDS = [9000002, 9000003, 9000004, 9000100, 9000101, 9000102, 9000103]
BARRIER_PID_MIN = 10_000_000
VEHICLE_PID_RANGE = (2_000_000, 8_999_999)

FINE_CONCRETE_PIDS = [10000001, 10000007]
COARSE_CONCRETE_PIDS = [10000013, 10000019]
FINE_REBAR_PIDS = [10000002, 10000008, 10000003, 10000009]
COARSE_REBAR_PIDS = [10000014, 10000020, 10000015, 10000021]
FINE_REINF_PIDS = [10000004, 10000010]
COARSE_REINF_PIDS = [10000016, 10000022]
FINE_STEEL_TUBE_PIDS = [10000005, 10000011]
COARSE_STEEL_TUBE_PIDS = [10000017, 10000023]
FINE_TLOK_PIDS = [10000006, 10000012]
COARSE_TLOK_PIDS = [10000018, 10000024]

FINE_PIDS = set(
    FINE_CONCRETE_PIDS + FINE_REBAR_PIDS + FINE_REINF_PIDS +
    FINE_STEEL_TUBE_PIDS + FINE_TLOK_PIDS
)
COARSE_PIDS = set(
    COARSE_CONCRETE_PIDS + COARSE_REBAR_PIDS + COARSE_REINF_PIDS +
    COARSE_STEEL_TUBE_PIDS + COARSE_TLOK_PIDS
)

# Part-family labels for segment_id assignment
PID_TO_PART_FAMILY = {}
for pid in FINE_CONCRETE_PIDS:   PID_TO_PART_FAMILY[pid] = "fine_concrete"
for pid in FINE_REBAR_PIDS:      PID_TO_PART_FAMILY[pid] = "fine_rebar"
for pid in FINE_REINF_PIDS:      PID_TO_PART_FAMILY[pid] = "fine_reinf"
for pid in FINE_STEEL_TUBE_PIDS: PID_TO_PART_FAMILY[pid] = "fine_steel_tube"
for pid in FINE_TLOK_PIDS:       PID_TO_PART_FAMILY[pid] = "fine_tlok"
for pid in COARSE_CONCRETE_PIDS:   PID_TO_PART_FAMILY[pid] = "coarse_concrete"
for pid in COARSE_REBAR_PIDS:      PID_TO_PART_FAMILY[pid] = "coarse_rebar"
for pid in COARSE_REINF_PIDS:      PID_TO_PART_FAMILY[pid] = "coarse_reinf"
for pid in COARSE_STEEL_TUBE_PIDS: PID_TO_PART_FAMILY[pid] = "coarse_steel_tube"
for pid in COARSE_TLOK_PIDS:       PID_TO_PART_FAMILY[pid] = "coarse_tlok"


# ---------------------------------------------------------------------------
# §2 — Robust field parser
# ---------------------------------------------------------------------------

def parse_fields(line: str, n_expected: int) -> list[str]:
    """Return n_expected fields from a k-file data line.
    Falls back to fixed-width 8-char slicing when space-delimited yields too few fields.
    """
    parts = line.split()
    if len(parts) >= n_expected:
        return parts
    return [line[i:i+8].strip() for i in range(0, len(line.rstrip()), 8)]


def _validate_pid(pid: int, context: str = "") -> None:
    """Fail loudly if a parsed PID looks like a corrupted fixed-width merge."""
    if pid > 100_000_000:
        raise ValueError(
            f"Parsed PID {pid} is absurdly large — fixed-width fallback failed. "
            f"Context: {context}"
        )


# ---------------------------------------------------------------------------
# §1.1 / §2 — k-file parser
# ---------------------------------------------------------------------------

@dataclass
class MeshData:
    """All geometry extracted from the k-file."""
    node_ids: np.ndarray          # (N,)  int64
    coords: np.ndarray            # (N,3) float64  — t=0 reference
    node_pid: np.ndarray          # (N,)  int32  — PID owning each node (first occurrence)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def node_index(self) -> dict[int, int]:
        """nid → row index into coords / node_ids."""
        return {int(nid): i for i, nid in enumerate(self.node_ids)}


def parse_kfile(kfile_path: str | Path) -> MeshData:
    """Parse *NODE and *ELEMENT_* sections from a k-file.

    Returns MeshData with reference coordinates and per-node PID assignment.
    """
    kfile_path = Path(kfile_path)
    logger.info("Parsing k-file: %s", kfile_path)

    node_id_list: list[int] = []
    coord_list: list[tuple[float, float, float]] = []
    nid_to_idx: dict[int, int] = {}

    # PID assignment: first element that references a node wins
    node_pid: dict[int, int] = {}

    current_keyword = None

    with open(kfile_path, "r") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith("$"):
                continue
            if line.startswith("*"):
                current_keyword = line.split()[0].upper()
                continue

            if current_keyword == "*NODE":
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                if nid in nid_to_idx:
                    continue
                idx = len(node_id_list)
                node_id_list.append(nid)
                coord_list.append((x, y, z))
                nid_to_idx[nid] = idx

            elif current_keyword is not None and current_keyword.startswith("*ELEMENT_"):
                # Generic handler covering SOLID, SHELL, BEAM, MASS, DISCRETE,
                # TSHELL, SEATBELT, NODAL_RIGID_BODY, etc.
                # Format for all: eid pid n1 [n2 ...] on the first data line.
                # Orientation / secondary lines have field[1] that is either not
                # an int or falls outside any valid PID range — silently skipped.
                fields = parse_fields(line, 3)
                if len(fields) < 2:
                    continue
                try:
                    pid = int(fields[1])
                except ValueError:
                    continue
                # Skip secondary/orientation lines (PID would be nonsensical)
                if pid < 1 or pid > 100_000_000:
                    continue
                _validate_pid(pid, f"line {lineno}: {line!r}")
                for nf in fields[2:]:
                    if not nf:
                        continue
                    try:
                        nid = int(nf)
                    except ValueError:
                        continue
                    if nid == 0:  # trailing padding
                        continue
                    if nid in nid_to_idx and nid not in node_pid:
                        node_pid[nid] = pid

    node_ids = np.array(node_id_list, dtype=np.int64)
    coords = np.array(coord_list, dtype=np.float64)
    # nodes with no element reference get PID=0
    pid_arr = np.array(
        [node_pid.get(int(nid), 0) for nid in node_ids], dtype=np.int32
    )

    mesh = MeshData(node_ids=node_ids, coords=coords, node_pid=pid_arr)
    logger.info("Parsed %d nodes", mesh.n_nodes)
    return mesh


# ---------------------------------------------------------------------------
# §4 — Region partitioning
# ---------------------------------------------------------------------------

def centerline_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Perpendicular distance from barrier centerline (mm)."""
    a, b, c = CENTERLINE
    return np.abs(a * x + b * y + c) / CENTERLINE_DENOM


def build_region_masks(mesh: MeshData) -> dict[str, np.ndarray]:
    """Return boolean masks (len = mesh.n_nodes) for each sampling region."""
    pid = mesh.node_pid
    coords = mesh.coords

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
        "barrier_fine":  mask_fine,
        "barrier_coarse": mask_coarse,
        "force_keep":    mask_force_keep,
        "veh_contact":   mask_veh_contact,
        "veh_near":      mask_veh_near,
        "veh_far":       mask_veh_far,
    }


# ---------------------------------------------------------------------------
# §5 — Samplers
# ---------------------------------------------------------------------------

@dataclass
class SamplerConfig:
    method: str = "fps"           # stride | random | poisson_disk | fps
    n_points: int = 10000
    seed: int = 42
    stride_order: str = "morton"  # file | morton | x
    poisson_radius: Optional[float] = None
    enforce_exact_n: bool = True
    density_weighted: bool = False
    density_d0: float = 600.0     # mm, decay length for density-weighted fps


def _morton_encode(coords: np.ndarray) -> np.ndarray:
    """Approximate Morton (Z-order) encoding via bit-interleaving of 21-bit coords."""
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    rng = np.where(mx - mn > 0, mx - mn, 1.0)
    norm = ((coords - mn) / rng * (2**21 - 1)).astype(np.uint64)
    x, y, z = norm[:, 0], norm[:, 1], norm[:, 2]

    def spread(v: np.ndarray) -> np.ndarray:
        v = v & np.uint64(0x1FFFFF)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
        return v

    return spread(x) | (spread(y) << np.uint64(1)) | (spread(z) << np.uint64(2))


def sample_stride(points: np.ndarray, n: int, cfg: SamplerConfig) -> np.ndarray:
    N = len(points)
    if cfg.stride_order == "file":
        order = np.arange(N)
    elif cfg.stride_order == "x":
        order = np.argsort(points[:, 0])
    else:  # morton
        order = np.argsort(_morton_encode(points))
    step = max(1, N // n)
    idx = order[::step]
    if len(idx) > n:
        idx = idx[:n]
    elif len(idx) < n and cfg.enforce_exact_n:
        remaining = np.setdiff1d(order, idx)
        idx = np.concatenate([idx, remaining[: n - len(idx)]])
    return idx


def sample_random(points: np.ndarray, n: int, cfg: SamplerConfig) -> np.ndarray:
    N = len(points)
    rng = np.random.default_rng(cfg.seed)
    return rng.choice(N, min(n, N), replace=False)


def sample_poisson_disk(points: np.ndarray, n: int, cfg: SamplerConfig) -> np.ndarray:
    N = len(points)
    rng = np.random.default_rng(cfg.seed)

    if cfg.poisson_radius is None:
        # Estimate r from target n and bounding box volume
        mn, mx = points.min(axis=0), points.max(axis=0)
        vol = np.prod(np.maximum(mx - mn, 1.0))
        r = (vol / n) ** (1.0 / 3.0) * 0.9
    else:
        r = cfg.poisson_radius

    order = rng.permutation(N)
    chosen: list[int] = []
    chosen_coords: list[np.ndarray] = []
    tree_pts: list[np.ndarray] = []

    tree = None
    for idx in order:
        pt = points[idx]
        if tree is None or len(chosen) == 0:
            chosen.append(idx)
            chosen_coords.append(pt)
            tree_pts.append(pt)
            if len(chosen) % 500 == 0:
                tree = cKDTree(np.array(tree_pts))
        else:
            if tree is not None:
                nn_dist, _ = tree.query(pt, k=1)
                if nn_dist >= r:
                    chosen.append(idx)
                    tree_pts.append(pt)
                    if len(chosen) % 500 == 0:
                        tree = cKDTree(np.array(tree_pts))

    chosen_arr = np.array(chosen)

    if cfg.enforce_exact_n:
        if len(chosen_arr) > n:
            chosen_arr = chosen_arr[:n]
        elif len(chosen_arr) < n:
            remaining = np.setdiff1d(np.arange(N), chosen_arr)
            chosen_arr = np.concatenate([chosen_arr, remaining[: n - len(chosen_arr)]])

    return chosen_arr


def sample_fps(
    points: np.ndarray,
    n: int,
    cfg: SamplerConfig,
    density: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Farthest-point sampling. Optionally density-weighted."""
    N = len(points)
    n = min(n, N)
    rng = np.random.default_rng(cfg.seed)

    chosen = np.zeros(n, dtype=np.int64)
    chosen[0] = rng.integers(0, N)
    min_dists = np.full(N, np.inf)

    for i in range(1, n):
        last = points[chosen[i - 1]]
        dists = np.sum((points - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)

        if cfg.density_weighted and density is not None:
            scores = min_dists * density
        else:
            scores = min_dists

        chosen[i] = int(np.argmax(scores))

    return chosen


def run_sampler(
    points: np.ndarray,
    n: int,
    cfg: SamplerConfig,
    density: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dispatch to the configured sampler."""
    if cfg.method == "stride":
        return sample_stride(points, n, cfg)
    elif cfg.method == "random":
        return sample_random(points, n, cfg)
    elif cfg.method == "poisson_disk":
        return sample_poisson_disk(points, n, cfg)
    elif cfg.method == "fps":
        return sample_fps(points, n, cfg, density)
    else:
        raise ValueError(f"Unknown sampler method: {cfg.method!r}")


# ---------------------------------------------------------------------------
# §5 — Per-part allocation within a region
# ---------------------------------------------------------------------------

PART_FAMILY_FULL_RETAIN = {"fine_rebar", "fine_reinf", "coarse_rebar", "coarse_reinf"}

def allocate_per_part(
    mesh: MeshData,
    region_mask: np.ndarray,
    n_total: int,
    cfg: SamplerConfig,
    split_by_part: bool = True,
    min_per_part: int = 50,
) -> np.ndarray:
    """Return local indices (into mesh arrays) of sampled nodes for a region.

    If split_by_part=True, distributes n_total across distinct PIDs by node-count
    proportion, with full-retain for small parts and a min_per_part floor.
    """
    region_idx = np.where(region_mask)[0]
    if len(region_idx) == 0:
        return region_idx

    if not split_by_part:
        pts = mesh.coords[region_idx]
        local = run_sampler(pts, n_total, cfg)
        return region_idx[local]

    # Group by PID
    pids_in_region = mesh.node_pid[region_idx]
    unique_pids = np.unique(pids_in_region)

    # Separate full-retain vs sampled parts
    retain_idx = []
    sample_parts: list[tuple[int, np.ndarray]] = []

    for pid in unique_pids:
        part_family = PID_TO_PART_FAMILY.get(int(pid), "")
        mask_part = pids_in_region == pid
        part_global_idx = region_idx[mask_part]
        if part_family in PART_FAMILY_FULL_RETAIN:
            retain_idx.append(part_global_idx)
        else:
            sample_parts.append((pid, part_global_idx))

    n_retained = sum(len(r) for r in retain_idx)
    budget_remaining = max(0, n_total - n_retained)

    # Proportional allocation
    total_sample_nodes = sum(len(p[1]) for p in sample_parts)
    sampled_idx = []
    used = 0

    for i, (pid, part_global_idx) in enumerate(sample_parts):
        if i == len(sample_parts) - 1:
            alloc = budget_remaining - used
        else:
            alloc = max(min_per_part, int(budget_remaining * len(part_global_idx) / max(total_sample_nodes, 1)))
        alloc = min(alloc, len(part_global_idx))
        alloc = max(alloc, 0)

        pts = mesh.coords[part_global_idx]
        local = run_sampler(pts, alloc, cfg)
        sampled_idx.append(part_global_idx[local])
        used += len(local)

    all_idx = (
        [r for r in retain_idx] + sampled_idx
    )
    if all_idx:
        return np.concatenate(all_idx)
    return np.array([], dtype=np.int64)


# ---------------------------------------------------------------------------
# §4 + §5 — Main sampling orchestration
# ---------------------------------------------------------------------------

@dataclass
class RegionConfig:
    name: str
    sampler: SamplerConfig
    split_by_part: bool = False
    min_per_part: int = 50
    reconstruction_mode: str = "interp"  # interp | rigid


DEFAULT_REGION_CONFIGS = [
    RegionConfig("barrier_fine",  SamplerConfig(n_points=40000, seed=42), split_by_part=True),
    RegionConfig("barrier_coarse", SamplerConfig(n_points=20000, seed=42), split_by_part=True),
    RegionConfig("veh_contact",   SamplerConfig(n_points=10000, seed=42)),
    RegionConfig("veh_near",      SamplerConfig(n_points=18000, seed=42)),
    RegionConfig("veh_far",       SamplerConfig(n_points=12000, seed=42)),
]


def sample_mesh(
    mesh: MeshData,
    region_configs: list[RegionConfig] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full sampling pipeline.

    Returns:
        sampled_global_idx  — row indices into mesh arrays  (≈100k)
        region_labels       — string label per sampled node
        segment_ids         — string segment id per sampled node
    """
    if region_configs is None:
        region_configs = DEFAULT_REGION_CONFIGS

    masks = build_region_masks(mesh)
    nid_to_idx = mesh.node_index()

    # Force-keep nodes first
    fk_mask = masks["force_keep"]
    fk_idx = np.where(fk_mask)[0]

    all_sampled: list[np.ndarray] = [fk_idx]
    all_labels: list[np.ndarray] = [np.full(len(fk_idx), "force_keep")]
    all_segments: list[np.ndarray] = []
    for idx in fk_idx:
        pid = int(mesh.node_pid[idx])
        all_segments.append(PID_TO_PART_FAMILY.get(pid, f"pid_{pid}"))

    already_chosen = set(fk_idx.tolist())

    for rcfg in region_configs:
        region_mask = masks.get(rcfg.name)
        if region_mask is None:
            logger.warning("Region %r not found in masks — skipping", rcfg.name)
            continue

        # Remove already-sampled nodes from this region
        exclude = np.array(list(already_chosen), dtype=np.int64)
        combined_mask = region_mask.copy()
        combined_mask[exclude] = False

        sampled = allocate_per_part(
            mesh, combined_mask, rcfg.sampler.n_points, rcfg.sampler,
            split_by_part=rcfg.split_by_part, min_per_part=rcfg.min_per_part,
        )

        if len(sampled) == 0:
            logger.warning("Region %r yielded 0 nodes", rcfg.name)
            continue

        all_sampled.append(sampled)
        all_labels.append(np.full(len(sampled), rcfg.name))

        seg_list = []
        for idx in sampled:
            pid = int(mesh.node_pid[idx])
            seg_list.append(PID_TO_PART_FAMILY.get(pid, f"pid_{pid}"))
        all_segments.extend(seg_list)

        already_chosen.update(sampled.tolist())
        logger.info("Region %r: %d nodes sampled", rcfg.name, len(sampled))

    sampled_idx = np.concatenate(all_sampled)
    labels = np.concatenate(all_labels)
    segments = np.array(all_segments + [""] * (len(sampled_idx) - len(all_segments)))

    # Deduplicate (should not be needed, but safety)
    _, unique_pos = np.unique(sampled_idx, return_index=True)
    sampled_idx = sampled_idx[unique_pos]
    labels = labels[unique_pos]
    segments = segments[unique_pos]

    logger.info("Total sampled nodes: %d", len(sampled_idx))
    return sampled_idx, labels, segments


# ---------------------------------------------------------------------------
# §6 — Reconstruction matrix W
# ---------------------------------------------------------------------------

@dataclass
class WConfig:
    knn: int = 6
    weight_mode: str = "inverse_distance"   # inverse_distance | rbf | barycentric
    tlok_seam_protect: bool = True


def _inverse_distance_weights(dists: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dists = np.atleast_1d(np.asarray(dists, dtype=np.float64))
    w = 1.0 / (dists + eps)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def _rbf_weights(dists: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dists = np.atleast_1d(np.asarray(dists, dtype=np.float64))
    sigma = dists.mean() + eps
    w = np.exp(-(dists / sigma) ** 2)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def _node_block(pid: int) -> str:
    """Map a PID to one of three reconstruction blocks: fine | coarse | vehicle."""
    if pid in FINE_PIDS:
        return "fine"
    if pid in COARSE_PIDS:
        return "coarse"
    return "vehicle"


def build_reconstruction_matrix(
    mesh: MeshData,
    sampled_idx: np.ndarray,
    cfg: WConfig | None = None,
) -> sp.csr_matrix:
    """Build sparse W matrix (N_full × N_sampled) on reference geometry.

    Three block-diagonal groups enforce the T_lok seam and barrier/vehicle
    separation required by §6.3:
      - fine   barrier block  (fine_* PIDs)
      - coarse barrier block  (coarse_* PIDs)
      - vehicle block         (everything else)

    Within each block, seam-protected kNN interpolation is used.
    Sampled-node rows are identity rows.
    """
    if cfg is None:
        cfg = WConfig()

    N_full = mesh.n_nodes
    N_samp = len(sampled_idx)

    sampled_set = set(sampled_idx.tolist())
    global_to_col = {int(s): c for c, s in enumerate(sampled_idx)}

    # Build one kd-tree per block from sampled nodes
    block_cols: dict[str, list[int]] = {"fine": [], "coarse": [], "vehicle": []}
    for col, global_row in enumerate(sampled_idx):
        pid = int(mesh.node_pid[global_row])
        block_cols[_node_block(pid)].append(col)

    block_trees: dict[str, tuple[cKDTree, np.ndarray]] = {}
    for block, cols in block_cols.items():
        if cols:
            cols_arr = np.array(cols, dtype=np.int64)
            pts = mesh.coords[sampled_idx[cols_arr]]
            block_trees[block] = (cKDTree(pts), cols_arr)
            logger.info("Block %r: %d sampled nodes → kd-tree built", block, len(cols_arr))

    rows: list[int] = []
    cols_list: list[int] = []
    data: list[float] = []

    X_ref = mesh.coords

    for i in range(N_full):
        if i in sampled_set:
            rows.append(i)
            cols_list.append(global_to_col[i])
            data.append(1.0)
            continue

        pid = int(mesh.node_pid[i])
        block = _node_block(pid)

        if block not in block_trees:
            # No sampled nodes in this block — fall back to vehicle block
            block = "vehicle"
        if block not in block_trees:
            continue  # mesh has no sampled nodes at all — skip

        tree, cols_arr = block_trees[block]
        k = max(1, min(cfg.knn, len(cols_arr)))
        dists, local_nn = tree.query(X_ref[i], k=k)
        dists = np.atleast_1d(np.asarray(dists, dtype=np.float64))
        local_nn = np.atleast_1d(np.asarray(local_nn, dtype=np.int64))

        if cfg.weight_mode == "rbf":
            w = _rbf_weights(dists)
        else:
            w = _inverse_distance_weights(dists)

        for weight, local_col in zip(w, local_nn):
            rows.append(i)
            cols_list.append(int(cols_arr[int(local_col)]))
            data.append(float(weight))

    W = sp.csr_matrix(
        (data, (rows, cols_list)),
        shape=(N_full, N_samp),
        dtype=np.float64,
    )
    logger.info("Built W: shape=%s, nnz=%d", W.shape, W.nnz)
    return W


# ---------------------------------------------------------------------------
# §1.2 — DisplacementSource interface
# ---------------------------------------------------------------------------

class DisplacementSource(ABC):
    """Abstract source of per-time-step nodal displacement (1.77M × 3)."""

    @abstractmethod
    def n_states(self) -> int: ...

    @abstractmethod
    def get_displacement(self, state_idx: int) -> np.ndarray:
        """Return displacement field (N_full, 3) at the given state index."""
        ...


class D3plotDisplacementSource(DisplacementSource):
    """Reads displacement from a d3plot file via lasso-python."""

    def __init__(self, d3plot_path: str | Path, x_ref: np.ndarray):
        try:
            from lasso.dyna import D3plot, ArrayType  # type: ignore
        except ImportError as e:
            raise ImportError("lasso-python is required for d3plot reading: pip install lasso-python") from e

        self._d3 = D3plot(str(d3plot_path))
        self._x_ref = x_ref
        # Shape: (n_states, n_nodes, 3)
        self._coords = self._d3.arrays[ArrayType.node_coordinates]

    def n_states(self) -> int:
        return self._coords.shape[0]

    def get_displacement(self, state_idx: int) -> np.ndarray:
        return self._coords[state_idx] - self._x_ref


# ---------------------------------------------------------------------------
# §7 — Closed-loop validation harness
# ---------------------------------------------------------------------------

def compute_physical_metrics(
    disp_recon: np.ndarray,
    disp_true: np.ndarray,
    x_ref: np.ndarray,
    impact_mask: np.ndarray,
    rigid_mask: np.ndarray,
) -> dict:
    """Compute per-state physical reconstruction metrics.

    Args:
        disp_recon: (N, 3) reconstructed displacement
        disp_true:  (N, 3) ground-truth displacement
        x_ref:      (N, 3) reference coordinates
        impact_mask: boolean mask for impact zone nodes
        rigid_mask:  boolean mask for rigid/far zone nodes
    """
    error = disp_recon - disp_true
    error_norm = np.linalg.norm(error, axis=1)

    # Dynamic deflection: max displacement magnitude in impact zone
    Dm_recon = float(np.linalg.norm(disp_recon[impact_mask], axis=1).max()) if impact_mask.any() else 0.0
    Dm_true  = float(np.linalg.norm(disp_true[impact_mask], axis=1).max()) if impact_mask.any() else 0.0

    # Working width: max lateral (y) displacement of barrier impact face
    Wm_recon = float(np.abs(disp_recon[impact_mask, 1]).max()) if impact_mask.any() else 0.0
    Wm_true  = float(np.abs(disp_true[impact_mask, 1]).max()) if impact_mask.any() else 0.0

    # Region-weighted L2
    l2_impact = float(np.sqrt((error_norm[impact_mask] ** 2).mean())) if impact_mask.any() else 0.0
    l2_rigid  = float(np.sqrt((error_norm[rigid_mask] ** 2).mean())) if rigid_mask.any() else 0.0

    return {
        "Dm_recon_mm": Dm_recon,
        "Dm_true_mm": Dm_true,
        "Dm_error_pct": abs(Dm_recon - Dm_true) / (Dm_true + 1e-12) * 100,
        "Wm_recon_mm": Wm_recon,
        "Wm_true_mm": Wm_true,
        "Wm_error_pct": abs(Wm_recon - Wm_true) / (Wm_true + 1e-12) * 100,
        "rmse_impact_mm": l2_impact,
        "rmse_rigid_mm": l2_rigid,
    }


def run_validation(
    W: sp.csr_matrix,
    sampled_idx: np.ndarray,
    mesh: MeshData,
    source: DisplacementSource,
    impact_mask: np.ndarray | None = None,
    rigid_mask: np.ndarray | None = None,
    max_states: int | None = None,
) -> dict:
    """Closed-loop validation pipeline (§7). Returns aggregated metrics."""
    N = mesh.n_nodes
    x_ref = mesh.coords

    if impact_mask is None:
        impact_mask = np.isin(mesh.node_pid, list(FINE_PIDS))
    if rigid_mask is None:
        vlo, vhi = VEHICLE_PID_RANGE
        rigid_mask = (mesh.node_pid >= vlo) & (mesh.node_pid <= vhi)
        d = centerline_distance(x_ref[:, 0], x_ref[:, 1])
        rigid_mask &= d >= 1000.0

    n_states = source.n_states()
    if max_states is not None:
        n_states = min(n_states, max_states)

    per_state: list[dict] = []
    for t in range(n_states):
        disp_true = source.get_displacement(t)          # (N, 3)
        disp_sampled = disp_true[sampled_idx]           # (100k, 3)
        disp_recon = W @ disp_sampled                   # (N, 3)

        metrics = compute_physical_metrics(
            disp_recon, disp_true, x_ref, impact_mask, rigid_mask
        )
        metrics["state"] = t
        per_state.append(metrics)

    # Aggregate
    keys = [k for k in per_state[0] if k != "state"]
    agg: dict = {k: float(np.mean([s[k] for s in per_state])) for k in keys}
    agg["max_Dm_error_pct"] = float(max(s["Dm_error_pct"] for s in per_state))
    agg["max_Wm_error_pct"] = float(max(s["Wm_error_pct"] for s in per_state))
    agg["per_state"] = per_state
    return agg


# ---------------------------------------------------------------------------
# §9 — Save / load outputs
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: str | Path,
    mesh: MeshData,
    sampled_idx: np.ndarray,
    labels: np.ndarray,
    segments: np.ndarray,
    W: sp.csr_matrix,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sampled_node_ids = mesh.node_ids[sampled_idx]
    np.save(out / "sampled_node_ids.npy", sampled_node_ids)
    np.save(out / "node_ref_coords.npy", mesh.coords)
    np.save(out / "node_ids.npy", mesh.node_ids)          # nid at each row — needed for nid→row lookup
    np.save(out / "sampled_node_part_ids.npy", mesh.node_pid[sampled_idx].astype(np.int32))
    sp.save_npz(str(out / "W.npz"), W)

    region_data = {
        "sampled_node_ids": sampled_node_ids.tolist(),
        "region_labels": labels.tolist(),
        "segment_ids": segments.tolist(),
    }
    with open(out / "region_assignment.json", "w") as f:
        json.dump(region_data, f)

    logger.info("Outputs saved to %s", out)


def load_outputs(output_dir: str | Path) -> dict:
    out = Path(output_dir)
    return {
        "sampled_node_ids": np.load(out / "sampled_node_ids.npy"),
        "node_ref_coords":  np.load(out / "node_ref_coords.npy"),
        "W": sp.load_npz(str(out / "W.npz")),
    }


# ---------------------------------------------------------------------------
# §10 — Acceptance checks
# ---------------------------------------------------------------------------

def acceptance_checks(mesh: MeshData, sampled_idx: np.ndarray, W: sp.csr_matrix) -> None:
    """Assert acceptance criteria from §10. Raises AssertionError on failure."""
    assert mesh.n_nodes == TOTAL_NODES, (
        f"Expected {TOTAL_NODES} nodes, got {mesh.n_nodes}"
    )

    # W row sums ≈ 1
    row_sums = np.array(W.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"W row sums not all 1.0 — max deviation {np.abs(row_sums - 1.0).max():.2e}"
    )

    # Rigid-body translation test: uniform displacement reconstructs exactly
    N_samp = W.shape[1]
    disp_samp = np.ones((N_samp, 3))
    disp_recon = W @ disp_samp
    assert np.allclose(disp_recon, 1.0, atol=1e-10), (
        "W fails rigid-body translation test"
    )

    # Force-keep nodes present
    sampled_pids = mesh.node_pid[sampled_idx]
    for pid in FORCE_KEEP_PIDS:
        assert pid in sampled_pids, f"Force-keep PID {pid} missing from sampled set"

    logger.info("All acceptance checks passed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Downsample FE mesh and build reconstruction matrix W.")
    parser.add_argument("kfile", help="Path to car_and_barriers.k")
    parser.add_argument("--output-dir", default="downsample_output", help="Output directory")
    parser.add_argument("--method", default="fps",
                        choices=["fps", "random", "stride", "poisson_disk"],
                        help="Sampling method")
    parser.add_argument("--knn", type=int, default=6, help="Neighbours for W interpolation")
    parser.add_argument("--weight-mode", default="inverse_distance",
                        choices=["inverse_distance", "rbf", "barycentric"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d3plot", default=None, help="Optional d3plot for validation")
    parser.add_argument("--validate-states", type=int, default=None,
                        help="Number of time states to validate (default: all)")
    args = parser.parse_args()

    # Override sampler method in all region configs
    region_configs = []
    for rcfg in DEFAULT_REGION_CONFIGS:
        new_sampler = SamplerConfig(
            method=args.method,
            n_points=rcfg.sampler.n_points,
            seed=args.seed,
        )
        region_configs.append(RegionConfig(
            name=rcfg.name,
            sampler=new_sampler,
            split_by_part=rcfg.split_by_part,
            min_per_part=rcfg.min_per_part,
        ))

    mesh = parse_kfile(args.kfile)
    sampled_idx, labels, segments = sample_mesh(mesh, region_configs)

    w_cfg = WConfig(knn=args.knn, weight_mode=args.weight_mode)
    W = build_reconstruction_matrix(mesh, sampled_idx, w_cfg)

    acceptance_checks(mesh, sampled_idx, W)
    save_outputs(args.output_dir, mesh, sampled_idx, labels, segments, W)

    if args.d3plot:
        source = D3plotDisplacementSource(args.d3plot, mesh.coords)
        report = run_validation(W, sampled_idx, mesh, source, max_states=args.validate_states)
        report_path = Path(args.output_dir) / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump({k: v for k, v in report.items() if k != "per_state"}, f, indent=2)
        logger.info("Validation report: %s", report_path)
        logger.info(
            "Dm error %.2f%% | Wm error %.2f%%",
            report["max_Dm_error_pct"], report["max_Wm_error_pct"],
        )


if __name__ == "__main__":
    main()
