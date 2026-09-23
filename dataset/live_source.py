"""
live_source.py — read fine-resolution (native d3plot dt) trajectory data
directly from a d3plot sequence, reusing the node subset/connectivity that
`dataset/build_dataset.py` already computed and stored in that case's stage-1
HDF5 (`/metadata`).

Why this exists: build_dataset.py applies `--frame-stride` when writing
`/states/*`, so a case's stage-1 h5 is missing exactly the fine intermediate
frames that stage-2 autoregressive training needs between two stage-1
reference frames. Redoing build_dataset.py's expensive region-aware FPS node
sampling live (per training run) would be wasteful and risks drifting from
the node subset the stage-1 model was actually trained on — so this module
never re-samples. It only re-derives:
  - global (full-mesh) element connectivity, via connectivity.load_connectivity()
    (cheap — one header-only d3plot open)
  - the sampled-node ROW indices into the d3plot's own node arrays, recovered
    by matching the h5's stored sampled_node_ids (k-file node IDs) against
    the d3plot's own node-ID array (see extract_live_trajectory step 6) — the
    h5 never stores this row mapping itself, only the IDs
and then calls the existing extract_frame_data() once per native frame in
the requested time range, exactly like build_dataset.py's Step 6, just with
frame-stride effectively 1 (every native frame kept).

IMPORTANT: the h5's own `shell_cells`/`solid_cells`/`beam_cells` datasets are
NOT reusable here — those are `connectivity.filter_and_remap_cells()` output
(sparse, LOCAL 0..N-1 indices, built only for visualization coverage). The
extract_frame_data() args need the GLOBAL, unfiltered connectivity instead,
which is why this module recomputes it from the d3plot header rather than
reading it back from the h5.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed
from lasso.dyna import ArrayType, D3plot

from .connectivity import load_connectivity
from .d3plot_io import copy_to_tmp, extract_frame_data, find_state_files, scan_times


@dataclass
class LiveTrajectory:
    times: np.ndarray               # (T_fine,) float64 [s], sorted, deduped
    positions: np.ndarray           # (T_fine, N, 3) float32 — deformed coords [mm]
    eff_plastic_strain: np.ndarray  # (T_fine, N) float32
    node_alive: np.ndarray          # (T_fine, N) uint8
    sampled_node_ids: np.ndarray    # (N,) int64 — passthrough from ref_h5, for validation/logging


# Process-wide cache: extracting a whole case is expensive (per-frame lasso
# opens), so both the normalization-stats pass and the dataset __init__ pass
# for the same (src_dir, ref_h5, t_start, t_end) share one extraction instead
# of paying for it twice within a training run.
_CACHE: dict[tuple, LiveTrajectory] = {}


def _dedup_sorted_in_range(
    entries: list[tuple[float, str, int]],
    t_lo: float,
    t_hi: float,
    eps: float = 1e-9,
) -> list[tuple[float, str, int]]:
    """Sort by time, drop entries outside [t_lo, t_hi], deduplicate by time.

    Unlike d3plot_io.select_frames, this keeps EVERY native frame in range —
    no stride is applied, since stage-2 needs the fine frames stage-1's
    --frame-stride discarded.
    """
    entries = sorted(entries, key=lambda e: e[0])
    out: list[tuple[float, str, int]] = []
    last_t: float | None = None
    for e in entries:
        if e[0] < t_lo - eps or e[0] > t_hi + eps:
            continue
        if last_t is None or abs(e[0] - last_t) > eps:
            out.append(e)
            last_t = e[0]
    return out


def extract_live_trajectory(
    src_dir: Path,
    ref_h5: Path,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
    n_jobs: int = 4,
    tmp: Path = Path("/tmp/d3plot_tmp_live"),
    use_cache: bool = True,
    allow_source_dir_mismatch: bool = False,
) -> LiveTrajectory:
    """Extract every native-resolution frame in [t_start, t_end] from the
    d3plot sequence at src_dir, using the node subset/connectivity already
    computed for this case in ref_h5.

    t_start/t_end default to ref_h5["states/times"][0]/[-1] — the whole span
    the stage-1 h5 covers — which is the recommended default (matches how
    stage-2 AR training already consumes one whole-case source, and gives
    enough frames for realistic sliding windows). Pass explicit values to
    debug on a shorter sub-range.
    """
    src_dir = Path(src_dir)
    ref_h5 = Path(ref_h5)
    key = (str(src_dir.resolve()), str(ref_h5.resolve()), t_start, t_end)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    tmp.mkdir(parents=True, exist_ok=True)

    # ── Read ref h5 metadata (node subset only — never the sparse cells) ──
    with h5py.File(ref_h5, "r") as f:
        mg = f["metadata"]
        sampled_node_ids = mg["sampled_node_ids"][:].astype(np.int64)
        source_dir_attr = mg.attrs.get("source_dir")
        n_nodes_attr = int(mg.attrs.get("n_nodes", len(sampled_node_ids)))
        if t_start is None or t_end is None:
            times_h5 = f["states/times"][:]
            if t_start is None:
                t_start = float(times_h5[0])
            if t_end is None:
                t_end = float(times_h5[-1])

    if not allow_source_dir_mismatch and source_dir_attr is not None:
        if str(src_dir.resolve()) != str(source_dir_attr):
            raise ValueError(
                f"{src_dir} does not match {ref_h5}'s recorded metadata.attrs['source_dir'] "
                f"({source_dir_attr}) — this h5's node subset/connectivity may not belong to "
                f"this d3plot case. Pass allow_source_dir_mismatch=True if the data was "
                f"legitimately moved/copied."
            )

    N = len(sampled_node_ids)
    if N != n_nodes_attr:
        raise ValueError(
            f"{ref_h5}: metadata/sampled_node_ids has {N} entries but "
            f"metadata.attrs['n_nodes']={n_nodes_attr} — inconsistent stage-1 h5."
        )

    # ── Enumerate + filter native frames ───────────────────────────────
    state_files = find_state_files(src_dir)
    if not state_files:
        raise FileNotFoundError(f"No d3plot state files in {src_dir}")
    all_entries = scan_times(state_files, tmp, n_jobs=n_jobs)
    selected = _dedup_sorted_in_range(all_entries, t_start, t_end)
    if len(selected) < 2:
        raise ValueError(
            f"Segment [{t_start}, {t_end}] contains only {len(selected)} native frame(s) in "
            f"{src_dir} — check that t_start/t_end (or {ref_h5}'s states/times) actually "
            f"bracket real simulation time for this case."
        )

    # ── Header probe: global connectivity + node-ID -> row-index map ──
    work_dir = Path(tempfile.mkdtemp(dir=tmp, prefix="header_"))
    try:
        copy_to_tmp(state_files[0], work_dir)
        d3 = D3plot(
            str(work_dir / "d3plot"),
            state_array_filter=[ArrayType.global_timesteps, ArrayType.node_ids],
        )
        conn = load_connectivity(d3)
        node_ids_full = d3.arrays.get(ArrayType.node_ids)
        if node_ids_full is None:
            raise RuntimeError(
                f"{src_dir}: d3plot header has no {ArrayType.node_ids} array — cannot map "
                f"{ref_h5}'s sampled_node_ids back to d3plot node rows."
            )
        del d3
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    id_to_row = {int(nid): i for i, nid in enumerate(node_ids_full)}
    missing = [int(nid) for nid in sampled_node_ids if int(nid) not in id_to_row]
    if missing:
        raise ValueError(
            f"{len(missing)} of {N} sampled node IDs from {ref_h5} were not found in "
            f"{src_dir}'s d3plot node array (e.g. {missing[:5]}) — {ref_h5} does not match "
            f"this d3plot sequence's mesh."
        )
    sampled_idx = np.array([id_to_row[int(nid)] for nid in sampled_node_ids], dtype=np.int64)
    if len(sampled_idx) != N:
        raise ValueError(f"sampled_idx length {len(sampled_idx)} != expected N={N}.")

    n_full = int(max(
        len(node_ids_full),
        conn.shell_conn.max(initial=-1) + 1,
        conn.solid_conn.max(initial=-1) + 1,
        conn.beam_conn.max(initial=-1) + 1,
    ))
    for name, c in (("shell", conn.shell_conn), ("solid", conn.solid_conn), ("beam", conn.beam_conn)):
        if len(c) > 0 and c.max() >= n_full:
            raise AssertionError(f"{name}_conn max index {c.max()} >= n_full {n_full} in {src_dir}")

    # ── Per-frame extraction (parallel, mirrors build_dataset.py Step 6) ──
    state_file_map = {p.name: p for p in state_files}
    extract_args = (sampled_idx, conn.shell_conn, conn.solid_conn, conn.beam_conn, n_full, tmp)
    if n_jobs == 1:
        results = [
            extract_frame_data(state_file_map[fname], sidx, *extract_args)
            for _, fname, sidx in selected
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(extract_frame_data)(state_file_map[fname], sidx, *extract_args)
            for _, fname, sidx in selected
        )

    times = np.array([t for t, _, _ in selected], dtype=np.float64)
    positions = np.stack([r[0] for r in results]).astype(np.float32)
    eff_plastic_strain = np.stack([r[1] for r in results]).astype(np.float32)
    node_alive = np.stack([r[2] for r in results]).astype(np.uint8)

    if positions.shape != (len(selected), N, 3):
        raise ValueError(
            f"Extracted positions shape {positions.shape} does not match expected "
            f"({len(selected)}, {N}, 3) for {src_dir} against {ref_h5}."
        )
    if not np.all(np.diff(times) > 0):
        raise AssertionError(f"{src_dir}: extracted times are not strictly increasing.")

    traj = LiveTrajectory(
        times=times,
        positions=positions,
        eff_plastic_strain=eff_plastic_strain,
        node_alive=node_alive,
        sampled_node_ids=sampled_node_ids,
    )
    if use_cache:
        _CACHE[key] = traj
    return traj
