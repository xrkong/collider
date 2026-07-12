"""
d3plot state-file plumbing, adapted from dataset/d3plot_to_h5_dt.py.

LS-DYNA splits one simulation across d3plot, d3plot01, d3plot02, ... :
the first file is a header (mesh/part layout) and the numbered files hold
state data only — lasso-python needs both copied together as `d3plot` /
`d3plot01` in the same directory to open a given state file in isolation.

lasso always opens ONE state file at a time this way (header + a single
numbered file), so memory use is bounded by one state's worth of arrays
regardless of how many d3plotNN files the simulation has — the full
multi-GB sequence is never loaded at once.

Parallel scanning/extraction (`n_jobs > 1`) runs each file in its own
isolated tmp subdirectory via tempfile.mkdtemp, so concurrent workers never
clobber each other's staged `d3plot`/`d3plot01` copies.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from lasso.dyna import ArrayType, D3plot


def decode(raw: object) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("ascii", errors="ignore").strip().strip("\x00")
    return str(raw).strip().strip("\x00")


def _suffix_number(path: Path) -> int:
    s = path.name.replace("d3plot", "", 1)
    return -1 if s == "" else (int(s) if s.isdigit() else 10 ** 12)


def find_state_files(src: Path) -> list[Path]:
    """Return d3plot01, d3plot02, ... sorted by numeric suffix (excludes the header file)."""
    files = [p for p in src.glob("d3plot*") if p.is_file() and p.name != "d3plot"]
    return sorted(files, key=_suffix_number)


def copy_to_tmp(state_file: Path, tmp: Path) -> None:
    """Stage src_dir/d3plot (header) + the given state file as tmp/d3plot, tmp/d3plot01."""
    src_dir = state_file.parent
    (tmp / "d3plot").unlink(missing_ok=True)
    (tmp / "d3plot01").unlink(missing_ok=True)
    shutil.copy2(src_dir / "d3plot", tmp / "d3plot")
    shutil.copy2(state_file, tmp / "d3plot01")


def close_d3(d3, tmp: Path) -> None:
    """Drop the lasso object and remove the staged copies."""
    if d3 is not None:
        del d3
    (tmp / "d3plot").unlink(missing_ok=True)
    (tmp / "d3plot01").unlink(missing_ok=True)


def _scan_one_file(state_file: Path, tmp_root: Path) -> list[tuple[float, str, int]]:
    """Worker unit for scan_times: stage + open one file in an isolated tmp dir."""
    work_dir = Path(tempfile.mkdtemp(dir=tmp_root, prefix="scan_"))
    entries: list[tuple[float, str, int]] = []
    try:
        copy_to_tmp(state_file, work_dir)
        try:
            d3 = D3plot(str(work_dir / "d3plot"), state_array_filter=[ArrayType.global_timesteps])
            t_arr = d3.arrays.get(ArrayType.global_timesteps, np.array([0.0]))
            entries = [(float(t), state_file.name, int(s)) for s, t in enumerate(t_arr)]
            del d3
        except RuntimeError as e:
            if "endmark" in str(e).lower():
                print(f"  WARNING: {state_file.name} skipped ({e})")
            else:
                raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return entries


def scan_times(
    state_files: list[Path], tmp: Path, n_jobs: int = 1
) -> list[tuple[float, str, int]]:
    """Open every state file just for timestamps; returns (time, filename, state_idx) tuples.

    n_jobs > 1 scans files concurrently via joblib (each file's read is fully
    independent — embarrassingly parallel). n_jobs=1 runs a plain loop.
    """
    print(f"Pass 1/2 — scanning {len(state_files)} state files … (n_jobs={n_jobs})")
    if n_jobs == 1:
        results = [_scan_one_file(f, tmp) for f in state_files]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_scan_one_file)(f, tmp) for f in state_files
        )
    entries = [e for sub in results for e in sub]
    print(f"  scanned {len(state_files)} files → {len(entries)} states total")
    return entries


def extract_node_array(
    state_file: Path,
    sidx: int,
    array_type,
    row_idx: np.ndarray,
    tmp_root: Path,
) -> np.ndarray:
    """Stage + open one state file in isolation, return array_type[sidx][row_idx].

    Used both sequentially and as the joblib worker unit for parallel
    per-frame extraction — each call only ever holds one state's arrays in
    memory (~tens of MB for this mesh), never the whole multi-GB sequence.
    """
    work_dir = Path(tempfile.mkdtemp(dir=tmp_root, prefix="extract_"))
    try:
        copy_to_tmp(state_file, work_dir)
        d3 = D3plot(
            str(work_dir / "d3plot"),
            state_array_filter=[ArrayType.global_timesteps, array_type],
        )
        arr_full = d3.arrays.get(array_type)
        if arr_full is None:
            raise RuntimeError(f"{array_type} missing in {state_file.name}")
        result = arr_full[sidx][row_idx].astype(np.float32)
        del d3
        return result
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def extract_frame_data(
    state_file: Path,
    sidx: int,
    sampled_idx: np.ndarray,
    shell_conn: np.ndarray,
    solid_conn: np.ndarray,
    beam_conn: np.ndarray,
    n_full: int,
    tmp_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage + open one state file, return (positions, eff_plastic_strain, node_alive)
    for sampled_idx.

    positions           : (N, 3) float32 — deformed coordinate (see the
                          node_displacement note in build_dataset.py: despite
                          the name it is the absolute coordinate, not a delta).
    eff_plastic_strain  : (N,) float32 — shell + solid element strain averaged
                          onto each sampled node using the FULL (unfiltered)
                          connectivity, so every sampled node gets a value
                          regardless of whether its neighbors survived
                          sampling (see connectivity.py docstring). Eroded
                          elements are excluded so a dead element's frozen
                          strain value can't keep polluting its neighbors.
    node_alive          : (N,) bool — True unless every element touching that
                          node has eroded (element_*_is_alive == 0). This
                          dataset does have some erosion despite SPEC's "no
                          erosion" assumption (~0.03% shell, ~0.04% solid by
                          the end of a 60km/h run) — node COUNT stays fixed
                          either way (LS-DYNA keeps eroded nodes' positions),
                          only this liveness flag changes.
    """
    from .connectivity import (
        average_elem_field, elem_is_alive, scatter_alive_to_nodes, scatter_scalar_to_nodes,
    )

    work_dir = Path(tempfile.mkdtemp(dir=tmp_root, prefix="extract_"))
    try:
        copy_to_tmp(state_file, work_dir)
        d3 = D3plot(
            str(work_dir / "d3plot"),
            state_array_filter=[
                ArrayType.global_timesteps,
                ArrayType.node_displacement,
                ArrayType.element_shell_effective_plastic_strain,
                ArrayType.element_solid_effective_plastic_strain,
                ArrayType.element_shell_is_alive,
                ArrayType.element_solid_is_alive,
                ArrayType.element_beam_is_alive,
            ],
        )

        disp_full = d3.arrays.get(ArrayType.node_displacement)
        if disp_full is None:
            raise RuntimeError(f"node_displacement missing in {state_file.name}")
        pos = disp_full[sidx][sampled_idx].astype(np.float32)

        alive_shell = elem_is_alive(d3.arrays.get(ArrayType.element_shell_is_alive))
        alive_solid = elem_is_alive(d3.arrays.get(ArrayType.element_solid_is_alive))
        alive_beam  = elem_is_alive(d3.arrays.get(ArrayType.element_beam_is_alive))
        if alive_shell is not None:
            alive_shell = alive_shell[: len(shell_conn)]
        if alive_solid is not None:
            alive_solid = alive_solid[: len(solid_conn)]
        if alive_beam is not None:
            alive_beam = alive_beam[: len(beam_conn)]

        node_sum = np.zeros(n_full, dtype=np.float64)
        node_count = np.zeros(n_full, dtype=np.int64)

        eps_shell_arr = d3.arrays.get(ArrayType.element_shell_effective_plastic_strain)
        if eps_shell_arr is not None and len(shell_conn) > 0:
            eps_shell = average_elem_field(eps_shell_arr[sidx])
            s, c = scatter_scalar_to_nodes(eps_shell, shell_conn, n_full, elem_alive=alive_shell)
            node_sum += s
            node_count += c

        eps_solid_arr = d3.arrays.get(ArrayType.element_solid_effective_plastic_strain)
        if eps_solid_arr is not None and len(solid_conn) > 0:
            eps_solid = average_elem_field(eps_solid_arr[sidx])
            s, c = scatter_scalar_to_nodes(eps_solid, solid_conn, n_full, elem_alive=alive_solid)
            node_sum += s
            node_count += c

        node_eps_full = np.zeros(n_full, dtype=np.float32)
        good = node_count > 0
        node_eps_full[good] = (node_sum[good] / node_count[good]).astype(np.float32)
        eps_node = node_eps_full[sampled_idx]

        node_alive_full = scatter_alive_to_nodes(
            [(alive_shell, shell_conn), (alive_solid, solid_conn), (alive_beam, beam_conn)],
            n_full,
        )
        # nodes touched by zero elements (shouldn't happen for real FE nodes,
        # but guard anyway) default to alive rather than a spurious False
        touched = np.zeros(n_full, dtype=bool)
        for conn in (shell_conn, solid_conn, beam_conn):
            if len(conn) > 0:
                touched[conn[conn >= 0].ravel()] = True
        node_alive_full[~touched] = True
        node_alive = node_alive_full[sampled_idx]

        del d3
        return pos, eps_node, node_alive
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def select_frames(
    entries: list[tuple[float, str, int]],
    stride: int,
    limit: int | None = None,
) -> list[tuple[float, str, int]]:
    """Sort by time, deduplicate, keep every `stride`-th entry, optionally cap the count."""
    entries = sorted(entries, key=lambda e: e[0])
    unique: list[tuple[float, str, int]] = []
    last_t: float | None = None
    for e in entries:
        if last_t is None or abs(e[0] - last_t) > 1e-9:
            unique.append(e)
            last_t = e[0]

    selected = unique[::stride]
    if limit is not None:
        selected = selected[:limit]

    times = [e[0] for e in selected]
    print(f"  {len(unique)} unique states → stride {stride} → {len(selected)} frames "
          f"(t = {times[0]*1e3:.2f}–{times[-1]*1e3:.2f} ms)")
    return selected
