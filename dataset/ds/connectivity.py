"""
Mesh connectivity extraction (via lasso-python) for PyVista visualization.

Element connectivity is constant across all frames (no erosion, SPEC §6.5),
so it's read once from the d3plot header and reused for every state. Two
distinct uses of the same connectivity:

1. eff_plastic_strain (full coverage): scatter-average ALL elements' strain
   onto their own nodes using the FULL, unfiltered connectivity, then
   subset to the sampled nodes. Every sampled node gets a value because
   every real FE node touches at least one element in the original mesh —
   this does not depend on whether the element's *other* nodes survived
   sampling.

2. visualization cells (sparse coverage): filtered to elements where EVERY
   node survived sampling, then remapped to local 0..N-1 indices for a
   PyVista mesh. This is necessarily sparse — region-aware FPS targets
   spread-out points, the opposite of what keeps an element's corners
   together — except for fully-retained parts (rebar/reinforcement, SPEC
   §4.1/§4.2) whose connectivity survives intact. Measured on this dataset:
   ~0.9% of shell elements, ~0.08% of solid elements, ~12.5% of beam
   elements (rebar/reinforcement) survive. Combine with a point-cloud
   overlay (using the eff_plastic_strain field) for full coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lasso.dyna import ArrayType, D3plot


@dataclass
class MeshConnectivity:
    shell_conn: np.ndarray   # (n_shell, 4) int32, 0-based global row indices (quad; tri = repeated last idx)
    solid_conn: np.ndarray   # (n_solid, 8) int32, 0-based global row indices (hex; degenerate = repeated idx)
    beam_conn: np.ndarray    # (n_beam, 2)  int32, 0-based global row indices (line endpoints only)


def load_connectivity(d3: D3plot) -> MeshConnectivity:
    """Read element-node connectivity from an opened D3plot header."""
    shell = d3.arrays.get(ArrayType.element_shell_node_indexes)
    solid = d3.arrays.get(ArrayType.element_solid_node_indexes)
    beam  = d3.arrays.get(ArrayType.element_beam_node_indexes)
    return MeshConnectivity(
        shell_conn=shell.astype(np.int32) if shell is not None else np.empty((0, 4), dtype=np.int32),
        solid_conn=solid.astype(np.int32) if solid is not None else np.empty((0, 8), dtype=np.int32),
        beam_conn=(beam[:, :2].astype(np.int32) if beam is not None else np.empty((0, 2), dtype=np.int32)),
    )


def filter_and_remap_cells(
    conn: np.ndarray, sampled_idx: np.ndarray, n_full: int
) -> np.ndarray:
    """Keep elements whose every node survived sampling; remap to local 0..N-1 indices."""
    if len(conn) == 0:
        return conn.reshape(0, conn.shape[1] if conn.ndim == 2 else 0)

    sampled_mask = np.zeros(n_full, dtype=bool)
    sampled_mask[sampled_idx] = True
    keep = sampled_mask[conn].all(axis=1)

    global_to_local = np.full(n_full, -1, dtype=np.int64)
    global_to_local[sampled_idx] = np.arange(len(sampled_idx))
    return global_to_local[conn[keep]]


def scatter_scalar_to_nodes(
    elem_scalar: np.ndarray,   # (n_elem,) float
    conn: np.ndarray,          # (n_elem, k) int, 0-based global row indices
    n_full: int,
    elem_alive: np.ndarray | None = None,   # (n_elem,) bool — exclude dead elements if given
) -> tuple[np.ndarray, np.ndarray]:
    """Average an element scalar field onto its own nodes (full mesh, every element).

    If elem_alive is given, eroded elements (alive == False) are excluded —
    otherwise a dead element's last-known (frozen) strain value would keep
    contributing to its nodes' average forever after it erodes, silently
    inflating strain at still-alive neighboring nodes.

    Returns (node_sum, node_count) so callers can combine contributions from
    multiple element types (shell + solid) before dividing.
    """
    node_sum = np.zeros(n_full, dtype=np.float64)
    node_count = np.zeros(n_full, dtype=np.int64)
    if len(conn) == 0:
        return node_sum, node_count

    k = conn.shape[1]
    nodes_flat = conn.ravel()
    vals_rep = np.repeat(np.asarray(elem_scalar, dtype=np.float64), k)
    valid = nodes_flat >= 0
    if elem_alive is not None:
        alive_rep = np.repeat(np.asarray(elem_alive, dtype=bool), k)
        valid = valid & alive_rep
    np.add.at(node_sum, nodes_flat[valid], vals_rep[valid])
    np.add.at(node_count, nodes_flat[valid], 1)
    return node_sum, node_count


def scatter_alive_to_nodes(
    elem_alive_and_conn: list[tuple[np.ndarray, np.ndarray]],
    n_full: int,
) -> np.ndarray:
    """OR-aggregate per-element alive flags onto their nodes (full mesh).

    A node counts as alive if AT LEAST ONE element touching it (across all
    element types passed in) is still alive — it only flips to eroded once
    every element sharing it has died. Pass [(alive_shell, shell_conn),
    (alive_solid, solid_conn), (alive_beam, beam_conn)].
    """
    node_alive = np.zeros(n_full, dtype=bool)
    for elem_alive, conn in elem_alive_and_conn:
        if len(conn) == 0 or elem_alive is None:
            continue
        k = conn.shape[1]
        nodes_flat = conn.ravel()
        alive_rep = np.repeat(np.asarray(elem_alive, dtype=bool), k)
        valid = nodes_flat >= 0
        np.logical_or.at(node_alive, nodes_flat[valid], alive_rep[valid])
    return node_alive


def average_elem_field(field: np.ndarray) -> np.ndarray:
    """Collapse integration-point/layer dims of a per-state element field to (n_elem,)."""
    f = np.asarray(field, dtype=np.float32)
    if f.ndim > 1:
        f = f.mean(axis=tuple(range(1, f.ndim)))
    return f


def elem_is_alive(field: np.ndarray | None) -> np.ndarray | None:
    """Convert a raw element_*_is_alive array to a bool mask.

    LS-DYNA encodes this as 0.0 == dead/eroded, any nonzero == alive (the
    nonzero value itself carries no further meaning — verified empirically:
    at t=0 every element reads nonzero with zero dead count; later states
    show exactly the eroded elements drop to 0.0).
    """
    if field is None:
        return None
    return np.asarray(field).reshape(-1) != 0
