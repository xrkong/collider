"""
k-file geometry parser — SPEC §2 (parsing gotchas) and §1.1 (inputs).

The k-file mixes space-delimited and fixed-width-8-char element blocks
within the *same* keyword (barrier *ELEMENT_SOLID/_BEAM lines are glued
digits, no spaces). `parse_fields` tries split() first and only falls back
to fixed-width slicing when that yields too few tokens — see SPEC §2 table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parse_fields(line: str, n_expected: int) -> list[str]:
    """Return >= n_expected fields from a k-file data line.

    Tries whitespace splitting first (cheap, correct for most lines); falls
    back to fixed 8-char-width slicing for the barrier blocks that have no
    spaces between fields at all.
    """
    parts = line.split()
    if len(parts) >= n_expected:
        return parts
    return [line[i:i + 8].strip() for i in range(0, len(line.rstrip()), 8)]


def _validate_pid(pid: int, context: str = "") -> None:
    """Fail loudly if a parsed PID looks like a corrupted fixed-width merge.

    A real PID never exceeds 8 digits in this mesh (max is 10000024); if the
    fixed-width fallback failed to trigger, glued digits produce an absurdly
    large integer instead — that's the signal to catch here.
    """
    if pid > 100_000_000:
        raise ValueError(
            f"Parsed PID {pid} is absurdly large — fixed-width fallback failed. "
            f"Context: {context}"
        )


@dataclass
class MeshData:
    """Geometry extracted from the k-file (t=0 reference state only)."""
    node_ids: np.ndarray   # (N,)  int64  — k-file node IDs, in file order
    coords: np.ndarray     # (N,3) float64 — reference (t=0) coordinates [mm]
    node_pid: np.ndarray   # (N,)  int32  — PID of the first element touching each node

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)


def parse_kfile(kfile_path: str | Path) -> MeshData:
    """Parse *NODE and every *ELEMENT_* block from a k-file.

    Returns reference coordinates and a per-node PID assignment (first
    element that references a node wins). A single generic handler covers
    all *ELEMENT_* keywords (SOLID, SHELL, BEAM, MASS, DISCRETE, ...) since
    they all share the `eid pid n1 [n2 ...]` layout on their first data line;
    secondary/orientation lines are skipped because their second field is
    not a plausible PID.
    """
    kfile_path = Path(kfile_path)
    logger.info("Parsing k-file: %s", kfile_path)

    node_id_list: list[int] = []
    coord_list: list[tuple[float, float, float]] = []
    nid_to_idx: dict[int, int] = {}
    node_pid: dict[int, int] = {}   # nid -> PID of first referencing element

    current_keyword: str | None = None

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
                nid_to_idx[nid] = len(node_id_list)
                node_id_list.append(nid)
                coord_list.append((x, y, z))

            elif current_keyword is not None and current_keyword.startswith("*ELEMENT_"):
                fields = parse_fields(line, 3)
                if len(fields) < 2:
                    continue
                try:
                    pid = int(fields[1])
                except ValueError:
                    continue
                if pid < 1 or pid > 100_000_000:
                    continue  # secondary/orientation line, not an element record
                _validate_pid(pid, f"line {lineno}: {line!r}")
                for nf in fields[2:]:
                    if not nf:
                        continue
                    try:
                        nid = int(nf)
                    except ValueError:
                        continue
                    if nid == 0:
                        continue  # trailing padding, not node id 0
                    if nid in nid_to_idx and nid not in node_pid:
                        node_pid[nid] = pid

    if not node_id_list:
        size = kfile_path.stat().st_size if kfile_path.exists() else -1
        raise ValueError(
            f"No *NODE data parsed from {kfile_path} (file size: {size} bytes). "
            f"The k-file is empty or missing its *NODE block — check with whoever "
            f"provided this dataset rather than re-running; there is nothing here "
            f"to sample from."
        )

    node_ids = np.array(node_id_list, dtype=np.int64)
    coords = np.array(coord_list, dtype=np.float64).reshape(-1, 3)
    pid_arr = np.array(
        [node_pid.get(int(nid), 0) for nid in node_ids], dtype=np.int32
    )

    mesh = MeshData(node_ids=node_ids, coords=coords, node_pid=pid_arr)
    logger.info("Parsed %d nodes", mesh.n_nodes)
    return mesh
