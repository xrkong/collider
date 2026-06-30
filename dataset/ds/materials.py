"""
k-file material parser, adapted from dataset/d3plot_to_h5_dt.py.

Single pass over *PART and *MAT_xxx[_TITLE] cards, joined on material id
(mid), keyed by part title so the result lines up with d3plot's
part_titles strings (d3plot part indices are sequential 1..P, not the
k-file PIDs, so name is the only reliable join key between the two files).
"""

from __future__ import annotations

import re
from pathlib import Path

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
MAT_TYPE_UNKNOWN = 8
MAT_TYPE_ID_LEGEND = (
    "0=plasticity 1=rigid 2=elastic 3=rubber "
    "4=concrete 5=foam 6=spotweld 7=spring/damper 8=unknown"
)


def _kfields(line: str) -> list[str]:
    """Parse a 10-char fixed-width k-file data line into tokens."""
    s = line.rstrip()
    if len(s) < 10:
        return s.split()
    chunks = [s[k:k + 10].strip() for k in range(0, len(s), 10)]
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


def parse_kfile_parts_and_materials(
    path: Path,
) -> tuple[dict[int, str], dict[str, dict]]:
    """
    Single pass over *PART and *MAT_xxx[_TITLE] cards.

    Returns
    -------
    pid_to_name    : {k-file PID -> part title}
                     PID is *PART's own first data-line field — the same PID
                     referenced by *ELEMENT_* lines, so it joins directly
                     with kfile_parser.MeshData.node_pid. (d3plot's
                     part_titles_ids is a sequential 1..P index, NOT this
                     PID, and cannot be used for this join.)
    name_to_props  : {part title -> {type_name, type_id, label, rho, E, nu, sigy}}
                     label is the precise *MAT_xxx_TITLE string when present,
                     else falls back to the bare MAT keyword.
    """
    parts: list[dict] = []
    mats: dict[int, dict] = {}

    try:
        with open(path, encoding="ascii", errors="ignore") as fh:
            lines = fh.readlines()
    except OSError as e:
        print(f"  WARNING: cannot read k-file {path}: {e}")
        return {}

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        kw = raw.strip().upper()

        if kw == "*PART":
            j = i + 1
            while j < len(lines) and lines[j].startswith("$"):
                j += 1
            part_title = ""
            if j < len(lines) and not lines[j].startswith("*"):
                part_title = lines[j].strip()
                j += 1
            while j < len(lines) and lines[j].startswith("$"):
                j += 1
            if j < len(lines) and not lines[j].startswith("*"):
                tok = _kfields(lines[j])
                # *PART data line: pid, secid, mid, eosid, hgid, grav, adpopt, tmid
                if len(tok) >= 3:
                    parts.append({"name": part_title, "pid": _ki(tok[0]), "mid": _ki(tok[2])})
                j += 1
            i = j
            continue

        if raw.startswith("*MAT_"):
            mat_kw = raw.strip()
            has_title = mat_kw.upper().endswith("_TITLE")
            mat_type = re.sub(r"_TITLE$", "", mat_kw, flags=re.IGNORECASE)
            mat_type = mat_type.upper().replace("*MAT_", "")

            j = i + 1
            mat_label = ""
            if has_title:
                while j < len(lines) and lines[j].startswith("$"):
                    j += 1
                if j < len(lines) and not lines[j].startswith("*"):
                    mat_label = lines[j].strip()
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
                    elif "BLATZ" in mat_type or "RUBBER" in mat_type or "FOAM" in mat_type:
                        E, nu, sigy = _kf(tok[2]) if len(tok) > 2 else 0.0, 0.0, 0.0
                    else:
                        E = _kf(tok[2]) if len(tok) > 2 else 0.0
                        nu = _kf(tok[3]) if len(tok) > 3 else 0.0
                        sigy = _kf(tok[4]) if len(tok) > 4 else 0.0
                    mats[mid] = {
                        "type_name": mat_type,
                        "type_id": _MAT_TYPE_ID_MAP.get(mat_type, MAT_TYPE_UNKNOWN),
                        "label": mat_label if mat_label else mat_type,
                        "rho": rho, "E": E, "nu": nu, "sigy": sigy,
                    }
                j += 1
            i = j
            continue

        i += 1

    name_to_props: dict[str, dict] = {}
    pid_to_name: dict[int, str] = {}
    for p in parts:
        pid_to_name[p["pid"]] = p["name"]
        props = mats.get(p["mid"])
        if props is not None:
            name_to_props[p["name"]] = props
    return pid_to_name, name_to_props
