"""
d3plot_inspect.py – Print a structured overview of every array / keyword
available in an LS-DYNA d3plot file set.

Covers:
  • Mesh metadata  – node counts, element counts per type
  • Part catalog   – part ID, title, element breakdown
  • All node arrays – coordinates, displacement, velocity, acceleration, …
  • All element arrays – stress (solid/shell), strain, history vars, …
  • All global arrays – timesteps, internal/kinetic energy, …
  • Material info  (if present: mat IDs, titles, type codes)

Usage
-----
  python tools/d3plot_inspect.py --src /path/to/d3plot/dir

  # Also load first state file so state arrays (stress, disp, …) are populated
  python tools/d3plot_inspect.py --src /path/to/d3plot/dir --load-states

  # Point directly at a single file instead of a directory
  python tools/d3plot_inspect.py --src /path/to/d3plot
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from lasso.dyna import ArrayType, D3plot

# ── Pretty-print helpers ───────────────────────────────────────────────────────

SEP  = "─" * 72
SEP2 = "═" * 72

def _shape_str(arr: np.ndarray) -> str:
    return "×".join(str(d) for d in arr.shape)

def _dtype_str(arr: np.ndarray) -> str:
    return str(arr.dtype)

def _range_str(arr: np.ndarray) -> str:
    """Show min/max for numeric arrays (clipped to avoid huge output)."""
    try:
        flat = arr.astype(np.float64).ravel()
        if flat.size == 0:
            return "empty"
        lo, hi = float(flat.min()), float(flat.max())
        return f"[{lo:.4g}, {hi:.4g}]"
    except Exception:
        return "n/a"

def _decode(b: object) -> str:
    if isinstance(b, (bytes, bytearray)):
        return b.decode("ascii", errors="ignore").strip().strip("\x00")
    return str(b).strip().strip("\x00")


def _print_array(name: str, arr: np.ndarray, indent: int = 4) -> None:
    pad = " " * indent
    shape = _shape_str(arr)
    dtype = _dtype_str(arr)
    rng   = _range_str(arr)
    print(f"{pad}{name:<48}  shape={shape:<20}  dtype={dtype:<8}  range={rng}")


# ── Collect all ArrayType names ────────────────────────────────────────────────

def _all_array_type_names() -> list[str]:
    """Return every public attribute name of ArrayType (the lasso enum/registry)."""
    return sorted(
        name for name in dir(ArrayType)
        if not name.startswith("_")
    )


# ── Open d3plot (header only or with first state) ─────────────────────────────

def _open_d3plot(src: Path, load_states: bool, tmp: Path | None) -> D3plot:
    """
    Open a D3plot object.

    If `src` is a directory, copy the header file and (optionally) the first
    state file to `tmp` so lasso doesn't follow the file-chain and read every
    state at once.
    """
    if src.is_dir():
        hdr_src = src / "d3plot"
        if not hdr_src.exists():
            sys.exit(f"ERROR: no 'd3plot' file found in {src}")
        assert tmp is not None
        hdr_dst = tmp / "d3plot"
        shutil.copy2(hdr_src, hdr_dst)

        if load_states:
            # find the first state file (d3plot01 or the earliest numeric one)
            state_files = sorted(
                [p for p in src.glob("d3plot0*") if p.is_file()],
                key=lambda p: p.name,
            )
            if state_files:
                shutil.copy2(state_files[0], tmp / "d3plot01")
                print(f"  (loading states from {state_files[0].name})")
        # open with no filter → loads every array present
        d3 = D3plot(str(hdr_dst))
    else:
        # single file given directly
        d3 = D3plot(str(src))

    return d3


# ── Section printers ──────────────────────────────────────────────────────────

def _print_mesh_summary(d3: D3plot) -> None:
    print(SEP2)
    print("  MESH SUMMARY")
    print(SEP2)

    def _n(key: str) -> int:
        arr = d3.arrays.get(key)
        return len(arr) if arr is not None else 0

    coords = d3.arrays.get(ArrayType.node_coordinates)
    n_nodes = len(coords) if coords is not None else 0

    counts = {
        "nodes"         : n_nodes,
        "solid elems"   : _n(ArrayType.element_solid_node_indexes),
        "shell elems"   : _n(ArrayType.element_shell_node_indexes),
        "beam elems"    : _n(ArrayType.element_beam_node_indexes),
        "thick shell"   : _n(ArrayType.element_tshell_node_indexes),
        "sph particles" : _n(ArrayType.sph_node_indexes) if hasattr(ArrayType, "sph_node_indexes") else 0,
        "parts"         : _n(ArrayType.part_titles),
    }
    for label, n in counts.items():
        if n:
            print(f"    {label:<22}: {n:,}")
    print()


def _print_parts(d3: D3plot) -> None:
    print(SEP)
    print("  PARTS")
    print(SEP)

    titles     = d3.arrays.get(ArrayType.part_titles)
    title_ids  = d3.arrays.get(ArrayType.part_titles_ids)

    solid_pi   = d3.arrays.get(ArrayType.element_solid_part_indexes)
    shell_pi   = d3.arrays.get(ArrayType.element_shell_part_indexes)
    beam_pi    = d3.arrays.get(ArrayType.element_beam_part_indexes)
    tshell_pi  = d3.arrays.get(ArrayType.element_tshell_part_indexes)

    def _count_per_part(pi: np.ndarray | None, n_parts: int) -> np.ndarray:
        if pi is None or n_parts == 0:
            return np.zeros(n_parts, dtype=np.int64)
        c = np.zeros(n_parts, dtype=np.int64)
        for idx in pi:
            if 0 <= idx < n_parts:
                c[idx] += 1
        return c

    if titles is None:
        print("  (no part_titles found)")
        print()
        return

    n = len(titles)
    ids = title_ids if title_ids is not None else np.arange(n)

    solid_c  = _count_per_part(solid_pi,  n)
    shell_c  = _count_per_part(shell_pi,  n)
    beam_c   = _count_per_part(beam_pi,   n)
    tshell_c = _count_per_part(tshell_pi, n)

    print(f"  {'idx':>5}  {'part_id':>8}  {'solid':>7}  {'shell':>7}  {'beam':>6}  {'tshell':>7}  name")
    print(f"  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*40}")
    for i in range(n):
        name = _decode(titles[i])
        pid  = int(ids[i]) if ids is not None else i
        print(f"  {i:>5}  {pid:>8}  {solid_c[i]:>7}  {shell_c[i]:>7}  "
              f"{beam_c[i]:>6}  {tshell_c[i]:>7}  {name}")
    print()


def _print_materials(d3: D3plot) -> None:
    print(SEP)
    print("  MATERIALS  (if present in file)")
    print(SEP)

    # lasso exposes material info through several optional arrays
    mat_candidates = [
        "material_type_numbers",
        "material_titles",
        "material_ids",
    ]
    found_any = False
    for key in mat_candidates:
        arr = d3.arrays.get(key)
        if arr is not None and len(arr):
            found_any = True
            _print_array(key, arr)
            if key == "material_titles":
                for i, v in enumerate(arr):
                    print(f"      [{i:>4}]  {_decode(v)}")

    # also check ArrayType attrs that mention 'material'
    for name in _all_array_type_names():
        if "material" in name.lower() and name not in mat_candidates:
            arr = d3.arrays.get(name)
            if arr is not None and len(arr):
                found_any = True
                _print_array(name, arr)

    if not found_any:
        print("  (no material arrays found – material info is usually embedded in the")
        print("   keyword file (.k/.key), not the d3plot binary)")
    print()


def _print_arrays_by_category(d3: D3plot) -> None:
    """Print every array present, grouped by prefix (node_, element_, global_, …)."""

    categories: dict[str, list[str]] = {}
    present: dict[str, np.ndarray] = {}

    for name in _all_array_type_names():
        val = getattr(ArrayType, name, None)
        if val is None:
            continue
        # ArrayType values are the actual string keys used in d3.arrays
        arr = d3.arrays.get(val)
        if arr is None:
            continue
        # Categorise by first word before '_'
        prefix = name.split("_")[0]
        categories.setdefault(prefix, []).append(name)
        present[name] = arr

    order = ["node", "element", "global", "part", "sph", "airbag", "rigid", "contact"]
    seen  = set()

    def _print_cat(prefix: str) -> None:
        if prefix not in categories:
            return
        seen.add(prefix)
        print(SEP)
        print(f"  ARRAYS  [{prefix.upper()}]")
        print(SEP)
        for name in sorted(categories[prefix]):
            _print_array(name, present[name])
        print()

    for p in order:
        _print_cat(p)

    # Catch-all for any remaining categories not in `order`
    for prefix in sorted(categories):
        if prefix not in seen:
            _print_cat(prefix)


def _print_all_array_types(d3: D3plot) -> None:
    """Also list every known ArrayType name, marking which ones are present."""
    print(SEP)
    print("  ALL KNOWN ArrayType FIELDS  (✓ = present in this file)")
    print(SEP)
    for name in _all_array_type_names():
        val = getattr(ArrayType, name, None)
        arr = d3.arrays.get(val) if val is not None else None
        tick = "✓" if arr is not None and (not hasattr(arr, "__len__") or len(arr) > 0) else " "
        shape = f"  [{_shape_str(arr)}]" if arr is not None else ""
        print(f"  [{tick}]  {name}{shape}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a structured overview of all arrays in a d3plot file set."
    )
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Directory containing d3plot (and d3plot01, …) OR path to d3plot file.",
    )
    parser.add_argument(
        "--load-states", action="store_true",
        help="Also copy the first state file so state arrays (stress, disp, …) are visible.",
    )
    parser.add_argument(
        "--all-fields", action="store_true",
        help="At the end, print a complete tick-list of every known ArrayType name.",
    )
    args = parser.parse_args()

    if not args.src.exists():
        sys.exit(f"ERROR: path not found: {args.src}")

    print()
    print(SEP2)
    print("  d3plot_inspect.py  –  LS-DYNA D3plot field overview")
    print(SEP2)
    print(f"  Source : {args.src.resolve()}")
    print(f"  States : {'yes (first state file)' if args.load_states else 'header only'}")
    print()

    with tempfile.TemporaryDirectory(prefix="d3inspect_") as _tmp:
        tmp = Path(_tmp)
        try:
            d3 = _open_d3plot(args.src, args.load_states, tmp)
        except Exception as exc:
            sys.exit(f"ERROR opening d3plot: {exc}")

        _print_mesh_summary(d3)
        _print_parts(d3)
        _print_materials(d3)
        _print_arrays_by_category(d3)
        if args.all_fields:
            _print_all_array_types(d3)

    print(SEP2)
    print("  Done.")
    print(SEP2)
    print()


if __name__ == "__main__":
    main()
