"""
Extract the downsampled node set from a k-file + d3plot sequence and write one HDF5 file.

Node selection comes from k_file_downsample.py outputs; no re-sampling is done here.

HDF5 layout
-----------
/metadata/
    sampled_node_ids       (N,)      int64    k-file node IDs
    ref_positions          (N, 3)    float32  t=0 reference coordinates [mm]
    region_id              (N,)      int32    0–5 integer (see region_id_legend attr)
    region_label           (N,)      bytes    "barrier_fine" / "veh_contact" / …
    node_part_id           (N,)      int32    k-file PID
    node_part_name         (N,)      bytes    d3plot part name
    node_mat_label         (N,)      bytes    material title (*MAT_xxx_TITLE) or type keyword
    node_mat_type_name     (N,)      bytes    MAT keyword without *MAT_
    node_mat_rho           (N,)      float32  density          [ton/mm³]
    node_mat_E             (N,)      float32  Young's modulus  [MPa]
    node_mat_nu            (N,)      float32  Poisson's ratio
    node_mat_sigy          (N,)      float32  yield stress     [MPa]
    node_mass              (N,)      float32  nodal mass [ton]  (part_mass/n_nodes; 0 if unavailable)
    attrs: n_nodes, n_frames, frame_stride, source_dir, kfile, region_id_legend

/states/
    times                  (T,)      float64  [s]
    positions              (T, N, 3) float32  deformed xyz [mm]

Usage
-----
python dataset/k_file_to_h5.py \\
    --downsample-dir out/ \\
    --src /path/to/fem/T_lok_F_shape_barrier_9_3_60km \\
    --tmp /tmp/d3plot_tmp \\
    --out /path/to/output.h5 \\
    --frame-stride 5
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
from lasso.dyna import ArrayType, D3plot


# ── region label → integer id ──────────────────────────────────────────────
REGION_ID_MAP: dict[str, int] = {
    "force_keep":     0,
    "barrier_fine":   1,
    "barrier_coarse": 2,
    "veh_contact":    3,
    "veh_near":       4,
    "veh_far":        5,
}
REGION_ID_LEGEND = (
    "0=force_keep 1=barrier_fine 2=barrier_coarse "
    "3=veh_contact 4=veh_near 5=veh_far"
)


# ── material parser (adapted from d3plot_to_h5_dt.py) ─────────────────────

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


def parse_kfile_materials(path: Path) -> dict[str, dict]:
    """
    Parse *PART + *MAT_xxx sections from a k-file.
    Returns {part_name -> {type_name, type_id, label, rho, E, nu, sigy}}.
    Keyed by part title so it matches d3plot part_titles strings.
    """
    parts: list[dict] = []
    mats:  dict[int, dict] = {}

    try:
        with open(path, encoding="ascii", errors="ignore") as fh:
            lines = fh.readlines()
    except OSError as e:
        print(f"  WARNING: cannot read k-file {path}: {e}")
        return {}

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        kw  = raw.strip().upper()

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
                if len(tok) >= 3:
                    parts.append({"name": part_title, "mid": _ki(tok[2])})
                j += 1
            i = j
            continue

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
                        E    = _kf(tok[2]) if len(tok) > 2 else 0.0
                        nu   = _kf(tok[3]) if len(tok) > 3 else 0.0
                        sigy = _kf(tok[4]) if len(tok) > 4 else 0.0
                    mats[mid] = {
                        "type_name": mat_type,
                        "type_id":   _MAT_TYPE_ID_MAP.get(mat_type, _MAT_TYPE_UNKNOWN),
                        "label":     mat_label if mat_label else mat_type,
                        "rho": rho, "E": E, "nu": nu, "sigy": sigy,
                    }
                j += 1
            i = j
            continue

        i += 1

    name_to_props: dict[str, dict] = {}
    for p in parts:
        props = mats.get(p["mid"])
        if props is not None:
            name_to_props[p["name"]] = props
    return name_to_props


# ── d3plot helpers ─────────────────────────────────────────────────────────

def _decode(raw: object) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("ascii", errors="ignore").strip().strip("\x00")
    return str(raw).strip().strip("\x00")


def _suffix_number(path: Path) -> int:
    s = path.name.replace("d3plot", "", 1)
    return -1 if s == "" else (int(s) if s.isdigit() else 10 ** 12)


def _find_state_files(src: Path) -> list[Path]:
    files = [p for p in src.glob("d3plot*") if p.is_file() and p.name != "d3plot"]
    return sorted(files, key=_suffix_number)


def _copy_to_tmp(src_file: Path, tmp: Path) -> None:
    src_dir = src_file.parent
    (tmp / "d3plot").unlink(missing_ok=True)
    (tmp / "d3plot01").unlink(missing_ok=True)
    shutil.copy2(src_dir / "d3plot", tmp / "d3plot")
    shutil.copy2(src_file, tmp / "d3plot01")


def _close_d3(d3, tmp: Path) -> None:
    if d3 is not None:
        del d3
    (tmp / "d3plot").unlink(missing_ok=True)
    (tmp / "d3plot01").unlink(missing_ok=True)


def _scan_times(state_files: list[Path], tmp: Path) -> list[tuple[float, str, int]]:
    entries: list[tuple[float, str, int]] = []
    print("Pass 1/2 — scanning state times …")
    for src in state_files:
        _copy_to_tmp(src, tmp)
        try:
            d3 = D3plot(str(tmp / "d3plot"), state_array_filter=[ArrayType.global_timesteps])
            t_arr = d3.arrays.get(ArrayType.global_timesteps, np.array([0.0]))
            for s, t in enumerate(t_arr):
                entries.append((float(t), src.name, int(s)))
            print(f"  {src.name}: {len(t_arr)} state(s), "
                  f"t = {float(t_arr[0])*1e3:.2f}–{float(t_arr[-1])*1e3:.2f} ms")
            del d3
        except RuntimeError as e:
            if "endmark" in str(e).lower():
                print(f"  WARNING: {src.name} skipped ({e})")
            else:
                raise
        finally:
            _close_d3(None, tmp)
    return entries


def _select_frames(
    entries: list[tuple[float, str, int]], stride: int, limit: int | None
) -> list[tuple[float, str, int]]:
    entries = sorted(entries, key=lambda e: e[0])
    # deduplicate
    unique, last_t = [], None
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


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract downsampled nodes from k-file + d3plot → HDF5."
    )
    parser.add_argument("--downsample-dir", type=Path, required=True,
                        help="Output dir from k_file_downsample.py (contains sampled_node_ids.npy etc.)")
    parser.add_argument("--src",  type=Path, required=True,
                        help="Directory containing d3plot, d3plot01, d3plot02 …")
    parser.add_argument("--tmp",  type=Path, default=Path("/tmp/d3plot_tmp"),
                        help="Scratch directory for single-file copies.")
    parser.add_argument("--out",  type=Path, required=True,
                        help="Output HDF5 file path.")
    parser.add_argument("--kfile", type=Path, default=None,
                        help="k-file path for material metadata. Auto-detected from --src if absent.")
    parser.add_argument("--frame-stride", type=int, default=5,
                        help="Keep every N-th frame (default 5).")
    parser.add_argument("--frame-limit",  type=int, default=None,
                        help="Cap total frames kept (after stride).")
    args = parser.parse_args()

    args.tmp.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for old in args.tmp.glob("d3plot*"):
        old.unlink()

    # ── 1. Load downsample outputs ─────────────────────────────────────────
    dd = args.downsample_dir
    print(f"Loading downsample outputs from {dd} …")
    sampled_nids   = np.load(dd / "sampled_node_ids.npy")           # (N,) k-file node IDs
    node_ids_all   = np.load(dd / "node_ids.npy")                   # (N_full,) k-file nids in row order
    ref_coords_all = np.load(dd / "node_ref_coords.npy")            # (N_full, 3)
    ra             = json.load(open(dd / "region_assignment.json"))
    region_labels  = np.array(ra["region_labels"])                   # (N,) str
    node_part_ids  = np.load(dd / "sampled_node_part_ids.npy") \
        if (dd / "sampled_node_part_ids.npy").exists() \
        else np.zeros(len(sampled_nids), dtype=np.int32)

    N = len(sampled_nids)
    print(f"  {N:,} sampled nodes")

    # nid → d3plot row index
    nid_to_row = {int(nid): i for i, nid in enumerate(node_ids_all)}
    sampled_rows = np.array([nid_to_row[int(n)] for n in sampled_nids], dtype=np.int64)

    ref_positions = ref_coords_all[sampled_rows].astype(np.float32)  # (N, 3)
    region_id     = np.array(
        [REGION_ID_MAP.get(r, -1) for r in region_labels], dtype=np.int32
    )

    # ── 2. d3plot header: part titles and initial part mass ────────────────
    print("Loading d3plot header …")
    hdr_tmp = args.tmp / "d3plot"
    hdr_tmp.unlink(missing_ok=True)
    shutil.copy2(args.src / "d3plot", hdr_tmp)
    try:
        d3hdr = D3plot(
            str(hdr_tmp),
            state_array_filter=[ArrayType.global_timesteps, ArrayType.part_mass],
        )
    finally:
        hdr_tmp.unlink(missing_ok=True)

    part_titles_raw = d3hdr.arrays.get(ArrayType.part_titles, np.array([]))
    part_pids_raw   = d3hdr.arrays.get(ArrayType.part_titles_ids, np.array([]))
    part_names_list = [_decode(x) for x in part_titles_raw]

    # PID → part name
    pid_to_name: dict[int, str] = {
        int(pid): name for pid, name in zip(part_pids_raw, part_names_list)
    }

    # per-node part name
    node_part_name = np.array(
        [pid_to_name.get(int(pid), f"pid_{pid}") for pid in node_part_ids]
    )

    # part mass from first available state (part_mass shape: (n_states, n_parts) or (n_parts,))
    part_mass_arr = d3hdr.arrays.get(ArrayType.part_mass)
    pid_to_mass: dict[int, float] = {}
    if part_mass_arr is not None:
        pm = np.asarray(part_mass_arr)
        if pm.ndim == 2:
            pm = pm[0]          # first state
        for pid, mass_val in zip(part_pids_raw, pm):
            pid_to_mass[int(pid)] = float(mass_val)

    # node_mass: part_mass / nodes_in_that_part (uniform distribution)
    # count nodes per PID in sampled set
    pid_node_count: dict[int, int] = {}
    for pid in node_part_ids:
        pid_node_count[int(pid)] = pid_node_count.get(int(pid), 0) + 1

    node_mass = np.array([
        pid_to_mass.get(int(pid), 0.0) / max(pid_node_count.get(int(pid), 1), 1)
        for pid in node_part_ids
    ], dtype=np.float32)

    del d3hdr

    # ── 3. Material metadata from k-file ──────────────────────────────────
    kfile = args.kfile
    if kfile is None:
        candidates = sorted(args.src.glob("*.k")) + sorted(args.src.glob("*.key"))
        kfile = candidates[0] if candidates else None
    if kfile is not None:
        print(f"Parsing k-file for material props: {kfile} …")
        name_to_props = parse_kfile_materials(kfile)
        print(f"  resolved {len(name_to_props)} unique part-material mappings")
    else:
        print("WARNING: no k-file found — material fields will be zero-filled")
        name_to_props = {}

    mat_type_name = np.full(N, "UNKNOWN", dtype=object)
    mat_label     = np.full(N, "UNKNOWN", dtype=object)
    mat_rho       = np.zeros(N, dtype=np.float32)
    mat_E         = np.zeros(N, dtype=np.float32)
    mat_nu        = np.zeros(N, dtype=np.float32)
    mat_sigy      = np.zeros(N, dtype=np.float32)

    for i, name in enumerate(node_part_name):
        p = name_to_props.get(str(name))
        if p:
            mat_type_name[i] = p["type_name"]
            mat_label[i]     = p.get("label", p["type_name"])
            mat_rho[i]       = p["rho"]
            mat_E[i]         = p["E"]
            mat_nu[i]        = p["nu"]
            mat_sigy[i]      = p["sigy"]

    # ── 4. Frame selection ─────────────────────────────────────────────────
    state_files = _find_state_files(args.src)
    if not state_files:
        raise FileNotFoundError(f"No d3plot state files in {args.src}")
    all_entries = _scan_times(state_files, args.tmp)
    selected    = _select_frames(all_entries, args.frame_stride, args.frame_limit)
    n_frames    = len(selected)

    # ── 5. Create HDF5 ────────────────────────────────────────────────────
    print(f"\nCreating {args.out}  ({n_frames} frames × {N:,} nodes) …")
    with h5py.File(args.out, "w") as h5f:

        # /metadata
        mg = h5f.require_group("metadata")
        mg.create_dataset("sampled_node_ids",   data=sampled_nids.astype(np.int64))
        mg.create_dataset("ref_positions",       data=ref_positions)
        mg.create_dataset("region_id",           data=region_id)
        mg.create_dataset("region_label",        data=region_labels.astype("S"))
        mg.create_dataset("node_part_id",        data=node_part_ids.astype(np.int32))
        mg.create_dataset("node_part_name",      data=node_part_name.astype("S"))
        mg.create_dataset("node_mat_label",      data=mat_label.astype("S"))
        mg.create_dataset("node_mat_type_name",  data=mat_type_name.astype("S"))
        mg.create_dataset("node_mat_rho",        data=mat_rho)
        mg.create_dataset("node_mat_E",          data=mat_E)
        mg.create_dataset("node_mat_nu",         data=mat_nu)
        mg.create_dataset("node_mat_sigy",       data=mat_sigy)
        mg.create_dataset("node_mass",           data=node_mass)
        mg.attrs["n_nodes"]          = N
        mg.attrs["n_frames"]         = n_frames
        mg.attrs["frame_stride"]     = args.frame_stride
        mg.attrs["region_id_legend"] = REGION_ID_LEGEND
        mg.attrs["source_dir"]       = str(args.src.resolve())
        mg.attrs["kfile"]            = str(kfile.resolve()) if kfile else "not_provided"
        mg.attrs["downsample_dir"]   = str(dd.resolve())
        mg.attrs["units"]            = "mm, ton, s, MPa"
        mg.attrs["node_mass_note"]   = "part_mass[t=0] / n_nodes_in_part (uniform distribution)"

        # /states (pre-allocated, chunked + gzip)
        sg   = h5f.require_group("states")
        ct   = min(16, n_frames)
        cn   = min(4096, N)
        sg.create_dataset("times",
                          shape=(n_frames,),         dtype="float64",
                          chunks=(ct,),              compression="gzip", compression_opts=4)
        sg.create_dataset("positions",
                          shape=(n_frames, N, 3),    dtype="float32",
                          chunks=(ct, cn, 3),        compression="gzip", compression_opts=4)

        # ── 6. Extract positions frame by frame ────────────────────────────
        print("\nPass 2/2 — extracting positions …")
        state_file_map = {p.name: p for p in state_files}
        current_name: str | None = None
        d3 = None

        try:
            for fi, (t, fname, sidx) in enumerate(selected):
                if current_name != fname:
                    _close_d3(d3, args.tmp)
                    _copy_to_tmp(state_file_map[fname], args.tmp)
                    current_name = fname
                    print(f"  loading {fname}  (frame {fi+1}/{n_frames})")
                    d3 = D3plot(
                        str(args.tmp / "d3plot"),
                        state_array_filter=[
                            ArrayType.global_timesteps,
                            ArrayType.node_displacement,
                        ],
                    )

                disp_full = d3.arrays.get(ArrayType.node_displacement)
                if disp_full is None:
                    raise RuntimeError(f"node_displacement missing in {fname}")

                disp_sel = disp_full[sidx][sampled_rows].astype(np.float32)  # (N, 3)
                pos      = ref_positions + disp_sel                           # (N, 3)

                h5f["states/times"][fi]     = t
                h5f["states/positions"][fi] = pos

                if fi % 20 == 0 or fi == n_frames - 1:
                    print(f"  [{fi+1:>5}/{n_frames}]  t = {t*1e3:.3f} ms")

        finally:
            _close_d3(d3, args.tmp)

    # ── Summary ────────────────────────────────────────────────────────────
    size_mb = args.out.stat().st_size / 1e6
    print(f"\n{'─'*60}")
    print(f"Done.  {args.out}")
    print(f"Size        : {size_mb:.1f} MB")
    print(f"Nodes (N)   : {N:,}")
    print(f"Frames (T)  : {n_frames}  (stride={args.frame_stride})")
    print(f"Layout      : /metadata  +  /states/{{times, positions}}")
    print(f"{'─'*60}")

    # quick sanity: print region breakdown
    for label, rid in sorted(REGION_ID_MAP.items(), key=lambda x: x[1]):
        cnt = (region_id == rid).sum()
        print(f"  {label:<20} {cnt:>7,} nodes")


if __name__ == "__main__":
    main()
