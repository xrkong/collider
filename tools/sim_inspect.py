#!/home/kong/miniconda3/envs/dyna_builder/bin/python
"""
sim_inspect.py  --  Full LS-DYNA simulation overview (keyword file + d3plot).

Run:
    python3 tools/sim_inspect.py --src /path/to/sim/dir
    python3 tools/sim_inspect.py --src /path/to/sim/dir --load-states
    python3 tools/sim_inspect.py --src /path/to/sim/dir --load-states --verbose

K-FILE sections:
  Simulation settings, parameters, initial velocity
  DATABASE_EXTENT_BINARY (which fields are written to d3plot)
  Materials (grouped by type, key properties per unique title)
  Sections (shell thickness range, elform codes)
  Parts catalog (pid / secid / mid / name / material type+title)
  Contacts

D3PLOT sections (requires --load-states for state arrays like stress/strain):
  Mesh summary (node/element counts)
  Parts table -- element counts + per-part mass, kinetic energy, velocity
  Part x Material cross-reference (d3plot part_id -> k-file material)
  Plastic strain summary (solid + shell, per part and global stats)
  Solid history variables summary (neiph extra vars, e.g. concrete damage)
  All arrays present: shape / dtype / value range, grouped by category
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from lasso.dyna import ArrayType, D3plot

SEP  = "-" * 72
SEP2 = "=" * 72


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _decode(b) -> str:
    if isinstance(b, (bytes, bytearray)):
        return b.decode("ascii", errors="ignore").strip().strip("\x00")
    return str(b).strip().strip("\x00")


def _fields(line: str) -> list:
    """LS-DYNA data line -> tokens using 10-char fixed-width fields."""
    s = line.rstrip()
    if len(s) < 10:
        return s.split()
    chunks = [s[k:k+10].strip() for k in range(0, len(s), 10)]
    chunks = [c for c in chunks if c]
    return chunks if len(chunks) > 1 else s.split()


def _f(tok: str) -> float:
    try:
        return float(tok)
    except (ValueError, TypeError):
        return 0.0


def _i(tok: str) -> int:
    try:
        return int(tok.split(".")[0])
    except (ValueError, TypeError):
        return 0


def _stats(arr: np.ndarray) -> str:
    """Return 'min / mean / max / std' string for a numeric array."""
    if arr is None or arr.size == 0:
        return "empty"
    f = arr.astype(np.float64).ravel()
    return (f"min={float(f.min()):.4g}  mean={float(f.mean()):.4g}"
            f"  max={float(f.max()):.4g}  std={float(f.std()):.4g}")


def _shape_str(arr) -> str:
    return "x".join(str(d) for d in arr.shape)


def _range_str(arr) -> str:
    try:
        f = arr.astype(np.float64).ravel()
        if f.size == 0:
            return "empty"
        return f"[{float(f.min()):.4g}, {float(f.max()):.4g}]"
    except Exception:
        return "n/a"


def _print_arr(name: str, arr, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}{name:<52}  shape={_shape_str(arr):<22}"
          f"  dtype={str(arr.dtype):<8}  range={_range_str(arr)}")


# ===========================================================================
# K-FILE PARSER
# ===========================================================================

def _parse_kfile(path: Path) -> dict:
    data: dict = {
        "title": "",
        "parameters": {},
        "termination": {},
        "timestep": {},
        "initial_velocity": [],
        "db_extent": {},
        "parts": [],        # {pid, secid, mid, name}
        "materials": [],    # {mid, kw, title, props}
        "sections": defaultdict(list),
        "contacts": [],
    }
    params: dict = {}

    def _resolve(s: str) -> str:
        for k, v in params.items():
            s = re.sub(rf"&{k}", v, s, flags=re.IGNORECASE)
        return s

    with open(path, encoding="ascii", errors="ignore") as fh:
        lines = fh.readlines()

    def _nxt(i: int):
        j = i + 1
        while j < len(lines):
            l = lines[j].rstrip("\n")
            if l and not l.startswith("$"):
                return l
            j += 1
        return None

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        kw  = raw.strip().upper()

        if kw == "*TITLE":
            j = i + 1
            while j < len(lines) and lines[j].startswith("$"):
                j += 1
            if j < len(lines):
                data["title"] = lines[j].strip()
            i = j + 1
            continue

        if kw in ("*PARAMETER", "*PARAMETER_EXPRESSION"):
            j = i + 1
            while j < len(lines):
                l = lines[j].rstrip("\n")
                if l.startswith("*"):
                    break
                if l.startswith("$"):
                    j += 1
                    continue
                parts = l.split()
                if len(parts) >= 3 and parts[0].upper() in ("R", "I"):
                    name = parts[1].upper()
                    params[name] = parts[2]
                    data["parameters"][name] = parts[2]
                j += 1
            i = j
            continue

        if kw == "*CONTROL_TERMINATION":
            l2 = _nxt(i)
            if l2:
                tok = _fields(l2)
                data["termination"] = {
                    "endtim": _f(tok[0]) if tok else 0.0,
                    "endcyc": _i(tok[1]) if len(tok) > 1 else 0,
                }
            i += 1
            continue

        if kw == "*CONTROL_TIMESTEP":
            l2 = _nxt(i)
            if l2:
                tok = _fields(l2)
                data["timestep"] = {
                    "dtinit": _f(tok[0]) if tok else 0.0,
                    "tssfac": _f(tok[1]) if len(tok) > 1 else 0.0,
                }
            i += 1
            continue

        if kw == "*INITIAL_VELOCITY_GENERATION":
            l2 = _nxt(i)
            if l2:
                tok = _fields(l2)
                data["initial_velocity"].append({
                    "id":   _i(tok[0]) if tok else 0,
                    "styp": _i(tok[1]) if len(tok) > 1 else 0,
                    "vx": _resolve(tok[3]) if len(tok) > 3 else "0",
                    "vy": _resolve(tok[4]) if len(tok) > 4 else "0",
                    "vz": _resolve(tok[5]) if len(tok) > 5 else "0",
                })
            i += 1
            continue

        if kw == "*DATABASE_EXTENT_BINARY":
            j, row = i + 1, 0
            while j < len(lines) and row < 2:
                l = lines[j].rstrip("\n")
                if l.startswith("*"):
                    break
                if l.startswith("$"):
                    j += 1
                    continue
                tok = _fields(l)
                if row == 0:
                    data["db_extent"] = {
                        "neiph":  _i(tok[0]) if tok else 0,
                        "neips":  _i(tok[1]) if len(tok) > 1 else 0,
                        "strflg": _i(tok[3]) if len(tok) > 3 else 0,
                        "sigflg": _i(tok[4]) if len(tok) > 4 else 0,
                        "epsflg": _i(tok[5]) if len(tok) > 5 else 0,
                        "rltflg": _i(tok[6]) if len(tok) > 6 else 0,
                        "engflg": _i(tok[7]) if len(tok) > 7 else 0,
                    }
                elif row == 1:
                    data["db_extent"].update({
                        "ieverp": _i(tok[1]) if len(tok) > 1 else 0,
                        "therm":  _i(tok[7]) if len(tok) > 7 else 0,
                    })
                row += 1
                j += 1
            i = j
            continue

        if kw == "*PART":
            j, title = i + 1, ""
            while j < len(lines):
                l = lines[j].rstrip("\n")
                if l.startswith("$"):
                    j += 1
                    continue
                title = l.strip()
                j += 1
                break
            while j < len(lines):
                l = lines[j].rstrip("\n")
                if l.startswith("$"):
                    j += 1
                    continue
                if l.startswith("*"):
                    break
                tok = _fields(l)
                if len(tok) >= 3:
                    data["parts"].append({
                        "pid":   _i(tok[0]),
                        "secid": _i(tok[1]),
                        "mid":   _i(tok[2]),
                        "name":  title,
                    })
                j += 1
                break
            i = j
            continue

        if raw.startswith("*MAT_"):
            mat_kw    = raw.strip()
            has_title = mat_kw.upper().endswith("_TITLE")
            mat_type  = re.sub(r"_TITLE$", "", mat_kw, flags=re.IGNORECASE).upper()
            j, mat_title = i + 1, ""
            if has_title:
                while j < len(lines):
                    l = lines[j].rstrip("\n")
                    if l.startswith("$"):
                        j += 1
                        continue
                    mat_title = l.strip()
                    j += 1
                    break
            props: dict = {}
            while j < len(lines):
                l = lines[j].rstrip("\n")
                if l.startswith("$"):
                    j += 1
                    continue
                if l.startswith("*"):
                    break
                tok = _fields(l)
                if len(tok) >= 2:
                    props["mid"] = _f(tok[0])
                    props["ro"]  = _f(tok[1])
                    if ("RIGID" in mat_type or "ELASTIC" in mat_type) and len(tok) >= 4:
                        props["e"]  = _f(tok[2])
                        props["pr"] = _f(tok[3])
                    elif "PLASTICITY" in mat_type and len(tok) >= 6:
                        props["e"]    = _f(tok[2])
                        props["pr"]   = _f(tok[3])
                        props["sigy"] = _f(tok[4])
                        props["etan"] = _f(tok[5])
                        props["fail"] = _f(tok[6]) if len(tok) > 6 else 0.0
                    elif "BLATZ" in mat_type and len(tok) >= 3:
                        props["g"]   = _f(tok[2])
                    elif "LOW_DENSITY_FOAM" in mat_type and len(tok) >= 3:
                        props["e"]   = _f(tok[2])
                    elif "CONCRETE" in mat_type and len(tok) >= 3:
                        props["pr"]  = _f(tok[2])
                j += 1
                break
            data["materials"].append({
                "mid": int(props.get("mid", 0)),
                "kw":  mat_type,
                "title": mat_title,
                "props": props,
            })
            i = j
            continue

        if raw.startswith("*SECTION_"):
            sec_kw   = re.sub(r"_TITLE$", "", raw.strip(), flags=re.IGNORECASE).upper()
            sec_type = sec_kw.replace("*SECTION_", "")
            j = i + 1
            if raw.upper().strip().endswith("_TITLE"):
                while j < len(lines) and lines[j].startswith("$"):
                    j += 1
                j += 1
            props_s: dict = {}
            rows = 0
            while j < len(lines) and rows < 2:
                l = lines[j].rstrip("\n")
                if l.startswith("$"):
                    j += 1
                    continue
                if l.startswith("*"):
                    break
                tok = _fields(l)
                if rows == 0 and len(tok) >= 2:
                    props_s["secid"]  = _i(tok[0])
                    props_s["elform"] = _i(tok[1])
                elif rows == 1 and sec_type == "SHELL" and tok:
                    props_s["t"] = _f(tok[0])
                rows += 1
                j += 1
            data["sections"][sec_type].append(props_s)
            i = j
            continue

        if raw.startswith("*CONTACT_"):
            ctype = re.sub(r"_ID$|_TITLE$", "", raw.strip(), flags=re.IGNORECASE).upper()
            data["contacts"].append(ctype)
            i += 1
            continue

        i += 1

    for iv in data["initial_velocity"]:
        for k in ("vx", "vy", "vz"):
            try:
                iv[k] = str(float(_resolve(iv[k])))
            except Exception:
                pass
    data["parameters"] = params
    return data


# ===========================================================================
# K-FILE PRINTERS
# ===========================================================================

def _k_sim_settings(d: dict) -> None:
    print(SEP2)
    print("  [K-FILE]  SIMULATION SETTINGS")
    print(SEP2)
    if d["title"]:
        print(f"  Title         : {d['title']}")
    t  = d.get("termination", {})
    ts = d.get("timestep", {})
    if t:
        print(f"  End time      : {t.get('endtim', '?')} s   "
              f"endcyc={t.get('endcyc', 0)}")
    if ts:
        print(f"  DT init/tssfac: {ts.get('dtinit', 0)} / {ts.get('tssfac', 0)}")
    params = d.get("parameters", {})
    if params:
        print()
        print("  PARAMETERS")
        for name, val in params.items():
            extra = ""
            if "VEL" in name.upper():
                try:
                    extra = f"  => {float(val)/1000*3.6:.1f} km/h"
                except Exception:
                    pass
            print(f"    {name:<14} = {val}{extra}")
    ivs = d.get("initial_velocity", [])
    if ivs:
        print()
        print("  INITIAL VELOCITY")
        for iv in ivs:
            vx = iv.get("vx", "0")
            try:
                vx_f   = float(vx)
                vx_str = f"{vx_f:.2f} mm/s  ({vx_f/1000*3.6:.1f} km/h)"
            except Exception:
                vx_str = vx
            print(f"    set_id={iv['id']}  styp={iv['styp']}  "
                  f"vx={vx_str}  vy={iv['vy']}  vz={iv['vz']}")
    print()


def _k_db_extent(d: dict) -> None:
    db = d.get("db_extent", {})
    if not db:
        return
    print(SEP)
    print("  [K-FILE]  DATABASE_EXTENT_BINARY  (what lands in d3plot)")
    print(SEP)

    def _flag(label: str, key: str, legend: dict) -> None:
        v = db.get(key, 0)
        print(f"    {label:<18}: {v}  ->  {legend.get(v, str(v))}")

    _flag("neiph",  "neiph",  {0: "no extra solid history vars"})
    _flag("neips",  "neips",  {0: "no extra shell history vars"})
    _flag("strflg", "strflg", {0: "no strain tensor", 1: "strain tensor written"})
    _flag("sigflg", "sigflg", {1: "local stress", 2: "global stress tensor"})
    _flag("epsflg", "epsflg", {0: "no eff. plastic strain", 1: "eff. plastic strain written"})
    _flag("rltflg", "rltflg", {1: "local strain", 2: "global strain tensor"})
    _flag("engflg", "engflg", {0: "no energy", 1: "internal+kinetic energy written"})
    _flag("ieverp", "ieverp", {0: "all states in one d3plot",
                                1: "each state -> separate d3plot file"})
    _flag("therm",  "therm",  {0: "no temperatures", 1: "temperatures written"})
    neiph = db.get("neiph", 0)
    if neiph > 0:
        print(f"\n    => neiph={neiph}: {neiph} extra integration-point history"
              " variables per solid element (e.g. concrete damage state vars)")
    print()


def _k_materials(d: dict, verbose: bool = False) -> None:
    mats = d.get("materials", [])
    if not mats:
        return
    by_type: dict = defaultdict(list)
    for m in mats:
        canonical = re.sub(r"^\*MAT_", "", m["kw"])
        by_type[canonical].append(m)

    print(SEP)
    print(f"  [K-FILE]  MATERIALS  ({len(mats)} total)")
    print(SEP)
    print("  Count by type:")
    for typ, lst in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"    {len(lst):>4}x  {typ}")
    print()
    print("  Unique titles per type (representative properties):")
    for typ, lst in sorted(by_type.items(), key=lambda x: -len(x[1])):
        by_title: dict = {}
        for m in lst:
            t = m["title"] or "(untitled)"
            if t not in by_title:
                by_title[t] = m
        print(f"\n  [{typ}]  {len(lst)} instances")
        for title, m in by_title.items():
            p   = m["props"]
            row = f"    '{title}'"
            if p.get("ro")   is not None: row += f"  rho={p['ro']:.3g} t/mm3"
            if p.get("e")    is not None: row += f"  E={p['e']:.4g} MPa"
            if p.get("pr")   is not None: row += f"  nu={p['pr']:.3g}"
            if p.get("sigy") is not None: row += f"  sigy={p['sigy']:.4g} MPa"
            if p.get("etan") and p["etan"] != 0: row += f"  Etan={p['etan']:.3g}"
            if p.get("fail") and p["fail"] != 0: row += f"  fail={p['fail']:.3g}"
            if p.get("g")    is not None: row += f"  G={p['g']:.3g} MPa"
            print(row)
    print()


def _k_sections(d: dict) -> None:
    secs = d.get("sections", {})
    if not secs:
        return
    print(SEP)
    print("  [K-FILE]  SECTIONS")
    print(SEP)
    for stype, slist in sorted(secs.items()):
        print(f"  [{stype}]  {len(slist)} instances")
        if stype == "SHELL":
            ts = sorted({s["t"] for s in slist if "t" in s})
            if ts:
                print(f"    Thickness range : {min(ts):.3g} - {max(ts):.3g} mm"
                      f"  ({len(ts)} unique values)")
                print(f"    Values          : {ts[:30]}{'...' if len(ts) > 30 else ''}")
            forms = sorted({s.get("elform", 0) for s in slist})
            print(f"    Elform codes    : {forms}")
        elif stype in ("SOLID", "BEAM"):
            forms = sorted({s.get("elform", 0) for s in slist})
            print(f"    Elform codes    : {forms}")
    print()


def _k_parts(d: dict, verbose: bool = False) -> None:
    parts = d.get("parts", [])
    mats  = {m["mid"]: m for m in d.get("materials", [])}

    def _mdesc(mid: int) -> str:
        m = mats.get(mid)
        if not m:
            return f"mid={mid}"
        typ = re.sub(r"^\*MAT_", "", m["kw"])
        return f"{typ}  '{m['title']}'" if m["title"] else typ

    print(SEP)
    print(f"  [K-FILE]  PARTS CATALOG  ({len(parts)} total)")
    print(SEP)

    if not verbose:
        # prefix groups
        prefixes: dict = defaultdict(int)
        for p in parts:
            pref = re.split(r"[_\s]", p["name"])[0] if p["name"] else "?"
            prefixes[pref] += 1
        top = sorted(prefixes.items(), key=lambda x: -x[1])[:20]
        print("  Name-prefix groups (top 20):")
        for pref, cnt in top:
            print(f"    {cnt:>4}x  {pref}")
        print()
        print(f"  {'pid':>9}  {'mid':>9}  {'material type + title':<50}  name")
        print(f"  {'-'*9}  {'-'*9}  {'-'*50}  {'-'*45}")
        for p in parts[:20]:
            print(f"  {p['pid']:>9}  {p['mid']:>9}  {_mdesc(p['mid']):<50}  {p['name']}")
        if len(parts) > 40:
            print(f"  ... ({len(parts)-40} parts omitted, use --verbose for full table)")
        for p in parts[-20:]:
            print(f"  {p['pid']:>9}  {p['mid']:>9}  {_mdesc(p['mid']):<50}  {p['name']}")
    else:
        print(f"  {'pid':>9}  {'secid':>9}  {'mid':>9}  {'material type + title':<50}  name")
        print(f"  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*50}  {'-'*45}")
        for p in parts:
            print(f"  {p['pid']:>9}  {p['secid']:>9}  {p['mid']:>9}"
                  f"  {_mdesc(p['mid']):<50}  {p['name']}")
    print()


def _k_contacts(d: dict) -> None:
    contacts = d.get("contacts", [])
    if not contacts:
        return
    print(SEP)
    print("  [K-FILE]  CONTACTS")
    print(SEP)
    counts: dict = defaultdict(int)
    for c in contacts:
        counts[c] += 1
    for ctype, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {n:>3}x  {ctype}")
    print()


# ===========================================================================
# D3PLOT HELPERS
# ===========================================================================

def _open_d3plot(src: Path, load_states: bool, tmp: Path) -> D3plot:
    hdr = src / "d3plot"
    if not hdr.exists():
        sys.exit(f"ERROR: no d3plot file in {src}")
    shutil.copy2(hdr, tmp / "d3plot")
    if load_states:
        state_files = sorted(
            [p for p in src.glob("d3plot0*") if p.is_file()],
            key=lambda p: p.name,
        )
        if state_files:
            shutil.copy2(state_files[0], tmp / "d3plot01")
            print(f"  (state data: {state_files[0].name})")
    return D3plot(str(tmp / "d3plot"))


def _all_array_names() -> list:
    return sorted(n for n in dir(ArrayType) if not n.startswith("_"))


# ===========================================================================
# D3PLOT PRINTERS
# ===========================================================================

def _d3_mesh_summary(d3: D3plot) -> None:
    print(SEP2)
    print("  [D3PLOT]  MESH SUMMARY")
    print(SEP2)

    def _n(key) -> int:
        a = d3.arrays.get(key)
        return len(a) if a is not None else 0

    coords  = d3.arrays.get(ArrayType.node_coordinates)
    n_nodes = len(coords) if coords is not None else 0
    items = [
        ("nodes",           n_nodes),
        ("solid elements",  _n(ArrayType.element_solid_node_indexes)),
        ("shell elements",  _n(ArrayType.element_shell_node_indexes)),
        ("beam elements",   _n(ArrayType.element_beam_node_indexes)),
        ("thick-shell",     _n(ArrayType.element_tshell_node_indexes)),
        ("parts (titles)",  _n(ArrayType.part_titles)),
        ("part_ids (incl. extras)", _n(ArrayType.part_ids)),
    ]
    for label, n in items:
        if n:
            print(f"    {label:<30}: {n:,}")
    print()


def _d3_parts_full(d3: D3plot, kparts: list, kmats: dict) -> None:
    """
    Comprehensive parts table:
      d3plot part_idx | part_id | elem counts | mass | KE | vel | k-file material
    """
    titles    = d3.arrays.get(ArrayType.part_titles)
    title_ids = d3.arrays.get(ArrayType.part_titles_ids)
    if titles is None:
        print("  (no part_titles in d3plot)")
        return

    n = len(titles)
    ids = np.asarray(title_ids) if title_ids is not None else np.arange(1, n + 1)

    # element counts per part
    def _ecount(pi_arr, n_parts: int) -> np.ndarray:
        c = np.zeros(n_parts, dtype=np.int64)
        if pi_arr is not None:
            for idx in pi_arr:
                if 0 <= int(idx) < n_parts:
                    c[int(idx)] += 1
        return c

    solid_c  = _ecount(d3.arrays.get(ArrayType.element_solid_part_indexes),  n)
    shell_c  = _ecount(d3.arrays.get(ArrayType.element_shell_part_indexes),  n)
    beam_c   = _ecount(d3.arrays.get(ArrayType.element_beam_part_indexes),   n)

    # per-part mass, KE, velocity (indexed by part_ids, not part_titles idx)
    part_ids_arr  = d3.arrays.get(ArrayType.part_ids)          # shape (P2,)
    part_mass_arr = d3.arrays.get(ArrayType.part_mass)         # shape (T, P2)
    part_ke_arr   = d3.arrays.get(ArrayType.part_kinetic_energy)
    part_vel_arr  = d3.arrays.get(ArrayType.part_velocity)     # shape (T, P2, 3)
    part_xref_arr = d3.arrays.get(ArrayType.part_ids_cross_references)  # maps P2 -> part_title_idx

    # build lookup: part_title_idx -> row in part_ids_arr
    xref_map: dict = {}
    if part_xref_arr is not None and part_ids_arr is not None:
        for row_idx, title_idx in enumerate(part_xref_arr):
            xref_map[int(title_idx)] = row_idx

    # k-file pid -> material description
    kpid_to_mat: dict = {}
    for kp in kparts:
        m = kmats.get(kp["mid"])
        if m:
            typ = re.sub(r"^\*MAT_", "", m["kw"])
            kpid_to_mat[kp["pid"]] = f"{typ}  '{m['title']}'" if m["title"] else typ
        else:
            kpid_to_mat[kp["pid"]] = f"mid={kp['mid']}"

    print(SEP)
    print(f"  [D3PLOT]  PARTS  ({n} titles)")
    print(SEP)
    hdr = (f"  {'idx':>5}  {'part_id':>8}  {'solid':>7}  {'shell':>7}  {'beam':>5}"
           f"  {'mass(t)':>10}  {'KE(J)':>12}  {'|vel|(mm/s)':>11}"
           f"  material (from k-file)  ->  name")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i in range(n):
        pid  = int(ids[i])
        name = _decode(titles[i])
        sc, shc, bc = int(solid_c[i]), int(shell_c[i]), int(beam_c[i])

        # mass / KE / vel from part_ids arrays
        mass_str = vel_str = ke_str = "  n/a"
        row_idx = xref_map.get(i)
        if row_idx is not None:
            if part_mass_arr is not None and part_mass_arr.ndim >= 2:
                mass_str = f"{float(part_mass_arr[-1, row_idx]):.4g}"
            if part_ke_arr is not None and part_ke_arr.ndim >= 2:
                ke_str = f"{float(part_ke_arr[-1, row_idx]):.4g}"
            if part_vel_arr is not None and part_vel_arr.ndim >= 3:
                v = part_vel_arr[-1, row_idx]
                vel_str = f"{float(np.linalg.norm(v)):.4g}"

        mat_desc = kpid_to_mat.get(pid, "")
        print(f"  {i:>5}  {pid:>8}  {sc:>7}  {shc:>7}  {bc:>5}"
              f"  {mass_str:>10}  {ke_str:>12}  {vel_str:>11}"
              f"  {mat_desc:<55}  {name}")
    print()


def _d3_plastic_strain(d3: D3plot, kparts: list, kmats: dict) -> None:
    """
    Detailed plastic strain report:
      - Global stats for solid and shell eff. plastic strain
      - Per-part breakdown (max plastic strain per part, sorted descending)
      - Solid history variables (neiph extra vars)
    """
    titles    = d3.arrays.get(ArrayType.part_titles)
    title_ids = d3.arrays.get(ArrayType.part_titles_ids)

    # k-file pid -> name
    pid_to_name: dict = {}
    for kp in kparts:
        pid_to_name[kp["pid"]] = kp["name"]
    pid_to_mat: dict = {}
    for kp in kparts:
        m = kmats.get(kp["mid"])
        if m:
            typ = re.sub(r"^\*MAT_", "", m["kw"])
            pid_to_mat[kp["pid"]] = f"{typ}  '{m['title']}'" if m["title"] else typ
        else:
            pid_to_mat[kp["pid"]] = f"mid={kp['mid']}"

    print(SEP)
    print("  [D3PLOT]  PLASTIC STRAIN & SOLID HISTORY VARIABLES")
    print(SEP)

    # ---- SOLID effective plastic strain ----
    solid_eps = d3.arrays.get(ArrayType.element_solid_effective_plastic_strain)
    if solid_eps is not None and solid_eps.size > 0:
        # shape: (T, n_elem, n_ip) or (T, n_elem, n_ip, 1) -- take last state, mean over IPs
        s = solid_eps[-1]           # (n_elem, ...) at last loaded state
        s = s.reshape(s.shape[0], -1).mean(axis=1)  # (n_elem,)
        print(f"\n  SOLID eff. plastic strain  (shape={solid_eps.shape})")
        print(f"    Global  : {_stats(s)}")

        # per-part breakdown
        solid_pi = d3.arrays.get(ArrayType.element_solid_part_indexes)
        if solid_pi is not None and titles is not None:
            n_parts = len(titles)
            ids     = np.asarray(title_ids) if title_ids is not None else np.arange(1, n_parts+1)
            part_max  = np.zeros(n_parts, dtype=np.float32)
            part_mean = np.zeros(n_parts, dtype=np.float32)
            part_cnt  = np.zeros(n_parts, dtype=np.int64)
            for ei, pi in enumerate(solid_pi):
                pi = int(pi)
                if 0 <= pi < n_parts:
                    v = float(s[ei])
                    if v > part_max[pi]:
                        part_max[pi] = v
                    part_mean[pi] += v
                    part_cnt[pi]  += 1
            nonzero = [(i, part_max[i], part_mean[i]/part_cnt[i] if part_cnt[i] else 0)
                       for i in range(n_parts) if part_max[i] > 0]
            nonzero.sort(key=lambda x: -x[1])
            if nonzero:
                print(f"    Parts with non-zero solid plastic strain  "
                      f"({len(nonzero)} / {n_parts}):")
                print(f"    {'idx':>5}  {'part_id':>8}  {'max_eps':>10}  {'mean_eps':>10}"
                      f"  {'n_elem':>7}  material  ->  name")
                print(f"    {'-'*5}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*55}")
                for i, pmax, pmean in nonzero:
                    pid  = int(ids[i])
                    name = _decode(titles[i])
                    mat  = pid_to_mat.get(pid, "")
                    print(f"    {i:>5}  {pid:>8}  {pmax:>10.4g}  {pmean:>10.4g}"
                          f"  {int(part_cnt[i]):>7}  {mat:<55}  {name}")
    else:
        print("\n  SOLID eff. plastic strain  : not present (load state file with --load-states)")

    # ---- SHELL effective plastic strain ----
    shell_eps = d3.arrays.get(ArrayType.element_shell_effective_plastic_strain)
    if shell_eps is not None and shell_eps.size > 0:
        # shape: (T, n_elem, n_layer) -- take last state, mean over layers
        s = shell_eps[-1]
        s = s.reshape(s.shape[0], -1).mean(axis=1)
        print(f"\n  SHELL eff. plastic strain  (shape={shell_eps.shape})")
        print(f"    Global  : {_stats(s)}")

        shell_pi = d3.arrays.get(ArrayType.element_shell_part_indexes)
        if shell_pi is not None and titles is not None:
            n_parts = len(titles)
            ids     = np.asarray(title_ids) if title_ids is not None else np.arange(1, n_parts+1)
            part_max  = np.zeros(n_parts, dtype=np.float32)
            part_mean = np.zeros(n_parts, dtype=np.float32)
            part_cnt  = np.zeros(n_parts, dtype=np.int64)
            for ei, pi in enumerate(shell_pi):
                pi = int(pi)
                if 0 <= pi < n_parts:
                    v = float(s[ei])
                    if v > part_max[pi]:
                        part_max[pi] = v
                    part_mean[pi] += v
                    part_cnt[pi]  += 1
            nonzero = [(i, part_max[i], part_mean[i]/part_cnt[i] if part_cnt[i] else 0)
                       for i in range(n_parts) if part_max[i] > 0]
            nonzero.sort(key=lambda x: -x[1])
            if nonzero:
                print(f"    Parts with non-zero shell plastic strain  "
                      f"({len(nonzero)} / {n_parts}):")
                print(f"    {'idx':>5}  {'part_id':>8}  {'max_eps':>10}  {'mean_eps':>10}"
                      f"  {'n_elem':>7}  material  ->  name")
                print(f"    {'-'*5}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*55}")
                for i, pmax, pmean in nonzero:
                    pid  = int(ids[i])
                    name = _decode(titles[i])
                    mat  = pid_to_mat.get(pid, "")
                    print(f"    {i:>5}  {pid:>8}  {pmax:>10.4g}  {pmean:>10.4g}"
                          f"  {int(part_cnt[i]):>7}  {mat:<55}  {name}")
    else:
        print("\n  SHELL eff. plastic strain  : not present")

    # ---- SOLID history variables (neiph extra vars, e.g. concrete damage) ----
    solid_hv = d3.arrays.get(ArrayType.element_solid_history_variables)
    if solid_hv is not None and solid_hv.size > 0:
        # shape: (T, n_elem, n_ip, n_hist) or (T, n_elem, 1, n_hist)
        s = solid_hv[-1]   # (n_elem, n_ip, n_hist)
        # collapse ip axis
        if s.ndim == 3:
            s_mean = s.mean(axis=1)  # (n_elem, n_hist)
        else:
            s_mean = s.reshape(s.shape[0], -1)
        n_hist = s_mean.shape[-1]
        print(f"\n  SOLID history variables  (shape={solid_hv.shape})"
              f"  -- {n_hist} var(s) per element (neiph={n_hist})")
        print(f"  These correspond to extra integration-point variables written")
        print(f"  by DATABASE_EXTENT_BINARY neiph= -- for concrete: damage state,")
        print(f"  effective plastic strain, damage scalar, etc.")
        for hv_idx in range(n_hist):
            col = s_mean[:, hv_idx]
            print(f"    hist_var[{hv_idx}]  : {_stats(col)}")

        # per-part breakdown for each history variable
        solid_pi = d3.arrays.get(ArrayType.element_solid_part_indexes)
        if solid_pi is not None and titles is not None:
            n_parts = len(titles)
            ids     = np.asarray(title_ids) if title_ids is not None else np.arange(1, n_parts+1)
            print()
            print(f"  History var max per part (parts with any nonzero value):")
            for hv_idx in range(n_hist):
                col   = s_mean[:, hv_idx]
                pmaxv = np.zeros(n_parts, dtype=np.float32)
                pcnt  = np.zeros(n_parts, dtype=np.int64)
                for ei, pi in enumerate(solid_pi):
                    pi = int(pi)
                    if 0 <= pi < n_parts:
                        v = float(col[ei])
                        if v > pmaxv[pi]:
                            pmaxv[pi] = v
                        pcnt[pi] += 1
                nonzero = [(i, pmaxv[i]) for i in range(n_parts) if pmaxv[i] > 0]
                nonzero.sort(key=lambda x: -x[1])
                if nonzero:
                    print(f"\n    hist_var[{hv_idx}]  ({len(nonzero)} parts with nonzero):")
                    for i, pmax in nonzero[:30]:
                        pid  = int(ids[i])
                        name = _decode(titles[i])
                        mat  = pid_to_mat.get(pid, "")
                        print(f"      part_idx={i:>4}  pid={pid:>8}  max={pmax:.4g}"
                              f"  {mat:<50}  {name}")
                    if len(nonzero) > 30:
                        print(f"      ... ({len(nonzero)-30} more parts)")
    else:
        print("\n  SOLID history variables    : not present")

    # ---- SOLID stress ----
    solid_stress = d3.arrays.get(ArrayType.element_solid_stress)
    if solid_stress is not None and solid_stress.size > 0:
        s = solid_stress[-1]  # (n_elem, n_ip, 6) or similar
        s6 = s.reshape(s.shape[0], -1, 6).mean(axis=1) if s.ndim == 3 else s.reshape(-1, 6)
        labels = ["sxx", "syy", "szz", "sxy", "syz", "sxz"]
        print(f"\n  SOLID stress  (shape={solid_stress.shape})  [MPa]")
        for ci, label in enumerate(labels):
            print(f"    {label}  : {_stats(s6[:, ci])}")
    else:
        print("\n  SOLID stress               : not present / all-zero")

    # ---- SHELL stress ----
    shell_stress = d3.arrays.get(ArrayType.element_shell_stress)
    if shell_stress is not None and shell_stress.size > 0:
        s = shell_stress[-1]
        s6 = s.reshape(s.shape[0], -1, 6).mean(axis=1) if s.ndim >= 3 else s
        labels = ["sxx", "syy", "szz", "sxy", "syz", "sxz"]
        print(f"\n  SHELL stress  (shape={shell_stress.shape})  [MPa]")
        for ci, label in enumerate(labels[:s6.shape[-1]]):
            print(f"    {label}  : {_stats(s6[:, ci])}")
    else:
        print("\n  SHELL stress               : not present")

    print()


def _d3_arrays_all(d3: D3plot) -> None:
    """Print every present array grouped by prefix, with shape/dtype/range."""
    categories: dict = {}
    present:    dict = {}
    for name in _all_array_names():
        val = getattr(ArrayType, name, None)
        if val is None:
            continue
        arr = d3.arrays.get(val)
        if arr is None:
            continue
        prefix = name.split("_")[0]
        categories.setdefault(prefix, []).append(name)
        present[name] = arr

    order = ["node", "element", "global", "part", "sph", "airbag", "rigid", "contact"]
    seen  = set()

    def _cat(prefix: str) -> None:
        if prefix not in categories:
            return
        seen.add(prefix)
        print(SEP)
        print(f"  [D3PLOT]  ARRAYS  [{prefix.upper()}]")
        print(SEP)
        for name in sorted(categories[prefix]):
            _print_arr(name, present[name])
        print()

    for p in order:
        _cat(p)
    for prefix in sorted(categories):
        if prefix not in seen:
            _cat(prefix)


def _d3_fields_checklist(d3: D3plot) -> None:
    """Full tick-list of every known ArrayType name."""
    print(SEP)
    print("  [D3PLOT]  ALL KNOWN ArrayType FIELDS  (Y = present in this file)")
    print(SEP)
    for name in _all_array_names():
        val = getattr(ArrayType, name, None)
        arr = d3.arrays.get(val) if val is not None else None
        tick  = "Y" if arr is not None and (not hasattr(arr, "__len__") or len(arr) > 0) else " "
        shape = f"  [{_shape_str(arr)}]" if arr is not None else ""
        print(f"  [{tick}]  {name}{shape}")
    print()


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full LS-DYNA simulation overview: keyword file + d3plot."
    )
    ap.add_argument("--src",   type=Path, default=None,
                    help="Simulation directory (auto-finds .k and d3plot).")
    ap.add_argument("--kfile", type=Path, default=None,
                    help="Explicit path to .k file.")
    ap.add_argument("--d3src", type=Path, default=None,
                    help="Explicit path to d3plot directory.")
    ap.add_argument("--load-states", action="store_true",
                    help="Load first state file so stress/strain arrays appear.")
    ap.add_argument("--all-fields", action="store_true",
                    help="Print full tick-list of every known ArrayType name.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print full part x material table (all parts).")
    args = ap.parse_args()

    if args.src is None and args.kfile is None and args.d3src is None:
        ap.error("Provide --src, or --kfile / --d3src.")

    kpath = args.kfile
    if kpath is None and args.src:
        k_files = sorted(args.src.glob("*.k")) or sorted(args.src.glob("*.key"))
        if k_files:
            kpath = k_files[0]

    d3dir = args.d3src or args.src

    print()
    print(SEP2)
    print("  sim_inspect.py  --  LS-DYNA full simulation overview")
    print(SEP2)
    if kpath:
        print(f"  K-file  : {kpath.resolve()}  ({kpath.stat().st_size/1e6:.1f} MB)")
    if d3dir:
        print(f"  D3plot  : {d3dir.resolve()}")
        print(f"  States  : {'yes (first state file)' if args.load_states else 'header only (use --load-states for stress/strain)'}")
    print()

    # ── Parse k-file ─────────────────────────────────────────────────────
    kd, kparts, kmats = {}, [], {}
    if kpath:
        if not kpath.exists():
            sys.exit(f"ERROR: k-file not found: {kpath}")
        print("  Parsing k-file ...", end="", flush=True)
        kd     = _parse_kfile(kpath)
        kparts = kd.get("parts", [])
        kmats  = {m["mid"]: m for m in kd.get("materials", [])}
        print(f" done.  ({len(kparts)} parts, {len(kmats)} materials)")
        print()
        _k_sim_settings(kd)
        _k_db_extent(kd)
        _k_materials(kd, verbose=args.verbose)
        _k_sections(kd)
        _k_parts(kd, verbose=args.verbose)
        _k_contacts(kd)

    # ── Load d3plot ───────────────────────────────────────────────────────
    if d3dir:
        if not d3dir.exists():
            sys.exit(f"ERROR: d3plot dir not found: {d3dir}")
        if not (d3dir / "d3plot").exists():
            sys.exit(f"ERROR: no d3plot file in {d3dir}")
        with tempfile.TemporaryDirectory(prefix="siminspect_") as _tmp:
            tmp = Path(_tmp)
            print("  Opening d3plot ...", end="", flush=True)
            try:
                d3 = _open_d3plot(d3dir, args.load_states, tmp)
            except Exception as exc:
                sys.exit(f"ERROR opening d3plot: {exc}")
            print(" done.")
            print()
            _d3_mesh_summary(d3)
            _d3_parts_full(d3, kparts, kmats)
            _d3_plastic_strain(d3, kparts, kmats)
            _d3_arrays_all(d3)
            if args.all_fields:
                _d3_fields_checklist(d3)

    print(SEP2)
    print("  Done.")
    print(SEP2)
    print()


if __name__ == "__main__":
    main()
