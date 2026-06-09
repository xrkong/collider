"""
k_inspect.py  –  Print a structured overview of an LS-DYNA keyword (.k) file.

Covers:
  • Simulation settings   – title, end-time, timestep, parameters, initial velocity
  • DATABASE_EXTENT_BINARY – exactly which fields are written to d3plot
  • Parts catalog          – pid / secid / mid / name for every *PART
  • Materials              – grouped by type with key properties (rho, E, nu, sigy, …)
  • Sections               – shell thickness, solid/beam formulations
  • Contacts               – type and count summary
  • Part × Material cross-reference (pid → material type + title)

Usage
-----
  python tools/k_inspect.py --src /path/to/simulation/dir
  python tools/k_inspect.py --src /path/to/car_and_barriers.k
  python tools/k_inspect.py --src /path/to/dir --verbose      # full part+mat tables
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# ─── helpers ──────────────────────────────────────────────────────────────────

SEP  = "-" * 72
SEP2 = "=" * 72

def _fields(line: str) -> list[str]:
    """Split a LS-DYNA data line into tokens.

    LS-DYNA keyword format uses 10-char fixed-width fields.  Adjacent numbers
    may share no whitespace (e.g. '  2000001 7.89e-9' occupies chars 1-20).
    We slice by 10-char chunks, falling back to whitespace split only when the
    line is clearly space-delimited (all chunks are blank except first).
    """
    stripped = line.rstrip()
    if len(stripped) < 10:
        return stripped.split()
    # try fixed-width: 10-char chunks
    chunks = [stripped[k:k+10].strip() for k in range(0, len(stripped), 10)]
    chunks = [c for c in chunks if c]
    # if we get only one non-empty chunk, fall back to whitespace split
    if len(chunks) <= 1:
        return stripped.split()
    return chunks

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


# ─── keyword file parser ───────────────────────────────────────────────────────

def _parse_kfile(path: Path) -> dict:
    """
    Single-pass parser.  Returns a dict with keys:
        title, parameters, termination, timestep, initial_velocity,
        db_extent, parts, materials, sections, contacts
    """
    data: dict = {
        "title": "",
        "parameters": {},           # name → value (str)
        "termination": {},
        "timestep": {},
        "initial_velocity": [],
        "db_extent": {},
        "parts": [],                # list of {pid, secid, mid, name}
        "materials": [],            # list of {mid, kw, title, props}
        "sections": defaultdict(list),   # type → list of {secid, props}
        "contacts": [],             # list of keyword strings
    }

    # resolve parameter references like &VEL_1
    params: dict[str, str] = {}

    def _resolve(s: str) -> str:
        for k, v in params.items():
            s = re.sub(rf"&{k}", v, s, flags=re.IGNORECASE)
        return s

    with open(path, encoding="ascii", errors="ignore") as fh:
        lines = fh.readlines()

    def _noncomment(i: int) -> str | None:
        """Return next non-comment, non-blank line after i."""
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

        # ── TITLE ──────────────────────────────────────────────────────────
        if kw == "*TITLE":
            j = i + 1
            while j < len(lines) and lines[j].startswith("$"):
                j += 1
            if j < len(lines):
                data["title"] = lines[j].strip()
            i = j + 1
            continue

        # ── PARAMETER / PARAMETER_EXPRESSION ──────────────────────────────
        if kw in ("*PARAMETER", "*PARAMETER_EXPRESSION"):
            j = i + 1
            while j < len(lines):
                l = lines[j].rstrip("\n")
                if l.startswith("*"):
                    break
                if l.startswith("$"):
                    j += 1
                    continue
                # "R NAME  value" or "R NAME  expression"
                parts = l.split()
                if len(parts) >= 3 and parts[0].upper() in ("R", "I"):
                    name = parts[1].upper()
                    val  = parts[2]
                    params[name] = val
                    data["parameters"][name] = val
                j += 1
            i = j
            continue

        # ── CONTROL_TERMINATION ────────────────────────────────────────────
        if kw == "*CONTROL_TERMINATION":
            line2 = _noncomment(i)
            if line2:
                tok = _fields(line2)
                data["termination"] = {
                    "endtim": _f(tok[0]) if len(tok) > 0 else 0.0,
                    "endcyc": _i(tok[1]) if len(tok) > 1 else 0,
                }
            i += 1
            continue

        # ── CONTROL_TIMESTEP ───────────────────────────────────────────────
        if kw == "*CONTROL_TIMESTEP":
            line2 = _noncomment(i)
            if line2:
                tok = _fields(line2)
                data["timestep"] = {
                    "dtinit": _f(tok[0]) if len(tok) > 0 else 0.0,
                    "tssfac": _f(tok[1]) if len(tok) > 1 else 0.0,
                }
            i += 1
            continue

        # ── INITIAL_VELOCITY_GENERATION ───────────────────────────────────
        if kw == "*INITIAL_VELOCITY_GENERATION":
            line2 = _noncomment(i)
            if line2:
                tok = _fields(line2)
                vx = _resolve(tok[3]) if len(tok) > 3 else "0"
                vy = _resolve(tok[4]) if len(tok) > 4 else "0"
                vz = _resolve(tok[5]) if len(tok) > 5 else "0"
                data["initial_velocity"].append({
                    "id":   _i(tok[0]) if len(tok) > 0 else 0,
                    "styp": _i(tok[1]) if len(tok) > 1 else 0,
                    "vx":   vx,  "vy": vy,  "vz": vz,
                })
            i += 1
            continue

        # ── DATABASE_EXTENT_BINARY ─────────────────────────────────────────
        if kw == "*DATABASE_EXTENT_BINARY":
            j = i + 1
            row = 0
            while j < len(lines) and row < 4:
                l = lines[j].rstrip("\n")
                if l.startswith("*"):
                    break
                if l.startswith("$"):
                    j += 1
                    continue
                tok = _fields(l)
                if row == 0:
                    data["db_extent"] = {
                        "neiph":  _i(tok[0]) if len(tok) > 0 else 0,
                        "neips":  _i(tok[1]) if len(tok) > 1 else 0,
                        "maxint": _i(tok[2]) if len(tok) > 2 else 0,
                        "strflg": _i(tok[3]) if len(tok) > 3 else 0,
                        "sigflg": _i(tok[4]) if len(tok) > 4 else 0,
                        "epsflg": _i(tok[5]) if len(tok) > 5 else 0,
                        "rltflg": _i(tok[6]) if len(tok) > 6 else 0,
                        "engflg": _i(tok[7]) if len(tok) > 7 else 0,
                    }
                elif row == 1:
                    data["db_extent"].update({
                        "cmpflg": _i(tok[0]) if len(tok) > 0 else 0,
                        "ieverp": _i(tok[1]) if len(tok) > 1 else 0,
                        "beamip": _i(tok[2]) if len(tok) > 2 else 0,
                        "dcomp":  _i(tok[3]) if len(tok) > 3 else 0,
                        "shge":   _i(tok[4]) if len(tok) > 4 else 0,
                        "stssz":  _i(tok[5]) if len(tok) > 5 else 0,
                        "therm":  _i(tok[7]) if len(tok) > 7 else 0,
                    })
                row += 1
                j += 1
            i = j
            continue

        # ── PART ──────────────────────────────────────────────────────────
        if kw == "*PART":
            # title line (skip comment lines)
            j = i + 1
            title = ""
            while j < len(lines):
                l = lines[j].rstrip("\n")
                if l.startswith("$"):
                    j += 1
                    continue
                title = l.strip()
                j += 1
                break
            # data line: pid secid mid eosid hgid grav adpopt tmid
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

        # ── MATERIAL CARDS ─────────────────────────────────────────────────
        if raw.startswith("*MAT_"):
            mat_kw = raw.strip()
            has_title = mat_kw.upper().endswith("_TITLE")
            mat_type  = re.sub(r"_TITLE$", "", mat_kw, flags=re.IGNORECASE).upper()

            j = i + 1
            mat_title = ""
            if has_title:
                while j < len(lines):
                    l = lines[j].rstrip("\n")
                    if l.startswith("$"):
                        j += 1
                        continue
                    mat_title = l.strip()
                    j += 1
                    break

            # first data row: mid  ro  (e  pr  sigy …)
            props: dict[str, float] = {}
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
                    if "RIGID" in mat_type and len(tok) >= 4:
                        props["e"]  = _f(tok[2])
                        props["pr"] = _f(tok[3])
                    elif "ELASTIC" in mat_type and len(tok) >= 4:
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
                "mid":   int(props.get("mid", 0)),
                "kw":    mat_type,
                "title": mat_title,
                "props": props,
            })
            i = j
            continue

        # ── SECTION cards ──────────────────────────────────────────────────
        if raw.startswith("*SECTION_"):
            sec_kw   = re.sub(r"_TITLE$", "", raw.strip(), flags=re.IGNORECASE).upper()
            sec_type = sec_kw.replace("*SECTION_", "")
            j = i + 1
            # skip title if present
            if raw.upper().strip().endswith("_TITLE"):
                while j < len(lines) and lines[j].startswith("$"):
                    j += 1
                j += 1  # title text line
            # first non-comment line = secid + formulation
            props: dict = {}
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
                    props["secid"]  = _i(tok[0])
                    props["elform"] = _i(tok[1])
                elif rows == 1 and sec_type == "SHELL" and len(tok) >= 1:
                    props["t"] = _f(tok[0])   # shell thickness t1
                rows += 1
                j += 1
            data["sections"][sec_type].append(props)
            i = j
            continue

        # ── CONTACT ────────────────────────────────────────────────────────
        if raw.startswith("*CONTACT_"):
            ctype = re.sub(r"_ID$|_TITLE$", "", raw.strip(), flags=re.IGNORECASE).upper()
            data["contacts"].append(ctype)
            i += 1
            continue

        i += 1

    # resolve any &PARAMETERs in initial velocity
    for iv in data["initial_velocity"]:
        for k in ("vx", "vy", "vz"):
            try:
                iv[k] = str(float(_resolve(iv[k])))
            except Exception:
                pass

    # fill parameters back
    data["parameters"] = params
    return data


# ─── printers ─────────────────────────────────────────────────────────────────

def _print_sim_settings(d: dict) -> None:
    print(SEP2)
    print("  SIMULATION SETTINGS")
    print(SEP2)
    if d["title"]:
        print(f"  Title        : {d['title']}")
    t = d.get("termination", {})
    ts = d.get("timestep", {})
    if t:
        print(f"  End time     : {t.get('endtim', '?')} s  "
              f"(endcyc={t.get('endcyc', 0)})")
    if ts:
        print(f"  DT init/tssfac: {ts.get('dtinit', 0)} / {ts.get('tssfac', 0)}")

    params = d.get("parameters", {})
    if params:
        print()
        print("  PARAMETERS")
        for name, val in params.items():
            # decode velocity parameter → km/h
            extra = ""
            if "VEL" in name.upper():
                try:
                    mm_s = float(val)
                    extra = f"  ({mm_s/1000*3.6:.1f} km/h)"
                except Exception:
                    pass
            print(f"    {name:<12} = {val}{extra}")

    ivs = d.get("initial_velocity", [])
    if ivs:
        print()
        print("  INITIAL VELOCITY")
        for iv in ivs:
            vx = iv.get("vx", "0")
            try:
                vx_f = float(vx)
                vx_str = f"{vx_f:.2f} mm/s  ({vx_f/1000*3.6:.1f} km/h)"
            except Exception:
                vx_str = vx
            print(f"    set_id={iv['id']}  styp={iv['styp']}  "
                  f"vx={vx_str}  vy={iv['vy']}  vz={iv['vz']}")
    print()


def _print_db_extent(d: dict) -> None:
    db = d.get("db_extent", {})
    if not db:
        return
    print(SEP)
    print("  DATABASE_EXTENT_BINARY  (fields written to d3plot)")
    print(SEP)

    def _flag(label: str, key: str, legend: dict) -> None:
        v = db.get(key, 0)
        desc = legend.get(v, str(v))
        print(f"    {label:<18}: {v}  → {desc}")

    _flag("neiph",  "neiph",  {0: "no extra solid history vars"})
    _flag("neips",  "neips",  {0: "no extra shell history vars"})
    _flag("strflg", "strflg", {0: "no strain", 1: "strain tensor", 2: "not written"})
    _flag("sigflg", "sigflg", {1: "local stress", 2: "global stress"})
    _flag("epsflg", "epsflg", {0: "no eff.pl.strain", 1: "eff. plastic strain written"})
    _flag("rltflg", "rltflg", {1: "local strain", 2: "global strain"})
    _flag("engflg", "engflg", {0: "no energy", 1: "internal+kinetic energy written"})
    _flag("ieverp", "ieverp", {0: "all states in one d3plot",
                                1: "each state in separate d3plot file"})
    _flag("therm",  "therm",  {0: "no temperature", 1: "temperatures written"})
    neiph = db.get("neiph", 0)
    if neiph > 0:
        print(f"\n    neiph={neiph}: {neiph} extra integration-pt history variables"
              " per solid element are in d3plot")
    print()


def _print_materials(d: dict, verbose: bool = False) -> None:
    mats = d.get("materials", [])
    if not mats:
        print("  (no materials found)")
        return

    # group by canonical type
    by_type: dict[str, list] = defaultdict(list)
    for m in mats:
        canonical = re.sub(r"^\*MAT_", "", m["kw"])
        by_type[canonical].append(m)

    print(SEP)
    print(f"  MATERIALS  ({len(mats)} total)")
    print(SEP)

    counts = {k: len(v) for k, v in sorted(by_type.items(), key=lambda x: -len(x[1]))}
    print("  Type summary:")
    for typ, cnt in counts.items():
        print(f"    {cnt:>4}×  {typ}")
    print()

    # unique titles per type
    print("  Unique material titles per type:")
    for typ in sorted(by_type, key=lambda k: -len(by_type[k])):
        mlist = by_type[typ]
        titles = sorted({m["title"] for m in mlist if m["title"]})
        # representative properties (first of each title)
        by_title: dict[str, dict] = {}
        for m in mlist:
            t = m["title"] or "(untitled)"
            if t not in by_title:
                by_title[t] = m
        print(f"\n  [{typ}]  {len(mlist)} instances")
        for title, m in by_title.items():
            p = m["props"]
            ro   = p.get("ro",   None)
            e    = p.get("e",    None)
            pr   = p.get("pr",   None)
            sigy = p.get("sigy", None)
            etan = p.get("etan", None)
            fail = p.get("fail", None)
            g    = p.get("g",    None)
            parts_str = ""
            if ro   is not None: parts_str += f"  ρ={ro:.3g} t/mm³"
            if e    is not None: parts_str += f"  E={e:.4g} MPa"
            if pr   is not None: parts_str += f"  ν={pr:.3g}"
            if sigy is not None: parts_str += f"  σy={sigy:.4g} MPa"
            if etan is not None and etan != 0: parts_str += f"  Etan={etan:.3g}"
            if fail is not None and fail != 0: parts_str += f"  fail={fail:.3g}"
            if g    is not None: parts_str += f"  G={g:.3g} MPa"
            print(f"    '{title}'{parts_str}")
    print()


def _print_sections(d: dict, verbose: bool = False) -> None:
    secs = d.get("sections", {})
    if not secs:
        return
    print(SEP)
    print("  SECTIONS")
    print(SEP)
    for stype, slist in sorted(secs.items()):
        print(f"  [{stype}]  {len(slist)} instances")
        if stype == "SHELL":
            thicknesses = sorted({s["t"] for s in slist if "t" in s})
            print(f"    Thickness range : {min(thicknesses):.3g} – {max(thicknesses):.3g} mm")
            print(f"    Unique values   : {thicknesses[:20]}{'…' if len(thicknesses)>20 else ''}")
            forms = sorted({s.get("elform",0) for s in slist})
            print(f"    Elform codes    : {forms}")
        elif stype == "SOLID":
            forms = sorted({s.get("elform",0) for s in slist})
            print(f"    Elform codes    : {forms}")
        elif stype == "BEAM":
            forms = sorted({s.get("elform",0) for s in slist})
            print(f"    Elform codes    : {forms}")
    print()


def _print_parts(d: dict, verbose: bool = False) -> None:
    parts = d.get("parts", [])
    mats  = {m["mid"]: m for m in d.get("materials", [])}
    print(SEP)
    print(f"  PARTS  ({len(parts)} total)")
    print(SEP)

    if not verbose:
        # brief: just show first/last few and prefix groups
        prefixes: dict[str, int] = defaultdict(int)
        for p in parts:
            pref = re.split(r"[_\s]", p["name"])[0] if p["name"] else "?"
            prefixes[pref] += 1
        top = sorted(prefixes.items(), key=lambda x: -x[1])[:15]
        print("  Part-name prefix groups (top 15):")
        for pref, cnt in top:
            print(f"    {cnt:>4}×  {pref}")
        print()
        print("  First 10 parts:")
        for p in parts[:10]:
            mid = p["mid"]
            m   = mats.get(mid, {})
            mat_desc = f"{m.get('kw','')}({m.get('title','')})" if m else f"mid={mid}"
            print(f"    pid={p['pid']:<9} mid={mid:<9}  {p['name']:<45}  [{mat_desc}]")
        print("  …")
        print(f"  Last 10 parts:")
        for p in parts[-10:]:
            mid = p["mid"]
            m   = mats.get(mid, {})
            mat_desc = f"{m.get('kw','')}({m.get('title','')})" if m else f"mid={mid}"
            print(f"    pid={p['pid']:<9} mid={mid:<9}  {p['name']:<45}  [{mat_desc}]")
    else:
        print(f"  {'pid':>9}  {'secid':>9}  {'mid':>9}  {'material':>35}  name")
        print(f"  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*35}  {'─'*45}")
        for p in parts:
            mid = p["mid"]
            m   = mats.get(mid, {})
            mat_desc = f"{re.sub(r'^.*MAT_','',m.get('kw','?'))}:{m.get('title','')}"[:35] if m else f"mid={mid}"
            print(f"  {p['pid']:>9}  {p['secid']:>9}  {mid:>9}  {mat_desc:>35}  {p['name']}")
    print()


def _print_contacts(d: dict) -> None:
    contacts = d.get("contacts", [])
    if not contacts:
        return
    print(SEP)
    print("  CONTACTS")
    print(SEP)
    counts: dict[str, int] = defaultdict(int)
    for c in contacts:
        counts[c] += 1
    for ctype, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {n:>3}×  {ctype}")
    print()


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a structured overview of an LS-DYNA keyword (.k) file."
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="Simulation directory (looks for *.k) or path to .k file.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full part × material cross-reference table.")
    args = parser.parse_args()

    if not args.src.exists():
        sys.exit(f"ERROR: path not found: {args.src}")

    # locate .k file
    if args.src.is_dir():
        k_files = sorted(args.src.glob("*.k"))
        if not k_files:
            k_files = sorted(args.src.glob("*.key"))
        if not k_files:
            sys.exit(f"ERROR: no .k / .key file found in {args.src}")
        kpath = k_files[0]
        if len(k_files) > 1:
            print(f"  (multiple .k files found; using {kpath.name})")
    else:
        kpath = args.src

    print()
    print(SEP2)
    print("  k_inspect.py  –  LS-DYNA keyword file overview")
    print(SEP2)
    print(f"  File   : {kpath.resolve()}")
    print(f"  Size   : {kpath.stat().st_size/1e6:.1f} MB  "
          f"({kpath.stat().st_size//1024:,} KB)")
    print()

    print("  Parsing … ", end="", flush=True)
    d = _parse_kfile(kpath)
    print("done.")
    print()

    _print_sim_settings(d)
    _print_db_extent(d)
    _print_materials(d, verbose=args.verbose)
    _print_sections(d, verbose=args.verbose)
    _print_parts(d, verbose=args.verbose)
    _print_contacts(d)

    print(SEP2)
    print("  Done.")
    print(SEP2)
    print()


if __name__ == "__main__":
    main()
