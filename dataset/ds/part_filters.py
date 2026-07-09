"""
Part exclusion filters — drop whole parts (by name pattern) from sampling
entirely, before any region/budget logic runs.

Motivating case: tires, rims, and wheel spindles rotate continuously
throughout the simulation. That rotation is large-amplitude, periodic motion
unrelated to the crash deformation the surrogate is meant to learn, and is
hard to predict from a fixed downsampled point cloud — so we drop those
parts from the candidate pool entirely rather than let them consume part of
the 100k node budget.

Patterns are matched case-insensitively as substrings against
metadata/node_part_name (k-file *PART title), or as a glob if the pattern
contains wildcard characters — mirrors dataset/d3plot_to_h5_dt.py's
_pattern_matches/_build_selected_part_mask convention.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import yaml


def pattern_matches(name_lower: str, pat: str) -> bool:
    if any(c in pat for c in "*?["):
        return fnmatch(name_lower, pat)
    return pat in name_lower


def resolve_exclude_pids(
    patterns: list[str],
    pid_to_name: dict[int, str],
) -> dict[int, str]:
    """Match patterns against every known part name.

    Returns {pid: name} for matched parts (not just a bare set) so callers
    can print exactly what got excluded.
    """
    matched: dict[int, str] = {}
    for pid, name in pid_to_name.items():
        low = name.lower()
        if any(pattern_matches(low, p.lower()) for p in patterns):
            matched[pid] = name
    return matched


def load_exclude_parts_config(path: str | Path) -> list[str]:
    """Load a YAML config of the form `exclude_parts: [pattern, ...]`."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    patterns = cfg.get("exclude_parts") or []
    if not isinstance(patterns, list):
        raise ValueError(f"{path}: 'exclude_parts' must be a list of name patterns")
    return [str(p) for p in patterns]


# ── Candidate extraction (used to (re)generate the YAML from a k-file) ────

DEFAULT_INCLUDE_PATTERNS = ["tire", "_rim", "spindle"]
# Substrings that would otherwise false-positive-match an include pattern:
# "wheel" patterns intentionally aren't in DEFAULT_INCLUDE_PATTERNS at all
# (would catch steering wheel, wheel well body panel, engine belt/fan/
# alternator pulleys — none of which are the rotating tire/wheel assembly).
DEFAULT_EXCLUDE_PATTERNS = [
    "wheelwell", "steeringwheel", "enginebeltwheel",
    "enginefanwheel", "alternatorwheel", "pullypumpwheel",
]


def extract_candidate_parts(
    pid_to_name: dict[int, str],
    include_patterns: list[str] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: list[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> dict[int, str]:
    """Find candidate rotating-part PIDs by name keyword, for review.

    Not meant to be trusted blindly — print the result and sanity-check
    against the k-file before writing it into a config (see
    write_exclude_parts_config / the generate_exclude_parts_config CLI).
    """
    matched: dict[int, str] = {}
    for pid, name in pid_to_name.items():
        low = name.lower()
        if any(p in low for p in include_patterns) and not any(p in low for p in exclude_patterns):
            matched[pid] = name
    return matched


def main() -> None:
    """CLI: print (and optionally write) candidate rotating-part patterns
    found in a k-file, for review before use as --exclude-parts-config.

    python -m dataset.ds.part_filters --kfile car_and_barriers.k
    python -m dataset.ds.part_filters --kfile car_and_barriers.k \\
        --out configs/data/exclude_parts_tires.yaml
    """
    import argparse

    from .materials import parse_kfile_parts_and_materials

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--kfile", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the matched PIDs/names as a reviewable YAML "
                             "(patterns themselves, not PIDs, are what's actually "
                             "used at sampling time).")
    args = parser.parse_args()

    pid_to_name, _ = parse_kfile_parts_and_materials(args.kfile)
    matched = extract_candidate_parts(pid_to_name)
    print(f"{len(matched)} candidate parts out of {len(pid_to_name)} total "
          f"(patterns: {DEFAULT_INCLUDE_PATTERNS}, excluding: {DEFAULT_EXCLUDE_PATTERNS}):")
    for pid, name in sorted(matched.items()):
        print(f"  {pid:>10}  {name}")

    if args.out:
        lines = [
            "# Parts excluded from sampling — continuously rotating tire/rim/spindle",
            "# components. See dataset/ds/part_filters.py docstring for rationale.",
            f"# Auto-generated from {args.kfile} — REVIEW before relying on this.",
            "#",
            f"# {len(matched)} parts matched out of {len(pid_to_name)} total:",
        ]
        for pid, name in sorted(matched.items()):
            lines.append(f"#   {pid:>10}  {name}")
        lines.append("")
        lines.append("exclude_parts:")
        for p in DEFAULT_INCLUDE_PATTERNS:
            lines.append(f"  - {p!r}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
