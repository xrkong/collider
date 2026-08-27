"""Overlay autoregressive-only GC/barrier kinematics across multiple rollout cases.

Companion to src/plot_gc_barrier.py: that script plots GT vs (one-step vs)
autoregressive for a SINGLE case. This script instead reads each case's
gc_barrier_autoregressive.csv (as written by src/rollout.py --mode
autoregressive) and overlays only the autoregressive-prediction curve from
every case on three combined figures — ORA_x, ORA_y, barrier resultant
displacement — one line per case, distinguished by a legend. Ground truth
and one-step are intentionally omitted here: the per-case GT/AR comparison
already covers those, this view is for comparing AR behavior across cases.

Usage:
    python src/plot_gc_barrier_multi.py \
        --cases train0_foo:outputs/rollouts_all/wj09/train0_foo \
                val0_bar:outputs/rollouts_all/wj09/val0_bar \
        --out-dir outputs/rollouts_all/wj09

To split cases into subgroups (e.g. one barrier design vs another) instead of
one figure with every case, run this once per group with --out-suffix /
--title-suffix so each group's 3 PNGs land side by side without overwriting:
    python src/plot_gc_barrier_multi.py --cases train0:... train1:... \
        --out-dir outputs/rollouts_all/wj09 \
        --out-suffix _new_barrier --title-suffix "New barrier (train0-train4)"
    python src/plot_gc_barrier_multi.py --cases train5:... val0:... \
        --out-dir outputs/rollouts_all/wj09 \
        --out-suffix _tlok --title-suffix "T-lok (train5-train7, val0)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.plot_gc_barrier import (  # noqa: E402
    _RCPARAMS, ASI_FILTER_WINDOW_MS, build_series, load_csv, window_frames_for,
)


def find_autoreg_csv(case_dir: Path) -> Path:
    matches = sorted(case_dir.glob("*gc_barrier_autoregressive.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one gc_barrier_autoregressive.csv in {case_dir}. "
            f"Found {[p.name for p in matches]}."
        )
    return matches[0]


def make_overlay_plot(y_label: str, title: str, series_by_label: dict,
                       attr: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab20", max(len(series_by_label), 1))
    for i, (label, series) in enumerate(series_by_label.items()):
        ax.plot(series.time, getattr(series, attr), color=cmap(i),
                linewidth=1.4, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay autoregressive-only GC/barrier kinematics across "
                     "multiple rollout cases")
    parser.add_argument("--cases", nargs="+", required=True, metavar="LABEL:DIR",
                        help="One or more 'label:rollout_dir' pairs, each dir "
                             "containing a gc_barrier_autoregressive.csv "
                             "(see src/rollout.py --mode autoregressive)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-ms", type=float, default=ASI_FILTER_WINDOW_MS,
                        help="Moving-average filter window in ms (default: 50ms, per EN 1317)")
    parser.add_argument("--out-suffix", default="",
                        help="Appended to each output filename stem, e.g. '_new_barrier' -> "
                             "gc_barrier_ora_x_all_cases_new_barrier.png. Use this to split "
                             "--cases into subgroups (one call per group) without each "
                             "group's PNGs overwriting the others'.")
    parser.add_argument("--title-suffix", default="",
                        help="Appended to each plot title in parentheses, e.g. "
                             "'New barrier (train0-train4)', to identify the subgroup.")
    args = parser.parse_args()

    plt.rcParams.update(_RCPARAMS)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    series_by_label = {}
    for entry in args.cases:
        if ":" not in entry:
            print(f"[Warn] --cases entry '{entry}' missing 'label:dir' — skipping")
            continue
        label, cdir = entry.split(":", 1)
        csv_path = find_autoreg_csv(Path(cdir))
        data = load_csv(csv_path)
        dt_seconds, window_frames = window_frames_for(data["time"], args.window_ms)
        series_by_label[label] = build_series(data, "pred", dt_seconds, window_frames)
        print(f"[Compare] loaded '{label}' from {csv_path}")

    if not series_by_label:
        raise SystemExit("No valid --cases entries found")

    title_tail = f" ({args.title_suffix})" if args.title_suffix else ""

    make_overlay_plot(
        "ORA$_x$ (g)", f"Longitudinal acceleration (ORA$_x$) — all cases (autoregressive){title_tail}",
        series_by_label, "ax_filt",
        args.out_dir / f"gc_barrier_ora_x_all_cases{args.out_suffix}.png")

    make_overlay_plot(
        "ORA$_y$ (g)", f"Lateral acceleration (ORA$_y$) — all cases (autoregressive){title_tail}",
        series_by_label, "ay_filt",
        args.out_dir / f"gc_barrier_ora_y_all_cases{args.out_suffix}.png")

    make_overlay_plot(
        "Displacement (mm)", f"Barrier resultant displacement — all cases (autoregressive){title_tail}",
        series_by_label, "disp",
        args.out_dir / f"gc_barrier_displacement_all_cases{args.out_suffix}.png")


if __name__ == "__main__":
    main()
