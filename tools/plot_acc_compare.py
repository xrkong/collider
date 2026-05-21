"""
plot_acc_compare.py  –  Compare acceleration magnitude over time across H5 files.

Shows how fast acceleration changes (its rate of change, i.e. jerk) and the raw
magnitude, so you can visually judge whether the field evolves smoothly or sharply.

Usage:
  python3 tools/plot_acc_compare.py /path/to/a.h5 /path/to/b.h5 ...
  python3 tools/plot_acc_compare.py /path/to/a.h5 /path/to/b.h5 --out results/compare.png
  python3 tools/plot_acc_compare.py /path/to/a.h5 --nodes 0 50 100  # specific node indices
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── helpers ────────────────────────────────────────────────────────────────────

def load_h5(path: str):
    with h5py.File(path, "r") as f:
        times = f["states/times"][:]           # (T,)
        acc   = f["states/acceleration"][:]    # (T, N, 3)
    return times, acc


def acc_magnitude(acc: np.ndarray) -> np.ndarray:
    """(T, N, 3) → (T, N) L2 magnitude."""
    return np.linalg.norm(acc, axis=-1)


def mean_acc_mag(acc: np.ndarray) -> np.ndarray:
    """(T, N, 3) → (T,) mean over nodes."""
    return acc_magnitude(acc).mean(axis=1)


def jerk_mag(times: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """
    Rate of change of acceleration (jerk magnitude), averaged over nodes.
    Returns (T,) — first point is set to 0 (forward diff).
    """
    mag = acc_magnitude(acc)           # (T, N)
    dt  = np.diff(times)               # (T-1,)
    dmag = np.diff(mag, axis=0)        # (T-1, N)
    jerk = np.abs(dmag) / dt[:, None]  # (T-1, N)
    mean_jerk = jerk.mean(axis=1)      # (T-1,)
    return np.concatenate([[0.0], mean_jerk])


# ── plot ───────────────────────────────────────────────────────────────────────

def plot_compare(files: list[str], node_indices=None, out_path=None, t_max_s: float = np.inf):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    ax_mag, ax_jerk, ax_xyz = axes

    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(len(files))]

    for i, (path, color) in enumerate(zip(files, colors)):
        label = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        times, acc = load_h5(path)

        # ── time crop ─────────────────────────────────────────────────────────
        t_mask = times <= t_max_s
        times  = times[t_mask]
        acc    = acc[t_mask]

        t_ms = times * 1e3   # convert s → ms for readability

        # ── panel 1: mean acceleration magnitude ──────────────────────────────
        if node_indices is not None:
            subset = acc[:, node_indices, :]
        else:
            subset = acc

        mag_mean = mean_acc_mag(subset)        # (T,)
        mag_std  = acc_magnitude(subset).std(axis=1)

        ax_mag.plot(t_ms, mag_mean, color=color, lw=1.5, label=label)
        ax_mag.fill_between(t_ms, mag_mean - mag_std, mag_mean + mag_std,
                            color=color, alpha=0.15)

        # ── panel 2: jerk (rate of change of acc magnitude) ───────────────────
        j = jerk_mag(times, subset)
        ax_jerk.plot(t_ms, j, color=color, lw=1.2, label=label)

        # ── panel 3: mean per-component acceleration ───────────────────────────
        comp_labels = ["X", "Y", "Z"]
        comp_styles = ["-", "--", ":"]
        for ci, (cl, cs) in enumerate(zip(comp_labels, comp_styles)):
            comp_mean = subset[:, :, ci].mean(axis=1)
            ax_xyz.plot(t_ms, comp_mean, color=color, ls=cs, lw=1.2,
                        label=f"{label} {cl}" if i == 0 else f"_{label} {cl}")

    # ── formatting ─────────────────────────────────────────────────────────────
    def _fmt(ax, ylabel, title):
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))
        ax.set_xlabel("Time (ms)", fontsize=9)

    node_info = f" (nodes {node_indices})" if node_indices is not None else " (all nodes)"
    _fmt(ax_mag,   "Mean |acc| (mm/s²)", f"Acceleration Magnitude{node_info}")
    _fmt(ax_jerk,  "Mean |Δacc|/Δt (mm/s³)", "Jerk — How fast does acceleration change?")
    _fmt(ax_xyz,   "Mean acc (mm/s²)",    "Per-component acceleration (X=solid, Y=dash, Z=dot)")

    # add component legend manually for panel 3
    for cs, cl in zip(["-", "--", ":"], ["X", "Y", "Z"]):
        ax_xyz.plot([], [], color="gray", ls=cs, lw=1.2, label=cl)
    ax_xyz.legend(fontsize=7, loc="upper right", ncol=3)

    plt.suptitle("Acceleration comparison across H5 files", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare acceleration magnitude & jerk across multiple H5 files.")
    parser.add_argument("h5_files", nargs="+", help="One or more output.h5 paths")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="Save figure to this path instead of showing interactively")
    parser.add_argument("--nodes", nargs="+", type=int, default=None,
                        help="Specific node indices to plot (default: all nodes)")
    parser.add_argument("--tmax", type=float, default=None, metavar="SECONDS",
                        help="Crop time axis to this upper limit in seconds (e.g. 0.9)")
    args = parser.parse_args()

    for p in args.h5_files:
        if not os.path.exists(p):
            sys.exit(f"File not found: {p}")

    print(f"Comparing {len(args.h5_files)} file(s):")
    for p in args.h5_files:
        with h5py.File(p, "r") as f:
            T = f["states/times"].shape[0]
            N = f["states/acceleration"].shape[1]
        print(f"  {p}  — T={T} frames, N={N} nodes")

    node_indices = np.array(args.nodes) if args.nodes else None
    t_max_s = args.tmax if args.tmax is not None else np.inf
    plot_compare(args.h5_files, node_indices=node_indices, out_path=args.out, t_max_s=t_max_s)


if __name__ == "__main__":
    main()
