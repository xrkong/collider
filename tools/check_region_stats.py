"""
Check whether per-region mean/std stats diverge from the global (all-region)
stats used for training normalization, and plot how mean/std evolve over
time frames per region.

Velocity/acceleration are derived from states/positions via the same
forward-difference convention as src/dataset.py and compute_global_stats
(dt = 1 frame), so this reflects exactly what training actually normalizes.
Regions come from metadata/region_id (see dataset/constants.py REGION_ID_MAP).

Usage:
  python3 tools/check_region_stats.py path/to/traj.h5
  python3 tools/check_region_stats.py h5_fps/*.h5 --fields velocity,acceleration
  python3 tools/check_region_stats.py path/to/traj.h5 --out-dir results/region_stats/
  python3 tools/check_region_stats.py path/to/traj.h5 --gap-ratio 2.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.constants import REGION_ID_MAP  # noqa: E402

ID_TO_LABEL = {v: k for k, v in REGION_ID_MAP.items()}
REGION_COLORS = plt.cm.tab10(np.linspace(0, 1, len(REGION_ID_MAP)))


def derive_field(pos: np.ndarray, field: str) -> np.ndarray:
    """pos: (T, N, 3) -> (T', N, 3) for the requested field, dt = 1 frame."""
    if field == "positions":
        return pos
    vel = np.diff(pos, axis=0)
    if field == "velocity":
        return vel
    if field == "acceleration":
        return np.diff(vel, axis=0)
    raise ValueError(f"Unknown field: {field}")


def load_h5(h5_path: Path, fields: list[str]) -> dict:
    with h5py.File(h5_path, "r") as f:
        pos = f["states/positions"][:].astype(np.float64)   # (T, N, 3)
        times = f["states/times"][:] if "states/times" in f else np.arange(pos.shape[0])
        region_id = f["metadata/region_id"][:].astype(np.int64)  # (N,)
    data = {field: derive_field(pos, field) for field in fields}
    return {"data": data, "times": times, "region_id": region_id}


def scalar_stats(x: np.ndarray) -> tuple[float, float]:
    """Pooled mean/std over every value in x (matches compute_global_stats)."""
    flat = x.reshape(-1)
    return float(flat.mean()), float(flat.std())


def print_stats_table(field: str, region_id: np.ndarray, data: np.ndarray, gap_ratio: float):
    g_mean, g_std = scalar_stats(data)
    g_cv = g_std / (abs(g_mean) + 1e-8)

    print(f"\n=== {field} — mean / std / |std/mean| by region ===")
    header = f"  {'region':<16}{'n_nodes':>9}{'mean':>14}{'std':>14}{'std/mean':>12}  flag"
    print(header)
    print(f"  {'GLOBAL':<16}{region_id.size:>9}{g_mean:>14.6f}{g_std:>14.6f}{g_cv:>12.3f}")

    for rid in sorted(ID_TO_LABEL):
        mask = region_id == rid
        n = int(mask.sum())
        label = ID_TO_LABEL[rid]
        if n == 0:
            print(f"  {label:<16}{n:>9}  (no nodes in this trajectory)")
            continue
        r_mean, r_std = scalar_stats(data[:, mask, :])
        r_cv = r_std / (abs(r_mean) + 1e-8)
        ratio = r_cv / g_cv if g_cv > 0 else float("inf")
        flag = "  <-- GAP" if (ratio > gap_ratio or ratio < 1.0 / gap_ratio) else ""
        print(f"  {label:<16}{n:>9}{r_mean:>14.6f}{r_std:>14.6f}{r_cv:>12.3f}{flag}")


def plot_field_over_time(field: str, times: np.ndarray, region_id: np.ndarray,
                          data: np.ndarray, out_dir: str | None, fmt: str, traj_name: str):
    """One figure: per-region mean line + std band, plus global overlay, vs time."""
    t = times[-data.shape[0]:]  # align derived-field length to trailing timestamps

    fig, ax = plt.subplots(figsize=(12, 6))

    g_mean = data.reshape(data.shape[0], -1).mean(axis=1)
    g_std = data.reshape(data.shape[0], -1).std(axis=1)
    ax.plot(t, g_mean, color="black", linewidth=1.8, linestyle="--", label="GLOBAL mean")
    ax.fill_between(t, g_mean - g_std, g_mean + g_std, color="black", alpha=0.08)

    for rid in sorted(ID_TO_LABEL):
        mask = region_id == rid
        if mask.sum() == 0:
            continue
        label = ID_TO_LABEL[rid]
        color = REGION_COLORS[rid]
        region_vals = data[:, mask, :].reshape(data.shape[0], -1)
        r_mean = region_vals.mean(axis=1)
        r_std = region_vals.std(axis=1)
        ax.plot(t, r_mean, color=color, linewidth=1.2, label=label)
        ax.fill_between(t, r_mean - r_std, r_mean + r_std, color=color, alpha=0.12)

    ax.set_title(f"{traj_name} — {field} mean ± std by region over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(field)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / f"{traj_name}_{field}.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  -> saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Check per-region vs global mean/std gap, and plot values over time.")
    parser.add_argument("h5_paths", nargs="+", help="One or more .h5 trajectory files")
    parser.add_argument("--fields", default="velocity,acceleration",
                        help="Comma-separated: positions,velocity,acceleration "
                             "(default: velocity,acceleration)")
    parser.add_argument("--gap-ratio", type=float, default=3.0,
                        help="Flag a region if its |std/mean| differs from the global "
                             "|std/mean| by more than this multiplicative factor (default: 3.0)")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help="Directory to save figures (default: show interactively)")
    parser.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    fields = [f.strip() for f in args.fields.split(",")]
    h5_paths = [Path(p) for p in args.h5_paths]

    # ── Pooled stats table across all provided trajectories ────────────────
    pooled_data = {f: [] for f in fields}
    pooled_region_id = []
    for p in h5_paths:
        print(f"Loading {p} ...")
        loaded = load_h5(p, fields)
        for f in fields:
            pooled_data[f].append(loaded["data"][f])
        pooled_region_id.append(loaded["region_id"])

    # region_id is static per node but nodes may differ across trajs — only
    # pool stats when all trajs share the same node set/count.
    same_regions = all(np.array_equal(pooled_region_id[0], r) for r in pooled_region_id)
    if same_regions:
        region_id = pooled_region_id[0]
        for f in fields:
            data = np.concatenate(pooled_data[f], axis=0)  # concat along time
            print_stats_table(f, region_id, data, args.gap_ratio)
    else:
        print("\n[Warning] region_id differs across trajectories — printing per-file stats.")
        for p, r in zip(h5_paths, pooled_region_id):
            for f in fields:
                loaded_f = derive_field(
                    h5py.File(p, "r")["states/positions"][:].astype(np.float64), f
                )
                print(f"\n--- {p.name} ---")
                print_stats_table(f, r, loaded_f, args.gap_ratio)

    # ── Per-trajectory time-series plots ────────────────────────────────────
    print("\nPlotting...")
    for p in h5_paths:
        loaded = load_h5(p, fields)
        traj_name = p.stem
        for f in fields:
            plot_field_over_time(f, loaded["times"], loaded["region_id"],
                                  loaded["data"][f], args.out_dir, args.fmt, traj_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
