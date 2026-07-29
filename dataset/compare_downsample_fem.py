"""Compare downsampled vs. raw FEM ground truth for one trajectory.

dataset/build_dataset.py (see its module docstring) writes two parallel
CSV groups per trajectory under <out>_analysis/:

  fem/<stem>_gc_barrier.csv, fem/<stem>_rotation.csv          native d3plot
      resolution, no --frame-stride / node sampling ("ground truth")
  downsampled/<stem>_gc_barrier.csv, downsampled/<stem>_rotation.csv
      read back from the finished HDF5 ("what the model actually trains on")

This script overlays the two for a given trajectory name and renders 7
PNGs — rotation (roll/pitch/yaw), barrier resultant displacement, ORA_x,
ORA_y, ASI — so a stride/sampling choice that's too coarse is visible
directly, no rollout needed. ORA_x/ORA_y/ASI are plotted RAW (no
moving-average filter) by default — pass --window-ms 50 for the EN 1317
50ms-filtered values instead.

Same math as dataset/gc_barrier.py's plot_gc_barrier_full_res / plot_rotation,
deliberately duplicated (not imported) rather than pulled in from there — see
src/plot_gc_barrier.py's docstring for the same reasoning: dataset/gc_barrier.py
imports lasso/joblib/scipy just to be importable at all (needed for its own
d3plot extraction), while this script only reads already-exported CSVs, so it
stays numpy+matplotlib only.

Usage (from the repo root; numpy+matplotlib only, no --nv needed — same
pattern as src/plot_gc_barrier.py's apptainer invocation in
configs/experiments/rollout_weitj.sh):
    apptainer exec --bind /raid /staging/proj_iim1/xrkong/container/collider.sif \\
        python -m dataset.compare_downsample_fem T_lok_F_shape_barrier_9_3_100km

    apptainer exec --bind /raid /staging/proj_iim1/xrkong/container/collider.sif \\
        python -m dataset.compare_downsample_fem T_lok_F_shape_barrier_9_3_100km \\
        --base-dir /raid/proj_iim1/xrkong/h5_fps_2ms_no_wheel \\
        --out-dir  /tmp/compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_BASE_DIR = Path("/raid/proj_iim1/xrkong/h5_fps_2ms_no_wheel")

# 1 g in mm/s^2 (dataset/constants.py's G_IN_MM_S2, duplicated — see module
# docstring) and EN 1317 ASI: ASI(t) = sqrt((ax/12)^2+(ay/9)^2+(az/10)^2),
# with ax/ay/az the 50ms moving-average-filtered accelerations in g.
G_IN_MM_S2 = 9806.65
ASI_LIMIT_X_G = 12.0
ASI_LIMIT_Y_G = 9.0
ASI_LIMIT_Z_G = 10.0
ASI_FILTER_WINDOW_MS = 50.0

# Times New Roman per user request (falls back to the metric-compatible
# Nimbus Roman, then Liberation Serif, if Times New Roman itself isn't
# installed — matplotlib silently drops to the next name in font.serif
# that resolves on this machine).
_RCPARAMS = {
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}

# dataviz skill's validated default palette, categorical slots 1/2 (blue/orange)
# — adjacent-pair CVD ΔE 9.1, clears the >=8 target. Downsampled also gets
# markers at its own (much sparser) sample times, so the stride is visible
# directly rather than just inferred from a smoother line.
STYLE_FEM = dict(color="#2a78d6", linestyle="-", linewidth=1.6,
                  label="FEM (full res)", zorder=2)
STYLE_DOWNSAMPLED = dict(color="#eb6834", linestyle="--", linewidth=1.4,
                          marker="o", markersize=3.5, markeredgewidth=0,
                          alpha=0.9, label="Downsampled", zorder=3)


def drop_near_duplicate_frames(
    times: np.ndarray, positions: np.ndarray, rel_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop samples whose time gap to the previous KEPT sample is a tiny
    fraction of the sequence's typical (median) gap — a near-duplicate
    terminal/restart-dump d3plot state, not real motion (see
    dataset/gc_barrier.py's identical helper)."""
    if len(times) < 3:
        return times, positions
    diffs = np.diff(times)
    median_dt = np.median(diffs)
    keep = [0]
    for i in range(1, len(times)):
        if times[i] - times[keep[-1]] >= rel_threshold * median_dt:
            keep.append(i)
    if len(keep) < len(times):
        print(f"[Compare] dropped {len(times) - len(keep)} near-duplicate frame(s) "
              f"(gap < {rel_threshold:.0%} of median dt={median_dt * 1000:.3f} ms) "
              f"— likely a terminal/restart-dump artifact, not real motion")
    keep = np.array(keep)
    return times[keep], positions[keep]


def central_diff_accel(pos: np.ndarray) -> np.ndarray:
    """acc[i] = pos[i+1] - 2*pos[i] + pos[i-1] (centered second difference),
    properly time-aligned with time[i] — no lag.

    dataset/gc_barrier.py's _forward_diff_padded instead does
    vel[i]=pos[i+1]-pos[i] then acc[k]=vel[k+1]-vel[k], which expands to the
    same pos[k+2]-2*pos[k+1]+pos[k] stencil but STORES the result at index
    k instead of k+1 (its own true center) — a one-frame lag. That's
    invisible when only one series is ever plotted (dataset/gc_barrier.py,
    src/plot_gc_barrier.py), but here FEM's frame is ~5ms and downsampled's
    is ~20ms, so the same one-frame lag becomes a different absolute-time
    shift on each curve — a spurious relative phase mismatch between the
    two overlaid ORA/ASI curves, not a real downsampling effect. Using the
    correctly-centered stencil directly (rather than forward-diff-of-forward-diff)
    avoids that confound. Edge frames (i=0, i=T-1) edge-hold the nearest
    interior value, since the centered stencil needs a neighbor on each side.
    """
    T = pos.shape[0]
    pos64 = pos.astype(np.float64)
    acc = np.zeros_like(pos64)
    if T >= 3:
        acc[1:-1] = pos64[2:] - 2 * pos64[1:-1] + pos64[:-2]
        acc[0] = acc[1]
        acc[-1] = acc[-2]
    return acc.astype(np.float32)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, edge-padded so the output length matches x."""
    window = max(1, int(round(window)))
    if window <= 1:
        return x.astype(float)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    xpad = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xpad, kernel, mode="valid")


def accel_to_g(accel_per_frame2: np.ndarray, dt_seconds: float) -> np.ndarray:
    """Convert centered-difference acceleration (mm/dt^2, dt = 1 frame) to g's."""
    return accel_per_frame2 / dt_seconds ** 2 / G_IN_MM_S2


def compute_asi(ax_g: np.ndarray, ay_g: np.ndarray, az_g: np.ndarray) -> np.ndarray:
    return np.sqrt((ax_g / ASI_LIMIT_X_G) ** 2 + (ay_g / ASI_LIMIT_Y_G) ** 2 +
                   (az_g / ASI_LIMIT_Z_G) ** 2)


def _paths(base_dir: Path, trajectory: str) -> dict[str, Path]:
    analysis_dir = base_dir / f"{trajectory}_analysis"
    fem_dir = analysis_dir / "fem"
    ds_dir = analysis_dir / "downsampled"
    return {
        "fem_gc_barrier":        fem_dir / f"{trajectory}_gc_barrier.csv",
        "fem_rotation":          fem_dir / f"{trajectory}_rotation.csv",
        "downsampled_gc_barrier": ds_dir / f"{trajectory}_gc_barrier.csv",
        "downsampled_rotation":   ds_dir / f"{trajectory}_rotation.csv",
    }


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — has dataset/build_dataset.py been run for this "
            f"trajectory? (see its module docstring for the <out>_analysis/ layout)")
    return path


def load_gc_barrier_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(_require(path), delimiter=",", names=True)


def load_rotation_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(_require(path), delimiter=",", names=True)


class Series:
    """One curve's derived quantities: filtered ORA_x/ORA_y, ASI, barrier
    resultant displacement — same definitions as dataset/gc_barrier.py's
    plot_gc_barrier_full_res, just packaged for a two-curve overlay."""

    def __init__(self, time: np.ndarray, ax_g: np.ndarray, ay_g: np.ndarray,
                 az_g: np.ndarray, disp_mag: np.ndarray, window_frames: int):
        self.time = time
        self.ax_filt = moving_average(ax_g, window_frames)
        self.ay_filt = moving_average(ay_g, window_frames)
        az_filt = moving_average(az_g, window_frames)
        self.asi = compute_asi(self.ax_filt, self.ay_filt, az_filt)
        self.disp = disp_mag


def window_frames_for(times: np.ndarray, window_ms: float) -> tuple[float, int]:
    """Each CSV has its own dt (FEM is native-res, e.g. ~5ms; downsampled is
    whatever --frame-stride build_dataset.py used, e.g. ~20ms) — convert the
    ms-based moving-average window to frame-count separately per series."""
    dt_seconds = float(np.median(np.diff(times)))
    window_frames = max(1, round((window_ms / 1000.0) / dt_seconds)) if window_ms > 0 else 1
    return dt_seconds, window_frames


def build_series(data: np.ndarray, window_ms: float, log_prefix: str) -> Series:
    """data: a gc_barrier.csv loaded via load_gc_barrier_csv (plain column
    scheme — gc_pos/vel/acc_x/y/z, barrier_pos/vel/acc_x/y/z, barrier_disp_x/y/z,
    time — no gt_/pred_ prefix, since both fem/ and downsampled/ use
    dataset/gc_barrier.py's export_gc_barrier_csv).

    Recomputes acceleration + displacement fresh from raw positions/time
    (does not trust the CSV's own gc_acc_x/y/z columns) after re-applying
    the near-duplicate-terminal-frame drop, same reasoning as
    dataset/gc_barrier.py's replot_from_csv."""
    times = data["time"].astype(np.float64)
    gc_pos = np.stack([data["gc_pos_x"], data["gc_pos_y"], data["gc_pos_z"]], axis=-1)
    barrier_pos = np.stack(
        [data["barrier_pos_x"], data["barrier_pos_y"], data["barrier_pos_z"]], axis=-1)
    positions = np.stack([gc_pos, barrier_pos], axis=1)   # (T, 2, 3)

    times, positions = drop_near_duplicate_frames(times, positions)
    dt_seconds, window_frames = window_frames_for(times, window_ms)
    if window_frames <= 1:
        print(f"{log_prefix} dt = {dt_seconds * 1000:.2f} ms, no smoothing "
              f"(raw per-frame values)")
    else:
        print(f"{log_prefix} dt = {dt_seconds * 1000:.2f} ms, moving-average window = "
              f"{window_frames} frame(s) (~{window_frames * dt_seconds * 1000:.1f} ms)")

    gc_acc = central_diff_accel(positions[:, 0])
    ax_g = accel_to_g(gc_acc[:, 0], dt_seconds)
    ay_g = accel_to_g(gc_acc[:, 1], dt_seconds)
    az_g = accel_to_g(gc_acc[:, 2], dt_seconds)

    barrier_disp = positions[:, 1] - positions[0, 1]
    disp_mag = np.sqrt(np.sum(barrier_disp ** 2, axis=-1))
    return Series(times, ax_g, ay_g, az_g, disp_mag, window_frames)


def make_plot(x_label: str, y_label: str, title: str,
              fem_curve: tuple[np.ndarray, np.ndarray],
              downsampled_curve: tuple[np.ndarray, np.ndarray],
              out_path: Path, hlines: list[tuple[float, str]] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_RCPARAMS)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(*fem_curve, **STYLE_FEM)
    ax.plot(*downsampled_curve, **STYLE_DOWNSAMPLED)
    for y, label in (hlines or []):
        ax.axhline(y, color="0.5", linestyle=":", linewidth=1.0, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Compare] plot saved -> {out_path}")


def run(trajectory: str, base_dir: Path, out_dir: Path, window_ms: float) -> None:
    paths = _paths(base_dir, trajectory)

    fem_data = load_gc_barrier_csv(paths["fem_gc_barrier"])
    ds_data = load_gc_barrier_csv(paths["downsampled_gc_barrier"])
    fem_rot = load_rotation_csv(paths["fem_rotation"])
    ds_rot = load_rotation_csv(paths["downsampled_rotation"])

    fem = build_series(fem_data, window_ms, "[Compare] FEM")
    ds = build_series(ds_data, window_ms, "[Compare] Downsampled")

    title_suffix = f" — {trajectory}"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_plot("Time (s)", "ORA$_x$ (g)", f"Longitudinal acceleration (ORA$_x$){title_suffix}",
               (fem.time, fem.ax_filt), (ds.time, ds.ax_filt),
               out_dir / "gc_barrier_ora_x.png")

    make_plot("Time (s)", "ORA$_y$ (g)", f"Lateral acceleration (ORA$_y$){title_suffix}",
               (fem.time, fem.ay_filt), (ds.time, ds.ay_filt),
               out_dir / "gc_barrier_ora_y.png")

    make_plot("Time (s)", "ASI (-)", f"Acceleration Severity Index (EN 1317){title_suffix}",
               (fem.time, fem.asi), (ds.time, ds.asi),
               out_dir / "gc_barrier_asi.png",
               hlines=[(1.0, "ASI = 1 (Class A limit)")])

    make_plot("Time (s)", "Displacement (mm)", f"Barrier resultant displacement{title_suffix}",
               (fem.time, fem.disp), (ds.time, ds.disp),
               out_dir / "gc_barrier_displacement.png")

    for axis_name, field in (("roll", "roll_deg"), ("pitch", "pitch_deg"), ("yaw", "yaw_deg")):
        make_plot("Time (s)", f"{axis_name.capitalize()} (deg)",
                   f"Vehicle {axis_name} rotation{title_suffix}",
                   (fem_rot["time"], fem_rot[field]), (ds_rot["time"], ds_rot[field]),
                   out_dir / f"rotation_{axis_name}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare downsampled vs. raw FEM ground truth (rotation, barrier "
                     "displacement, ORA_x/y, ASI) for one trajectory.")
    parser.add_argument("trajectory", help="Trajectory/output stem, e.g. "
                        "T_lok_F_shape_barrier_9_3_100km (matches <out>.h5's filename stem "
                        "from dataset/build_dataset.py)")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR,
                        help=f"Directory containing <trajectory>_analysis/ "
                             f"(default: {DEFAULT_BASE_DIR})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to save the 7 PNGs (default: "
                             "<base-dir>/<trajectory>_analysis/comparison)")
    parser.add_argument("--window-ms", type=float, default=0.0,
                        help="Moving-average filter window in ms, applied separately per "
                             "series at its own dt (default: 0 = raw, unfiltered per-frame "
                             f"values; pass {ASI_FILTER_WINDOW_MS:g} for the EN 1317 "
                             "ASI-standard filter)")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.base_dir / f"{args.trajectory}_analysis" / "comparison")
    run(args.trajectory, args.base_dir, out_dir, args.window_ms)


if __name__ == "__main__":
    main()
