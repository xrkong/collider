"""Plot vehicle-GC / barrier kinematics: ground truth vs one-step vs autoregressive.

Reads the gc_barrier_onestep.csv and gc_barrier_autoregressive.csv produced
by src/rollout.py (see src/gc_barrier.py for how the two tracked points —
vehicle GC and barrier reference — are located and exported) and renders four
academic-style figures, each overlaying ground-truth / one-step / autoregressive
curves:

  1. ORA_x  — vehicle GC longitudinal acceleration (g)
  2. ORA_y  — vehicle GC lateral acceleration (g)
  3. ASI    — EN 1317 Acceleration Severity Index (dimensionless)
  4. Barrier resultant displacement (mm)

Ground truth is read from the gt_* columns already embedded in the one-step
CSV (identical to the autoregressive CSV's gt_* columns — both cover the
same underlying frame range) — this is the DOWNSAMPLED ground truth, i.e.
whatever --frame-stride the training h5 used.

Optionally overlays a 5th (well, 4th-drawn, see below) reference: the FEM
full native-resolution ground truth CSV produced by dataset/gc_barrier.py
(<out_stem>_analysis/<out_stem>_gc_barrier.csv, next to the training h5 —
NOT under outputs/rollouts/) via --fem-csv. Since one-step is designed to
closely track the downsampled GT, plotting both as solid lines of similar
weight makes them visually indistinguishable — so when --fem-csv is given,
the downsampled GT is drawn as markers only (at its own sparser timestamps)
against the FEM curve's continuous line, and one-step's dashed line is
checked against those markers instead of fighting a second solid line for
the same pixels.

Usage:
    python src/plot_gc_barrier.py outputs/rollouts/wj06
    python src/plot_gc_barrier.py outputs/rollouts/wj06 --stem T_lok_F_shape_barrier_9_3_60km
    python src/plot_gc_barrier.py outputs/rollouts/wj06 \
        --fem-csv /raid/proj_iim1/xrkong/h5_fps_2ms_no_wheel/T_lok_F_shape_barrier_9_3_60km_analysis/T_lok_F_shape_barrier_9_3_60km_gc_barrier.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.constants import G_IN_MM_S2  # 1 g in mm/s^2, for unit conversion

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_RCPARAMS = {
    "font.family":     "DejaVu Serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

# EN 1317 Acceleration Severity Index: ASI(t) = sqrt((ax/12)^2 + (ay/9)^2 + (az/10)^2),
# with ax/ay/az the 50ms moving-average-filtered accelerations in g.
ASI_LIMIT_X_G = 12.0
ASI_LIMIT_Y_G = 9.0
ASI_LIMIT_Z_G = 10.0
ASI_FILTER_WINDOW_MS = 50.0

# Fixed categorical colors (dataviz skill's validated default palette, slots 1-3),
# each paired with a distinct linestyle so curves stay distinguishable in
# grayscale print / for colorblind readers.
STYLE_GT = dict(color="#2a78d6", linestyle="-",  linewidth=1.8, label="Ground truth")
STYLE_OS = dict(color="#008300", linestyle="--", linewidth=1.6, label="One-step")
STYLE_AR = dict(color="#e87ba4", linestyle="-.", linewidth=1.6, label="Autoregressive")

# Used only when --fem-csv is given (see module docstring): the downsampled
# GT switches from a solid line to markers-only (it's now visibly a SUBSET
# of the continuous FEM curve, not a competing line), and the FEM curve
# itself is a thin neutral-ink backdrop rather than a 4th categorical color
# — it's the reference everything else is checked against, not another
# model output.
STYLE_FEM      = dict(color="0.35", linestyle="-", linewidth=1.2, alpha=0.8,
                       label="FEM ground truth (full res)", zorder=1)
STYLE_GT_MARKER = dict(color="#2a78d6", linestyle="none", marker="o", markersize=4.5,
                        markeredgewidth=0, label="Ground truth (downsampled)", zorder=3)


def find_csvs(rollout_dir: Path, stem: str | None) -> tuple[Path | None, Path]:
    """Autoregressive CSV is required; one-step is optional (absent for a
    rollout.py run with --mode autoregressive, e.g. rollout_all_weitj.slurm)."""
    prefix = f"{stem}_" if stem else "*"
    onestep_matches = sorted(rollout_dir.glob(f"{prefix}gc_barrier_onestep.csv"))
    autoreg_matches = sorted(rollout_dir.glob(f"{prefix}gc_barrier_autoregressive.csv"))
    if len(autoreg_matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one gc_barrier_autoregressive.csv in {rollout_dir}. "
            f"Found {[p.name for p in autoreg_matches]}. "
            f"Pass --stem to disambiguate a multi-test-set folder."
        )
    if len(onestep_matches) > 1:
        raise FileNotFoundError(
            f"Expected at most one gc_barrier_onestep.csv in {rollout_dir}. "
            f"Found {[p.name for p in onestep_matches]}. "
            f"Pass --stem to disambiguate a multi-test-set folder."
        )
    return (onestep_matches[0] if onestep_matches else None), autoreg_matches[0]


def load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def _drop_near_duplicate_frames(
    times: np.ndarray, positions: np.ndarray, rel_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop samples whose time gap to the previous KEPT sample is a tiny
    fraction of the sequence's typical (median) gap — a near-duplicate
    terminal/restart-dump d3plot state, not real motion (see
    dataset/gc_barrier.py, which has the same helper — deliberately
    duplicated here rather than imported, to keep this script's
    dependencies to numpy+matplotlib; dataset/gc_barrier.py pulls in
    lasso/joblib/scipy just to be importable at all)."""
    if len(times) < 3:
        return times, positions
    diffs = np.diff(times)
    median_dt = np.median(diffs)
    keep = [0]
    for i in range(1, len(times)):
        if times[i] - times[keep[-1]] >= rel_threshold * median_dt:
            keep.append(i)
    if len(keep) < len(times):
        print(f"[Plot] dropped {len(times) - len(keep)} near-duplicate frame(s) "
              f"(gap < {rel_threshold:.0%} of median dt={median_dt * 1000:.3f} ms) "
              f"— likely a terminal/restart-dump artifact, not real motion")
    keep = np.array(keep)
    return times[keep], positions[keep]


def _forward_diff_padded(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """vel[i]=pos[i+1]-pos[i], acc[i]=vel[i+1]-vel[i], tail-padded — same
    convention as src/rollout.py's _derive_padded_kinematics / dataset/
    gc_barrier.py's _forward_diff_padded (duplicated, see note above)."""
    T = pos.shape[0]
    pos64 = pos.astype(np.float64)
    vel = np.zeros_like(pos64)
    acc = np.zeros_like(pos64)
    n_vel_valid = max(T - 1, 0)
    n_acc_valid = max(T - 2, 0)
    if n_vel_valid > 0:
        vel[:n_vel_valid] = np.diff(pos64, axis=0)
        vel[n_vel_valid:] = vel[n_vel_valid - 1]
    if n_acc_valid > 0:
        acc[:n_acc_valid] = np.diff(vel[:n_vel_valid], axis=0)
        acc[n_acc_valid:] = acc[n_acc_valid - 1]
    return vel.astype(np.float32), acc.astype(np.float32)


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
    """Convert forward-difference acceleration (mm/dt^2, dt = 1 frame) to g's."""
    return accel_per_frame2 / dt_seconds ** 2 / G_IN_MM_S2


def compute_asi(ax_g: np.ndarray, ay_g: np.ndarray, az_g: np.ndarray) -> np.ndarray:
    return np.sqrt((ax_g / ASI_LIMIT_X_G) ** 2 + (ay_g / ASI_LIMIT_Y_G) ** 2 +
                   (az_g / ASI_LIMIT_Z_G) ** 2)


class Series:
    """One curve's derived quantities: filtered ORA_x/ORA_y, ASI, displacement."""

    def __init__(self, time: np.ndarray, ax_g: np.ndarray, ay_g: np.ndarray,
                 az_g: np.ndarray, disp_mag: np.ndarray, window_frames: int):
        self.time = time
        self.ax_filt = moving_average(ax_g, window_frames)
        self.ay_filt = moving_average(ay_g, window_frames)
        az_filt = moving_average(az_g, window_frames)
        self.asi = compute_asi(self.ax_filt, self.ay_filt, az_filt)
        self.disp = disp_mag


def build_series(data: np.ndarray, prefix: str, dt_seconds: float,
                  window_frames: int) -> Series:
    ax_g = accel_to_g(data[f"{prefix}_gc_acc_x"], dt_seconds)
    ay_g = accel_to_g(data[f"{prefix}_gc_acc_y"], dt_seconds)
    az_g = accel_to_g(data[f"{prefix}_gc_acc_z"], dt_seconds)
    dx = data[f"{prefix}_barrier_disp_x"]
    dy = data[f"{prefix}_barrier_disp_y"]
    dz = data[f"{prefix}_barrier_disp_z"]
    disp_mag = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return Series(data["time"], ax_g, ay_g, az_g, disp_mag, window_frames)


def build_series_plain(data: np.ndarray, window_ms: float) -> Series:
    """Same job as build_series, but for the FEM full-res CSV from
    dataset/gc_barrier.py (columns with no gt_/pred_ prefix: gc_pos/vel/acc_x/y/z,
    barrier_pos/vel/acc_x/y/z, barrier_disp_x/y/z, time — ground-truth only,
    no model).

    Deliberately does NOT trust that CSV's own gc_acc_x/y/z (or
    barrier_disp_x/y/z) columns — an older FEM CSV, extracted before
    dataset/gc_barrier.py's near-duplicate-terminal-frame fix, has the
    resulting spurious huge acceleration spike baked into exactly those
    columns. Recomputes both fresh from raw positions + time instead, after
    the same duplicate-frame drop dataset/build_dataset.py itself now
    applies at extraction time (so a stale CSV plots correctly here even
    without re-running the (expensive) extraction)."""
    times = data["time"].astype(np.float64)
    gc_pos = np.stack([data["gc_pos_x"], data["gc_pos_y"], data["gc_pos_z"]], axis=-1)
    barrier_pos = np.stack(
        [data["barrier_pos_x"], data["barrier_pos_y"], data["barrier_pos_z"]], axis=-1)
    positions = np.stack([gc_pos, barrier_pos], axis=1)   # (T, 2, 3)

    times, positions = _drop_near_duplicate_frames(times, positions)
    dt_seconds, window_frames = window_frames_for(times, window_ms)
    print(f"[Plot] FEM full-res dt = {dt_seconds * 1000:.2f} ms, moving-average window = "
          f"{window_frames} frame(s) (~{window_frames * dt_seconds * 1000:.1f} ms)")

    _, gc_acc = _forward_diff_padded(positions[:, 0])
    ax_g = accel_to_g(gc_acc[:, 0], dt_seconds)
    ay_g = accel_to_g(gc_acc[:, 1], dt_seconds)
    az_g = accel_to_g(gc_acc[:, 2], dt_seconds)

    barrier_disp = positions[:, 1] - positions[0, 1]
    disp_mag = np.sqrt(np.sum(barrier_disp ** 2, axis=-1))
    return Series(times, ax_g, ay_g, az_g, disp_mag, window_frames)


def window_frames_for(times: np.ndarray, window_ms: float) -> tuple[float, int]:
    """Each CSV has its own dt (the FEM CSV is native-res, ~5ms; the rollout
    CSVs are whatever --frame-stride the training h5 used, e.g. ~20ms) — the
    ms-based moving-average window must be converted to frame-count
    separately per series, not shared, or the "50ms" filter would be the
    wrong width for whichever series doesn't match."""
    dt_seconds = float(np.median(np.diff(times)))
    window_frames = max(1, round((window_ms / 1000.0) / dt_seconds)) if window_ms > 0 else 1
    return dt_seconds, window_frames


def make_plot(x_label: str, y_label: str, title: str, curves: list,
              out_path: Path, hlines: list[tuple[float, str]] | None = None,
              ylim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for time, values, style in curves:
        ax.plot(time, values, **style)
    for y, label in (hlines or []):
        ax.axhline(y, color="0.5", linestyle=":", linewidth=1.0, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot vehicle-GC / barrier kinematics: GT vs one-step vs autoregressive")
    parser.add_argument("rollout_dir", type=Path,
                        help="Folder containing gc_barrier_onestep.csv / "
                             "gc_barrier_autoregressive.csv (e.g. outputs/rollouts/wj06)")
    parser.add_argument("--stem", default=None,
                        help="Test-set filename prefix, for multi-test-set folders "
                             "(e.g. 'T_lok_F_shape_barrier_9_3_60km')")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to save the PNGs (defaults to rollout_dir)")
    parser.add_argument("--window-ms", type=float, default=ASI_FILTER_WINDOW_MS,
                        help="Moving-average filter window in ms (default: 50ms, per EN 1317)")
    parser.add_argument("--fem-csv", type=Path, default=None,
                        help="Optional: the FEM full native-resolution ground-truth CSV from "
                             "dataset/gc_barrier.py (<out_stem>_analysis/<out_stem>_gc_barrier.csv, "
                             "next to the training h5 — a separate directory tree from "
                             "outputs/rollouts/). Overlays it as a 4th curve and switches the "
                             "downsampled GT to markers-only so it doesn't visually merge with "
                             "one-step's near-identical line.")
    args = parser.parse_args()

    plt.rcParams.update(_RCPARAMS)

    onestep_path, autoreg_path = find_csvs(args.rollout_dir, args.stem)
    onestep_data = load_csv(onestep_path) if onestep_path is not None else None
    autoreg_data = load_csv(autoreg_path)

    # GT lives in both CSVs' gt_* columns (identical range) — fall back to the
    # autoregressive CSV when one-step wasn't run (--mode autoregressive).
    gt_source = onestep_data if onestep_data is not None else autoreg_data

    dt_seconds, window_frames = window_frames_for(gt_source["time"], args.window_ms)
    print(f"[Plot] dt = {dt_seconds * 1000:.2f} ms, moving-average window = "
          f"{window_frames} frame(s) (~{window_frames * dt_seconds * 1000:.1f} ms)")

    gt = build_series(gt_source, "gt", dt_seconds, window_frames)
    os_ = build_series(onestep_data, "pred", dt_seconds, window_frames) if onestep_data is not None else None
    ar = build_series(autoreg_data, "pred", dt_seconds, window_frames)

    fem = None
    if args.fem_csv is not None:
        fem_data = load_csv(args.fem_csv)
        fem = build_series_plain(fem_data, args.window_ms)

    # Only switch the downsampled-GT style to markers-only when the FEM
    # backdrop is actually present — with no --fem-csv, behavior/appearance
    # is unchanged from before (a solid GT line).
    gt_style = STYLE_GT_MARKER if fem is not None else STYLE_GT

    out_dir = args.out_dir or args.rollout_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    name_suffix = f" — {args.stem or args.rollout_dir.name}"

    def _curves(attr: str) -> list:
        curves = []
        if fem is not None:
            curves.append((fem.time, getattr(fem, attr), STYLE_FEM))
        curves.append((gt.time, getattr(gt, attr), gt_style))
        if os_ is not None:
            curves.append((os_.time, getattr(os_, attr), STYLE_OS))
        curves.append((ar.time, getattr(ar, attr), STYLE_AR))
        return curves

    make_plot(
        "Time (s)", "ORA$_x$ (g)", f"Longitudinal acceleration (ORA$_x$){name_suffix}",
        _curves("ax_filt"), out_dir / "gc_barrier_ora_x.png")

    make_plot(
        "Time (s)", "ORA$_y$ (g)", f"Lateral acceleration (ORA$_y$){name_suffix}",
        _curves("ay_filt"), out_dir / "gc_barrier_ora_y.png")

    make_plot(
        "Time (s)", "ASI (–)", f"Acceleration Severity Index (EN 1317){name_suffix}",
        _curves("asi"), out_dir / "gc_barrier_asi.png",
        hlines=[(1.0, "ASI = 1 (Class A limit)")])

    make_plot(
        "Time (s)", "Displacement (mm)", f"Barrier resultant displacement{name_suffix}",
        _curves("disp"), out_dir / "gc_barrier_displacement.png")


if __name__ == "__main__":
    main()
