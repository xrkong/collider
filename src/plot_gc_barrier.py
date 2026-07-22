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
same underlying frame range).

Usage:
    python src/plot_gc_barrier.py outputs/rollouts/wj06
    python src/plot_gc_barrier.py outputs/rollouts/wj06 --stem T_lok_F_shape_barrier_9_3_60km
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


def find_csvs(rollout_dir: Path, stem: str | None) -> tuple[Path, Path]:
    prefix = f"{stem}_" if stem else "*"
    onestep_matches = sorted(rollout_dir.glob(f"{prefix}gc_barrier_onestep.csv"))
    autoreg_matches = sorted(rollout_dir.glob(f"{prefix}gc_barrier_autoregressive.csv"))
    if len(onestep_matches) != 1 or len(autoreg_matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one gc_barrier_onestep.csv and one "
            f"gc_barrier_autoregressive.csv in {rollout_dir}. Found "
            f"onestep={[p.name for p in onestep_matches]}, "
            f"autoregressive={[p.name for p in autoreg_matches]}. "
            f"Pass --stem to disambiguate a multi-test-set folder."
        )
    return onestep_matches[0], autoreg_matches[0]


def load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


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


def make_plot(x_label: str, y_label: str, title: str, curves: list,
              out_path: Path, hlines: list[tuple[float, str]] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for time, values, style in curves:
        ax.plot(time, values, **style)
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
    args = parser.parse_args()

    plt.rcParams.update(_RCPARAMS)

    onestep_path, autoreg_path = find_csvs(args.rollout_dir, args.stem)
    onestep_data = load_csv(onestep_path)
    autoreg_data = load_csv(autoreg_path)

    dt_seconds = float(np.median(np.diff(onestep_data["time"])))
    window_frames = max(1, round((args.window_ms / 1000.0) / dt_seconds))
    print(f"[Plot] dt = {dt_seconds * 1000:.2f} ms, moving-average window = "
          f"{window_frames} frame(s) (~{window_frames * dt_seconds * 1000:.1f} ms)")

    gt = build_series(onestep_data, "gt", dt_seconds, window_frames)
    os_ = build_series(onestep_data, "pred", dt_seconds, window_frames)
    ar = build_series(autoreg_data, "pred", dt_seconds, window_frames)

    out_dir = args.out_dir or args.rollout_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    name_suffix = f" — {args.stem or args.rollout_dir.name}"

    make_plot(
        "Time (s)", "ORA$_x$ (g)", f"Longitudinal acceleration (ORA$_x$){name_suffix}",
        [(gt.time, gt.ax_filt, STYLE_GT),
         (os_.time, os_.ax_filt, STYLE_OS),
         (ar.time, ar.ax_filt, STYLE_AR)],
        out_dir / "gc_barrier_ora_x.png")

    make_plot(
        "Time (s)", "ORA$_y$ (g)", f"Lateral acceleration (ORA$_y$){name_suffix}",
        [(gt.time, gt.ay_filt, STYLE_GT),
         (os_.time, os_.ay_filt, STYLE_OS),
         (ar.time, ar.ay_filt, STYLE_AR)],
        out_dir / "gc_barrier_ora_y.png")

    make_plot(
        "Time (s)", "ASI (–)", f"Acceleration Severity Index (EN 1317){name_suffix}",
        [(gt.time, gt.asi, STYLE_GT),
         (os_.time, os_.asi, STYLE_OS),
         (ar.time, ar.asi, STYLE_AR)],
        out_dir / "gc_barrier_asi.png",
        hlines=[(1.0, "ASI = 1 (Class A limit)")])

    make_plot(
        "Time (s)", "Displacement (mm)", f"Barrier resultant displacement{name_suffix}",
        [(gt.time, gt.disp, STYLE_GT),
         (os_.time, os_.disp, STYLE_OS),
         (ar.time, ar.disp, STYLE_AR)],
        out_dir / "gc_barrier_displacement.png")


if __name__ == "__main__":
    main()
