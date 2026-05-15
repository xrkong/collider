#!/usr/bin/env python3
"""Visualize per-frame velocity, acceleration, and stress statistics from an HDF5 file.

The script expects the layout written by `dataset/d3plot_to_h5.py`:

  /states/times          (T,)
  /states/velocity       (T, N, 3)
  /states/acceleration   (T, N, 3)
  /states/stress         (T, N, 6)

For each frame, it computes the node-wise L2 magnitude of each field and then
plots the per-frame minimum, mean, and maximum as cleaner line charts with
min-max bands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator


FIELD_SPECS: tuple[tuple[str, str], ...] = (
    ("velocity", "Velocity magnitude"),
    ("acceleration", "Acceleration magnitude"),
    ("stress", "Stress magnitude"),
)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
        }
    )


def _load_field(h5f: h5py.File, field_name: str) -> np.ndarray:
    dataset_path = f"states/{field_name}"
    if dataset_path not in h5f:
        raise KeyError(f"Missing dataset: {dataset_path}")
    dataset = cast(h5py.Dataset, h5f[dataset_path])
    return np.asarray(dataset[:])


def _resolve_output_paths(input_h5: Path, output_arg: Path | None) -> tuple[Path, Path]:
    """Return (magnitude_path, component_path) for a file or directory output argument."""
    if output_arg is None:
        base_dir = input_h5.parent
        stem = input_h5.stem
        return base_dir / f"{stem}_magnitude.png", base_dir / f"{stem}_components.png"

    if output_arg.exists() and output_arg.is_dir():
        stem = input_h5.stem
        return output_arg / f"{stem}_magnitude.png", output_arg / f"{stem}_components.png"

    if output_arg.suffix.lower() == ".png":
        return output_arg.with_name(f"{output_arg.stem}_magnitude.png"), output_arg.with_name(
            f"{output_arg.stem}_components.png"
        )

    output_arg.mkdir(parents=True, exist_ok=True)
    stem = input_h5.stem
    return output_arg / f"{stem}_magnitude.png", output_arg / f"{stem}_components.png"


def _compute_frame_stats(field_data: np.ndarray) -> np.ndarray:
    """Return per-frame [min, mean, max] statistics for a (T, N, D) field."""
    if field_data.ndim != 3:
        raise ValueError(f"Expected a 3D array shaped (T, N, D), got {field_data.shape}")

    magnitudes = np.linalg.norm(field_data.astype(np.float64), axis=-1)
    frame_min = magnitudes.min(axis=1)
    frame_mean = magnitudes.mean(axis=1)
    frame_max = magnitudes.max(axis=1)
    return np.stack([frame_min, frame_mean, frame_max], axis=1)


def _global_field_stats(field_data: np.ndarray) -> tuple[float, float, float, float]:
    """Return global component and magnitude min/max for a (T, N, D) field."""
    if field_data.ndim != 3:
        raise ValueError(f"Expected a 3D array shaped (T, N, D), got {field_data.shape}")

    data = field_data.astype(np.float64)
    magnitudes = np.linalg.norm(data, axis=-1)
    return float(np.nanmin(data)), float(np.nanmax(data)), float(np.nanmin(magnitudes)), float(np.nanmax(magnitudes))


def _compute_percentiles(field_data: np.ndarray) -> tuple[float, float, float]:
    """Return 50th (median), 90th, and 95th percentiles of magnitude for a (T, N, D) field."""
    if field_data.ndim != 3:
        raise ValueError(f"Expected a 3D array shaped (T, N, D), got {field_data.shape}")

    data = field_data.astype(np.float64)
    magnitudes = np.linalg.norm(data, axis=-1)
    magnitudes_flat = magnitudes.flatten()
    
    p50 = float(np.percentile(magnitudes_flat, 50))
    p90 = float(np.percentile(magnitudes_flat, 90))
    p95 = float(np.percentile(magnitudes_flat, 95))
    
    return p50, p90, p95


def _plot_field(ax: Axes, frame_stats: np.ndarray, title: str, color: str) -> None:
    frame_count = frame_stats.shape[0]
    frame_idx = np.arange(frame_count)

    ax.fill_between(frame_idx, frame_stats[:, 0], frame_stats[:, 2], color=color, alpha=0.16, linewidth=0)
    ax.plot(frame_idx, frame_stats[:, 0], color=color, alpha=0.45, linewidth=1.0, label="min")
    ax.plot(frame_idx, frame_stats[:, 1], color=color, linewidth=2.1, label="mean")
    ax.plot(frame_idx, frame_stats[:, 2], color=color, alpha=0.75, linewidth=1.0, linestyle="--", label="max")

    ax.set_title(title)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Magnitude")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.legend(frameon=False, ncol=3, loc="upper right")


def _compute_component_stats(field_data: np.ndarray) -> np.ndarray:
    """Return per-frame [x_min, x_mean, x_max, y_min, y_mean, y_max, z_min, z_mean, z_max] for (T, N, 3) field."""
    if field_data.ndim != 3 or field_data.shape[2] != 3:
        raise ValueError(f"Expected a 3D array shaped (T, N, 3), got {field_data.shape}")

    data = field_data.astype(np.float64)
    results = []
    for comp_idx in range(3):
        comp_data = data[:, :, comp_idx]
        results.append(comp_data.min(axis=1))
        results.append(comp_data.mean(axis=1))
        results.append(comp_data.max(axis=1))
    return np.stack(results, axis=1)


def _plot_component_field(ax: Axes, component_stats: np.ndarray, title: str) -> None:
    """Plot x/y/z components with min/mean/max bands. component_stats shape: (T, 9)."""
    frame_count = component_stats.shape[0]
    frame_idx = np.arange(frame_count)
    colors_xyz = ["#1f77b4", "#d62728", "#2ca02c"]
    comp_names = ["x", "y", "z"]

    for comp_i in range(3):
        x_min = component_stats[:, 3 * comp_i]
        x_mean = component_stats[:, 3 * comp_i + 1]
        x_max = component_stats[:, 3 * comp_i + 2]

        ax.fill_between(frame_idx, x_min, x_max, color=colors_xyz[comp_i], alpha=0.12, linewidth=0)
        ax.plot(frame_idx, x_mean, color=colors_xyz[comp_i], linewidth=2.0, label=f"{comp_names[comp_i]}-mean", marker="o", markersize=2, markevery=max(1, frame_count // 20))

    ax.set_title(title)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Component value")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.legend(frameon=False, ncol=3, loc="best")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)


def main() -> None:
    _set_plot_style()

    parser = argparse.ArgumentParser(
        description="Plot per-frame min/mean/max statistics for velocity, acceleration, and stress from output.h5."
    )
    parser.add_argument("input_h5", type=Path, help="Path to the HDF5 file produced by d3plot_to_h5.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the figure. Defaults to <input_stem>_stats.png next to the input file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    args = parser.parse_args()

    if not args.input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {args.input_h5}")

    magnitude_path, component_path = _resolve_output_paths(args.input_h5, args.output)

    with h5py.File(args.input_h5, "r") as h5f:
        times = np.asarray(cast(h5py.Dataset, h5f["states/times"])[:]) if "states/times" in h5f else None
        field_stats: list[tuple[str, str, np.ndarray]] = []
        for field_name, label in FIELD_SPECS:
            field_data = _load_field(h5f, field_name)
            comp_min, comp_max, mag_min, mag_max = _global_field_stats(field_data)
            print(
                f"[{field_name}] component min/max = {comp_min:.6e} / {comp_max:.6e}, "
                f"magnitude min/max = {mag_min:.6e} / {mag_max:.6e}"
            )
            
            # Calculate and print percentiles
            p50, p90, p95 = _compute_percentiles(field_data)
            print(
                f"[{field_name}] percentiles: p50 (median) = {p50:.6e}, p90 = {p90:.6e}, p95 = {p95:.6e}"
            )
            
            field_stats.append((field_name, label, _compute_frame_stats(field_data)))

        # Prepare component-wise analysis: velocity, acceleration, displacement
        print("\n[Component-wise analysis]")
        vel_data = _load_field(h5f, "velocity")
        acc_data = _load_field(h5f, "acceleration")
        pos_data = _load_field(h5f, "positions")
        ref_pos_data = np.asarray(cast(h5py.Dataset, h5f["metadata/ref_positions"])[:])
        disp_data = pos_data - ref_pos_data[np.newaxis, :, :]

        vel_comp_stats = _compute_component_stats(vel_data)
        acc_comp_stats = _compute_component_stats(acc_data)
        disp_comp_stats = _compute_component_stats(disp_data)

        component_fields = [
            ("velocity", "Velocity x/y/z components", vel_comp_stats),
            ("acceleration", "Acceleration x/y/z components", acc_comp_stats),
            ("displacement", "Displacement x/y/z components", disp_comp_stats),
        ]

    frame_count = field_stats[0][2].shape[0]
    for _, _, stats in field_stats[1:]:
        if stats.shape[0] != frame_count:
            raise ValueError("All fields must have the same number of frames")

    # Create two figures: one for magnitude, one for components
    fig1, axes1 = plt.subplots(len(field_stats), 1, figsize=(14, 10), sharex=True)
    if len(field_stats) == 1:
        axes1 = [axes1]

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for ax, (_, label, stats), color in zip(axes1, field_stats, colors):
        _plot_field(ax, stats, label, color)

    axes1[-1].set_xlabel("Frame index")

    if times is not None and times.shape[0] == frame_count:
        tick_step = max(1, frame_count // 10)
        tick_positions = np.arange(0, frame_count, tick_step)
        tick_labels = [f"{i}\n{times[i]:.3g}s" for i in tick_positions]
        axes1[-1].set_xticks(tick_positions)
        axes1[-1].set_xticklabels(tick_labels, rotation=0)
    else:
        tick_step = max(1, frame_count // 10)
        tick_positions = np.arange(0, frame_count, tick_step)
        tick_labels = [str(i) for i in tick_positions]
        axes1[-1].set_xticks(np.arange(0, frame_count, tick_step))

    fig1.suptitle("Per-frame min / mean / max statistics", y=0.995, fontsize=15)
    fig1.tight_layout(rect=(0, 0, 1, 0.98))
    fig1.savefig(magnitude_path, dpi=200, bbox_inches="tight")
    print(f"Saved magnitude figure to {magnitude_path}")

    # Component-wise figure
    fig2, axes2 = plt.subplots(len(component_fields), 1, figsize=(14, 10), sharex=True)
    if len(component_fields) == 1:
        axes2 = [axes2]

    for ax, (_, label, stats) in zip(axes2, component_fields):
        _plot_component_field(ax, stats, label)

    axes2[-1].set_xlabel("Frame index")

    if times is not None and times.shape[0] == frame_count:
        axes2[-1].set_xticks(tick_positions)
        axes2[-1].set_xticklabels(tick_labels, rotation=0)
    else:
        axes2[-1].set_xticks(np.arange(0, frame_count, tick_step))

    fig2.suptitle("Component-wise (x/y/z) distributions", y=0.995, fontsize=15)
    fig2.tight_layout(rect=(0, 0, 1, 0.98))
    fig2.savefig(component_path, dpi=200, bbox_inches="tight")
    print(f"Saved components figure to {component_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()