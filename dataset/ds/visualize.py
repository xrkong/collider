"""
visualize.py — animate a build_dataset.py HDF5 output as a top-down GIF.

Reads /states/positions + /states/times + /metadata/region_label, colors
each sampled node by region, and writes one frame per kept time state.
Axis limits are fixed to the full position range across every frame so the
view doesn't rescale as the vehicle travels — that's the whole point of a
sanity-check GIF: confirm the downsampled set tracks the real motion.

Usage
-----
python -m dataset.ds.visualize --h5 output.h5 --out anim.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REGION_STYLE: dict[str, dict] = {
    "barrier_fine":   dict(color="#e63946", label="Barrier fine",   zorder=5, s=1.5, alpha=0.8),
    "barrier_coarse": dict(color="#f4a261", label="Barrier coarse", zorder=4, s=1.5, alpha=0.8),
    "veh_contact":    dict(color="#2a9d8f", label="Veh contact",    zorder=3, s=1.2, alpha=0.7),
    "veh_near":       dict(color="#457b9d", label="Veh near",       zorder=2, s=0.7, alpha=0.5),
    "veh_far":        dict(color="#adb5bd", label="Veh far",        zorder=1, s=0.4, alpha=0.35),
    "force_keep":     dict(color="#f72585", label="Force-keep",     zorder=6, s=25,  alpha=1.0),
}
DEFAULT_STYLE = dict(color="#cccccc", label="other", zorder=0, s=0.3, alpha=0.3)


def make_gif(
    h5_path: Path,
    out_path: Path,
    frame_stride: int = 1,
    fps: int = 10,
    max_frames: int = 80,
    dpi: int = 100,
) -> None:
    """Render /states/positions over time as an animated top-down (X-Y) GIF."""
    with h5py.File(h5_path, "r") as f:
        positions = f["states/positions"][:]                       # (T, N, 3)
        times = f["states/times"][:]                                # (T,)
        region_label = f["metadata/region_label"][:].astype(str)    # (N,)

    T = positions.shape[0]
    idx = np.arange(0, T, frame_stride)
    if len(idx) > max_frames:
        idx = np.linspace(0, T - 1, max_frames).astype(int)

    x_all, y_all = positions[:, :, 0], positions[:, :, 1]
    pad = 0.05 * max(x_all.max() - x_all.min(), y_all.max() - y_all.min(), 1.0)
    xlim = (float(x_all.min() - pad), float(x_all.max() + pad))
    ylim = (float(y_all.min() - pad), float(y_all.max() + pad))

    present_regions = list(dict.fromkeys(region_label))

    print(f"Rendering {len(idx)} GIF frames (of {T} available) …")
    frames: list[Image.Image] = []
    for n, fi in enumerate(idx):
        fig, ax = plt.subplots(figsize=(11, 6), dpi=dpi)
        for reg in present_regions:
            mask = region_label == reg
            st = REGION_STYLE.get(reg, DEFAULT_STYLE)
            ax.scatter(positions[fi, mask, 0], positions[fi, mask, 1],
                      c=st["color"], s=st["s"], alpha=st["alpha"],
                      linewidths=0, zorder=st["zorder"], label=st["label"])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"t = {times[fi] * 1e3:.1f} ms   (frame {fi + 1}/{T})")
        if n == 0:
            ax.legend(loc="upper right", fontsize=7, markerscale=4, framealpha=0.9)
        fig.tight_layout()

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(img).convert("RGB"))
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f"Saved GIF: {out_path}  ({len(frames)} frames, {out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a build_dataset.py HDF5 output as a GIF.")
    parser.add_argument("--h5",  type=Path, required=True, help="HDF5 file from build_dataset.py.")
    parser.add_argument("--out", type=Path, required=True, help="Output .gif path.")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Use every N-th stored frame (further subsample beyond the h5's own stride).")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=80,
                        help="Cap total GIF frames (resamples evenly if exceeded).")
    args = parser.parse_args()
    make_gif(args.h5, args.out, args.frame_stride, args.fps, args.max_frames)


if __name__ == "__main__":
    main()
