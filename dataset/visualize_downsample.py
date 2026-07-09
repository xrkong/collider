"""
Visualize the downsampled point set produced by k_file_downsample.py.

Usage:
    python dataset/visualize_downsample.py out/
    python dataset/visualize_downsample.py out/ --kfile car_and_barriers.k
    python dataset/visualize_downsample.py out/ --save vis.png   # combined + per-panel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# ── region colour map ──────────────────────────────────────────────────────
REGION_STYLE: dict[str, dict] = {
    "barrier_fine":   dict(color="#e63946", label="Barrier fine",   zorder=5, s=1.5, alpha=0.8),
    "barrier_coarse": dict(color="#f4a261", label="Barrier coarse", zorder=4, s=1.5, alpha=0.8),
    "veh_contact":    dict(color="#2a9d8f", label="Veh contact",    zorder=3, s=1.2, alpha=0.7),
    "veh_near":       dict(color="#457b9d", label="Veh near",       zorder=2, s=0.7, alpha=0.5),
    "veh_far":        dict(color="#adb5bd", label="Veh far",        zorder=1, s=0.4, alpha=0.35),
    "force_keep":     dict(color="#f72585", label="Force-keep",     zorder=6, s=25,  alpha=1.0),
}
DEFAULT_STYLE = dict(color="#cccccc", label="other", zorder=0, s=0.3, alpha=0.3)

IMPACT_BBOX_REGIONS = {"barrier_fine", "veh_contact", "force_keep"}
IMPACT_PAD_MM = 800


# ── data loading ───────────────────────────────────────────────────────────

def load_node_ids(out_dir: Path, kfile: str | None) -> np.ndarray:
    p = out_dir / "node_ids.npy"
    if p.exists():
        return np.load(p)
    if kfile is None:
        raise FileNotFoundError(
            f"{p} not found. Re-run k_file_downsample.py (now saves node_ids.npy) "
            "or pass --kfile."
        )
    print(f"node_ids.npy missing — parsing *NODE from {kfile} …")
    nids: list[int] = []
    in_node = False
    with open(kfile) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("$"):
                continue
            if line.startswith("*"):
                in_node = line.split()[0].upper() == "*NODE"
                continue
            if in_node:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        nids.append(int(parts[0]))
                    except ValueError:
                        pass
    arr = np.array(nids, dtype=np.int64)
    np.save(p, arr)
    print(f"  cached {len(arr):,} node IDs → {p}")
    return arr


def load_data(out_dir: Path, kfile: str | None):
    coords_full = np.load(out_dir / "node_ref_coords.npy")
    sampled_nids = np.load(out_dir / "sampled_node_ids.npy")
    ra = json.load(open(out_dir / "region_assignment.json"))
    regions = np.array(ra["region_labels"])

    node_ids = load_node_ids(out_dir, kfile)
    nid_to_row = {int(nid): i for i, nid in enumerate(node_ids)}
    rows = np.array([nid_to_row[int(n)] for n in sampled_nids], dtype=np.int64)
    coords = coords_full[rows]
    return coords, regions


def impact_bbox(x, y, z, regions):
    mask = np.isin(regions, list(IMPACT_BBOX_REGIONS))
    p = IMPACT_PAD_MM
    return dict(
        xlo=x[mask].min() - p,  xhi=x[mask].max() + p,
        ylo=y[mask].min() - p,  yhi=y[mask].max() + p,
        zlo=max(z[mask].min() - 150, 0.0), zhi=z[mask].max() + 150,
    )


# ── drawing primitives ─────────────────────────────────────────────────────

def _scatter2d(ax, x, y, regions, xlabel, ylabel, xlim=None, ylim=None):
    for reg in list(dict.fromkeys(regions)):
        mask = regions == reg
        st = REGION_STYLE.get(reg, DEFAULT_STYLE)
        ax.scatter(x[mask], y[mask], c=st["color"], s=st["s"], alpha=st["alpha"],
                   linewidths=0, zorder=st["zorder"], rasterized=True)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")          # same mm scale on both axes
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def _scatter3d(ax, x, y, z, regions, bb, present_regions):
    iz = (
        (x >= bb["xlo"]) & (x <= bb["xhi"]) &
        (y >= bb["ylo"]) & (y <= bb["yhi"]) &
        (z >= bb["zlo"]) & (z <= bb["zhi"])
    )
    rng = np.random.default_rng(0)
    for reg in present_regions:
        mask = iz & (regions == reg)
        st = REGION_STYLE.get(reg, DEFAULT_STYLE)
        idx = np.where(mask)[0]
        if len(idx) > 8000:
            idx = rng.choice(idx, 8000, replace=False)
        if len(idx) == 0:
            continue
        ax.scatter(x[idx], y[idx], z[idx], c=st["color"],
                   s=max(st["s"] * 4, 2), alpha=min(st["alpha"] + 0.15, 1.0),
                   linewidths=0, depthshade=True, rasterized=True)
    ax.set_xlabel("X (mm)", fontsize=7)
    ax.set_ylabel("Y (mm)", fontsize=7)
    ax.set_zlabel("Z (mm)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(bb["xlo"], bb["xhi"])
    ax.set_ylim(bb["ylo"], bb["yhi"])
    ax.set_zlim(bb["zlo"], bb["zhi"])
    # equal physical scale on all three axes
    ax.set_box_aspect([
        bb["xhi"] - bb["xlo"],
        bb["yhi"] - bb["ylo"],
        bb["zhi"] - bb["zlo"],
    ])


def _add_legend(fig, present_regions):
    patches = [
        mpatches.Patch(color=REGION_STYLE.get(r, DEFAULT_STYLE)["color"],
                       label=REGION_STYLE.get(r, DEFAULT_STYLE)["label"])
        for r in present_regions
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))


# ── panel definitions (one dict per panel) ─────────────────────────────────

def build_panels(x, y, z, regions, bb, present_regions):
    """Return list of (key, title, draw_fn, is_3d) for each panel."""

    def ov(ax):
        _scatter2d(ax, x, y, regions, "X (mm)", "Y (mm)")
        ax.add_patch(Rectangle(
            (bb["xlo"], bb["ylo"]), bb["xhi"] - bb["xlo"], bb["yhi"] - bb["ylo"],
            linewidth=1.4, edgecolor="black", facecolor="none", linestyle="--", zorder=10,
        ))

    def iz_xy(ax):
        _scatter2d(ax, x, y, regions, "X (mm)", "Y (mm)",
                   xlim=(bb["xlo"], bb["xhi"]), ylim=(bb["ylo"], bb["yhi"]))

    def iz_xz(ax):
        _scatter2d(ax, x, z, regions, "X (mm)", "Z (mm)",
                   xlim=(bb["xlo"], bb["xhi"]), ylim=(bb["zlo"], bb["zhi"]))

    def iz_yz(ax):
        _scatter2d(ax, y, z, regions, "Y (mm)", "Z (mm)",
                   xlim=(bb["ylo"], bb["yhi"]), ylim=(bb["zlo"], bb["zhi"]))

    def iz_3d(ax):
        _scatter3d(ax, x, y, z, regions, bb, present_regions)

    return [
        ("overview_xy", "Overview (X–Y, full)  — dashed = impact zone", ov,    False),
        ("impact_xy",   "Impact zone — top (X–Y)",                       iz_xy, False),
        ("impact_xz",   "Impact zone — side (X–Z)",                      iz_xz, False),
        ("impact_yz",   "Impact zone — front (Y–Z)",                     iz_yz, False),
        ("impact_3d",   "Impact zone — 3D",                              iz_3d, True),
    ]


# ── figure sizing helpers ──────────────────────────────────────────────────

def _panel_figsize(key, bb, base=6):
    """Estimate a reasonable standalone figure size (w, h) for each panel."""
    xr = bb["xhi"] - bb["xlo"]
    yr = bb["yhi"] - bb["ylo"]
    zr = bb["zhi"] - bb["zlo"]
    if key == "overview_xy":
        # full barrier ≈ 53 m wide × 25 m tall
        return (14, 7)
    if key == "impact_xy":
        ratio = xr / yr
        return (base * ratio, base)
    if key == "impact_xz":
        ratio = xr / zr
        h = max(base * 0.5, base / ratio)
        return (base, h)
    if key == "impact_yz":
        ratio = yr / zr
        h = max(base * 0.5, base / ratio)
        return (base, h)
    if key == "impact_3d":
        return (8, 6)
    return (base, base)


# ── public entry point ─────────────────────────────────────────────────────

def visualize(out_dir: Path, kfile: str | None, save: str | None) -> None:
    print("Loading data …")
    coords, regions = load_data(out_dir, kfile)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    present_regions = list(dict.fromkeys(regions))
    bb = impact_bbox(x, y, z, regions)

    print(f"  {len(x):,} sampled nodes  "
          f"x=[{x.min():.0f}, {x.max():.0f}]  "
          f"y=[{y.min():.0f}, {y.max():.0f}]  "
          f"z=[{z.min():.0f}, {z.max():.0f}] mm")
    print(f"  Impact bbox  x=[{bb['xlo']:.0f}, {bb['xhi']:.0f}]  "
          f"y=[{bb['ylo']:.0f}, {bb['yhi']:.0f}]  "
          f"z=[{bb['zlo']:.0f}, {bb['zhi']:.0f}] mm")

    panels = build_panels(x, y, z, regions, bb, present_regions)
    count_str = "  ".join(
        f"{REGION_STYLE.get(r, DEFAULT_STYLE)['label']}: {(regions == r).sum():,}"
        for r in present_regions
    )

    # ── save each panel independently ─────────────────────────────────────
    if save:
        stem = Path(save).with_suffix("")
        for key, title, draw_fn, is_3d in panels:
            fs = _panel_figsize(key, bb)
            fig_p = plt.figure(figsize=fs)
            if is_3d:
                ax_p = fig_p.add_subplot(111, projection="3d")
            else:
                ax_p = fig_p.add_subplot(111)
            draw_fn(ax_p)
            ax_p.set_title(title, fontsize=9)
            _add_legend(fig_p, present_regions)
            out_path = f"{stem}_{key}.png"
            fig_p.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig_p)
            print(f"  Saved panel → {out_path}")

    # ── combined figure ────────────────────────────────────────────────────
    # layout: row0=overview, row1=three 2D zooms, row2=3D
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"Downsampled mesh — {len(x):,} nodes  |  {count_str}",
        fontsize=9, y=0.99,
    )

    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                          left=0.05, right=0.97, top=0.96, bottom=0.07)

    panel_axes = {
        "overview_xy": fig.add_subplot(gs[0, :]),
        "impact_xy":   fig.add_subplot(gs[1, 0]),
        "impact_xz":   fig.add_subplot(gs[1, 1]),
        "impact_yz":   fig.add_subplot(gs[1, 2]),
        "impact_3d":   fig.add_subplot(gs[2, :], projection="3d"),
    }

    for key, title, draw_fn, _ in panels:
        ax = panel_axes[key]
        draw_fn(ax)
        ax.set_title(title, fontsize=9)

    _add_legend(fig, present_regions)

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved combined → {save}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", help="Directory from k_file_downsample.py")
    parser.add_argument("--kfile", default=None,
                        help="k-file path (only if node_ids.npy is absent)")
    parser.add_argument("--save", default=None,
                        help="Save to this path (also saves each panel as <stem>_<key>.png)")
    args = parser.parse_args()
    visualize(Path(args.out_dir), args.kfile, args.save)


if __name__ == "__main__":
    main()
