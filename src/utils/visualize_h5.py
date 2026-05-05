"""
visualize_h5.py  -  Visualise decimated d3plot HDF5 crash-simulation data.

Steps
-----
  1. Load node positions and part names from the HDF5 file.
  2. Map every node to one of 9 structural categories (matching the
     section headings in required_parts.config) via substring matching.
  3. For each sampled time frame render a 2-panel matplotlib figure
     (side view XZ  +  top view XY) and save as SVG.
     Scatter points are rasterised inside the SVG so file sizes stay small.
  4. Convert each SVG to PNG with cairosvg (or fall back to a direct PNG
     render) and stitch all frames into an animated GIF with Pillow.

Usage
-----
    python visualize_h5.py [--h5 PATH] [--out-dir DIR] [--gif PATH]
                           [--every N] [--dpi N] [--fps N] [--max-pts N]
                           [--pt-size F]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

try:
    import cairosvg
    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_H5      = Path("/home/kong/datasets/barrier/h5/output.h5")
DEFAULT_OUT_DIR = Path("/home/kong/datasets/barrier/h5/frames")
DEFAULT_GIF     = Path("/home/kong/datasets/barrier/h5/crash.gif")
DEFAULT_EVERY   = 1        # use every N-th stored frame  (1 = all)
DEFAULT_DPI     = 120
DEFAULT_FPS     = 10
DEFAULT_MAX_PTS = 80_000   # randomly subsample nodes if more than this
DEFAULT_PT_SIZE = 1.0      # scatter marker size in points^2


# ── Category -> colour map  (order matches required_parts.config sections) ─────
#
# Each tuple: (display_name, [substring_patterns], hex_colour)
# Matching: first category whose ANY pattern is a substring of the
# lowercased part name wins (same rule as the extractor).

CATEGORIES: list[tuple[str, list[str], str]] = [
    ("Barrier",
     ["concrete_fine_mesh"],
     "#E63946"),   # vivid red

    ("Front Crash",
     ["frontface","bumper","bumpersteel","bumpercover","bumperhousing",
      "hood","hoodinner","fender","windshield",
      "railfront","noserail","frontrail","railfrontplate","railfrontplaterear",
      "raillargefront","raillargeinnerfront","raillargefrontouter","railmid",
      "xmemberfront","xmembermidfront","xmemberrearmiddlebottom","framexmemberplates",
      "radiator","radiatorframe","radiatorframebottom","radiatorframebrkttop",
      "radiatorside","radiatorsolid","radiatornullshell","condensor"],
     "#F4A261"),   # orange

    ("Occupant Cage",
     ["firewall","firewalltray","firewallsupport","ipbeamfirewallbrkt",
      "floor","floorfront","rearfloor","floorxmember","floorlongitudesupport",
      "floorsupport","floorbottomplate",
      "rocker","rockerinner","rockerfront","rockerinternal","rockerfrontinnersuppport",
      "apillar","bpillar","cpillar","pillar",
      "cabrail","cabrailupper","cabraillower","cabraillatitudebar",
      "cabrailconnection","cabrailpanel","cabrailframeconnector",
      "roof","roofrail","roofrailfront","roofrailrear","roofrailxmember",
      "sidepanel","backwall","cabpanel"],
     "#2A9D8F"),   # teal

    ("Doors",
     ["doorfront","doorrear","doorouter","doorinner","frontdoorinner",
      "doorfrontbar","doorrearlowerbar","doorrearuppersupport",
      "doorfrtlongitudesuprt","doorfrontlonguppersprt","doorrearlatitudesuprt",
      "windowframe","windowguide","lockplate","hinge"],
     "#457B9D"),   # steel blue

    ("Interior",
     ["ipbeam","dash","dashcover","dashfan","dashscreen","dashcenter",
      "dashbottom","dashcompartment","glovecompartment",
      "airbag","airbagbrkt","airbagbkt","airbagcover",
      "steering","steeringwheel","steeringcolmn","steeringcolumn","steeringrack"],
     "#A8DADC"),   # light cyan

    ("Seats & Belts",
     ["seat","seatdriver","seadriver","backseat","seatback","seatfoam","foam",
      "driverseat","passengerseat","rearseat","headrest","seatbelt",
      "seatdriverouterrail","seatdriverinnerrail"],
     "#E9C46A"),   # amber

    ("Dummy & Sensors",
     ["dummy","dummy_beams","lap_strap","shoulder_strap",
      "accelerometer","accelerometers"],
     "#F72585"),   # hot pink

    ("Powertrain",
     ["engine","engineoilpan","enginemount","enginemountrubber",
      "transmission","transmissionoilpan","transmissionmount",
      "battery","batterymount","fusebox",
      "brake","brakebooster","brakeboostermnt","brakefluidcontainer"],
     "#6A4C93"),   # purple

    ("Suspension & Wheels",
     ["aarm","upperarm","lowerarm","spindle","diskfront","upright",
      "tire","rim","swaybar","shockhousingfront","shocksupport",
      "suspensionbracketinner"],
     "#43AA8B"),   # green

    ("Unknown",    [], "#888888"),   # catch-all
]

CAT_NAMES  = [c[0] for c in CATEGORIES]
CAT_PATS   = [c[1] for c in CATEGORIES]
CAT_COLORS = [c[2] for c in CATEGORIES]


# ── Category assignment ────────────────────────────────────────────────────────

def _assign_categories(raw_names: np.ndarray) -> np.ndarray:
    """
    Map each node's raw part name (bytes or str) to a category index.
    Returns int32 array of shape (N,).
    """
    unknown_idx = len(CATEGORIES) - 1
    cat_idx = np.full(len(raw_names), unknown_idx, dtype=np.int32)

    decoded = []
    for r in raw_names:
        if isinstance(r, (bytes, bytearray, np.bytes_)):
            decoded.append(r.decode("utf-8", errors="ignore").strip().lower())
        else:
            decoded.append(str(r).strip().lower())

    for ci, pats in enumerate(CAT_PATS):
        if not pats:
            continue
        for ni, name in enumerate(decoded):
            if cat_idx[ni] == unknown_idx:   # first match wins
                if any(p in name for p in pats):
                    cat_idx[ni] = ci

    return cat_idx


def _cat_to_rgba(cat_idx: np.ndarray) -> np.ndarray:
    """Convert category index array -> (N, 4) float32 RGBA."""
    palette = np.array(
        [matplotlib.colors.to_rgba(c) for c in CAT_COLORS], dtype=np.float32
    )
    return palette[cat_idx]


# ── Figure rendering ───────────────────────────────────────────────────────────

_BG = "#0d0d0d"


def _make_fig(
    pos: np.ndarray,
    colors: np.ndarray,
    cat_idx: np.ndarray,
    t_ms: float,
    frame_no: int,
    bounds: dict,
    pt_size: float,
) -> plt.Figure:
    """Build and return a matplotlib Figure (caller saves / closes it)."""
    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 6),
        facecolor=_BG,
        gridspec_kw={"wspace": 0.05},
    )

    views = [
        ("Side view  (X - Z)", pos[:, 0], pos[:, 2],
         "X  [mm]", "Z  [mm]", bounds["x"], bounds["z"]),
        ("Top view   (X - Y)", pos[:, 0], pos[:, 1],
         "X  [mm]", "Y  [mm]", bounds["x"], bounds["y"]),
    ]

    for ax, (title, xv, yv, xl, yl, xb, yb) in zip(axes, views):
        ax.set_facecolor(_BG)
        ax.scatter(xv, yv, c=colors, s=pt_size, linewidths=0,
                   rasterized=True, zorder=2)
        ax.set_xlim(xb)
        ax.set_ylim(yb)
        ax.set_aspect("equal")
        ax.set_xlabel(xl, color="#aaaaaa", fontsize=8)
        ax.set_ylabel(yl, color="#aaaaaa", fontsize=8)
        ax.set_title(title, color="#dddddd", fontsize=9, pad=4)
        ax.tick_params(colors="#666666", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # legend: show only categories present in this node subset
    present = np.unique(cat_idx)
    handles = [Patch(facecolor=CAT_COLORS[ci], label=CAT_NAMES[ci])
               for ci in present]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 5),
        fontsize=7,
        facecolor="#1a1a1a",
        edgecolor="#444444",
        labelcolor="#cccccc",
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    # timestamp
    fig.text(
        0.5, 0.97,
        f"t = {t_ms:.1f} ms    (frame {frame_no})",
        ha="center", va="top",
        color="#ffffff", fontsize=11, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2, foreground="#000000")],
    )
    fig.subplots_adjust(bottom=0.18, top=0.93)
    return fig


def _fig_to_svg_bytes(fig: plt.Figure, dpi: int) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _fig_to_pil(fig: plt.Figure, dpi: int) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()   # .copy() so buf can be GC'd


def _svg_bytes_to_pil(svg_bytes: bytes, dpi: int) -> Image.Image:
    png = cairosvg.svg2png(bytestring=svg_bytes, dpi=dpi)
    return Image.open(io.BytesIO(png)).convert("RGBA")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise decimated d3plot HDF5 data -> SVG frames + GIF."
    )
    parser.add_argument("--h5",      type=Path,  default=DEFAULT_H5)
    parser.add_argument("--out-dir", type=Path,  default=DEFAULT_OUT_DIR,
                        help="Directory for SVG frames.")
    parser.add_argument("--gif",     type=Path,  default=DEFAULT_GIF,
                        help="Output animated GIF path.")
    parser.add_argument("--every",   type=int,   default=DEFAULT_EVERY,
                        help="Render every N-th frame from the HDF5 (1 = all).")
    parser.add_argument("--dpi",     type=int,   default=DEFAULT_DPI,
                        help="Render resolution.")
    parser.add_argument("--fps",     type=float, default=DEFAULT_FPS,
                        help="GIF playback speed (frames/sec).")
    parser.add_argument("--max-pts", type=int,   default=DEFAULT_MAX_PTS,
                        help="Max nodes to plot; excess randomly subsampled.")
    parser.add_argument("--pt-size", type=float, default=DEFAULT_PT_SIZE,
                        help="Scatter marker area in points^2.")
    args = parser.parse_args()

    if not HAS_PIL:
        sys.exit("ERROR: Pillow is required.  Run:  pip install Pillow")
    if not args.h5.exists():
        sys.exit(f"ERROR: HDF5 not found: {args.h5}")
    if not HAS_CAIRO:
        print("WARNING: cairosvg not found. GIF will be rendered from PNG "
              "directly (SVGs are still saved).  To enable SVG->PNG "
              "rasterisation: pip install cairosvg")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.gif.parent.mkdir(parents=True, exist_ok=True)

    # ── Load metadata (not state data) ────────────────────────────────────────
    print(f"Opening {args.h5} ...")
    with h5py.File(args.h5, "r") as h5f:
        times     = h5f["states/times"][:]             # (T,)  float64
        positions = h5f["states/positions"]            # (T, N, 3) – read lazily
        part_names = h5f["metadata/node_part_name"][:] # (N,)

        T, N, _ = positions.shape
        print(f"  Total frames : {T},  nodes : {N}")

        # ── frame selection ───────────────────────────────────────────────────
        frame_indices = list(range(0, T, args.every))
        print(f"  Rendering    : {len(frame_indices)} frames  (every={args.every})")

        # ── node subsampling for display ───────────────────────────────────────
        if N > args.max_pts:
            rng = np.random.default_rng(42)
            vis_idx = np.sort(rng.choice(N, size=args.max_pts, replace=False))
            print(f"  Node display : {N:,} -> {args.max_pts:,} (random subsample)")
        else:
            vis_idx = np.arange(N)
            print(f"  Node display : {N:,} (all)")

        # ── category assignment (done once, before frame loop) ────────────────
        print("  Assigning structural categories ...")
        vis_names   = part_names[vis_idx]
        cat_idx_vis = _assign_categories(vis_names)
        colors_vis  = _cat_to_rgba(cat_idx_vis)

        print("  Category breakdown:")
        for ci in range(len(CATEGORIES)):
            cnt = int(np.sum(cat_idx_vis == ci))
            if cnt > 0:
                print(f"    [{ci}] {CAT_NAMES[ci]:<26}  {cnt:>8,} nodes  "
                      f"({100*cnt/len(vis_idx):.1f}%)")

        # ── global axis bounds (sampled from a few frames) ────────────────────
        print("  Computing stable axis bounds ...")
        probe_step  = max(1, len(frame_indices) // 10)
        probe_idxs  = frame_indices[::probe_step]
        probe_pos   = np.concatenate(
            [positions[fi][vis_idx] for fi in probe_idxs], axis=0
        )
        pad = 100.0   # mm margin
        bounds = {
            "x": (float(probe_pos[:, 0].min()) - pad,
                  float(probe_pos[:, 0].max()) + pad),
            "y": (float(probe_pos[:, 1].min()) - pad,
                  float(probe_pos[:, 1].max()) + pad),
            "z": (float(probe_pos[:, 2].min()) - pad,
                  float(probe_pos[:, 2].max()) + pad),
        }
        del probe_pos

        # ── frame loop ────────────────────────────────────────────────────────
        gif_pil_frames: list[Image.Image] = []

        for render_no, fi in enumerate(frame_indices):
            t_ms      = float(times[fi]) * 1e3
            pos_frame = positions[fi][vis_idx]   # (N_vis, 3)

            print(f"  [{render_no+1:>4}/{len(frame_indices)}]  "
                  f"fi={fi}  t={t_ms:7.2f} ms", end="\r", flush=True)

            fig = _make_fig(pos_frame, colors_vis, cat_idx_vis,
                            t_ms, render_no, bounds, args.pt_size)

            # -- save SVG
            svg_bytes = _fig_to_svg_bytes(fig, args.dpi)
            svg_path  = args.out_dir / f"frame_{render_no:05d}.svg"
            svg_path.write_bytes(svg_bytes)

            # -- PNG for GIF
            if HAS_CAIRO:
                # rasterise SVG we already have (consistent look)
                try:
                    pil_img = _svg_bytes_to_pil(svg_bytes, args.dpi)
                except Exception as e:
                    print(f"\n  cairosvg failed ({e}), falling back to PNG render")
                    fig2 = _make_fig(pos_frame, colors_vis, cat_idx_vis,
                                     t_ms, render_no, bounds, args.pt_size)
                    pil_img = _fig_to_pil(fig2, args.dpi)
            else:
                # re-render as PNG directly (fig was already closed above)
                fig2 = _make_fig(pos_frame, colors_vis, cat_idx_vis,
                                 t_ms, render_no, bounds, args.pt_size)
                pil_img = _fig_to_pil(fig2, args.dpi)

            gif_pil_frames.append(pil_img.convert("RGB"))

    print()  # newline after \r progress

    # ── Build animated GIF ────────────────────────────────────────────────────
    print(f"Building GIF ({len(gif_pil_frames)} frames, {args.fps} fps) ...")
    duration_ms = int(1000 / args.fps)

    # convert to palette mode for compact GIF
    p_frames = [
        img.convert("P", palette=Image.ADAPTIVE, colors=256)
        for img in gif_pil_frames
    ]

    p_frames[0].save(
        args.gif,
        save_all=True,
        append_images=p_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    size_mb = args.gif.stat().st_size / 1e6
    print(f"\nDone.")
    print(f"  SVG frames : {args.out_dir}/  ({len(frame_indices)} files)")
    print(f"  GIF        : {args.gif}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
