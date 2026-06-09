#!/usr/bin/env python3
"""
Plastic-strain / material GT visualization with cumulative erosion masking.

Reads a new-format h5 that contains:
  states/plastic_strain  (T, N)    effective plastic strain [-]
  metadata/node_mat_type_id   (N,) integer material class
  metadata/node_mat_type_name (N,) material keyword string
  metadata/node_mat_sigy      (N,) yield stress [MPa]

Color modes (--color-by):
  strain     -- effective plastic strain [-], erosion threshold in real units
  material   -- discrete material type (no erosion)
  velocity   -- velocity magnitude [mm/frame]
  displacement -- displacement from frame-0 [mm]

Erosion (strain mode only):
  --threshold 0.5  means: once a node's plastic strain exceeds 0.5 (50%)
  it is removed from the scene and optionally highlighted with --erode-only.
  This is a real physical unit — no normalization.

Examples:
    # All nodes colored by plastic strain, erode at eps_p > 0.5
    python tools/vis_strain_erosion.py \
        --h5 /home/kong/datasets/barrier/h5dt_50ns_5fs_mat/T_lok_F_shape_barrier_9_3_100km/output.h5 
        --color-by strain 
        --threshold 0.5 
        --gif

    # Only show eroded nodes (damage map)
    python tools/vis_strain_erosion.py \
        --h5 /home/kong/datasets/barrier/h5dt_50ns_5fs_mat/T_lok_F_shape_barrier_9_3_100km/output.h5 \
        --color-by strain --threshold 0.5 \
        --erode-only --gif

    # Barrier connectors only, colored by material type
    python tools/vis_strain_erosion.py \
        --h5 /home/kong/datasets/barrier/h5dt_50ns_5fs_mat/T_lok_F_shape_barrier_9_3_100km/output.h5 \
        --parts "steel tube" "T lok" "concrete" \
        --color-by material --gif

    # Vehicle only (exclude barrier beams), colored by strain
    python tools/vis_strain_erosion.py \
        --h5 /home/kong/datasets/barrier/h5dt_50ns_5fs/T_lok_F_shape_barrier_9_3_100km/output.h5 \
        --exclude-parts "steel tube" "T lok" "anchor rebar" "reinforcement" \
        --color-by strain --threshold 0.5 --gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    _VIS = True
except ImportError:
    _VIS = False
    print("ERROR: matplotlib required")

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False
    print("ERROR: Pillow required")


# ── Style ─────────────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family":     "Times New Roman",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

# Coarse material type names (kept for fallback / console summary)
_MAT_ID_NAMES = {
    0: "Plasticity", 1: "Rigid", 2: "Elastic", 3: "Rubber",
    4: "Concrete",   5: "Foam",  6: "Spotweld", 7: "Spring/Damper", 8: "Unknown",
}


# ── Part selection ─────────────────────────────────────────────────────────────

def match_parts(node_part_name: np.ndarray, patterns: list[str]) -> np.ndarray:
    mask = np.zeros(len(node_part_name), dtype=bool)
    for i, name in enumerate(node_part_name):
        nl = name.lower()
        if any(p.lower() in nl for p in patterns):
            mask[i] = True
    return mask


def build_selection_mask(
    node_part_name: np.ndarray,
    include: list[str] | None,
    exclude: list[str] | None,
) -> np.ndarray:
    N = len(node_part_name)
    mask = match_parts(node_part_name, include) if include else np.ones(N, dtype=bool)
    if exclude:
        mask &= ~match_parts(node_part_name, exclude)
    return mask


# ── Rendering ──────────────────────────────────────────────────────────────────

X_LIM = (-14000, 20000)
Y_LIM = (-10000,  8000)
Z_LIM =   (-500,  4000)


def _make_fig(dpi):
    plt.rcParams.update(_RCPARAMS)

    x_span = X_LIM[1] - X_LIM[0]
    z_span = Z_LIM[1] - Z_LIM[0]
    y_span = Y_LIM[1] - Y_LIM[0]

    fig_w  = 7.0
    upi    = x_span / fig_w
    plot_h = (z_span + y_span) / upi
    fig_h  = plot_h + 2.4

    fig, axs = plt.subplots(
        2, 1, figsize=(fig_w, fig_h), dpi=dpi,
        gridspec_kw={"height_ratios": [z_span, y_span], "hspace": 0.18},
    )
    fig.patch.set_facecolor("white")

    for ax in axs:
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=10)
        ax.set_aspect("equal", adjustable="box")
        for sp in ax.spines.values():
            sp.set_edgecolor("#aaaaaa")

    axs[0].set_xlim(X_LIM); axs[0].set_ylim(Z_LIM)
    axs[1].set_xlim(X_LIM); axs[1].set_ylim(Y_LIM)
    axs[0].set_ylabel("Z [mm]", color="black")
    axs[1].set_ylabel("Y [mm]", color="black")
    axs[1].set_xlabel("X [mm]", color="black")

    return fig, axs


def _save_legend_svg(label_to_color: dict, out_path: Path) -> None:
    """Save a standalone legend as SVG (transparent background, Times New Roman)."""
    plt.rcParams.update(_RCPARAMS)
    n = len(label_to_color)
    fig = plt.figure(figsize=(4.5, max(n * 0.32 + 0.4, 1.0)))
    elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
               markersize=10, label=lbl)
        for lbl, c in label_to_color.items()
    ]
    fig.legend(handles=elements, loc="center",
               fontsize=9, frameon=False, labelcolor="black")
    plt.axis("off")
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(str(svg_path), format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[Legend] SVG → {svg_path}")


def render_strain(
    positions:      np.ndarray,      # (T, N, 3)
    scalar_field:   np.ndarray,      # (T, N)  plastic strain (or vel / disp)
    sel_mask:       np.ndarray,      # (N,) bool
    *,
    label:          str   = "Eff. Plastic Strain [-]",
    threshold:      float = 0.5,     # in real units (not normalized)
    apply_erosion:  bool  = True,
    erode_only:     bool  = False,
    vmin:           float = 0.0,
    vmax:           float = 2.0,
    out_dir:        Path,
    fps:            int   = 10,
    max_frames:     int   = 200,
    dpi:            int   = 150,
    save_gif:       bool  = True,
    save_pngs:      bool  = True,
) -> None:

    T   = min(positions.shape[0], max_frames)
    N   = positions.shape[1]

    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "frames"
    if save_pngs:
        png_dir.mkdir(exist_ok=True)

    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axs = _make_fig(dpi)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs.tolist(), orientation="horizontal",
                        fraction=0.025, pad=0.02, aspect=50)
    cbar.set_label(label, color="black")
    cbar.ax.tick_params(colors="black", labelsize=10)

    # Background scatter (non-selected nodes, light gray)
    bg_mask = ~sel_mask
    has_bg  = bg_mask.any() and not erode_only
    if has_bg:
        bp0 = positions[0][bg_mask]
        bg_xz = axs[0].scatter(bp0[:, 0], bp0[:, 2], c="#cccccc", s=0.1, alpha=0.5, linewidths=0)
        bg_xy = axs[1].scatter(bp0[:, 0], bp0[:, 1], c="#cccccc", s=0.1, alpha=0.5, linewidths=0)
    else:
        bg_xz = bg_xy = None

    sel_xz = axs[0].scatter([], [], c=[], cmap=cmap, norm=norm, s=1.8, linewidths=0, alpha=0.9)
    sel_xy = axs[1].scatter([], [], c=[], cmap=cmap, norm=norm, s=1.8, linewidths=0, alpha=0.9)

    title_xz = axs[0].set_title("", color="black", fontsize=12, pad=4, loc="left")
    axs[1].set_title("X–Y Plane", color="black", fontsize=11, pad=3)

    fig.canvas.draw()

    eroded     = np.zeros(N, dtype=bool)
    gif_frames: list[Image.Image] = []

    for t in range(T):
        pos_t = positions[t]
        val_t = scalar_field[t]

        if apply_erosion:
            # Threshold directly in real units (no normalization)
            eroded |= sel_mask & (val_t > threshold)

        if erode_only:
            active = sel_mask & eroded
        elif apply_erosion:
            active = sel_mask & ~eroded
        else:
            active = sel_mask

        # Update background
        if has_bg and not erode_only:
            bp = pos_t[bg_mask]
            bg_xz.set_offsets(np.c_[bp[:, 0], bp[:, 2]])
            bg_xy.set_offsets(np.c_[bp[:, 0], bp[:, 1]])

        # Update colored nodes
        if active.any():
            ap = pos_t[active]
            ac = val_t[active]
            sel_xz.set_offsets(np.c_[ap[:, 0], ap[:, 2]])
            sel_xy.set_offsets(np.c_[ap[:, 0], ap[:, 1]])
            sel_xz.set_array(ac)
            sel_xy.set_array(ac)
        else:
            sel_xz.set_offsets(np.empty((0, 2)))
            sel_xy.set_offsets(np.empty((0, 2)))
            sel_xz.set_array(np.array([]))
            sel_xy.set_array(np.array([]))

        n_active  = int(active.sum())
        n_eroded  = int((sel_mask & eroded).sum())
        n_sel     = int(sel_mask.sum())

        if erode_only:
            title_xz.set_text(f"X–Z   frame {t+1:03d}/{T}   eroded {n_eroded}/{n_sel}")
        elif apply_erosion:
            title_xz.set_text(f"X–Z   frame {t+1:03d}/{T}   alive {n_active}/{n_sel}   eroded {n_eroded}")
        else:
            title_xz.set_text(f"X–Z   frame {t+1:03d}/{T}   nodes {n_active}")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img  = Image.fromarray(rgba).convert("RGB")
        gif_frames.append(img)

        if save_pngs:
            img.save(png_dir / f"frame_{t:04d}.png")

        if (t + 1) % 20 == 0 or t == T - 1:
            if apply_erosion:
                print(f"  [{t+1:3d}/{T}]  alive={n_active}  eroded={n_eroded}")
            else:
                print(f"  [{t+1:3d}/{T}]  nodes={n_active}")

    plt.close(fig)
    _save_output(gif_frames, out_dir, fps, save_gif, save_pngs)


def render_material(
    positions:      np.ndarray,   # (T, N, 3)
    mat_label:      np.ndarray,   # (N,) str — precise label e.g. "Steel - 300"
    plastic_strain: np.ndarray,   # (T, N) — always used for erosion
    sel_mask:       np.ndarray,   # (N,) bool
    *,
    threshold:      float = 0.5,  # plastic strain erosion threshold (real units)
    erode_only:     bool  = False, # show ONLY eroded nodes; hide surviving nodes
    out_dir:        Path,
    fps:            int   = 10,
    max_frames:     int   = 200,
    dpi:            int   = 150,
    save_gif:       bool  = True,
    save_pngs:      bool  = True,
) -> None:

    T = min(positions.shape[0], max_frames)
    N = positions.shape[1]

    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "frames"
    if save_pngs:
        png_dir.mkdir(exist_ok=True)

    # Build label → color mapping from unique labels in the selection
    labels_sel   = mat_label[sel_mask]
    unique_labels = sorted(set(labels_sel.tolist()))
    n_labels      = len(unique_labels)

    # Use tab20 + tab20b combined for up to 40 distinct colors
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cm = plt.get_cmap(cmap_name)
        palette.extend(cm(i / 20) for i in range(20))
    label_to_color = {lbl: palette[i % len(palette)]
                      for i, lbl in enumerate(unique_labels)}

    print(f"  {n_labels} distinct material labels in selection:")
    for lbl in unique_labels:
        n = int((labels_sel == lbl).sum())
        print(f"    {lbl:<50} {n:>7,} nodes")

    # Per-node RGBA (fixed across all frames)
    sel_colors = np.array([label_to_color[lbl] for lbl in labels_sel])  # (n_sel, 4)

    fig, axs = _make_fig(dpi)

    # Legend saved as separate SVG (not embedded in figure)
    _save_legend_svg(label_to_color, out_dir / "material_legend")

    # Background (non-selected nodes, light gray; hidden in erode_only mode)
    bg_mask = ~sel_mask
    if bg_mask.any() and not erode_only:
        bp0 = positions[0][bg_mask]
        axs[0].scatter(bp0[:, 0], bp0[:, 2], c="#cccccc", s=0.1, alpha=0.5, linewidths=0)
        axs[1].scatter(bp0[:, 0], bp0[:, 1], c="#cccccc", s=0.1, alpha=0.5, linewidths=0)

    sp0   = positions[0][sel_mask]
    sc_xz = axs[0].scatter(sp0[:, 0], sp0[:, 2], c=sel_colors, s=1.5, linewidths=0, alpha=0.9)
    sc_xy = axs[1].scatter(sp0[:, 0], sp0[:, 1], c=sel_colors, s=1.5, linewidths=0, alpha=0.9)

    title_xz = axs[0].set_title("", color="black", fontsize=12, pad=4, loc="left")
    axs[1].set_title("X–Y Plane  (color = material label)", color="black", fontsize=11, pad=3)

    # Erosion preview
    strain_sel = plastic_strain[:, sel_mask]   # (T, n_sel)
    n_erode    = int((strain_sel > threshold).any(axis=0).sum())
    n_sel      = int(sel_mask.sum())
    print(f"  Strain erosion threshold={threshold}: {n_erode}/{n_sel} nodes will erode "
          f"({100*n_erode/n_sel:.1f}%)")

    fig.canvas.draw()
    gif_frames: list[Image.Image] = []
    eroded = np.zeros(N, dtype=bool)   # cumulative across frames

    for t in range(T):
        # Update cumulative erosion from plastic strain
        eroded |= sel_mask & (plastic_strain[t] > threshold)
        active   = (sel_mask & eroded) if erode_only else (sel_mask & ~eroded)

        n_alive  = int((sel_mask & ~eroded).sum())
        n_eroded = int((sel_mask & eroded).sum())

        if active.any():
            ap = positions[t][active]
            # colors: select from sel_colors using local indices within sel_mask
            sel_idx    = np.where(sel_mask)[0]
            active_local = np.where(active[sel_mask])[0]
            ac = sel_colors[active_local]
            sc_xz.set_offsets(np.c_[ap[:, 0], ap[:, 2]])
            sc_xy.set_offsets(np.c_[ap[:, 0], ap[:, 1]])
            sc_xz.set_color(ac)
            sc_xy.set_color(ac)
        else:
            sc_xz.set_offsets(np.empty((0, 2)))
            sc_xy.set_offsets(np.empty((0, 2)))

        title_xz.set_text(
            f"X–Z   frame {t+1:03d}/{T}   "
            f"alive {n_alive}/{n_sel}   eroded {n_eroded}  (ε_p > {threshold})"
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img  = Image.fromarray(rgba).convert("RGB")
        gif_frames.append(img)

        if save_pngs:
            img.save(png_dir / f"frame_{t:04d}.png")

        if (t + 1) % 20 == 0 or t == T - 1:
            print(f"  [{t+1:3d}/{T}]  alive={n_alive}  eroded={n_eroded}")

    plt.close(fig)
    _save_output(gif_frames, out_dir, fps, save_gif, save_pngs)


def _save_output(gif_frames, out_dir, fps, save_gif, save_pngs):
    if save_gif and gif_frames:
        gif_path = out_dir / "vis.gif"
        gif_frames[0].save(
            gif_path, save_all=True, append_images=gif_frames[1:],
            duration=int(1000 / fps), loop=0,
        )
        print(f"GIF → {gif_path}")
    if save_pngs:
        print(f"PNGs → {out_dir / 'frames'}  ({len(gif_frames)} frames)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plastic strain / material visualization with erosion masking"
    )
    parser.add_argument("--h5",           required=True,
                        help="Path to output.h5 (new format with plastic_strain + material)")
    parser.add_argument("--parts",        nargs="+", default=None,
                        help="Include only parts matching these substrings (default: all).")
    parser.add_argument("--exclude-parts", nargs="+", default=None,
                        help="Remove parts matching these substrings from selection.")
    parser.add_argument("--color-by",     default="strain",
                        choices=["strain", "material", "velocity", "displacement"],
                        help="Field to color nodes by (default: strain).")
    parser.add_argument("--threshold",    type=float, default=0.5,
                        help="Plastic strain erosion threshold in real units [-]. "
                             "0.5 = 50%% eff. plastic strain. Only for --color-by strain. "
                             "(default 0.5)")
    parser.add_argument("--vmax",         type=float, default=None,
                        help="Colormap max. Default: 99th-percentile of selected nodes.")
    parser.add_argument("--vmin",         type=float, default=0.0,
                        help="Colormap min (default 0).")
    parser.add_argument("--erode-only",   action="store_true",
                        help="Show ONLY eroded nodes; all others invisible. "
                             "Nodes accumulate each frame as more get eroded.")
    parser.add_argument("--fps",          type=int,   default=10)
    parser.add_argument("--max-frames",   type=int,   default=200)
    parser.add_argument("--dpi",          type=int,   default=150)
    parser.add_argument("--gif",          action="store_true")
    parser.add_argument("--no-pngs",      action="store_true")
    parser.add_argument("--out",          default=None)
    args = parser.parse_args()

    if not _VIS or not _PIL:
        return

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {h5_path.parent.name}/output.h5 ...")
    with h5py.File(h5_path, "r") as f:
        positions       = f["states/positions"][:].astype(np.float32)     # (T, N, 3)
        velocity_raw    = f["states/velocity"][:].astype(np.float32)      # (T, N, 3)
        plastic_strain  = f["states/plastic_strain"][:].astype(np.float32)# (T, N)
        node_part_name  = np.array([
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f["metadata/node_part_name"][:]
        ])
        mat_type_id     = f["metadata/node_mat_type_id"][:].astype(np.int32)  # (N,)
        mat_type_name   = np.array([
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f["metadata/node_mat_type_name"][:]
        ])
        # precise label — "Steel - 300", "T lok", etc. (falls back to type_name if absent)
        if "metadata/node_mat_label" in f:
            mat_label = np.array([
                x.decode() if isinstance(x, bytes) else str(x)
                for x in f["metadata/node_mat_label"][:]
            ])
        else:
            print("  [warn] node_mat_label not found in h5 — falling back to node_mat_type_name")
            mat_label = mat_type_name

    T, N, _ = positions.shape
    print(f"  {T} frames  {N} nodes")

    # ── Part selection ────────────────────────────────────────────────────────
    sel_mask = build_selection_mask(node_part_name, args.parts, args.exclude_parts)
    n_sel    = int(sel_mask.sum())

    if n_sel == 0:
        print("ERROR: no nodes selected after include/exclude filtering.")
        return

    mode_str = "global" if not args.parts and not args.exclude_parts else "filtered"
    print(f"\nSelection ({mode_str}): {n_sel} / {N} nodes")
    unique_sel, ucounts = np.unique(node_part_name[sel_mask], return_counts=True)
    for cnt, name in sorted(zip(ucounts, unique_sel), reverse=True)[:15]:
        print(f"  {cnt:>7,}  {name}")
    if len(unique_sel) > 15:
        print(f"  ... ({len(unique_sel)} distinct parts total)")

    # Material summary for selection
    print("\nMaterial labels in selection:")
    for lbl in sorted(np.unique(mat_label[sel_mask])):
        n = int((mat_label[sel_mask] == lbl).sum())
        print(f"  {lbl:<50} {n:>7,} nodes")

    # ── Build scalar field + render config ────────────────────────────────────
    color_by      = args.color_by
    apply_erosion = False

    if color_by == "strain":
        scalar_field  = plastic_strain                                   # (T, N), real units
        field_label   = "Eff. Plastic Strain [-]"
        apply_erosion = True
        print(f"\nPlastic strain on selected nodes:")
        sel_ps = plastic_strain[:, sel_mask]
        for p in [50, 75, 90, 95, 99]:
            print(f"  p{p:2d} = {np.percentile(sel_ps, p):.6f}")
        print(f"  max = {sel_ps.max():.6f}")
        print(f"  nonzero = {100*(sel_ps>0).mean():.1f}%")

        vmax = args.vmax if args.vmax is not None else float(np.percentile(sel_ps, 99))
        vmin = args.vmin
        if vmax <= vmin:
            vmax = vmin + 1.0

        # Erosion preview
        n_erode = int((sel_ps > args.threshold).any(axis=0).sum())
        print(f"\nErosion threshold: {args.threshold} (real units, no normalization)")
        print(f"Nodes that will erode: {n_erode}/{n_sel} ({100*n_erode/n_sel:.1f}%)")
        if n_erode > 0:
            fe = (sel_ps > args.threshold).argmax(axis=0)
            will = (sel_ps > args.threshold).any(axis=0)
            print(f"First erosion at frame {fe[will].min()} (earliest) — {fe[will].max()} (latest)")

        print(f"Colormap: [{vmin:.4f}, {vmax:.4f}]  (strain)")

    elif color_by == "material":
        scalar_field = None   # not used for material mode
        field_label  = "Material type"
        vmin = vmax  = 0.0

    elif color_by == "velocity":
        scalar_field = np.linalg.norm(velocity_raw, axis=-1)            # (T, N)
        field_label  = "Velocity magnitude [mm/frame]"
        sel_v = scalar_field[:, sel_mask]
        vmax = args.vmax if args.vmax is not None else float(np.percentile(sel_v, 99))
        vmin = args.vmin
        if vmax <= vmin: vmax = vmin + 1.0
        print(f"\nColormap: [{vmin:.4f}, {vmax:.4f}]  (velocity)")

    else:  # displacement
        scalar_field = np.linalg.norm(positions - positions[0:1], axis=-1)  # (T, N)
        field_label  = "Displacement from t=0 [mm]"
        sel_d = scalar_field[:, sel_mask]
        vmax = args.vmax if args.vmax is not None else float(np.percentile(sel_d, 99))
        vmin = args.vmin
        if vmax <= vmin: vmax = vmin + 1.0
        print(f"\nColormap: [{vmin:.4f}, {vmax:.4f}]  (displacement)")

    # ── Output dir ────────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out or
        project_root / "outputs" / "strain_vis" /
        f"{h5_path.parent.name}_{color_by}")
    print(f"\nOutput → {out_dir}")

    # ── Render ────────────────────────────────────────────────────────────────
    if color_by == "material":
        render_material(
            positions      = positions,
            mat_label      = mat_label,
            plastic_strain = plastic_strain,
            sel_mask       = sel_mask,
            threshold      = args.threshold,
            erode_only     = args.erode_only,
            out_dir        = out_dir,
            fps            = args.fps,
            max_frames     = args.max_frames,
            dpi            = args.dpi,
            save_gif       = args.gif,
            save_pngs      = not args.no_pngs,
        )
    else:
        render_strain(
            positions     = positions,
            scalar_field  = scalar_field,
            sel_mask      = sel_mask,
            label         = field_label,
            threshold     = args.threshold,
            apply_erosion = apply_erosion,
            erode_only    = args.erode_only,
            vmin          = vmin,
            vmax          = vmax,
            out_dir       = out_dir,
            fps           = args.fps,
            max_frames    = args.max_frames,
            dpi           = args.dpi,
            save_gif      = args.gif,
            save_pngs     = not args.no_pngs,
        )
    print("Done.")


if __name__ == "__main__":
    main()
