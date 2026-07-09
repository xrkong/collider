"""
visualize_pyvista.py — render a build_dataset.py HDF5 frame (or animation)
with PyVista.

Builds a PyVista mesh from the sparse cell connectivity (shell_cells /
solid_cells / beam_cells — see build_dataset.py's HDF5 layout docstring and
connectivity.py for why coverage is necessarily partial) plus a point-cloud
overlay of every sampled node, both colored by eff_plastic_strain. The
cells render as a real surface (smooth-shaded); points fill in everywhere
sampling didn't keep a complete element.

Usage
-----
python -m dataset.ds.visualize_pyvista \
    --h5 /home/kong/datasets/barrier/h5_fps/T_lok_F_shape_barrier_9_3_60km.h5 \
    --frame 20 \
    --out /home/kong/datasets/barrier/h5_fps/frame20.png

python -m dataset.ds.visualize_pyvista --h5 output.h5 --frame 0 --interactive
python -m dataset.ds.visualize_pyvista \
    --h5 /home/kong/datasets/barrier/h5_poisson_disk_fs10/T_lok_F_shape_barrier_9_3_60km.h5 \
    --gif \
    --out anim.gif


python -m dataset.ds.visualize_pyvista \
    --h5 /home/kong/datasets/barrier/h5_stride_by_file/T_lok_F_shape_barrier_9_3_60km.h5 \
    --frame 20 \
    --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from PIL import Image


def _focus_bounds(
    points: np.ndarray,            # (N, 3) single frame, or (T, N, 3) trajectory
    region_label: np.ndarray,      # (N,) — always per-node, regardless of points' shape
    focus_region: list[str] | None,
    pad: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Bounding box (xmin,xmax,ymin,ymax,zmin,zmax) around the given regions, padded.

    For a (T, N, 3) trajectory this unions the box across every frame, so a
    camera fit from it stays valid for the whole animation, not just frame 0.
    Returns None (full-extent view) if focus_region is empty/unset or matches
    nothing — callers should fall back to the default auto-fit camera.
    """
    if not focus_region:
        return None
    mask = np.isin(region_label, focus_region)
    if not mask.any():
        print(f"WARNING: --focus-region {focus_region} matched 0 nodes — showing full extent instead.")
        return None
    p = points[:, mask, :] if points.ndim == 3 else points[mask]
    mn = p.reshape(-1, 3).min(axis=0)
    mx = p.reshape(-1, 3).max(axis=0)
    return (mn[0] - pad, mx[0] + pad, mn[1] - pad, mx[1] + pad, mn[2] - pad, mx[2] + pad)


def _apply_camera(plotter: pv.Plotter, bounds, zoom: float) -> None:
    """Set a deterministic top-down orthographic view, optionally tight-fit to `bounds`.

    Uses parallel (orthographic) projection with a manually computed
    parallel_scale rather than PyVista's reset_camera(bounds=...): under the
    default perspective projection, reset_camera(bounds=...) does not
    reliably crop the view to just the given box — empirically, far-away
    geometry stayed fully visible even after a large parallel_scale change,
    because perspective rendering doesn't use parallel_scale at all. Forcing
    orthographic + computing the scale ourselves makes the zoom exact and
    independent of camera distance.
    """
    plotter.enable_parallel_projection()
    plotter.camera_position = "xy"
    if bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        half_w, half_h = (xmax - xmin) / 2, (ymax - ymin) / 2
        w, h = plotter.window_size
        aspect = (w / h) if h else 1.0
        half_h_needed = max(half_h, half_w / aspect, 1.0)
        depth = max(zmax - zmin, 1.0) * 5 + half_h_needed * 5
        plotter.camera.focal_point = (cx, cy, cz)
        plotter.camera.position = (cx, cy, cz + depth)
        plotter.camera.parallel_scale = half_h_needed * 1.05  # 5% margin
    else:
        plotter.reset_camera()
    if zoom and zoom != 1.0:
        plotter.camera.zoom(zoom)


def build_frame_mesh(
    h5_path: Path,
    frame: int = -1,
) -> tuple[pv.PolyData | None, pv.UnstructuredGrid | None, pv.PolyData, dict]:
    """Load one frame and build PyVista objects for it.

    Returns (shell_mesh, solid_mesh, point_cloud, info). Either mesh may be
    None if that cell type has zero surviving elements (commonly true for
    solid_mesh — see connectivity.py).
    """
    with h5py.File(h5_path, "r") as f:
        n_frames = f["metadata"].attrs["n_frames"]
        fi = frame if frame >= 0 else n_frames + frame
        positions = f["states/positions"][fi]                  # (N, 3)
        strain = f["states/eff_plastic_strain"][fi]             # (N,)
        time_s = float(f["states/times"][fi])
        shell_cells = f["metadata/shell_cells"][:]               # (Es, 4) local idx
        solid_cells = f["metadata/solid_cells"][:]               # (Eh, 8) local idx
        region_label = f["metadata/region_label"][:].astype(str)

    info = {"frame": int(fi), "n_frames": int(n_frames), "time_ms": time_s * 1e3,
            "n_nodes": len(positions), "n_shell_cells": len(shell_cells),
            "n_solid_cells": len(solid_cells)}

    shell_mesh = None
    if len(shell_cells) > 0:
        n_cells = len(shell_cells)
        faces = np.hstack([
            np.full((n_cells, 1), 4, dtype=np.int64), shell_cells.astype(np.int64),
        ]).ravel()
        shell_mesh = pv.PolyData(positions, faces)
        shell_mesh.point_data["eff_plastic_strain"] = strain

    solid_mesh = None
    if len(solid_cells) > 0:
        n_cells = len(solid_cells)
        cells = np.hstack([
            np.full((n_cells, 1), 8, dtype=np.int64), solid_cells.astype(np.int64),
        ]).ravel()
        cell_types = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        solid_mesh = pv.UnstructuredGrid(cells, cell_types, positions)
        solid_mesh.point_data["eff_plastic_strain"] = strain

    point_cloud = pv.PolyData(positions)
    point_cloud.point_data["eff_plastic_strain"] = strain
    point_cloud.point_data["region_label"] = region_label

    return shell_mesh, solid_mesh, point_cloud, info


def render_frame(
    h5_path: Path,
    frame: int = -1,
    out_path: Path | None = None,
    interactive: bool = False,
    point_size: float = 3.0,
    cmap: str = "coolwarm",
    window_size: tuple[int, int] = (1400, 900),
    focus_region: list[str] | None = None,
    focus_pad: float = 800.0,
    zoom: float = 1.0,
) -> None:
    """Render one frame: cell surface(s) smooth-shaded + point cloud overlay,
    both colored by eff_plastic_strain, on a white background.

    focus_region restricts the camera to a tight box around just those
    region(s) (e.g. ["barrier_fine", "veh_contact", "force_keep"] for the
    impact zone) instead of fitting the whole scene; zoom is an extra
    multiplier on top of whatever view that produces.
    """
    shell_mesh, solid_mesh, point_cloud, info = build_frame_mesh(h5_path, frame)
    print(f"Frame {info['frame']}/{info['n_frames']}  t = {info['time_ms']:.1f} ms  "
          f"nodes={info['n_nodes']:,}  shell_cells={info['n_shell_cells']:,}  "
          f"solid_cells={info['n_solid_cells']:,}")

    clim = (float(point_cloud.point_data["eff_plastic_strain"].min()),
            float(point_cloud.point_data["eff_plastic_strain"].max()))
    if clim[0] == clim[1]:
        clim = (clim[0], clim[0] + 1e-6)

    plotter = pv.Plotter(off_screen=not interactive, window_size=window_size)
    plotter.background_color = "white"

    plotter.add_mesh(
        point_cloud, scalars="eff_plastic_strain", cmap=cmap, clim=clim,
        render_points_as_spheres=True, point_size=point_size,
        show_scalar_bar=(shell_mesh is None and solid_mesh is None),
    )

    if solid_mesh is not None:
        surf = solid_mesh.extract_surface()
        plotter.add_mesh(
            surf, scalars="eff_plastic_strain", cmap=cmap, clim=clim,
            smooth_shading=True, show_edges=False, show_scalar_bar=False,
        )
    if shell_mesh is not None:
        plotter.add_mesh(
            shell_mesh, scalars="eff_plastic_strain", cmap=cmap, clim=clim,
            smooth_shading=True, show_edges=False,
            show_scalar_bar=True, scalar_bar_args={"title": "eff. plastic strain"},
        )

    plotter.add_text(f"t = {info['time_ms']:.1f} ms", position="upper_left", font_size=10, color="black")
    bounds = _focus_bounds(point_cloud.points, point_cloud.point_data["region_label"],
                           focus_region, focus_pad)
    _apply_camera(plotter, bounds, zoom)

    if interactive:
        plotter.show()
    else:
        out_path = out_path or h5_path.with_suffix(f".frame{info['frame']}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(out_path))
        print(f"Saved: {out_path}")
    plotter.close()


def make_gif(
    h5_path: Path,
    out_path: Path,
    frame_stride: int = 1,
    fps: int = 8,
    max_frames: int = 60,
    cmap: str = "coolwarm",
    point_size: float = 3.0,
    window_size: tuple[int, int] = (1400, 900),
    focus_region: list[str] | None = None,
    focus_pad: float = 800.0,
    zoom: float = 1.0,
) -> None:
    """Animate every kept frame in the same mesh + point-cloud style as render_frame.

    Camera and color scale are computed once (from the focus/auto-fit bounds
    + zoom, and the 99.5th percentile of strain over the whole trajectory)
    and then held fixed across all frames, so only the geometry and coloring
    change — otherwise a per-frame autoscaled clim/camera would make frames
    impossible to compare against each other. focus_region's bounding box is
    unioned across every kept frame so it stays valid for the whole
    animation even as the focused nodes move (see _focus_bounds).
    """
    with h5py.File(h5_path, "r") as f:
        n_frames = int(f["metadata"].attrs["n_frames"])
        times = f["states/times"][:]
        positions_all = f["states/positions"][:]            # (T, N, 3)
        eps_all = f["states/eff_plastic_strain"][:]          # (T, N)
        shell_cells = f["metadata/shell_cells"][:]
        solid_cells = f["metadata/solid_cells"][:]
        region_label = f["metadata/region_label"][:].astype(str)

    idx = np.arange(0, n_frames, frame_stride)
    if len(idx) > max_frames:
        idx = np.linspace(0, n_frames - 1, max_frames).astype(int)

    # 99.5th percentile, not max — a single outlier element would otherwise
    # wash out the color gradient for every other node in the animation.
    clim = (0.0, float(np.percentile(eps_all, 99.5)))
    if clim[1] <= clim[0]:
        clim = (0.0, 1e-6)

    shell_faces = None
    if len(shell_cells) > 0:
        shell_faces = np.hstack([
            np.full((len(shell_cells), 1), 4, dtype=np.int64), shell_cells.astype(np.int64),
        ]).ravel()

    solid_vtk_cells, solid_cell_types = None, None
    if len(solid_cells) > 0:
        solid_vtk_cells = np.hstack([
            np.full((len(solid_cells), 1), 8, dtype=np.int64), solid_cells.astype(np.int64),
        ]).ravel()
        solid_cell_types = np.full(len(solid_cells), pv.CellType.HEXAHEDRON, dtype=np.uint8)

    # Union the focus box across every kept frame so the camera stays valid
    # for the whole animation even as the focused nodes move.
    bounds = _focus_bounds(positions_all, region_label, focus_region, focus_pad)
    print(f"Rendering {len(idx)} PyVista frames (of {n_frames} available), "
          f"clim=(0, {clim[1]:.4g})"
          + (f", focus_region={focus_region} bounds={tuple(round(b) for b in bounds)}" if bounds else "")
          + (f", zoom={zoom}" if zoom != 1.0 else "") + " …")

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    frames: list[Image.Image] = []
    fixed_camera = None

    for fi in idx:
        pos = positions_all[fi]
        eps = eps_all[fi]

        plotter.clear()
        plotter.background_color = "white"

        point_cloud = pv.PolyData(pos)
        point_cloud.point_data["eff_plastic_strain"] = eps
        plotter.add_mesh(
            point_cloud, scalars="eff_plastic_strain", cmap=cmap, clim=clim,
            render_points_as_spheres=True, point_size=point_size,
            show_scalar_bar=True, scalar_bar_args={"title": "eff. plastic strain"},
        )

        if solid_vtk_cells is not None:
            solid_mesh = pv.UnstructuredGrid(solid_vtk_cells, solid_cell_types, pos)
            solid_mesh.point_data["eff_plastic_strain"] = eps
            plotter.add_mesh(
                solid_mesh.extract_surface(), scalars="eff_plastic_strain",
                cmap=cmap, clim=clim, smooth_shading=True, show_edges=False,
                show_scalar_bar=False,
            )

        if shell_faces is not None:
            shell_mesh = pv.PolyData(pos, shell_faces)
            shell_mesh.point_data["eff_plastic_strain"] = eps
            plotter.add_mesh(
                shell_mesh, scalars="eff_plastic_strain", cmap=cmap, clim=clim,
                smooth_shading=True, show_edges=False, show_scalar_bar=False,
            )

        plotter.add_text(f"t = {times[fi] * 1e3:.1f} ms   (frame {fi + 1}/{n_frames})",
                         position="upper_left", font_size=10, color="black")

        if fixed_camera is None:
            _apply_camera(plotter, bounds, zoom)
            fixed_camera = plotter.camera_position
        else:
            plotter.camera_position = fixed_camera

        img = plotter.screenshot(return_img=True)
        frames.append(Image.fromarray(img))

    plotter.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f"Saved GIF: {out_path}  ({len(frames)} frames, {out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a build_dataset.py HDF5 frame (or animation) with PyVista.")
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=-1, help="Frame index (negative = from the end).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: <h5>.frameN.png, or <h5>.pv.gif with --gif).")
    parser.add_argument("--interactive", action="store_true", help="Open an interactive PyVista window instead.")
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--gif", action="store_true", help="Render an animated GIF instead of a single frame.")
    parser.add_argument("--frame-stride", type=int, default=1, help="(--gif) use every N-th stored frame.")
    parser.add_argument("--fps", type=int, default=8, help="(--gif) frames per second.")
    parser.add_argument("--max-frames", type=int, default=60, help="(--gif) cap total GIF frames.")
    parser.add_argument("--focus-region", default=None,
                        help="Comma-separated region names to zoom into, e.g. "
                             "'barrier_fine,veh_contact,force_keep' for the impact zone "
                             "(region_label values: force_keep, barrier_fine, barrier_coarse, "
                             "veh_contact, veh_near, veh_far). Camera fits tightly to just "
                             "these nodes' bounding box (+ --focus-pad) instead of the full scene.")
    parser.add_argument("--focus-pad", type=float, default=800.0,
                        help="Padding in mm around the --focus-region bounding box (default 800).")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Extra camera zoom multiplier on top of the auto-fit/focus view "
                             "(>1 zooms in, <1 zooms out). Combine with --focus-region for the "
                             "tightest detail view.")
    args = parser.parse_args()
    focus_region = args.focus_region.split(",") if args.focus_region else None

    if args.gif:
        out = args.out or args.h5.with_suffix(".pv.gif")
        make_gif(args.h5, out, args.frame_stride, args.fps, args.max_frames,
                 args.cmap, args.point_size,
                 focus_region=focus_region, focus_pad=args.focus_pad, zoom=args.zoom)
    else:
        render_frame(args.h5, args.frame, args.out, args.interactive, args.point_size, args.cmap,
                     focus_region=focus_region, focus_pad=args.focus_pad, zoom=args.zoom)


if __name__ == "__main__":
    main()
