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
python -m dataset.ds.visualize_pyvista --h5 output.h5 --frame -1 --out frame.png
python -m dataset.ds.visualize_pyvista --h5 output.h5 --frame 0 --interactive
python -m dataset.ds.visualize_pyvista \
    --h5 /home/kong/datasets/barrier/h5_downsampled/T_lok_F_shape_barrier_9_3_60km.h5 
    --gif \
    --out anim.gif


python -m dataset.ds.visualize_pyvista \
    --h5 /home/kong/datasets/barrier/h5_downsampled/T_lok_F_shape_barrier_9_3_60km.h5 \
    --frame -1 \
    --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from PIL import Image


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
) -> None:
    """Render one frame: cell surface(s) smooth-shaded + point cloud overlay,
    both colored by eff_plastic_strain, on a white background.
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
    plotter.camera_position = "xy"

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
) -> None:
    """Animate every kept frame in the same mesh + point-cloud style as render_frame.

    Camera and color scale are computed once (from frame 0's camera fit and
    the 99.5th percentile of strain over the whole trajectory) and then held
    fixed across all frames, so only the geometry and coloring change —
    otherwise a per-frame autoscaled clim/camera would make frames
    impossible to compare against each other.
    """
    with h5py.File(h5_path, "r") as f:
        n_frames = int(f["metadata"].attrs["n_frames"])
        times = f["states/times"][:]
        positions_all = f["states/positions"][:]            # (T, N, 3)
        eps_all = f["states/eff_plastic_strain"][:]          # (T, N)
        shell_cells = f["metadata/shell_cells"][:]
        solid_cells = f["metadata/solid_cells"][:]

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

    print(f"Rendering {len(idx)} PyVista frames (of {n_frames} available), "
          f"clim=(0, {clim[1]:.4g}) …")

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
            plotter.camera_position = "xy"
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
    args = parser.parse_args()

    if args.gif:
        out = args.out or args.h5.with_suffix(".pv.gif")
        make_gif(args.h5, out, args.frame_stride, args.fps, args.max_frames,
                 args.cmap, args.point_size)
    else:
        render_frame(args.h5, args.frame, args.out, args.interactive, args.point_size, args.cmap)


if __name__ == "__main__":
    main()
