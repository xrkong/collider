"""
build_dataset.py — k-file + d3plot → one downsampled HDF5 file.

Mirrors the pipeline shape of dataset/d3plot_to_h5_dt.py (part/node
selection → frame selection → per-frame extraction → HDF5 write), but the
node selection is the SPEC region-aware sampler (see SPEC/SPEC_sampling_reconstruction.md
§4-5) instead of a YAML part-pattern + stride filter.

Pipeline
--------
  Step 1 – Parse k-file part names + material props      (materials.py)
  Step 2 – Optional part exclusion (--exclude-parts-config, e.g. rotating
           tires/rims/spindles — see part_filters.py)
  Step 3 – Parse k-file geometry, then region-aware       (kfile_parser.py,
           ~100k node sampling                            regions.py, samplers.py, sampling.py)
  Step 4 – d3plot header: part mass + mesh connectivity  (d3plot_io.py, connectivity.py)
  Step 5 – Time scan + frame-stride selection            (d3plot_io.py)
  Step 6 – Per-frame position + eff_plastic_strain       (this file, d3plot_io.py)
  Step 7 – Write one HDF5 (metadata + states)            (this file)
  Step 8 – Vehicle-GC / barrier kinematics CSV + plots,  (gc_barrier.py)
           at full native d3plot resolution (no
           --frame-stride) — side artifact, always run,
           doesn't touch the HDF5 above

HDF5 layout
-----------
  /metadata/
      sampled_node_ids       (N,)      int64    k-file node IDs
      ref_positions          (N, 3)    float32  t=0 reference coords [mm]
      region_id              (N,)      int32    0–5 (see region_id_legend attr)
      region_label           (N,)      bytes    "barrier_fine" / "veh_contact" / ...
      node_part_id           (N,)      int32    k-file PID
      node_part_name         (N,)      bytes    d3plot part name
      node_mat_label         (N,)      bytes    material title or type keyword
      node_mat_type_name     (N,)      bytes    MAT keyword without *MAT_
      node_mat_rho           (N,)      float32  density          [ton/mm^3]
      node_mat_E             (N,)      float32  Young's modulus  [MPa]
      node_mat_nu            (N,)      float32  Poisson's ratio
      node_mat_sigy          (N,)      float32  yield stress     [MPa]
      node_mass              (N,)      float32  nodal mass [ton] (part_mass / n_nodes_in_part)
      shell_cells            (Es, 4)   int32    quad cell connectivity, LOCAL 0..N-1 indices
                                                 (tri elements: last index repeats the 3rd)
      solid_cells            (Eh, 8)   int32    hex cell connectivity, LOCAL 0..N-1 indices
      beam_cells              (Eb, 2)   int32    line cell connectivity, LOCAL 0..N-1 indices
                                                 (mostly rebar/reinforcement — fully retained
                                                 by sampling, so this connectivity is intact)
      attrs: n_nodes, n_frames, frame_stride, source_dir, kfile, region_id_legend,
             n_shell_cells, n_solid_cells, n_beam_cells, cell_coverage_note

  /states/
      times                  (T,)      float64  [s]
      positions              (T, N, 3) float32  deformed xyz [mm]
      eff_plastic_strain     (T, N)    float32  shell+solid element strain averaged onto
                                                 each sampled node, eroded elements excluded
                                                 (full coverage — see connectivity.py docstring)
      node_alive             (T, N)    uint8    1 unless every element touching that node has
                                                 eroded (element_*_is_alive == 0); this dataset
                                                 does have some erosion despite SPEC's "no
                                                 erosion" assumption (~0.03% shell / ~0.04%
                                                 solid elements by the end of a 60km/h run) —
                                                 node COUNT stays fixed regardless (LS-DYNA
                                                 keeps eroded nodes' positions), only liveness
                                                 changes; mask training loss with this where
                                                 needed

  shell_cells/solid_cells/beam_cells are sparse: PyVista visualization should
  overlay a point cloud (e.g. colored by eff_plastic_strain) for full
  coverage, with the cells drawn on top wherever they exist. See
  dataset/visualize_pyvista.py.

Side outputs (under <out_stem>_analysis/ next to --out, always written)
----------------------------------------------------------------------
  <out_stem>_analysis/
    fem/                                ground truth, full native d3plot
                                         resolution — no --frame-stride,
                                         no node sampling
      <out_stem>_gc_barrier.csv         vehicle-GC + barrier reference
                                         point kinematics, every d3plot state
      <out_stem>_gc_barrier_plots/      ORA_x, ORA_y, ASI (EN 1317),
                                         barrier displacement PNGs derived
                                         from that CSV
      <out_stem>_rotation.csv           vehicle roll/pitch/yaw (deg),
                                         Kabsch-fit on the 8-node CG
                                         cluster (PID 9000100), every
                                         d3plot state
      <out_stem>_rotation_plots/        rotation_roll/pitch/yaw.png
                                         derived from that CSV
    downsampled/                        same CSV+plots, but read back from
                                         the sampled/strided HDF5 above —
                                         "what the model actually sees"
      <out_stem>_gc_barrier.csv
      <out_stem>_gc_barrier_plots/
      <out_stem>_rotation.csv
      <out_stem>_rotation_plots/
      <out_stem>.gif                    top-down rollout GIF (--gif),
                                         rendered from the same HDF5

  See dataset/gc_barrier.py (run_gc_barrier_full_res / run_gc_barrier_downsampled).

Usage
-----
conda activate collider
conda run -n collider python -m dataset.build_dataset \
    --kfile  /home/kong/datasets/barrier/fem/T_lok_F_shape_barrier_9_3_60km/car_and_barriers.k \
    --src    /home/kong/datasets/barrier/fem/T_lok_F_shape_barrier_9_3_60km \
    --out    /home/kong/datasets/barrier/h5_fps_no_wheels/T_lok_F_shape_barrier_9_3_60km.h5 \
    --method fps \
    --seed 42 \
    --exclude-parts-config configs/data/exclude_parts_tires.yaml \
    --frame-stride 10 --n-jobs 8 \
    --gif
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed
from lasso.dyna import ArrayType, D3plot

from .connectivity import filter_and_remap_cells, load_connectivity
from .constants import REGION_ID_LEGEND, REGION_ID_MAP
from .d3plot_io import (
    close_d3, copy_to_tmp, extract_frame_data, find_state_files, scan_times, select_frames,
)
from .gc_barrier import run_gc_barrier_downsampled, run_gc_barrier_full_res
from .kfile_parser import parse_kfile
from .materials import parse_kfile_parts_and_materials
from .part_filters import load_exclude_parts_config, resolve_exclude_pids
from .sampling import DEFAULT_REGION_CONFIGS, RegionConfig, sample_mesh
from .samplers import SamplerConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a k-file + d3plot sequence into one downsampled HDF5 file."
    )
    parser.add_argument("--kfile", type=Path, required=True,
                        help="LS-DYNA keyword file (car_and_barriers.k) — geometry + materials.")
    parser.add_argument("--src",  type=Path, required=True,
                        help="Directory containing d3plot, d3plot01, d3plot02, ...")
    parser.add_argument("--tmp",  type=Path, default=Path("/tmp/d3plot_tmp"),
                        help="Scratch directory for single-state-file copies.")
    parser.add_argument("--out",  type=Path, required=True,
                        help="Output HDF5 file path.")
    parser.add_argument("--frame-stride", type=int, default=5,
                        help="Keep every N-th time state (default 5).")
    parser.add_argument("--frame-limit",  type=int, default=None,
                        help="Cap total frames kept, applied after stride.")
    parser.add_argument("--method", default="fps",
                        choices=["fps", "random", "stride", "poisson_disk"],
                        help="Node-sampling method applied to every region (SPEC §5). "
                             "Region budgets/floors are unchanged — only the within-region "
                             "selection algorithm changes. Default fps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the sampler (random/poisson_disk/fps).")
    parser.add_argument("--exclude-parts-config", type=Path, default=None,
                        help="YAML with an 'exclude_parts' list of name patterns (e.g. "
                             "configs/data/exclude_parts_tires.yaml) — parts matching any "
                             "pattern are dropped from sampling entirely, before region/budget "
                             "logic runs. Use dataset/part_filters.py to (re)generate one "
                             "for a new k-file.")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Parallel workers for d3plot time-scan + frame extraction "
                             "(joblib; each file is read independently). 1 = sequential.")
    parser.add_argument("--compression-level", type=int, default=1, choices=range(0, 10),
                        help="gzip level for HDF5 datasets (default 1: fast, still compresses "
                             "well — level 4+ is much slower for little extra size benefit here).")
    parser.add_argument("--gif", action="store_true",
                        help="Render an animated top-down GIF of the result once the HDF5 is written "
                             "(saved under <out_stem>_analysis/ unless --gif-out is given).")
    parser.add_argument("--gif-out", type=Path, default=None,
                        help="GIF output path (default: <out_stem>_analysis/<out_stem>.gif).")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-max-frames", type=int, default=80,
                        help="Cap GIF frames (resamples evenly across kept frames if exceeded).")
    parser.add_argument("--gc-barrier-window-ms", type=float, default=50.0,
                        help="Moving-average window (ms) applied to ORA_x/ORA_y/ASI in the "
                             "GC/barrier plots (default 50, per EN 1317). Set to 0 to plot "
                             "raw, unfiltered per-frame values instead.")
    args = parser.parse_args()

    args.tmp.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for old in args.tmp.glob("d3plot*"):
        old.unlink()
    for old_dir in list(args.tmp.glob("scan_*")) + list(args.tmp.glob("extract_*")):
        if old_dir.is_dir():
            shutil.rmtree(old_dir, ignore_errors=True)

    # ══════════════════════════════════════════════════════════════════════
    # Step 1-3 — parse k-file geometry + part names/materials, optional
    # part exclusion, then region-aware sampling
    # ══════════════════════════════════════════════════════════════════════
    # d3plot's ArrayType.part_titles_ids is a sequential 1..P index, NOT the
    # real k-file PID — it cannot be used to join part names. The k-file's
    # own *PART block has the real PID as its first field, so parse that
    # directly instead (same parse pass also resolves *MAT_xxx properties).
    # Needed before sampling (not just for metadata) so --exclude-parts-config
    # can resolve name patterns to PIDs before any region/budget logic runs.
    print(f"Parsing k-file for part names + material props: {args.kfile} …")
    pid_to_name, name_to_props = parse_kfile_parts_and_materials(args.kfile)
    print(f"  resolved {len(pid_to_name)} parts, {len(name_to_props)} with material props")

    exclude_pids: set[int] = set()
    if args.exclude_parts_config:
        patterns = load_exclude_parts_config(args.exclude_parts_config)
        matched = resolve_exclude_pids(patterns, pid_to_name)
        exclude_pids = set(matched)
        print(f"Excluding {len(matched)} parts matching {patterns} "
              f"(from {args.exclude_parts_config}):")
        for pid, name in sorted(matched.items()):
            print(f"  {pid:>10}  {name}")

    # Region budgets/floors (§4) stay fixed; only the within-region sampler
    # method changes, so the comparison across methods is fair (SPEC §5).
    region_configs = [
        RegionConfig(
            name=rc.name,
            sampler=SamplerConfig(
                method=args.method, n_points=rc.sampler.n_points, seed=args.seed,
            ),
            split_by_part=rc.split_by_part,
            min_per_part=rc.min_per_part,
        )
        for rc in DEFAULT_REGION_CONFIGS
    ]
    print(f"Sampling method: {args.method}  (seed={args.seed})")

    mesh = parse_kfile(args.kfile)
    sampled_idx, region_labels, _segments = sample_mesh(
        mesh, region_configs, exclude_pids=exclude_pids
    )
    N = len(sampled_idx)

    sampled_nids  = mesh.node_ids[sampled_idx]
    ref_positions = mesh.coords[sampled_idx].astype(np.float32)
    node_part_ids = mesh.node_pid[sampled_idx].astype(np.int32)
    region_id     = np.array([REGION_ID_MAP.get(r, -1) for r in region_labels], dtype=np.int32)

    print(f"\nSampled {N:,} nodes total:")
    for label, rid in sorted(REGION_ID_MAP.items(), key=lambda x: x[1]):
        print(f"  {label:<20} {(region_id == rid).sum():>7,}")

    node_part_name = np.array(
        [pid_to_name.get(int(pid), f"pid_{pid}") for pid in node_part_ids]
    )

    mat_type_name = np.full(N, "UNKNOWN", dtype=object)
    mat_label     = np.full(N, "UNKNOWN", dtype=object)
    mat_rho       = np.zeros(N, dtype=np.float32)
    mat_E         = np.zeros(N, dtype=np.float32)
    mat_nu        = np.zeros(N, dtype=np.float32)
    mat_sigy      = np.zeros(N, dtype=np.float32)
    for i, name in enumerate(node_part_name):
        p = name_to_props.get(str(name))
        if p:
            mat_type_name[i] = p["type_name"]
            mat_label[i]     = p.get("label", p["type_name"])
            mat_rho[i]       = p["rho"]
            mat_E[i]         = p["E"]
            mat_nu[i]        = p["nu"]
            mat_sigy[i]      = p["sigy"]

    # ══════════════════════════════════════════════════════════════════════
    # Step 4 — nodal mass (d3plot part_mass) + mesh connectivity (d3plot
    # element_*_node_indexes). Both need a state file attached — the header
    # alone always reports part_mass empty — so stage header + first state
    # file together, same as the extraction loop does later. NOTE:
    # ArrayType.part_ids (not part_titles_ids) holds the real PIDs aligned
    # 1:1 with part_mass — verified against known PIDs.
    # ══════════════════════════════════════════════════════════════════════
    print("\nLoading d3plot header (+ first state, for part mass + connectivity) …")
    state_files_probe = find_state_files(args.src)
    if not state_files_probe:
        raise FileNotFoundError(f"No d3plot state files in {args.src}")
    copy_to_tmp(state_files_probe[0], args.tmp)
    try:
        d3hdr = D3plot(
            str(args.tmp / "d3plot"),
            state_array_filter=[ArrayType.global_timesteps, ArrayType.part_mass],
        )
        conn = load_connectivity(d3hdr)
    finally:
        close_d3(None, args.tmp)

    part_ids_raw  = d3hdr.arrays.get(ArrayType.part_ids, np.array([]))
    part_mass_arr = d3hdr.arrays.get(ArrayType.part_mass)
    pid_to_mass: dict[int, float] = {}
    if part_mass_arr is not None and len(part_ids_raw) > 0:
        pm = np.asarray(part_mass_arr)
        if pm.ndim == 2:
            pm = pm[0]  # state 0
        pid_to_mass = {int(pid): float(m) for pid, m in zip(part_ids_raw, pm)}

    n_full = int(max(
        mesh.n_nodes,
        conn.shell_conn.max(initial=-1) + 1,
        conn.solid_conn.max(initial=-1) + 1,
        conn.beam_conn.max(initial=-1) + 1,
    ))
    del d3hdr

    pid_node_count: dict[int, int] = {}
    for pid in node_part_ids:
        pid_node_count[int(pid)] = pid_node_count.get(int(pid), 0) + 1
    node_mass = np.array([
        pid_to_mass.get(int(pid), 0.0) / max(pid_node_count.get(int(pid), 1), 1)
        for pid in node_part_ids
    ], dtype=np.float32)

    print(f"  connectivity: {len(conn.shell_conn):,} shell / {len(conn.solid_conn):,} solid / "
          f"{len(conn.beam_conn):,} beam elements total")
    shell_cells = filter_and_remap_cells(conn.shell_conn, sampled_idx, n_full)
    solid_cells = filter_and_remap_cells(conn.solid_conn, sampled_idx, n_full)
    beam_cells  = filter_and_remap_cells(conn.beam_conn, sampled_idx, n_full)
    print(f"  surviving cells (all nodes sampled): {len(shell_cells):,} shell "
          f"({100*len(shell_cells)/max(len(conn.shell_conn),1):.2f}%) / "
          f"{len(solid_cells):,} solid ({100*len(solid_cells)/max(len(conn.solid_conn),1):.2f}%) / "
          f"{len(beam_cells):,} beam ({100*len(beam_cells)/max(len(conn.beam_conn),1):.2f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # Step 5 — time scan + frame-stride selection
    # ══════════════════════════════════════════════════════════════════════
    state_files = state_files_probe
    all_entries = scan_times(state_files, args.tmp, n_jobs=args.n_jobs)
    selected = select_frames(all_entries, args.frame_stride, args.frame_limit)
    n_frames = len(selected)

    # ══════════════════════════════════════════════════════════════════════
    # Step 6-7 — write HDF5: static metadata first, then per-frame positions
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nCreating {args.out}  ({n_frames} frames x {N:,} nodes) …")
    with h5py.File(args.out, "w") as h5f:
        mg = h5f.require_group("metadata")
        mg.create_dataset("sampled_node_ids",  data=sampled_nids.astype(np.int64))
        mg.create_dataset("ref_positions",     data=ref_positions)
        mg.create_dataset("region_id",         data=region_id)
        mg.create_dataset("region_label",      data=region_labels.astype("S"))
        mg.create_dataset("node_part_id",      data=node_part_ids)
        mg.create_dataset("node_part_name",    data=node_part_name.astype("S"))
        mg.create_dataset("node_mat_label",    data=mat_label.astype("S"))
        mg.create_dataset("node_mat_type_name", data=mat_type_name.astype("S"))
        mg.create_dataset("node_mat_rho",      data=mat_rho)
        mg.create_dataset("node_mat_E",        data=mat_E)
        mg.create_dataset("node_mat_nu",       data=mat_nu)
        mg.create_dataset("node_mat_sigy",     data=mat_sigy)
        mg.create_dataset("node_mass",         data=node_mass)
        mg.create_dataset("shell_cells",       data=shell_cells.astype(np.int32))
        mg.create_dataset("solid_cells",       data=solid_cells.astype(np.int32))
        mg.create_dataset("beam_cells",        data=beam_cells.astype(np.int32))
        mg.attrs["n_nodes"]          = N
        mg.attrs["n_frames"]         = n_frames
        mg.attrs["frame_stride"]     = args.frame_stride
        mg.attrs["sampling_method"]  = args.method
        mg.attrs["sampling_seed"]    = args.seed
        mg.attrs["region_id_legend"] = REGION_ID_LEGEND
        mg.attrs["source_dir"]       = str(args.src.resolve())
        mg.attrs["kfile"]            = str(args.kfile.resolve())
        mg.attrs["units"]            = "mm, ton, s, MPa"
        mg.attrs["node_mass_note"]   = "part_mass[t=0] / n_nodes_in_part (uniform distribution)"
        mg.attrs["n_shell_cells"]    = len(shell_cells)
        mg.attrs["n_solid_cells"]    = len(solid_cells)
        mg.attrs["n_beam_cells"]     = len(beam_cells)
        mg.attrs["cell_coverage_note"] = (
            "cells kept only where ALL element nodes survived sampling — sparse by "
            "construction (FPS spreads points out); overlay a point cloud colored by "
            "eff_plastic_strain for full coverage, see visualize_pyvista.py"
        )

        sg = h5f.require_group("states")
        ct, cn = min(16, n_frames), min(4096, N)
        gz = args.compression_level
        sg.create_dataset("times", shape=(n_frames,), dtype="float64",
                          chunks=(ct,), compression="gzip", compression_opts=gz)
        sg.create_dataset("positions", shape=(n_frames, N, 3), dtype="float32",
                          chunks=(ct, cn, 3), compression="gzip", compression_opts=gz)
        sg.create_dataset("eff_plastic_strain", shape=(n_frames, N), dtype="float32",
                          chunks=(ct, cn), compression="gzip", compression_opts=gz)
        sg.create_dataset("node_alive", shape=(n_frames, N), dtype="uint8",
                          chunks=(ct, cn), compression="gzip", compression_opts=gz)

        # NOTE: despite the name, lasso's ArrayType.node_displacement holds
        # the deformed nodal COORDINATE directly (verified: a fixed ground
        # node has disp == ref exactly; a moving node's "delta" grows to
        # >10 m, far beyond any physical displacement — i.e. it is not a
        # delta at all). extract_frame_data returns it as-is; do NOT add
        # ref_positions on top or it gets double-counted.
        print(f"\nPass 2/2 — extracting {n_frames} frames … (n_jobs={args.n_jobs})")
        state_file_map = {p.name: p for p in state_files}
        tasks = [(t, fname, sidx) for t, fname, sidx in selected]

        extract_args = (
            sampled_idx, conn.shell_conn, conn.solid_conn, conn.beam_conn, n_full, args.tmp,
        )
        if args.n_jobs == 1:
            results = [
                extract_frame_data(state_file_map[fname], sidx, *extract_args)
                for _, fname, sidx in tasks
            ]
        else:
            results = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(extract_frame_data)(state_file_map[fname], sidx, *extract_args)
                for _, fname, sidx in tasks
            )

        n_eroded_final = 0
        for fi, ((pos, eps, alive), (t, fname, _sidx)) in enumerate(zip(results, tasks)):
            h5f["states/times"][fi] = t
            h5f["states/positions"][fi] = pos
            h5f["states/eff_plastic_strain"][fi] = eps
            h5f["states/node_alive"][fi] = alive.astype(np.uint8)
            n_eroded_final = int((~alive).sum())
            if fi % 20 == 0 or fi == n_frames - 1:
                print(f"  [{fi + 1:>5}/{n_frames}]  t = {t * 1e3:.3f} ms  (file {fname})  "
                      f"eroded_nodes={n_eroded_final}")

    size_mb = args.out.stat().st_size / 1e6
    print(f"\n{'-'*60}")
    print(f"Done.  {args.out}")
    print(f"Size        : {size_mb:.1f} MB")
    print(f"Nodes (N)   : {N:,}")
    print(f"Frames (T)  : {n_frames}  (stride={args.frame_stride})")
    print(f"Cells       : {len(shell_cells):,} shell / {len(solid_cells):,} solid / {len(beam_cells):,} beam")
    print(f"Eroded nodes: {n_eroded_final:,} / {N:,} at final frame")
    print(f"Layout      : /metadata  +  /states/{{times, positions, eff_plastic_strain, node_alive}}")
    print(f"{'-'*60}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 8 — vehicle-GC / barrier kinematics CSV + plots (ORA_x, ORA_y,
    # ASI, displacement), plus vehicle rigid-body rotation (roll/pitch/yaw,
    # Kabsch-fit on the 8-node CG cluster) — independent side artifact of
    # the main HDF5 above, which stays exactly as before. See
    # dataset/gc_barrier.py.
    #
    # Two groups under <out_stem>_analysis/, next to the .h5 itself:
    #   fem/          raw FEM/d3plot ground truth, full native resolution
    #                 (no --frame-stride, no node sampling)
    #   downsampled/  the same CSV+plots but read back from the sampled/
    #                 strided HDF5 above — "what the model actually
    #                 sees" — plus the rollout GIF (also from that HDF5)
    # ══════════════════════════════════════════════════════════════════════
    analysis_dir = args.out.parent / f"{args.out.stem}_analysis"
    fem_analysis_dir = analysis_dir / "fem"
    downsampled_analysis_dir = analysis_dir / "downsampled"
    fem_analysis_dir.mkdir(parents=True, exist_ok=True)
    downsampled_analysis_dir.mkdir(parents=True, exist_ok=True)

    run_gc_barrier_full_res(
        mesh, state_files, all_entries, args.tmp, args.n_jobs,
        out_csv=fem_analysis_dir / f"{args.out.stem}_gc_barrier.csv",
        out_plot_dir=fem_analysis_dir / f"{args.out.stem}_gc_barrier_plots",
        out_rotation_csv=fem_analysis_dir / f"{args.out.stem}_rotation.csv",
        out_rotation_plot_dir=fem_analysis_dir / f"{args.out.stem}_rotation_plots",
        title_suffix=f" — {args.out.stem}", window_ms=args.gc_barrier_window_ms,
    )
    run_gc_barrier_downsampled(
        args.out,
        out_csv=downsampled_analysis_dir / f"{args.out.stem}_gc_barrier.csv",
        out_plot_dir=downsampled_analysis_dir / f"{args.out.stem}_gc_barrier_plots",
        out_rotation_csv=downsampled_analysis_dir / f"{args.out.stem}_rotation.csv",
        out_rotation_plot_dir=downsampled_analysis_dir / f"{args.out.stem}_rotation_plots",
        title_suffix=f" — {args.out.stem}", window_ms=args.gc_barrier_window_ms,
    )

    if args.gif:
        from .visualize import make_gif
        gif_path = args.gif_out or (downsampled_analysis_dir / f"{args.out.stem}.gif")
        make_gif(args.out, gif_path, fps=args.gif_fps, max_frames=args.gif_max_frames)


if __name__ == "__main__":
    main()
