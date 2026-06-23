"""
Remove suspension-part nodes from h5 trajectories.

Suspension part keywords are read from configs/data/sampling_config.yaml
(the `required_parts.suspension` list).  Any node whose part name contains
one of those keywords (case-insensitive substring match) is dropped.

Handles all metadata fields automatically:
  - Per-node fields (first dim == N_nodes): filtered by keep mask
  - Index fields (*_idx, first dim != N_nodes): remapped to new node indices
  - Pattern / string fields and per-part lists: copied or filtered appropriately
  - states datasets: filtered along node axis (axis 1 for 3-D, axis 1 for 2-D)

Usage (single trajectory):
    python dataset/remove_suspension.py \\
        --src /data/.../h5dt_50ns_10fs_mat/T_lok_F_shape_barrier_9_3_100km \\
        --dst /data/.../h5dt_50ns_10fs_mat_no_suspension/T_lok_F_shape_barrier_9_3_100km

Usage (batch — all sub-dirs of SRC_ROOT → DST_ROOT):
    python dataset/remove_suspension.py --batch [--workers N]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLING_CFG = PROJECT_ROOT / "configs" / "data" / "sampling_config.yaml"

SRC_ROOT = Path("/data/curtin_ciraee/curtin_xiangrui/data/h5dt_50ns_10fs_mat")
DST_ROOT = Path("/data/curtin_ciraee/curtin_xiangrui/data/h5dt_50ns_10fs_mat_no_suspension")


# ── Suspension keyword list ────────────────────────────────────────────────────

def load_suspension_keywords(cfg_path: Path) -> list[str]:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    keywords = cfg["required_parts"]["suspension"]
    return [k.lower() for k in keywords]


# ── Node keep mask ─────────────────────────────────────────────────────────────

def build_keep_mask(node_part_name: np.ndarray, suspension_kw: list[str]) -> np.ndarray:
    """Return bool mask (N,): True = keep, False = suspension node."""
    keep = np.ones(len(node_part_name), dtype=bool)
    for i, raw in enumerate(node_part_name):
        name = (raw.decode() if isinstance(raw, bytes) else str(raw)).lower()
        if any(kw in name for kw in suspension_kw):
            keep[i] = False
    return keep


# ── Single-trajectory conversion ───────────────────────────────────────────────

def convert_one(src_dir: Path, dst_dir: Path, suspension_kw: list[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata.json unchanged
    src_json = src_dir / "metadata.json"
    if src_json.exists():
        shutil.copy2(src_json, dst_dir / "metadata.json")

    src_h5 = src_dir / "output.h5"
    dst_h5 = dst_dir / "output.h5"

    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        # ── Build node keep mask ───────────────────────────────────────────
        raw_names   = src["metadata/node_part_name"][:]
        keep        = build_keep_mask(raw_names, suspension_kw)
        N_orig      = len(keep)
        N_new       = int(keep.sum())

        # old-index → new-index remapping table (−1 = removed)
        old_to_new  = np.full(N_orig, -1, dtype=np.int64)
        old_to_new[keep] = np.arange(N_new, dtype=np.int64)

        # ── States group ──────────────────────────────────────────────────
        dst_states = dst.create_group("states")
        for ds_name, ds in src["states"].items():
            data = ds[()]
            if data.ndim >= 2 and data.shape[1] == N_orig:
                # Per-node axis is axis 1 (T, N, ...) or (T, N)
                data = data[:, keep]
            # times and other 1-D non-node arrays are copied verbatim
            dst_states.create_dataset(
                ds_name, data=data,
                compression="gzip", compression_opts=4, chunks=True,
            )

        # ── Metadata group ────────────────────────────────────────────────
        dst_meta = dst.create_group("metadata")

        # Identify suspension part IDs so we can clean up part_ids/part_names
        node_part_id   = src["metadata/node_part_id"][:]
        susp_part_ids  = set(node_part_id[~keep].tolist())
        kept_part_ids  = set(node_part_id[keep].tolist())
        # A part is fully removed only if it has NO kept nodes
        drop_part_ids  = susp_part_ids - kept_part_ids

        part_ids_arr   = src["metadata/part_ids"][:]
        part_keep_mask = np.array([pid not in drop_part_ids for pid in part_ids_arr])

        for ds_name, ds in src["metadata"].items():
            data = ds[()]

            if ds_name in ("part_ids", "part_names"):
                # Per-part arrays: remove fully-dropped suspension parts
                filtered = data[part_keep_mask]
                dst_meta.create_dataset(ds_name, data=filtered)
                continue

            if data.ndim >= 1 and data.shape[0] == N_orig:
                # Per-node metadata (node_part_id, node_part_name, ref_positions,
                # node_mat_*, node_global_idx, node_part_label, …)
                filtered = data[keep]
                dst_meta.create_dataset(ds_name, data=filtered)
                continue

            # Index remapping fields: *_idx arrays whose values are node indices
            if ds_name.endswith("_idx") and data.ndim == 1 and data.dtype.kind in ("i", "u"):
                remapped = old_to_new[data]
                valid    = remapped >= 0          # drop indices that pointed to removed nodes
                dst_meta.create_dataset(ds_name, data=remapped[valid])
                continue

            # Everything else (pattern strings, scalars, etc.): copy verbatim
            dst_meta.create_dataset(ds_name, data=data)

    removed = N_orig - N_new
    print(f"  {src_dir.name}: {N_orig} → {N_new} nodes  (removed {removed} suspension)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Remove suspension nodes from h5 trajectories")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch",  action="store_true",
                      help=f"Process all sub-dirs of {SRC_ROOT} → {DST_ROOT}")
    mode.add_argument("--src",    type=Path, metavar="DIR",
                      help="Single source trajectory directory")
    parser.add_argument("--dst",  type=Path, metavar="DIR", default=None,
                        help="Single destination directory (required with --src)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for --batch mode (default: 1)")
    args = parser.parse_args()

    suspension_kw = load_suspension_keywords(SAMPLING_CFG)
    print(f"[Config] Suspension keywords ({len(suspension_kw)}): {suspension_kw}")

    if args.batch:
        cases = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])
        print(f"[Batch] {len(cases)} trajectories: {SRC_ROOT} → {DST_ROOT}")

        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            futures = {}
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for src_dir in cases:
                    fut = pool.submit(convert_one, src_dir, DST_ROOT / src_dir.name, suspension_kw)
                    futures[fut] = src_dir.name
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    name = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        print(f"  ERROR {name}: {exc}")
        else:
            for src_dir in tqdm(cases, desc="Processing"):
                try:
                    convert_one(src_dir, DST_ROOT / src_dir.name, suspension_kw)
                except Exception as exc:
                    print(f"  ERROR {src_dir.name}: {exc}")
    else:
        if args.dst is None:
            parser.error("--dst is required when using --src")
        convert_one(args.src, args.dst, suspension_kw)

    print("Done.")


if __name__ == "__main__":
    main()
