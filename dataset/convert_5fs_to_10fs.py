"""
Convert h5dt_50ns_5fs_mat → h5dt_50ns_10fs_mat by keeping only even-indexed frames
(0, 2, 4, ...), effectively doubling the frame stride from 5fs to 10fs.

Usage:
    python dataset/convert_5fs_to_10fs.py [--workers N]
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

SRC_ROOT = Path("/data/curtin_ciraee/curtin_xiangrui/data/h5dt_50ns_5fs_mat")
DST_ROOT = Path("/data/curtin_ciraee/curtin_xiangrui/data/h5dt_50ns_10fs_mat")

STATES_KEY = "states"
METADATA_KEY = "metadata"


def convert_one(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata.json unchanged if present
    src_json = src_dir / "metadata.json"
    if src_json.exists():
        shutil.copy2(src_json, dst_dir / "metadata.json")

    src_h5 = src_dir / "output.h5"
    dst_h5 = dst_dir / "output.h5"

    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        # --- metadata group: copy verbatim ---
        src.copy(METADATA_KEY, dst)

        # --- states group: subsample first axis ---
        dst_states = dst.create_group(STATES_KEY)
        src_states = src[STATES_KEY]

        for ds_name, ds in src_states.items():
            data = ds[()]  # load fully; datasets are manageable in RAM
            if data.ndim >= 1 and data.shape[0] > 1:
                # Keep frames 0, 2, 4, ... (delete 1, 3, 5, ...)
                data = data[::2]
            dst_states.create_dataset(
                ds_name,
                data=data,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample h5 frame stride 5fs→10fs")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    args = parser.parse_args()

    cases = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(cases)} cases in {SRC_ROOT}")

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for src_dir in cases:
                dst_dir = DST_ROOT / src_dir.name
                fut = pool.submit(convert_one, src_dir, dst_dir)
                futures[fut] = src_dir.name

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                name = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    print(f"  ERROR {name}: {exc}")
    else:
        for src_dir in tqdm(cases, desc="Converting"):
            dst_dir = DST_ROOT / src_dir.name
            try:
                convert_one(src_dir, dst_dir)
            except Exception as exc:
                print(f"  ERROR {src_dir.name}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
