"""
trim_h5.py  –  Keep only the first fraction of frames from an output.h5 file.

Usage:
  python3 tools/trim_h5.py input.h5 output.h5            # default 10%
  python3 tools/trim_h5.py input.h5 output.h5 --scale 0.1
  python3 tools/trim_h5.py input.h5 output.h5 --frames 50
"""

import argparse
import sys
import math

import h5py
import numpy as np

# datasets in /states that have a leading time dimension
STATE_FIELDS = ["times", "positions", "velocity", "acceleration", "stress"]


def trim_h5(src_path: str, dst_path: str, k: int) -> None:
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:

        # ── /metadata: copy everything unchanged ──────────────────────────────
        src.copy("metadata", dst)

        # update n_frames to reflect the trim
        dst["metadata"].attrs["n_frames"] = k

        # ── /states: copy only first k frames ─────────────────────────────────
        sg = dst.require_group("states")
        for field in STATE_FIELDS:
            ds_src = src[f"states/{field}"]
            data   = ds_src[:k]             # read first k frames

            # preserve same chunking / compression scheme
            chunks = ds_src.chunks
            if chunks is not None:
                # clamp time chunk to k (can't exceed dataset size)
                chunks = (min(chunks[0], k),) + chunks[1:]

            sg.create_dataset(
                field,
                data=data,
                chunks=chunks,
                compression=ds_src.compression,
                compression_opts=ds_src.compression_opts,
            )

        # copy any extra datasets in /states we don't know about
        for name in src["states"]:
            if name not in STATE_FIELDS:
                src.copy(f"states/{name}", sg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim an output.h5 to the first N frames for training.")
    parser.add_argument("input",  help="Source output.h5")
    parser.add_argument("output", help="Destination trimmed .h5")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scale",  type=float, default=0.1, metavar="FRAC",
                       help="Fraction of frames to keep (default: 0.1 = 10%%)")
    group.add_argument("--frames", type=int,   default=None, metavar="K",
                       help="Exact number of frames to keep")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        total_frames = int(f["metadata"].attrs["n_frames"])
        times        = f["states/times"][:]

    if args.frames is not None:
        k = args.frames
    else:
        k = max(1, math.ceil(total_frames * args.scale))

    if k > total_frames:
        sys.exit(f"Requested {k} frames but file only has {total_frames}.")

    t_start = times[0]  * 1e3
    t_end   = times[k-1] * 1e3

    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Frames : {k} / {total_frames}  "
          f"({k/total_frames*100:.1f}%,  t = {t_start:.2f} – {t_end:.2f} ms)")

    trim_h5(args.input, args.output, k)

    import os
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Done.  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
