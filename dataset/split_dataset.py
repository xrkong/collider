#!/usr/bin/env python3
"""
H5 Dataset Builder for BVC Simulations.

Reads raw HDF5 simulation files (states/positions, velocity, acceleration, stress)
and produces processed train / val / test splits:

  - train / val : sliding windows of (context_length + prediction_horizon) frames
  - test        : full trajectory slices for autoregressive rollout evaluation

Data is written in **raw physical units** (no normalisation applied).
Per-field mean/std and displacement/acceleration statistics are computed from
TRAINING data only and stored in metadata.json so that the dataloader / model
can normalise on the fly however it wants.

Output layout:

    <output_dir>/
        train/train_data_000.h5   (window groups + sim_metadata group)
        valid/valid_data_000.h5   (window groups + sim_metadata group)
        test/test_data_000.h5     (full-trajectory groups + sim_metadata group)
        metadata.json
        metadata/metadata.json

Each output h5 file contains a `sim_metadata/<sim_id>/` group with
`barrier_idx` and `frontface_idx` for every sim that contributed at least
one window to that file. Window groups carry `sim_id` in their attrs so the
dataloader can look up the corresponding mask.

To normalise downstream:
    x_norm = (x_raw - mean) / std
where mean/std come from metadata["normalization_stats"][field_name].
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELDS = [
    ("states/positions",    "positions",    3),
    ("states/velocity",     "velocity",     3),
    ("states/acceleration", "acceleration", 3),
    ("states/stress",       "stress",       6),
]


# ---------------------------------------------------------------------------
# Time-axis helpers
# ---------------------------------------------------------------------------

def _sliding_windows(T: int, window_size: int, stride: int = 1):
    """Return list of (start, end) frame index pairs."""
    if T < window_size:
        return []
    starts = range(0, T - window_size + 1, stride)
    return [(s, s + window_size) for s in starts]


def _finite_diff(arr: np.ndarray) -> np.ndarray:
    """arr shape (T, N, D) → (T-1, N, D)."""
    return arr[1:] - arr[:-1]


def _preprocess_time_axis(
    arr: np.ndarray,
    max_frames_per_sim: int | None,
    frame_skip: int,
) -> np.ndarray:
    """
    Apply max-frame truncation then frame-skip subsampling.

    Order matters: max_frames_per_sim is interpreted in *raw* frames so the
    user can think in terms of the original simulation length, not the
    post-subsample length.
    """
    if max_frames_per_sim is not None and max_frames_per_sim > 0:
        arr = arr[:max_frames_per_sim]
    if frame_skip > 1:
        arr = arr[::frame_skip]
    return arr


def _extract_dt_from_h5(h5_path: Path, uniformity_tol: float = 1e-4) -> float | None:
    """
    Read /states/times from an h5 file and infer the simulation timestep.

    Returns the mean of np.diff(times); None if /states/times is missing or
    has fewer than 2 entries. Prints a warning when the timestep is
    non-uniform beyond `uniformity_tol` (relative std of np.diff).
    """
    with h5py.File(h5_path, "r") as f:
        if "states/times" not in f:
            return None
        times = f["states/times"][:]

    if times.size < 2:
        return None

    diffs = np.diff(times)
    dt_mean = float(diffs.mean())

    if dt_mean <= 0:
        print(f"  Warning: non-monotonic times in {h5_path.name} (dt_mean={dt_mean})")
        return None

    rel_std = float(diffs.std() / dt_mean) if dt_mean else 0.0
    if rel_std > uniformity_tol:
        print(f"  Warning: non-uniform timestep in {h5_path.name} "
              f"(rel std = {rel_std:.2e}, mean dt = {dt_mean:.6e}); using mean.")

    return dt_mean


def _resolve_dt(h5_files: list[Path], user_dt: float | None) -> float | None:
    """
    Decide which dt to use:
      - If the user passed --dt, use that and sanity-check against file 0.
      - Otherwise auto-detect from the first file and cross-check the rest.
    """
    if user_dt is not None:
        file_dt = _extract_dt_from_h5(h5_files[0])
        if file_dt is not None and abs(file_dt - user_dt) / file_dt > 1e-3:
            print(f"  Warning: --dt={user_dt} differs from file dt={file_dt:.6e} "
                  f"({h5_files[0].name}). Using user-provided value.")
        return user_dt

    dt = _extract_dt_from_h5(h5_files[0])
    if dt is None:
        print(f"  Warning: could not auto-detect dt from {h5_files[0].name} "
              f"(missing or too-short /states/times). dt will be null in metadata.")
        return None

    print(f"  dt auto-detected from {h5_files[0].name}: {dt:.6e} s")

    for other in h5_files[1:]:
        other_dt = _extract_dt_from_h5(other)
        if other_dt is None:
            continue
        if abs(other_dt - dt) / dt > 1e-3:
            print(f"  Warning: dt mismatch in {other.name}: "
                  f"{other_dt:.6e} vs {dt:.6e} (using first-file value)")

    return dt


# ---------------------------------------------------------------------------
# Per-sim metadata helpers (barrier / frontface masks)
# ---------------------------------------------------------------------------

def _read_sim_masks(h5_path: Path) -> dict[str, np.ndarray]:
    """
    Read per-sim node-level masks from a raw h5 file.

    Returns a dict with int64 index arrays. Returns empty dict if the masks
    aren't present (caller decides whether that's fatal).
    """
    masks: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for key in ("barrier_idx", "frontface_idx"):
            ds_path = f"/metadata/{key}"
            if ds_path in f:
                masks[key] = f[ds_path][:].astype(np.int64)
    return masks


# ---------------------------------------------------------------------------
# Statistics (computed for metadata only — NOT applied to written data)
# ---------------------------------------------------------------------------

class RunningStats:
    """Online Welford mean/variance accumulator over (M, D) chunks."""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, chunk: np.ndarray):
        chunk = chunk.astype(np.float64)
        for x in chunk:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def finalize(self):
        if self.n < 2:
            return self.mean, np.ones_like(self.mean)
        return self.mean, np.sqrt(self.M2 / self.n)


def _compute_all_stats(raw_train_windows: dict[str, list[np.ndarray]]):
    """Compute per-field (mean, std) plus displacement/acceleration stats."""
    stats: dict[str, dict] = {}

    for field, arrays in raw_train_windows.items():
        if not arrays:
            continue
        dim = arrays[0].shape[-1]
        acc = RunningStats(dim)
        for arr in arrays:
            acc.update(arr.reshape(-1, dim))
        mean, std = acc.finalize()
        std = np.where(std < 1e-8, 1.0, std)
        stats[field] = {"mean": mean.tolist(), "std": std.tolist()}

    pos_arrays = raw_train_windows.get("positions", [])
    all_disp, all_acc_d = [], []
    for pos in pos_arrays:
        if pos.shape[0] < 2:
            continue
        disp = _finite_diff(pos)
        all_disp.append(disp.reshape(-1, 3))
        if disp.shape[0] >= 2:
            acc_d = _finite_diff(disp)
            all_acc_d.append(acc_d.reshape(-1, 3))

    disp_acc_stats: dict = {}
    if all_disp:
        disp_arr = np.concatenate(all_disp, axis=0)
        disp_acc_stats["displacement_mean"] = disp_arr.mean(axis=0).tolist()
        disp_acc_stats["displacement_std"] = disp_arr.std(axis=0).tolist()
    if all_acc_d:
        acc_arr = np.concatenate(all_acc_d, axis=0)
        disp_acc_stats["acceleration_mean"] = acc_arr.mean(axis=0).tolist()
        disp_acc_stats["acceleration_std"] = acc_arr.std(axis=0).tolist()

    return stats, disp_acc_stats


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _split_windows_temporal(windows, train_ratio, val_ratio, split_gap):
    """
    Time-internal split with a `split_gap` of windows skipped between splits
    to prevent train/val/test windows from sharing frames.

    Returns (train_wins, val_wins, test_start_window_idx_or_None).
    """
    n = len(windows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_wins = windows[:n_train]

    val_start = n_train + split_gap
    val_end = val_start + n_val
    val_wins = windows[val_start:val_end] if val_start < n else []

    test_start_idx = val_end + split_gap
    if test_start_idx >= n:
        test_start_idx = None

    return train_wins, val_wins, test_start_idx


def _split_simulations(n_sims, train_ratio, val_ratio):
    """Whole-sim split. Returns (train_idx, val_idx, test_idx) as index lists."""
    n_train = int(n_sims * train_ratio)
    n_val = int(n_sims * val_ratio)
    if n_train + n_val >= n_sims and n_sims >= 3:
        n_val = max(1, n_val)
        n_train = n_sims - n_val - 1
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n_sims))
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def _open_split_file(output_dir: Path, split: str, file_idx: int) -> h5py.File:
    path = output_dir / split / f"{split}_data_{file_idx:03d}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(path, "w")


def _write_window(h5file, window_idx, data, attrs):
    grp = h5file.create_group(f"window_{window_idx:06d}")
    for key, arr in data.items():
        grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
    for k, v in attrs.items():
        grp.attrs[k] = v


def _ensure_sim_metadata(
    h5file: h5py.File,
    sim_id: int,
    sim_masks: dict[str, np.ndarray],
):
    """
    Ensure /sim_metadata/<sim_id>/ exists in `h5file` and contains the
    per-sim mask arrays. Idempotent — safe to call before every window write.

    Stored once per output file per sim, regardless of how many windows the
    sim contributes.
    """
    if not sim_masks:
        return
    grp_path = f"sim_metadata/{sim_id}"
    if grp_path in h5file:
        return
    g = h5file.create_group(grp_path)
    for key, arr in sim_masks.items():
        g.create_dataset(key, data=arr)


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Cast to float32 for storage; keeps physical units."""
    return arr.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_dataset(
    input_dir: str,
    output_dir: str,
    # --- HIGH priority ---
    context_length: int = 5,
    prediction_horizon: int = 1,
    frame_skip: int = 1,
    dt: float | None = None,
    # --- MEDIUM priority ---
    split_mode: str = "temporal",          # {"temporal", "by_simulation"}
    split_gap: int = 0,                    # in *windows*, only used when split_mode == "temporal"
    max_frames_per_sim: int | None = None, # truncate raw frames per sim
    # --- existing knobs ---
    stride: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    windows_per_file: int = 500,
    require_masks: bool = True,
):
    # ---------------- Validation ----------------
    if split_mode not in ("temporal", "by_simulation"):
        sys.exit(f"Unknown split_mode: {split_mode!r}")
    if frame_skip < 1:
        sys.exit("frame_skip must be >= 1")
    if prediction_horizon < 1:
        sys.exit("prediction_horizon must be >= 1")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        sys.exit(f"No .h5 files found in {input_dir}")

    print(f"Found {len(h5_files)} simulation file(s):")
    for f in h5_files:
        print(f"  {f.name}")

    # Resolve dt: user override > auto-detect from /states/times
    print("\nResolving timestep...")
    dt = _resolve_dt(h5_files, dt)

    window_size = context_length + prediction_horizon
    effective_dt = dt * frame_skip if dt is not None else None

    print(f"\nConfig:")
    print(f"  context_length     = {context_length}")
    print(f"  prediction_horizon = {prediction_horizon}")
    print(f"  window_size        = {window_size}")
    print(f"  stride             = {stride}")
    print(f"  frame_skip         = {frame_skip}  (raw dt={dt}, effective dt={effective_dt})")
    print(f"  split_mode         = {split_mode}")
    if split_mode == "temporal":
        print(f"  split_gap          = {split_gap} windows")
    print(f"  max_frames_per_sim = {max_frames_per_sim}")
    print(f"  output: RAW (un-normalised) — stats stored in metadata for downstream use")

    # -----------------------------------------------------------------------
    # Pre-pass — load per-sim masks (small, do it once)
    # -----------------------------------------------------------------------
    print("\nLoading per-sim barrier / frontface masks...")
    all_sim_masks: dict[int, dict[str, np.ndarray]] = {}
    for sim_idx, h5_path in enumerate(h5_files):
        masks = _read_sim_masks(h5_path)
        if not masks:
            msg = (f"  {h5_path.name}: no /metadata/barrier_idx or "
                   f"/metadata/frontface_idx found")
            if require_masks:
                sys.exit(msg + "  (set require_masks=False to allow)")
            print(msg + "  (skipping mask propagation for this sim)")
        else:
            print(f"  {h5_path.name}: barrier={masks.get('barrier_idx', np.array([])).size}, "
                  f"frontface={masks.get('frontface_idx', np.array([])).size}")
        all_sim_masks[sim_idx] = masks

    # -----------------------------------------------------------------------
    # PASS 1 — Compute stats from TRAINING data only (for metadata)
    # -----------------------------------------------------------------------
    print("\n[Pass 1] Scanning training windows to compute statistics...")

    all_windows: dict[str, list[np.ndarray]] = {f[1]: [] for f in FIELDS}

    if split_mode == "by_simulation":
        train_sims, val_sims, test_sims = _split_simulations(
            len(h5_files), train_ratio, val_ratio
        )
        sim_split = {}
        for i in train_sims: sim_split[i] = "train"
        for i in val_sims:   sim_split[i] = "valid"
        for i in test_sims:  sim_split[i] = "test"
        print(f"  by_simulation split: "
              f"{len(train_sims)} train, {len(val_sims)} val, {len(test_sims)} test sims")
    else:
        sim_split = None

    for sim_idx, h5_path in enumerate(h5_files):
        with h5py.File(h5_path, "r") as f:
            pos = _preprocess_time_axis(
                f["states/positions"][:], max_frames_per_sim, frame_skip
            )
            T_eff = pos.shape[0]

            if split_mode == "temporal":
                windows = _sliding_windows(T_eff, window_size, stride)
                n_train = int(len(windows) * train_ratio)
                train_wins = windows[:n_train]
            else:  # by_simulation
                if sim_split[sim_idx] != "train":
                    continue
                windows = _sliding_windows(T_eff, window_size, stride)
                train_wins = windows

            if not train_wins:
                continue

            for raw_key, out_key, _ in FIELDS:
                arr = _preprocess_time_axis(
                    f[raw_key][:], max_frames_per_sim, frame_skip
                )
                for s, e in train_wins:
                    all_windows[out_key].append(arr[s:e])

    norm_stats, disp_acc_stats = _compute_all_stats(all_windows)
    del all_windows

    print("  Per-field statistics (computed on training data, NOT applied to output):")
    for field, v in norm_stats.items():
        mean_fmt = [f"{x:.4e}" for x in v["mean"]]
        std_fmt  = [f"{x:.4e}" for x in v["std"]]
        print(f"    {field:14s}  mean={mean_fmt}  std={std_fmt}")

    # -----------------------------------------------------------------------
    # PASS 2 — Write raw windows
    # -----------------------------------------------------------------------
    print("\n[Pass 2] Writing raw dataset...")

    split_names = ("train", "valid", "test")
    h5_handles   = {s: None for s in split_names}
    file_indices = {s: 0    for s in split_names}
    win_in_file  = {s: 0    for s in split_names}
    global_idx   = {s: 0    for s in split_names}
    test_traj_lens: list[int] = []

    def get_handle(split: str) -> h5py.File:
        if h5_handles[split] is None or win_in_file[split] >= windows_per_file:
            if h5_handles[split] is not None:
                h5_handles[split].close()
            h5_handles[split] = _open_split_file(output_dir, split, file_indices[split])
            file_indices[split] += 1
            win_in_file[split] = 0
        return h5_handles[split]

    for sim_idx, h5_path in enumerate(h5_files):
        print(f"\n--- Processing {h5_path.name} (sim {sim_idx}) ---")

        with h5py.File(h5_path, "r") as f:
            raw = {
                out_key: _preprocess_time_axis(
                    f[raw_key][:], max_frames_per_sim, frame_skip
                )
                for raw_key, out_key, _ in FIELDS
            }

        T_eff, N, _ = raw["positions"].shape
        print(f"  Frames (effective): {T_eff}, Nodes: {N}")

        windows = _sliding_windows(T_eff, window_size, stride)
        if not windows:
            print(f"  Skipping: not enough frames for one window of size {window_size}")
            continue

        sim_masks = all_sim_masks.get(sim_idx, {})

        base_attrs = {
            "sim_id": sim_idx,
            "num_particles": N,
            "batch_id": sim_idx,
            "normalised": False,
            "frame_skip": frame_skip,
        }

        if split_mode == "temporal":
            train_wins, val_wins, test_start_idx = _split_windows_temporal(
                windows, train_ratio, val_ratio, split_gap
            )

            if test_start_idx is not None:
                test_start = windows[test_start_idx][0]
            else:
                test_start = max(0, T_eff - window_size)
            test_start = max(0, test_start - context_length)
            test_end = T_eff

            print(f"  Windows: {len(windows)} → "
                  f"{len(train_wins)} train / {len(val_wins)} val "
                  f"(gap={split_gap})")
            print(f"  Test trajectory: frames {test_start}..{test_end-1} "
                  f"({test_end - test_start} frames)")

            # train
            for s, e in train_wins:
                h5 = get_handle("train")
                _ensure_sim_metadata(h5, sim_idx, sim_masks)
                _write_window(
                    h5, global_idx["train"],
                    {k: _to_float32(raw[k][s:e]) for k in raw},
                    {**base_attrs, "window_start_frame": s},
                )
                global_idx["train"] += 1
                win_in_file["train"] += 1

            # val
            for s, e in val_wins:
                h5 = get_handle("valid")
                _ensure_sim_metadata(h5, sim_idx, sim_masks)
                _write_window(
                    h5, global_idx["valid"],
                    {k: _to_float32(raw[k][s:e]) for k in raw},
                    {**base_attrs, "window_start_frame": s},
                )
                global_idx["valid"] += 1
                win_in_file["valid"] += 1

            # test trajectory (one per sim)
            if test_end - test_start >= window_size:
                h5 = get_handle("test")
                _ensure_sim_metadata(h5, sim_idx, sim_masks)
                _write_window(
                    h5, global_idx["test"],
                    {k: _to_float32(raw[k][test_start:test_end]) for k in raw},
                    {**base_attrs,
                     "window_start_frame": test_start,
                     "window_name": h5_path.stem},
                )
                global_idx["test"] += 1
                win_in_file["test"] += 1
                test_traj_lens.append(test_end - test_start)

        else:  # split_mode == "by_simulation"
            assignment = sim_split[sim_idx]
            print(f"  Assigned to: {assignment}")

            if assignment in ("train", "valid"):
                for s, e in windows:
                    h5 = get_handle(assignment)
                    _ensure_sim_metadata(h5, sim_idx, sim_masks)
                    _write_window(
                        h5, global_idx[assignment],
                        {k: _to_float32(raw[k][s:e]) for k in raw},
                        {**base_attrs, "window_start_frame": s},
                    )
                    global_idx[assignment] += 1
                    win_in_file[assignment] += 1
                print(f"  Wrote {len(windows)} {assignment} windows")
            else:  # test
                h5 = get_handle("test")
                _ensure_sim_metadata(h5, sim_idx, sim_masks)
                _write_window(
                    h5, global_idx["test"],
                    {k: _to_float32(raw[k][:T_eff]) for k in raw},
                    {**base_attrs,
                     "window_start_frame": 0,
                     "window_name": h5_path.stem},
                )
                global_idx["test"] += 1
                win_in_file["test"] += 1
                test_traj_lens.append(T_eff)
                print(f"  Wrote 1 test trajectory ({T_eff} frames)")

    for h in h5_handles.values():
        if h is not None:
            h.close()

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------
    test_traj_len = test_traj_lens[0] if test_traj_lens else window_size

    metadata = {
        "config": {
            "window_length":       test_traj_len,
            "context_length":      context_length,
            "prediction_horizon":  prediction_horizon,
            "window_size":         window_size,
            "stride":              stride,
            "frame_skip":          frame_skip,
            "dt":                  dt,
            "effective_dt":        effective_dt,
            "max_frames_per_sim":  max_frames_per_sim,
            "split_mode":          split_mode,
            "split_gap":           split_gap,
            "train_ratio":         train_ratio,
            "val_ratio":           val_ratio,
            "test_ratio":          round(1.0 - train_ratio - val_ratio, 4),
            "normalised":          False,
        },
        # Per-field mean/std computed from training data — apply downstream as:
        #   x_norm = (x_raw - mean) / std
        "normalization_stats": norm_stats,
        # Displacement / acceleration stats for MultiScaleSimulator._time_diff()
        "global_stats": disp_acc_stats,
        "num_particle_types": 1,
        "dataset_stats": {
            "num_simulations":   len(h5_files),
            "train_windows":     global_idx["train"],
            "val_windows":       global_idx["valid"],
            "test_trajectories": global_idx["test"],
            "test_traj_lengths": test_traj_lens,
        },
        # Per-sim mask sizes for quick sanity checking from metadata alone.
        "sim_mask_info": {
            str(sim_idx): {
                key: int(arr.size) for key, arr in masks.items()
            }
            for sim_idx, masks in all_sim_masks.items()
            if masks
        },
    }

    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    for meta_path in [output_dir / "metadata.json",
                      output_dir / "metadata" / "metadata.json"]:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {meta_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Dataset built successfully (RAW values, normalisation deferred)")
    print(f"  Output:        {output_dir}")
    print(f"  Train windows: {global_idx['train']}")
    print(f"  Val windows:   {global_idx['valid']}")
    print(f"  Test trajs:    {global_idx['test']}")
    if test_traj_lens:
        print(f"  Test traj lens: min={min(test_traj_lens)}, "
              f"max={max(test_traj_lens)}, mean={np.mean(test_traj_lens):.1f}")
    print(f"  Effective dt:  {effective_dt}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build BVC dataset from raw HDF5 files (raw values out; stats in metadata)"
    )
    parser.add_argument("--input_dir",  default="/home/kong/datasets/barrier/h5")
    parser.add_argument("--output_dir", default="./dataset/data_processed")

    # ----- HIGH priority -----
    parser.add_argument("--context_length",     type=int,   default=5,
                        help="Number of past frames fed to the model")
    parser.add_argument("--prediction_horizon", type=int,   default=1,
                        help="Number of future frames per training target "
                             "(window_size = context_length + prediction_horizon)")
    parser.add_argument("--frame_skip",         type=int,   default=1,
                        help="Subsample every k-th frame from raw simulation. "
                             "Effective dt = raw dt * frame_skip.")
    parser.add_argument("--dt",                 type=float, default=None,
                        help="Raw simulation timestep in seconds. "
                             "If omitted, auto-detected from /states/times in the "
                             "first input h5. Stored in metadata.")

    # ----- MEDIUM priority -----
    parser.add_argument("--split_mode", choices=("temporal", "by_simulation"),
                        default="temporal",
                        help="'temporal': split frames within each sim. "
                             "'by_simulation': whole sims go to one split.")
    parser.add_argument("--split_gap", type=int, default=0,
                        help="Number of windows skipped between train/val and "
                             "val/test in temporal mode (prevents leakage).")
    parser.add_argument("--max_frames_per_sim", type=int, default=None,
                        help="Truncate each simulation to at most N raw frames "
                             "(applied before frame_skip). Useful for debugging.")

    # ----- existing knobs -----
    parser.add_argument("--stride",           type=int,   default=1)
    parser.add_argument("--train_ratio",      type=float, default=0.8)
    parser.add_argument("--val_ratio",        type=float, default=0.1)
    parser.add_argument("--windows_per_file", type=int,   default=500)
    parser.add_argument("--allow_missing_masks", action="store_true",
                        help="Don't fail if /metadata/barrier_idx or "
                             "/metadata/frontface_idx is missing in an input file.")

    args = parser.parse_args()

    build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
        frame_skip=args.frame_skip,
        dt=args.dt,
        split_mode=args.split_mode,
        split_gap=args.split_gap,
        max_frames_per_sim=args.max_frames_per_sim,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        windows_per_file=args.windows_per_file,
        require_masks=not args.allow_missing_masks,
    )


if __name__ == "__main__":
    main()