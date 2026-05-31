from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.utils.data

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")


# ── Normalization stats (legacy — used by rollout / evaluate) ─────────────────
class NormStats:
    FEATURES = ["positions", "velocity", "acceleration", "stress"]

    def __init__(
        self,
        metadata_path: str | Path,
        acc_scale: Optional[float] = None,
    ):
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"metadata.json not found: {path}")

        with open(path) as f:
            meta = json.load(f)

        raw = meta.get("field_stats", {})
        self._mean: Dict[str, np.ndarray] = {}
        self._std:  Dict[str, np.ndarray] = {}

        for feat in self.FEATURES:
            if feat not in raw:
                raise KeyError(f"Feature '{feat}' missing from normalization_stats in {path}")
            self._mean[feat] = np.array(raw[feat]["mean"], dtype=np.float32)
            self._std[feat]  = np.array(raw[feat]["std"],  dtype=np.float32)

            zero_mask = self._std[feat] < 1e-8
            if zero_mask.any():
                print(f"Warning: near-zero std in '{feat}' dims {np.where(zero_mask)[0]} "
                      f"— clamped to 1.0")
                self._std[feat][zero_mask] = 1.0

        self._acc_scale: Optional[float] = float(acc_scale) if acc_scale else None
        if self._acc_scale is not None:
            print(f"[NormStats] acceleration uses asinh transform, scale={self._acc_scale:.2e}")
        else:
            print("[NormStats] acceleration uses z-score (no acc_scale provided)")

    def normalize(self, feature: str, arr: np.ndarray) -> np.ndarray:
        return ((arr - self._mean[feature]) / self._std[feature]).astype(np.float32)

    def denormalize(self, feature: str, arr: np.ndarray) -> np.ndarray:
        return (arr * self._std[feature] + self._mean[feature]).astype(np.float32)

    def denormalize_tensor(self, feature: str, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self._mean[feature], dtype=t.dtype, device=t.device)
        std  = torch.tensor(self._std[feature],  dtype=t.dtype, device=t.device)
        return t * std + mean

def load_norm_stats(metadata_path: str | Path) -> Optional[NormStats]:
    try:
        return NormStats(metadata_path)
    except FileNotFoundError:
        return None


# ── Global stats (multi-traj) ─────────────────────────────────────────────────

# Default fields match h5 states/ group keys and metadata.json field_stats keys.
_DEFAULT_NORM_FIELDS = ["positions", "velocity", "acceleration"]


def compute_global_stats(traj_h5_paths: list[str], fields: list[str]) -> dict:
    """One-pass Welford mean/std over (frames × nodes × dims) per field.

    Args:
        traj_h5_paths: list of output.h5 paths
        fields: list of field names matching states/ group keys
                e.g. ["positions", "velocity", "acceleration"]

    Returns:
        {field: {"mean": float, "std": float}}
    """
    accum = {f: {"n": 0, "mean": 0.0, "M2": 0.0} for f in fields}
    for p in traj_h5_paths:
        with h5py.File(p, "r") as fh:
            for field in fields:
                h5_key = f"states/{field}"
                if h5_key not in fh:
                    raise KeyError(f"{p}: missing dataset '{h5_key}'")
                x = np.asarray(fh[h5_key][...]).reshape(-1).astype(np.float64)
                n_b = x.size
                if n_b == 0:
                    continue
                mean_b = float(x.mean())
                var_b  = float(x.var())
                s      = accum[field]
                n_a    = s["n"]
                delta  = mean_b - s["mean"]
                n_new  = n_a + n_b
                s["mean"] = (n_a * s["mean"] + n_b * mean_b) / n_new
                s["M2"]  += var_b * n_b + delta ** 2 * n_a * n_b / n_new
                s["n"]    = n_new
    return {
        f: {"mean": s["mean"], "std": (s["M2"] / max(s["n"], 1)) ** 0.5}
        for f, s in accum.items()
    }


def load_or_compute_global_stats(
    train_dirs: list[str],
    cache_path: Path,
    fields: list[str],
) -> dict:
    """Load cached global stats or recompute from train dirs.

    Cache key = sorted resolved train_dirs + sorted fields.
    Recomputes when either changes.
    """
    key = sorted(str(Path(d).resolve()) for d in train_dirs)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("key") == key and cached.get("fields") == sorted(fields):
            print(f"[GlobalStats] Loaded cached stats from {cache_path}")
            return cached["stats"]

    print(f"[GlobalStats] Computing global stats over {len(train_dirs)} train traj(s) ...")
    h5_paths = [_resolve_traj_dir(d)["h5"] for d in train_dirs]
    stats    = compute_global_stats(h5_paths, fields)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"key": key, "fields": sorted(fields), "stats": stats}, indent=2
    ))
    print(f"[GlobalStats] Stats saved to {cache_path}")
    for field, s in stats.items():
        print(f"  {field}: mean={s['mean']:.6f}  std={s['std']:.6f}")
    return stats


# ── Trajectory dir resolver ───────────────────────────────────────────────────

def _resolve_traj_dir(d: str | Path) -> dict:
    """Resolve a trajectory dir to {h5, metadata} paths. Fails loudly."""
    d = Path(d)
    if not d.is_dir():
        raise FileNotFoundError(f"Trajectory dir not found: {d}")
    h5, meta = d / "output.h5", d / "metadata.json"
    missing = [p.name for p in (h5, meta) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{d} missing required files: {missing}")
    return {"h5": str(h5), "metadata": str(meta)}


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int):
        pass


# ── Helpers for collision-feature plumbing ────────────────────────────────────

def _load_sim_masks(h5_path: Path) -> Dict[int, np.ndarray]:
    """Read /sim_metadata/<sim_id>/barrier_idx for every sim in the file."""
    masks: Dict[int, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        if "sim_metadata" not in f:
            return masks
        for sid in f["sim_metadata"]:
            ds_path = f"sim_metadata/{sid}/barrier_idx"
            if ds_path in f:
                masks[int(sid)] = f[ds_path][:].astype(np.int64)
    return masks


def _normalize_dist_feature(
    coll_feat: np.ndarray,
    dist_mean: float,
    dist_std: float,
) -> np.ndarray:
    """Z-score the distance channel only; leave the collision flag untouched."""
    out = coll_feat.copy()
    out[..., 0] = (out[..., 0] - dist_mean) / max(dist_std, 1e-8)
    return out


# ── BVC Trajectory-Sliced Training Dataset ────────────────────────────────────

class BVCSlicedDataset(BaseDataset):
    """Full-trajectory dataset with sliced-window sampling for push-forward training.

    Supports multiple trajectories. Sliding windows never cross traj boundaries.
    Each __getitem__ returns a window starting at index t:
        - frames [t, t + input_frames):                    velocity input (normalized)
        - frames [t + input_frames, t + input_frames + K): K acceleration targets (normalized)
        - frames [t, t + input_frames):                    raw positions for SDF
        - frames [t + input_frames, t + input_frames + K): raw positions for push-forward SDF
        - frame  t + input_frames - 1:                     raw velocity (for integration)

    Args:
        cfg requires:
            - cfg["data"]["paths"]:          list[str], h5 file paths
            - cfg["data"]["metadata_paths"]: list[str], metadata.json paths
            - cfg["data"]["input_frames"]:   int, default 5
            - cfg["data"]["normalize"]:      bool, default True
            - cfg["train"]["push_forward_k"]: int, default 1
        stats: global normalization stats dict {field: {"mean": float, "std": float}}.
               If None and normalize=True, falls back to first traj's metadata.json.
               CRITICAL: val dataset must receive train stats, not its own metadata stats.
    """

    INPUT_KEY  = "states/velocity"
    POS_KEY    = "states/positions"
    TARGET_KEY = "states/acceleration"

    def __init__(self, cfg: dict, *, stats: dict | None = None):
        super().__init__(cfg)
        data_cfg  = cfg["data"]
        train_cfg = cfg.get("train", {})

        # ── Resolve paths ──────────────────────────────────────────────────
        paths = data_cfg.get("paths") or data_cfg.get("path")
        if paths is None:
            raise ValueError("BVCSlicedDataset requires cfg['data']['paths']")
        if isinstance(paths, str):
            paths = [paths]
        self.h5_paths = [Path(p) for p in paths]
        for p in self.h5_paths:
            if not p.exists():
                raise FileNotFoundError(f"H5 file not found: {p}")

        metadata_paths = data_cfg.get("metadata_paths", [])
        if isinstance(metadata_paths, str):
            metadata_paths = [metadata_paths]

        # ── Config ────────────────────────────────────────────────────────
        self.input_frames = int(data_cfg.get("input_frames", 5))
        self.K = int(train_cfg.get("push_forward_k", 1))
        if self.K < 1:
            raise ValueError(f"push_forward_k must be >= 1, got {self.K}")
        self.window_len = self.input_frames + self.K

        # ── Normalization ──────────────────────────────────────────────────
        # Priority: external stats dict > first-traj metadata fallback > no-norm
        self._stats_dict: dict | None = None          # global scalar stats
        self._norm_stats: NormStats | None = None     # legacy per-dim stats (fallback)

        if data_cfg.get("normalize", True):
            if stats is not None:
                self._stats_dict = stats
                print(f"[BVCSlicedDataset] Using external global stats "
                      f"(fields: {list(stats.keys())})")
            elif metadata_paths:
                self._norm_stats = NormStats(
                    metadata_paths[0],
                    acc_scale=data_cfg.get("acc_scale"),
                )
                print(f"[BVCSlicedDataset] Fallback: norm stats from {metadata_paths[0]}")
            else:
                print("[BVCSlicedDataset] Warning: normalize=True but no stats available")

        # ── Load trajectories into memory ──────────────────────────────────
        self._trajectories: list[dict] = []
        per_traj_windows: list[int] = []

        for idx, p in enumerate(self.h5_paths):
            with h5py.File(p, "r") as f:
                vel  = f[self.INPUT_KEY][:].astype(np.float32)    # (T, N, 3)
                acc  = f[self.TARGET_KEY][:].astype(np.float32)   # (T, N, 3)
                pos  = f[self.POS_KEY][:].astype(np.float32)      # (T, N, 3)

            T = vel.shape[0]
            if T < self.window_len:
                raise ValueError(
                    f"Trajectory {p} has only {T} frames, need >= {self.window_len} "
                    f"(input_frames={self.input_frames} + K={self.K})"
                )

            vel_norm = self._normalize("velocity",     vel)
            acc_norm = self._normalize("acceleration", acc)

            vel_derived     = np.diff(pos, axis=0)          # (T-1, N, 3)
            acc_derived_raw = np.diff(vel_derived, axis=0)  # (T-2, N, 3)
            vel_derived_norm = self._normalize("velocity",     vel_derived)
            acc_derived_norm = self._normalize("acceleration", acc_derived_raw)

            n_windows = max(0, T - self.window_len)
            per_traj_windows.append(n_windows)

            dir_name = p.parent.name
            meta_label = ""
            if idx < len(metadata_paths):
                try:
                    with open(metadata_paths[idx]) as mf:
                        mc = json.load(mf).get("config", {})
                    speed = mc.get("source", "")
                    meta_label = f"  [{speed.split('/')[-1]}]" if speed else ""
                except Exception:
                    pass
            print(f"[BVCSlicedDataset] traj[{idx}] {dir_name}{meta_label} "
                  f"— T={T}, windows={n_windows}")

            self._trajectories.append({
                "vel_norm":          vel_norm,
                "vel_phys":          vel,
                "acc_norm":          acc_norm,
                "pos_phys":          pos,
                "T":                 T,
                "path":              str(p),
                "vel_derived_norm":  vel_derived_norm,
                "vel_derived_phys":  vel_derived,
                "acc_derived_norm":  acc_derived_norm,
                "T_derived":         T - 2,
            })

        # ── Build global (traj_idx, start_idx) index ──────────────────────
        # Windows are built per-traj; they never cross traj boundaries.
        self._index_map: list[tuple[int, int]] = []
        for ti, n_windows in enumerate(per_traj_windows):
            for si in range(n_windows):
                self._index_map.append((ti, si))

        n_trajs = len(self._trajectories)
        total_w = len(self._index_map)
        print(f"[BVCSlicedDataset] {n_trajs} traj(s) | {total_w} total windows "
              f"| per-traj: {per_traj_windows}")
        print(f"[BVCSlicedDataset] input_frames={self.input_frames}, K={self.K}, "
              f"window_len={self.window_len}")

    def _normalize(self, field: str, arr: np.ndarray) -> np.ndarray:
        """Apply normalization using whichever stats source is active."""
        if self._stats_dict is not None:
            s = self._stats_dict.get(field)
            if s is None:
                return arr.astype(np.float32)
            return ((arr - s["mean"]) / max(s["std"], 1e-8)).astype(np.float32)
        if self._norm_stats is not None:
            return self._norm_stats.normalize(field, arr)
        return arr.astype(np.float32)

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int):
        """Derived-kinematics version: vel = pos[t+1]-pos[t], acc = vel[t+1]-vel[t].

        Forward-difference convention (dt = 1 frame). Trainer integrates as:
            x_{i+1} = x_i + v_i
            v_{i+1} = v_i + a_i
        with x_0 = input_pos[:, -1, :] and v_0 = v_last_phys.
        """
        traj_idx, start = self._index_map[idx]
        traj = self._trajectories[traj_idx]
        T_in = self.input_frames
        K    = self.K
        end  = start + T_in + K

        # Input: derived velocity (normalized, flattened)
        vel_in = traj["vel_derived_norm"][start : start + T_in]    # (T_in, N, 3)
        N = vel_in.shape[1]
        x_vel = vel_in.transpose(1, 0, 2).reshape(N, -1)           # (N, T_in*3)

        # Last input frame physical velocity for integration
        v_last_phys = traj["vel_derived_phys"][start + T_in - 1]   # (N, 3)

        # Target: derived acceleration (K frames, normalized)
        acc_start  = start + T_in - 1
        acc_future = traj["acc_derived_norm"][acc_start : acc_start + K]  # (K, N, 3)
        future_acc = acc_future.transpose(1, 0, 2)                 # (N, K, 3)

        # Positions (raw, for SDF)
        input_pos  = traj["pos_phys"][start : start + T_in]        # (T_in, N, 3)
        future_pos = traj["pos_phys"][start + T_in : end]          # (K, N, 3)
        future_pos = future_pos.transpose(1, 0, 2)                 # (N, K, 3)
        input_pos  = input_pos.transpose(1, 0, 2)                  # (N, T_in, 3)

        return (
            torch.from_numpy(np.ascontiguousarray(x_vel)),         # (N, T_in*3)
            torch.from_numpy(np.ascontiguousarray(future_acc)),    # (N, K, 3)
            torch.from_numpy(np.ascontiguousarray(input_pos)),     # (N, T_in, 3)
            torch.from_numpy(np.ascontiguousarray(future_pos)),    # (N, K, 3)
            torch.from_numpy(np.ascontiguousarray(v_last_phys)),   # (N, 3)
        )


# ── DataLoader factory ────────────────────────────────────────────────────────

_DATASET_MAP = {
    "bvc_sliced": BVCSlicedDataset,
}


def build_dataloader(
    cfg: dict,
    dirs: list[str] | str,
    *,
    shuffle: bool,
    batch_size: int,
    stats: dict | None = None,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from a list of trajectory dirs.

    Args:
        cfg:        full experiment config dict
        dirs:       one or more trajectory dirs (each must contain output.h5 + metadata.json)
        shuffle:    whether to shuffle windows across all trajs
        batch_size: batch size
        stats:      global normalization stats from load_or_compute_global_stats.
                    MUST be train stats even for val loader.
    """
    data_cfg = cfg.get("data", {})

    dataset_type = data_cfg.get("dataset_type", "bvc_sliced")
    dataset_cls  = _DATASET_MAP.get(dataset_type)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")

    if isinstance(dirs, str):
        dirs = [dirs]
    if not dirs:
        raise ValueError("build_dataloader needs at least one trajectory dir")

    trajectories = [_resolve_traj_dir(d) for d in dirs]
    ds_cfg = {**cfg, "data": {
        **data_cfg,
        "paths":          [t["h5"]       for t in trajectories],
        "metadata_paths": [t["metadata"] for t in trajectories],
    }}

    dataset = dataset_cls(ds_cfg, stats=stats)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = data_cfg.get("num_workers", 0),
        pin_memory  = data_cfg.get("pin_memory", True),
    )
