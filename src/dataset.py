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



# ── Normalization stats ───────────────────────────────────────────────────────
class NormStats:
    FEATURES = ["positions", "velocity", "acceleration", "stress"]

    def __init__(
        self,
        metadata_path: str | Path,
        acc_scale: Optional[float] = None,   # ← 新增
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

        # ── asinh transform for acceleration (opt-in) ──
        self._acc_scale: Optional[float] = float(acc_scale) if acc_scale else None
        if self._acc_scale is not None:
            print(f"[NormStats] acceleration uses asinh transform, scale={self._acc_scale:.2e}")
        else:
            print("[NormStats] acceleration uses z-score (no acc_scale provided)")

    def normalize(self, feature: str, arr: np.ndarray) -> np.ndarray:
        # if feature == "acceleration" and self._acc_scale is not None:
        #     return np.arcsinh(arr / self._acc_scale).astype(np.float32)
        return ((arr - self._mean[feature]) / self._std[feature]).astype(np.float32)

    def denormalize(self, feature: str, arr: np.ndarray) -> np.ndarray:
        # if feature == "acceleration" and self._acc_scale is not None:
        #     return (self._acc_scale * np.sinh(arr)).astype(np.float32)
        return (arr * self._std[feature] + self._mean[feature]).astype(np.float32)

    def denormalize_tensor(self, feature: str, t: torch.Tensor) -> torch.Tensor:
        if feature == "acceleration" and self._acc_scale is not None:
            return self._acc_scale * torch.sinh(t)
        mean = torch.tensor(self._mean[feature], dtype=t.dtype, device=t.device)
        std  = torch.tensor(self._std[feature],  dtype=t.dtype, device=t.device)
        return t * std + mean

def load_norm_stats(metadata_path: str | Path) -> Optional[NormStats]:
    try:
        return NormStats(metadata_path)
    except FileNotFoundError:
        return None


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
    """Read /sim_metadata/<sim_id>/barrier_idx for every sim in the file.

    Returns {sim_id: barrier_idx_array}. Empty dict if no sim_metadata group.
    """
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
    """Z-score the distance channel only; leave the collision flag untouched.

    coll_feat shape: (..., 2) where [..., 0]=dist, [..., 1]=flag
    """
    out = coll_feat.copy()
    out[..., 0] = (out[..., 0] - dist_mean) / max(dist_std, 1e-8)
    return out


# ── BVC Training Dataset ──────────────────────────────────────────────────────
class BVCDataset(BaseDataset):
    """HDF5 dataset for TransolverNet training (velocity + collision → acceleration).

    Each window has 6 frames (T=6). Splits into:
        - x (input):  velocity + collision features from first 5 frames
                      → (N, 5 * (3 + 2)) = (N, 25)
        - y (target): acceleration at 6th frame                → (N, 3)

    Per-frame collision features (per node):
        [dist_to_nearest_barrier, is_collision_flag (0/1)]

    Input layout per node (15 dims):
        [vx_t0, vy_t0, vz_t0,
         vx_t1, vy_t1, vz_t1, 
         ...,
         vx_t4, vy_t4, vz_t4]

    Args:
        cfg: Must have:
            - cfg["data"]["path"]
            - cfg["data"]["metadata_path"]
            - cfg["data"]["normalize"]            (default True)
    """

    INPUT_FEATURE  = "velocity"
    SDF_FEATURE = "positions"  # positions needed for sdf

    TARGET_FEATURE = "acceleration"
    INPUT_FRAMES   = 10
    TARGET_FRAME   = 10  # 11th frame, 0-indexed

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        data_cfg = cfg["data"]

        self.h5_path = Path(data_cfg["path"])
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        # ── Normalization stats ──
        self._stats: Optional[NormStats] = None
        if data_cfg.get("normalize", True):
            meta_path = data_cfg.get("metadata_path")
            if meta_path:
                self._stats = NormStats(
                    meta_path,
                    acc_scale=data_cfg.get("acc_scale"),
                )
                print(f"[Dataset] Normalization enabled — stats loaded from {meta_path}")
            else:
                print("Warning: normalize=True but metadata_path not set — skipping normalization")

        # ── Collision feature config ──
        self._dist_mean: Optional[float] = data_cfg.get("dist_mean")
        self._dist_std:  Optional[float] = data_cfg.get("dist_std")

        # ── Build window key index ──
        with h5py.File(self.h5_path, "r") as f:
            self._keys = sorted(k for k in f.keys() if k.startswith("window_"))
        if not self._keys:
            raise ValueError(f"No window groups found in {self.h5_path}")

        # ── Retrival of barrier_idx from source data 
        
        # File handle — opened lazily per worker
        self._file: Optional[h5py.File] = None

    def __len__(self) -> int:
        return len(self._keys)

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def _resolve_barrier_idx(self, grp) -> np.ndarray:
        """Look up the barrier idx for this window's sim."""
        sim_id = int(grp.attrs["sim_id"])
        if sim_id not in self._barrier_idx_per_sim:
            raise KeyError(
                f"sim_id={sim_id} from window attrs has no entry in "
                f"sim_metadata. Available: {list(self._barrier_idx_per_sim)}"
            )
        return self._barrier_idx_per_sim[sim_id]

    def __getitem__(self, idx):
        f   = self._get_file()
        grp = f[self._keys[idx]]

        # ── Velocity (input) — first 5 frames ──
        vel = grp[self.INPUT_FEATURE][:self.INPUT_FRAMES].astype(np.float32)  # (5, N, 3)
        if self._stats is not None:
            vel_norm = self._stats.normalize(self.INPUT_FEATURE, vel)
        else:
            vel_norm = vel

        # ── Positions for the same 5 frames (for collision feature) ──
        pos = grp[self.SDF_FEATURE][:self.INPUT_FRAMES].astype(np.float32)         # (5, N, 3)
        # Note: use RAW positions (not normalized) because the threshold is
        # in physical units. If you ever change to normalized positions,
        # threshold and dist_mean/std must be in the same space.

        # ── Collision features ──
        # barrier_idx = self._resolve_barrier_idx(grp)
        # coll_feat = compute_collision_features_numpy(
        #     pos, barrier_idx, threshold=self.collision_threshold
        # )  # (5, N, 2)

        # if self.normalize_dist and self._dist_mean is not None and self._dist_std is not None:
        #     coll_feat = _normalize_dist_feature(coll_feat, self._dist_mean, self._dist_std)

        # ── Concatenate per-frame features: [vel(3) | dist(1) | flag(1)] = 5 dims ──
        # Shape: (5, N, 5)
        per_frame = np.concatenate([vel_norm], axis=-1)

        # (T, N, C) → (N, T, C) → (N, T*C)
        N = per_frame.shape[1]
        x = per_frame.transpose(1, 0, 2).reshape(N, -1)   # (N, TxC)

        # ── Target: acceleration at 6th frame ──
        acc = grp[self.TARGET_FEATURE][self.TARGET_FRAME].astype(np.float32)  # (N, 3)
        if self._stats is not None:
            acc = self._stats.normalize(self.TARGET_FEATURE, acc)
        y = acc

        # ── Also return the collision flag at target frame for loss weighting ──
        # Compute flag at the target frame (frame 5)
        # pos_target = grp["positions"][self.TARGET_FRAME:self.TARGET_FRAME + 1].astype(np.float32)  # (1, N, 3)
        # coll_target = compute_collision_features_numpy(
        #     pos_target, barrier_idx, threshold=self.collision_threshold
        # )  # (1, N, 2)
        # target_flag = coll_target[0, :, 1]  # (N,)

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
            torch.from_numpy(np.ascontiguousarray(pos))
        )

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ── BVC Full-Trajectory Dataset ───────────────────────────────────────────────

class BVCFullTrajectoryDataset(BaseDataset):
    """Full-trajectory dataset for autoregressive rollout evaluation.

    Returns::

        {
          "positions":    FloatTensor (T, N, 3),
          "velocity":     FloatTensor (T, N, 3),
          "acceleration": FloatTensor (T, N, 3),
          "stress":       FloatTensor (T, N, 6),
          "meta":         {"window_name": str, "window_idx": int, "sim_id": int},
        }

    `collision` is computed from RAW positions; the dist channel is in
    physical units. `positions/velocity/acceleration/stress` are normalized
    if normalize=True (collision feature is NOT normalized — handle that
    downstream if needed).
    """

    FEATURES = ["positions", "velocity", "acceleration", "stress"]

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        data_cfg = cfg["data"]

        self.h5_path = Path(data_cfg["path"])
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        self._stats: Optional[NormStats] = None
        if data_cfg.get("normalize", True):
            meta_path = data_cfg.get("metadata_path")
            if meta_path:
                self._stats = NormStats(
                    meta_path,
                    acc_scale=data_cfg.get("acc_scale"),
                )

        # self.collision_threshold = float(data_cfg.get("collision_threshold", 100.0))

        with h5py.File(self.h5_path, "r") as f:
            self._keys = sorted(k for k in f.keys() if k.startswith("window_"))
        if not self._keys:
            raise ValueError(f"No window groups found in {self.h5_path}")

        self._barrier_idx_per_sim = _load_sim_masks(self.h5_path)
        if not self._barrier_idx_per_sim:
            raise ValueError(
                f"No /sim_metadata/<sim_id>/barrier_idx found in {self.h5_path}."
            )

        self._file: Optional[h5py.File] = None

    def __len__(self) -> int:
        return len(self._keys)

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> Dict[str, object]:
        f     = self._get_file()
        wname = self._keys[idx]
        grp   = f[wname]
        sim_id = int(grp.attrs["sim_id"])
        # barrier_idx = self._barrier_idx_per_sim[sim_id]

        data: Dict[str, object] = {}
        # Need RAW positions to compute collision; keep a reference
        raw_pos = None
        for feat in self.FEATURES:
            arr = grp[feat][:].astype(np.float32)   # (T, N, C)
            if feat == "positions":
                raw_pos = arr.copy()
            if self._stats is not None:
                arr = self._stats.normalize(feat, arr)
            data[feat] = torch.from_numpy(arr)

        # coll = compute_collision_features_numpy(
        #     raw_pos, barrier_idx, threshold=self.collision_threshold
        # )  # (T, N, 2)
        # data["collision"]   = torch.from_numpy(coll)
        # data["barrier_idx"] = torch.from_numpy(barrier_idx.astype(np.int64))
        data["meta"] = {"window_name": wname, "window_idx": idx, "sim_id": sim_id}
        return data

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

# ── BVC Trajectory-Sliced Training Dataset ────────────────────────────────────
class BVCSlicedDataset(BaseDataset):
    """Full-trajectory dataset with sliced-window sampling for push-forward training.
    
    Loads one or more full trajectories into memory at init. Each __getitem__ 
    returns a window starting at index t:
        - frames [t, t + input_frames):                    velocity input (normalized)
        - frames [t + input_frames, t + input_frames + K): K acceleration targets (normalized)
        - frames [t, t + input_frames):                    raw positions for first-step SDF
        - frames [t + input_frames, t + input_frames + K): raw positions for push-forward SDF
        - frame  t + input_frames - 1:                     raw velocity (for integration)
    
    Args:
        cfg requires:
            - cfg["data"]["paths"]: list[str] or single str — h5 file paths
            - cfg["data"]["metadata_path"]: for global normalization stats
            - cfg["data"]["input_frames"]: int, default 5
            - cfg["data"]["normalize"]: bool, default True
            - cfg["train"]["push_forward_k"]: int, default 1
    """
    
    INPUT_KEY  = "states/velocity"
    POS_KEY    = "states/positions"
    TARGET_KEY = "states/acceleration"
    
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        data_cfg  = cfg["data"]
        train_cfg = cfg.get("train", {})
        
        # ── Resolve paths ──
        paths = data_cfg.get("paths") or data_cfg.get("path")
        if paths is None:
            raise ValueError("BVCSlicedDataset requires cfg['data']['paths'] or ['path']")
        if isinstance(paths, str):
            paths = [paths]
        self.h5_paths = [Path(p) for p in paths]
        for p in self.h5_paths:
            if not p.exists():
                raise FileNotFoundError(f"H5 file not found: {p}")
        
        # ── Config ──
        self.input_frames = int(data_cfg.get("input_frames", 5))
        self.K = int(train_cfg.get("push_forward_k", 1))
        if self.K < 1:
            raise ValueError(f"push_forward_k must be >= 1, got {self.K}")
        self.window_len = self.input_frames + self.K
        
        # ── Normalization (always from metadata.json) ──
        self._stats: Optional[NormStats] = None
        if data_cfg.get("normalize", True):
            meta_path = data_cfg.get("metadata_path")
            if not meta_path:
                raise ValueError("normalize=True requires metadata_path in cfg")
            self._stats = NormStats(
                meta_path,
                acc_scale=data_cfg.get("acc_scale"),
            )
            print(f"[BVCSlicedDataset] Normalization stats loaded from {meta_path}")
        
        # ── Load trajectories into memory ──
        # Pre-normalize velocity and acceleration (used directly as input/target).
        # Keep positions raw (SDF computed in physical units downstream).
        # Keep raw velocity too (needed in physical units for integration).
        self._trajectories: list[dict] = []
        for p in self.h5_paths:
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
            
            if self._stats is not None:
                vel_norm = self._stats.normalize("velocity",     vel)
                acc_norm = self._stats.normalize("acceleration", acc)
            else:
                vel_norm = vel
                acc_norm = acc
            
            self._trajectories.append({
                "vel_norm":  vel_norm,   # (T, N, 3) normalized
                "vel_phys":  vel,        # (T, N, 3) physical, for integration
                "acc_norm":  acc_norm,   # (T, N, 3) normalized target
                "pos_phys":  pos,        # (T, N, 3) physical, for SDF
                "T":         T,
                "path":      str(p),
            })
        
        # ── Build (traj_idx, start_idx) index ──
        # Each valid start spans [0, T - window_len], inclusive.
        self._index_map: list[tuple[int, int]] = []
        for ti, traj in enumerate(self._trajectories):
            n_starts = traj["T"] - self.window_len + 1
            for si in range(n_starts):
                self._index_map.append((ti, si))
        
        print(f"[BVCSlicedDataset] {len(self._trajectories)} trajectory(ies), "
              f"{len(self._index_map)} total windows")
        print(f"[BVCSlicedDataset] input_frames={self.input_frames}, K={self.K}, "
              f"window_len={self.window_len}")
    
    def __len__(self) -> int:
        return len(self._index_map)
    
    def __getitem__(self, idx):
        traj_idx, start = self._index_map[idx]
        traj = self._trajectories[traj_idx]
        T_in = self.input_frames
        K    = self.K
        end  = start + T_in + K
        
        # ── Input velocity (normalized, flattened) ──
        vel_in = traj["vel_norm"][start : start + T_in]            # (T_in, N, 3)
        N = vel_in.shape[1]
        x_vel = vel_in.transpose(1, 0, 2).reshape(N, -1)           # (N, T_in*3)
        
        # ── Last input frame physical velocity (for integration) ──
        v_last_phys = traj["vel_phys"][start + T_in - 1]           # (N, 3)
        
        # ── Future acceleration targets (K frames, normalized) ──
        acc_future = traj["acc_norm"][start + T_in : end]          # (K, N, 3)
        future_acc = acc_future.transpose(1, 0, 2)                 # (N, K, 3)
        
        # ── Positions (raw, for SDF) ──
        input_pos  = traj["pos_phys"][start : start + T_in]        # (T_in, N, 3)
        future_pos = traj["pos_phys"][start + T_in : end]          # (K, N, 3)
        future_pos = future_pos.transpose(1, 0, 2)                 # (N, K, 3)


        input_pos = input_pos.transpose(1, 0, 2)                 # (T_in, N, 3)->(N, T_in, 3)
        
        return (
            torch.from_numpy(np.ascontiguousarray(x_vel)),         # (N, T_in*3)
            torch.from_numpy(np.ascontiguousarray(future_acc)),    # (N, K, 3)
            torch.from_numpy(np.ascontiguousarray(input_pos)),     # (N, T_in, 3)
            torch.from_numpy(np.ascontiguousarray(future_pos)),    # (N, K, 3)
            torch.from_numpy(np.ascontiguousarray(v_last_phys)),   # (N, 3)
        )


# ── DataLoader factory ────────────────────────────────────────────────────────

_DATASET_MAP = {
    "bvc":            BVCDataset,
    "bvc_trajectory": BVCFullTrajectoryDataset,
    "bvc_sliced":     BVCSlicedDataset,
}

def build_dataloader(cfg: dict, split: str = "train") -> torch.utils.data.DataLoader:
    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    
    dataset_type = data_cfg.get("dataset_type", "bvc")
    dataset_cls  = _DATASET_MAP.get(dataset_type)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")
    
    if dataset_type == "bvc_sliced":
        # New-style: explicit paths in config; supports per-split override.
        # e.g. data.train_paths / data.val_paths, or single data.paths for both.
        split_paths = data_cfg.get(f"{split}_paths") or data_cfg.get("paths")
        if split_paths is None:
            raise ValueError(
                f"bvc_sliced requires data.{split}_paths or data.paths in cfg"
            )
        split_cfg = {**cfg, "data": {**data_cfg, "paths": split_paths}}
    else:
        # Legacy pre-windowed
        base_path = Path(data_cfg.get("base_path", "dataset/data_processed"))
        h5_path   = base_path / f"{split}" / f"{split}_data_000.h5"
        split_cfg = {**cfg, "data": {**data_cfg, "path": str(h5_path)}}
    
    dataset = dataset_cls(split_cfg)
    
    is_train   = (split == "train")
    batch_size = train_cfg.get("batch_size", 1) if is_train else 1
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = is_train,
        num_workers = data_cfg.get("num_workers", 0),
        pin_memory  = data_cfg.get("pin_memory", True),
    )