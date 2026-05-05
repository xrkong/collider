# 迁移自: src/dataset.py
# 改动内容:
#   - 删除 BVCWindowDataset（数据已预切窗口，滑窗逻辑多余）
#   - 删除 _BaseBVCMixin（normalize_feature 是 no-op，数据已预归一化）
#   - 删除 H5Dataset（项目未使用）
#   - BVCDataset: 修复 h5 文件句柄问题（lazy open，支持 num_workers > 0）
#   - BVCDataset: 添加 z-score normalization，从 metadata.json 读取 stats
#   - BVCFullTrajectoryDataset: 去掉 Mixin，简化为直接读取，同样支持 normalize
#   - build_dataloader: 默认 dataset_type 改为 "bvc"

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
    """Holds per-feature z-score stats loaded from metadata.json.

    Reads ``normalization_stats`` section:
    ::

        {
          "normalization_stats": {
            "positions":    {"mean": [x, y, z],          "std": [x, y, z]},
            "velocity":     {"mean": [x, y, z],          "std": [x, y, z]},
            "acceleration": {"mean": [x, y, z],          "std": [x, y, z]},
            "stress":       {"mean": [s0..s5],            "std": [s0..s5]}
          }
        }

    Args:
        metadata_path: Path to ``metadata.json``.

    Example::

        stats = NormStats("datasets/dataset_v1/metadata.json")
        x_norm = stats.normalize("positions", x)   # (T, N, 3) or (N, 3)
        x_raw  = stats.denormalize("positions", x_norm)
    """

    FEATURES = ["positions", "velocity", "acceleration", "stress"]

    def __init__(self, metadata_path: str | Path):
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"metadata.json not found: {path}")

        with open(path) as f:
            meta = json.load(f)

        raw = meta.get("normalization_stats", {})
        self._mean: Dict[str, np.ndarray] = {}
        self._std:  Dict[str, np.ndarray] = {}

        for feat in self.FEATURES:
            if feat not in raw:
                raise KeyError(
                    f"Feature '{feat}' missing from normalization_stats in {path}"
                )
            self._mean[feat] = np.array(raw[feat]["mean"], dtype=np.float32)
            self._std[feat]  = np.array(raw[feat]["std"],  dtype=np.float32)

            # Guard against zero std (would cause NaN)
            zero_mask = self._std[feat] < 1e-8
            if zero_mask.any():
                print(f"Warning: near-zero std in '{feat}' dims {np.where(zero_mask)[0]} "
                      f"— clamped to 1.0")
                self._std[feat][zero_mask] = 1.0

    def normalize(self, feature: str, arr: np.ndarray) -> np.ndarray:
        """Z-score normalize. Works for shapes (..., C)."""
        return (arr - self._mean[feature]) / self._std[feature]

    def denormalize(self, feature: str, arr: np.ndarray) -> np.ndarray:
        """Invert z-score. Works for shapes (..., C)."""
        return arr * self._std[feature] + self._mean[feature]

    def denormalize_tensor(self, feature: str, t: torch.Tensor) -> torch.Tensor:
        """Denormalize a torch tensor (for use in loss / rollout)."""
        mean = torch.tensor(self._mean[feature], dtype=t.dtype, device=t.device)
        std  = torch.tensor(self._std[feature],  dtype=t.dtype, device=t.device)
        return t * std + mean


def load_norm_stats(metadata_path: str | Path) -> Optional[NormStats]:
    """Load NormStats if path exists; return None if not found (normalize disabled)."""
    try:
        return NormStats(metadata_path)
    except FileNotFoundError:
        return None


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    """Abstract dataset base class.

    Args:
        cfg: Config dict with at least ``cfg["data"]["path"]``.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int):
        pass


# ── BVC Training Dataset ──────────────────────────────────────────────────────

class BVCDataset(BaseDataset):
    """HDF5 dataset for TransolverNet training.

    Reads pre-windowed h5 files where each window has 6 frames.
    Splits into:
        - x (input):  first 5 frames, all features concatenated → (N, 75)
        - y (target): 6th frame, all features concatenated      → (N, 15)

    Input feature layout  (75 dims per node):
        positions    5 × 3 = 15
        velocity     5 × 3 = 15
        acceleration 5 × 3 = 15
        stress       5 × 6 = 30

    Target feature layout (15 dims per node):
        positions    1 × 3 = 3
        velocity     1 × 3 = 3
        acceleration 1 × 3 = 3
        stress       1 × 6 = 6

    Args:
        cfg: Must have:
            - ``cfg["data"]["path"]``: path to h5 file
            - ``cfg["data"]["metadata_path"]``: path to metadata.json
            - ``cfg["data"]["normalize"]``: bool, default True

    Note:
        h5 file is opened lazily per ``__getitem__`` call to support
        ``num_workers > 0`` in DataLoader.
    """

    FEATURES     = ["positions", "velocity", "acceleration", "stress"]
    INPUT_FRAMES = 5
    TARGET_FRAME = 5   # 6th frame, 0-indexed

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        data_cfg = cfg["data"]

        self.h5_path = Path(data_cfg["path"])
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        # Load normalization stats
        self._stats: Optional[NormStats] = None
        if data_cfg.get("normalize", True):
            meta_path = data_cfg.get("metadata_path")
            if meta_path:
                self._stats = NormStats(meta_path)
                print(f"[Dataset] Normalization enabled — stats loaded from {meta_path}")
            else:
                print("Warning: normalize=True but metadata_path not set — skipping normalization")

        # Build key index without holding the file open
        with h5py.File(self.h5_path, "r") as f:
            self._keys = sorted(k for k in f.keys() if k.startswith("window_"))

        if not self._keys:
            raise ValueError(f"No window groups found in {self.h5_path}")

        # File handle — opened lazily per worker in __getitem__
        self._file: Optional[h5py.File] = None

    def __len__(self) -> int:
        return len(self._keys)

    def _get_file(self) -> h5py.File:
        """Lazy-open the h5 file. Each DataLoader worker gets its own handle."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx):
        f   = self._get_file()
        grp = f[self._keys[idx]]

        # ── Input: 前5帧（不变）─────────────────────────
        x_parts = []
        for feat in self.FEATURES:
            arr = grp[feat][:self.INPUT_FRAMES].astype(np.float32)
            if self._stats is not None:
                arr = self._stats.normalize(feat, arr)
            x_parts.append(arr)
        x = np.concatenate(x_parts, axis=-1)   # (5, N, 15)
        x = x.reshape(x.shape[1], -1)           # (N, 75)

        # ── Target: 残差 = frame 6 - frame 5 ─────────────
        y_parts = []
        for feat in self.FEATURES:
            frame_6 = grp[feat][self.TARGET_FRAME].astype(np.float32)   # (N, C)
            frame_5 = grp[feat][self.INPUT_FRAMES - 1].astype(np.float32) # (N, C)
            
            if self._stats is not None:
                frame_6 = self._stats.normalize(feat, frame_6)
                frame_5 = self._stats.normalize(feat, frame_5)
            
            residual = frame_6 - frame_5   # (N, C) 在 normalized space 算残差
            y_parts.append(residual)

        y = np.concatenate(y_parts, axis=-1)    # (N, 15)

        return torch.from_numpy(x), torch.from_numpy(y)

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ── BVC Full-Trajectory Dataset ───────────────────────────────────────────────

class BVCFullTrajectoryDataset(BaseDataset):
    """Full-trajectory dataset for autoregressive rollout evaluation.

    Each ``__getitem__`` returns a complete window as a dict of tensors,
    suitable for step-by-step rollout evaluation.

    Returns::

        {
          "positions":    FloatTensor (6, N, 3),  # normalized if stats provided
          "velocity":     FloatTensor (6, N, 3),
          "acceleration": FloatTensor (6, N, 3),
          "stress":       FloatTensor (6, N, 6),
          "meta":         {"window_name": str, "window_idx": int},
        }

    Args:
        cfg: Must have:
            - ``cfg["data"]["path"]``: path to h5 file
            - ``cfg["data"]["metadata_path"]``: path to metadata.json
            - ``cfg["data"]["normalize"]``: bool, default True
    """

    FEATURES = ["positions", "velocity", "acceleration", "stress"]

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        data_cfg = cfg["data"]

        self.h5_path = Path(data_cfg["path"])
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        # Load normalization stats (shared with BVCDataset — same metadata.json)
        self._stats: Optional[NormStats] = None
        if data_cfg.get("normalize", True):
            meta_path = data_cfg.get("metadata_path")
            if meta_path:
                self._stats = NormStats(meta_path)

        with h5py.File(self.h5_path, "r") as f:
            self._keys = sorted(k for k in f.keys() if k.startswith("window_"))

        if not self._keys:
            raise ValueError(f"No window groups found in {self.h5_path}")

        self._file: Optional[h5py.File] = None

    def __len__(self) -> int:
        return len(self._keys)

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return full 6-frame window for rollout evaluation."""
        f     = self._get_file()
        wname = self._keys[idx]
        grp   = f[wname]

        data = {}
        for feat in self.FEATURES:
            arr = grp[feat][:].astype(np.float32)   # (6, N, C)
            if self._stats is not None:
                arr = self._stats.normalize(feat, arr)
            data[feat] = torch.from_numpy(arr)

        data["meta"] = {"window_name": wname, "window_idx": idx}
        return data

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ── DataLoader factory ────────────────────────────────────────────────────────

_DATASET_MAP = {
    "bvc":            BVCDataset,
    "bvc_trajectory": BVCFullTrajectoryDataset,
}


def build_dataloader(cfg: dict, split: str = "train") -> torch.utils.data.DataLoader:
    """Build a DataLoader for the given split.

    Expects h5 files at:
        ``{cfg["data"]["base_path"]}/{split}/{split}_data.h5``

    Args:
        cfg:   Full config dict with ``cfg["data"]`` and ``cfg["train"]`` sections.
        split: ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        torch.utils.data.DataLoader

    Example::

        train_loader = build_dataloader(cfg, split="train")
        val_loader   = build_dataloader(cfg, split="val")

        for x, y in train_loader:
            # x: (B, N, 75)
            # y: (B, N, 15)
    """
    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    dataset_type = data_cfg.get("dataset_type", "bvc")
    dataset_cls  = _DATASET_MAP.get(dataset_type)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            f"Available: {list(_DATASET_MAP.keys())}"
        )

    # Build per-split h5 path: base_path/split/split_data.h5
    base_path = Path(data_cfg.get("base_path", "dataset/data_processed"))
    h5_path   = base_path / f"{split}" / f"{split}_data_000.h5" #TODO: support multiple files per split (e.g. _000, _001, ...)

    split_cfg = {**cfg, "data": {**data_cfg, "path": str(h5_path)}}
    dataset   = dataset_cls(split_cfg)

    is_train  = (split == "train")
    batch_size = train_cfg.get("batch_size", 1) if is_train else 1

    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = is_train,
        num_workers = data_cfg.get("num_workers", 0),
        pin_memory  = data_cfg.get("pin_memory", True),
    )