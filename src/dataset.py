# 迁移自: dataset/bvc_data_loader.py
# 改动内容:
#   - 添加 BaseDataset 抽象类和 build_dataloader() 工厂函数
#   - BVCWindowDataset / BVCFullTrajectoryDataset 继承 BaseDataset，__init__ 接收 cfg
#   - 原有 __getitem__ 逻辑保留不变
#   - normalize_feature() 保留但标注为 no-op（数据已由 h5_dataset_builder.py 预归一化）

from __future__ import annotations

import abc
import json
from pathlib import Path

import torch
import torch.utils.data
import numpy as np

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    """Abstract dataset. Subclasses must implement ``load_data`` and ``__getitem__``.

    Args:
        cfg: Config dict or OmegaConf DictConfig.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def load_data(self):
        """Build in-memory data or a lazy index."""

    @abc.abstractmethod
    def __getitem__(self, idx: int):
        """Return a single sample."""

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


# ── Generic H5 dataset ────────────────────────────────────────────────────────

class H5Dataset(BaseDataset):
    """Generic HDF5 dataset returning ``(x, y)`` tensor pairs.

    Expects top-level datasets ``"x"`` and ``"y"`` inside the HDF5 file.

    Args:
        cfg: Must have ``cfg["data"]["path"]``.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._x = self._y = None
        self.load_data()

    def load_data(self):
        path = self.cfg["data"]["path"]
        with h5py.File(path, "r") as f:
            self._x = torch.from_numpy(f["x"][:]).float()
            self._y = torch.from_numpy(f["y"][:]).float()

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        """Returns: Tuple[Tensor, Tensor] — ``(x[idx], y[idx])``"""
        return self._x[idx], self._y[idx]


# ── BVC normalization mixin ───────────────────────────────────────────────────

class _BaseBVCMixin:
    """Normalization helpers shared by BVC dataset classes.

    NOTE: With h5_dataset_builder.py the data is already z-score normalised at
    build time. ``normalize_feature()`` is effectively a no-op in the current
    pipeline because ``global_stats`` only contains displacement/acceleration
    keys, not per-field keys like ``positions_mean``.
    """

    def _load_normalization_stats(self, data_dir: str) -> dict:
        for candidate in [
            Path(data_dir).parent / "metadata" / "metadata.json",
            Path(data_dir).parent / "metadata.json",
        ]:
            if candidate.exists():
                with open(candidate) as f:
                    return json.load(f).get("global_stats") or {}
        return {}

    def normalize_feature(self, feature: np.ndarray, feature_name: str) -> np.ndarray:
        stats = getattr(self, "_stats", None)
        if not stats:
            return feature
        mean_k, std_k = f"{feature_name}_mean", f"{feature_name}_std"
        if mean_k not in stats:
            return feature
        mean, std = stats[mean_k], stats[std_k]
        return feature if std == 0 else (feature - mean) / std


# ── BVC Window Dataset ────────────────────────────────────────────────────────

class BVCWindowDataset(_BaseBVCMixin, BaseDataset):
    """Sliding-window HDF5 dataset for BVC training/validation.

    Each ``__getitem__`` returns::

        {
          "context":    {feat: Tensor(context_length, N, C)},
          "prediction": {feat: Tensor(1, N, C)},  # last frame
          "meta":       {"window_idx": int, "num_particles": int, "window_name": str},
        }

    Args:
        cfg: Must have ``cfg["data"]["path"]`` pointing to a split directory
             (e.g. ``dataset/data_processed/train``).
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg["data"]
        self.data_dir       = Path(d["path"])
        self.context_length = d.get("context_length", 5)
        self.normalize      = d.get("normalize", True)
        self._stats = self._load_normalization_stats(str(self.data_dir)) if self.normalize else {}

        split_name = self.data_dir.name
        h5_files   = sorted(self.data_dir.glob(f"{split_name}_data_*.h5"))
        if not h5_files:
            fb = self.data_dir / f"{split_name}_data.h5"
            h5_files = [fb] if fb.exists() else []
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")

        self.h5_files = h5_files
        self._file_window_map: list = []
        self._total_windows = 0
        self._available_features: list[str] = []
        self._window_length = self._num_particles = self._spatial_dim = None

        for h5f in h5_files:
            with h5py.File(h5f, "r") as f:
                wnames = sorted(k for k in f if k.startswith("window_"))
                self._file_window_map.append((h5f, wnames))
                self._total_windows += len(wnames)
                if wnames and not self._available_features:
                    self._available_features = list(f[wnames[0]].keys())
                if wnames and self._window_length is None and "positions" in f[wnames[0]]:
                    s = f[wnames[0]]["positions"][:]
                    self._window_length = s.shape[0]
                    self._num_particles = s.shape[1]
                    self._spatial_dim   = s.shape[2] if s.ndim > 2 else 1

        lf = d.get("load_features", None)
        self.features = [x for x in lf if x in self._available_features] if lf else self._available_features
        self.load_data()

    def load_data(self):
        pass  # index built in __init__; lazy loading per __getitem__

    def __len__(self):
        return self._total_windows

    def __getitem__(self, idx: int) -> dict:
        cumulative = 0
        for h5f, wnames in self._file_window_map:
            if cumulative + len(wnames) > idx:
                wname = wnames[idx - cumulative]
                break
            cumulative += len(wnames)

        with h5py.File(h5f, "r") as f:
            grp = f[wname]
            data = {}
            for feat in self.features:
                if feat in grp:
                    arr = self.normalize_feature(grp[feat][:], feat)
                    data[feat] = torch.from_numpy(arr).float()
            n_particles = int(grp.attrs.get("num_particles", 0))

        return {
            "context":    {k: v[: self.context_length] for k, v in data.items()},
            "prediction": {k: v[-1:]                   for k, v in data.items()},
            "meta": {"window_idx": idx, "num_particles": n_particles, "window_name": wname},
        }


# ── BVC Full-Trajectory Dataset ───────────────────────────────────────────────

class BVCFullTrajectoryDataset(_BaseBVCMixin, BaseDataset):
    """Full-trajectory HDF5 dataset for BVC evaluation / autoregressive rollout.

    Each ``__getitem__`` returns::

        {
          "data": {feat: Tensor(T, N, C)},
          "meta": {"window_idx": int, "num_particles": int, "batch_id": int, "window_name": str},
        }

    Args:
        cfg: Must have ``cfg["data"]["path"]`` pointing to a split directory.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg["data"]
        self.data_dir  = Path(d["path"])
        self.normalize = d.get("normalize", True)
        self._stats    = self._load_normalization_stats(str(self.data_dir)) if self.normalize else {}

        split_name = self.data_dir.name
        h5_files   = sorted(self.data_dir.glob(f"{split_name}_data_*.h5"))
        if not h5_files:
            fb = self.data_dir / f"{split_name}_data.h5"
            h5_files = [fb] if fb.exists() else []
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")

        self.h5_files = h5_files
        self._file_window_map: list = []
        self._total_windows = 0
        self._available_features: list[str] = []

        for h5f in h5_files:
            with h5py.File(h5f, "r") as f:
                wnames = sorted(k for k in f if k.startswith("window_"))
                self._file_window_map.append((h5f, wnames))
                self._total_windows += len(wnames)
                if wnames and not self._available_features:
                    self._available_features = list(f[wnames[0]].keys())

        lf = d.get("load_features", None)
        self.features = [x for x in lf if x in self._available_features] if lf else self._available_features
        self.load_data()

    def load_data(self):
        pass

    def __len__(self):
        return self._total_windows

    def __getitem__(self, idx: int) -> dict:
        cumulative = 0
        for h5f, wnames in self._file_window_map:
            if cumulative + len(wnames) > idx:
                wname = wnames[idx - cumulative]
                break
            cumulative += len(wnames)

        with h5py.File(h5f, "r") as f:
            grp = f[wname]
            data = {}
            for feat in self.features:
                if feat in grp:
                    arr = self.normalize_feature(grp[feat][:], feat)
                    data[feat] = torch.from_numpy(arr).float()
            n_particles = int(grp.attrs.get("num_particles", 0))
            batch_id    = int(grp.attrs.get("batch_id", 0))

        return {
            "data": data,
            "meta": {
                "window_idx": idx, "num_particles": n_particles,
                "batch_id": batch_id, "window_name": wname,
            },
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

_DATASET_MAP = {
    "bvc_window":     BVCWindowDataset,
    "bvc_trajectory": BVCFullTrajectoryDataset,
    "h5":             H5Dataset,
}


def build_dataloader(cfg: dict, split: str = "train") -> torch.utils.data.DataLoader:
    """Build a DataLoader for the given split.

    Args:
        cfg:   Full config dict with ``cfg["data"]`` and ``cfg["train"]`` sections.
        split: ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        torch.utils.data.DataLoader
    """
    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    dataset_type = data_cfg.get("dataset_type", "bvc_window")
    dataset_cls  = _DATASET_MAP.get(dataset_type, BVCWindowDataset)

    base_path = data_cfg.get("base_path", data_cfg.get("path", "dataset/data_processed"))
    split_cfg = {**data_cfg, "path": str(Path(base_path) / split)}
    split_cfg_wrapped = {**cfg, "data": split_cfg}

    dataset = dataset_cls(split_cfg_wrapped)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = train_cfg.get("batch_size", 1) if split == "train" else 1,
        shuffle     = (split == "train"),
        num_workers = data_cfg.get("num_workers", 0),
        pin_memory  = data_cfg.get("pin_memory", True),
    )
