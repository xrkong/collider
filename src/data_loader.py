# 迁移自: sgnn/transolver/static_graph_data_loader.py
# 改动内容:
#   - 导入 BVCWindowDataset / BVCFullTrajectoryDataset 改为 src.dataset（不再依赖外部 dataset/）
#   - 导入 MultiScaleConfig / MultiScaleGraph 改为 models.multi_scale_graph
#   - 移除 Taylor Impact 2D 遗留函数（multi_scale_collate_fn / get_multi_scale_data_loader_by_*）
#   - 保留全部 BVC 相关类和函数，逻辑不变

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.utils.data

from src.dataset import BVCWindowDataset, BVCFullTrajectoryDataset
from models.multi_scale_graph import MultiScaleConfig, MultiScaleGraph


# ── Static graph builder ──────────────────────────────────────────────────────

def build_static_multi_scale_graph(
    initial_positions: torch.Tensor,
    num_scales: int = 3,
    window_size: int = 3,
    radius_multiplier: float = 2.0,
) -> Dict[str, Any]:
    """Build a static multi-scale graph from initial particle positions.

    Always computed on CPU (graph topology is device-independent).

    Args:
        initial_positions: ``(N, 3)`` positions.
        num_scales:         Number of hierarchy levels.
        window_size:        Spatial sub-sampling stride for coarser levels.
        radius_multiplier:  Connectivity radius multiplier.

    Returns:
        Dict with ``graph_hierarchy``, ``grid2mesh_edges``,
        ``mesh2mesh_edges``, ``mesh2grid_edges``.
    """
    config = MultiScaleConfig(
        num_scales=num_scales,
        window_size=window_size,
        radius_multiplier=radius_multiplier,
    )
    graph = MultiScaleGraph(config)
    return graph.create_all_edges(initial_positions.cpu())


# ── Collate function ──────────────────────────────────────────────────────────

def _bvc_collate(batch: List[Dict]) -> Dict:
    """Collate a list of BVC window samples into a batched dict."""
    out = {"context": {}, "prediction": {}, "meta": [], "graph": None}
    for sample in batch:
        out["meta"].append(sample["meta"])
        for feat, t in sample["context"].items():
            out["context"].setdefault(feat, []).append(t)
        for feat, t in sample["prediction"].items():
            out["prediction"].setdefault(feat, []).append(t)
        if "graph" in sample and out["graph"] is None:
            out["graph"] = sample["graph"]
    out["context"]    = {k: torch.stack(v) for k, v in out["context"].items()}
    out["prediction"] = {k: torch.stack(v) for k, v in out["prediction"].items()}
    return out


# ── Multi-scale BVC Window Dataset ───────────────────────────────────────────

class MultiScaleBVCWindowDataset(BVCWindowDataset):
    """BVCWindowDataset extended with static multi-scale graph per window.

    Builds all graphs at init time (one per window).
    """

    def __init__(
        self,
        cfg: dict,
        num_scales: int = 3,
        window_size: int = 3,
        radius_multiplier: float = 2.0,
    ):
        super().__init__(cfg)
        self._num_scales        = num_scales
        self._window_size       = window_size
        self._radius_multiplier = radius_multiplier

        print(f"Building {len(self)} static multi-scale graphs ...")
        self._static_graphs: Dict[int, Optional[Dict]] = {}
        for i in range(len(self)):
            try:
                sample    = super().__getitem__(i)
                positions = sample["context"].get("positions")
                if positions is not None:
                    self._static_graphs[i] = build_static_multi_scale_graph(
                        positions[0], num_scales, window_size, radius_multiplier
                    )
            except Exception as exc:
                print(f"  Warning: graph {i} failed: {exc}")
                self._static_graphs[i] = None
        n_ok = sum(1 for g in self._static_graphs.values() if g is not None)
        print(f"  Built {n_ok}/{len(self)} graphs (scales={num_scales}, window={window_size})")

    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)
        if self._static_graphs.get(idx) is not None:
            sample["graph"] = self._static_graphs[idx]
        return sample


# ── Multi-scale BVC Full-Trajectory Dataset ───────────────────────────────────

class MultiScaleBVCFullTrajectoryDataset(BVCFullTrajectoryDataset):
    """BVCFullTrajectoryDataset extended with a static multi-scale graph."""

    def __init__(
        self,
        cfg: dict = None,
        data_dir: str = None,
        num_scales: int = 3,
        window_size: int = 3,
        radius_multiplier: float = 2.0,
    ):
        # Accept either cfg dict or bare data_dir string for backward compat
        if cfg is None and data_dir is not None:
            cfg = {"data": {"path": data_dir}}
        super().__init__(cfg)
        self._num_scales        = num_scales
        self._window_size       = window_size
        self._radius_multiplier = radius_multiplier

        print(f"Building {len(self)} trajectory graphs ...")
        self._static_graphs: Dict[int, Optional[Dict]] = {}
        for i in range(len(self)):
            try:
                traj      = super().__getitem__(i)
                positions = traj["data"].get("positions")
                if positions is not None:
                    self._static_graphs[i] = build_static_multi_scale_graph(
                        positions[0], num_scales, window_size, radius_multiplier
                    )
            except Exception as exc:
                print(f"  Warning: graph {i} failed: {exc}")
                self._static_graphs[i] = None
        n_ok = sum(1 for g in self._static_graphs.values() if g is not None)
        print(f"  Built {n_ok}/{len(self)} trajectory graphs")

    def __getitem__(self, idx: int) -> Dict:
        traj = super().__getitem__(idx)
        if self._static_graphs.get(idx) is not None:
            traj["graph"] = self._static_graphs[idx]
        return traj


# ── DataLoader factories ──────────────────────────────────────────────────────

def get_multi_scale_bvc_data_loader_by_windows(
    data_dir: str,
    context_length: int = 5,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    num_scales: int = 3,
    window_size: int = 3,
    radius_multiplier: float = 2.0,
    split: str = "train",
) -> torch.utils.data.DataLoader:
    """DataLoader for BVC sliding-window training.

    Args:
        data_dir:         Base directory containing ``train/``, ``valid/``, ``test/``.
        context_length:   Number of input context frames.
        batch_size:       Training batch size.
        shuffle:          Whether to shuffle.
        num_workers:      DataLoader workers.
        pin_memory:       Pin memory for faster GPU transfer.
        num_scales:       Hierarchy levels.
        window_size:      Spatial sub-sampling stride.
        radius_multiplier: Connectivity radius multiplier.
        split:            ``"train"``, ``"valid"``, or ``"test"``.

    Returns:
        torch.utils.data.DataLoader
    """
    cfg = {"data": {"path": str(Path(data_dir) / split), "context_length": context_length}}
    dataset = MultiScaleBVCWindowDataset(
        cfg, num_scales=num_scales,
        window_size=window_size, radius_multiplier=radius_multiplier,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_bvc_collate,
    )


def get_multi_scale_bvc_data_loader_by_trajectories(
    data_dir: str,
    num_workers: int = 0,
    pin_memory: bool = True,
    num_scales: int = 3,
    window_size: int = 3,
    radius_multiplier: float = 2.0,
    split: str = "test",
) -> torch.utils.data.DataLoader:
    """DataLoader for BVC full-trajectory evaluation.

    Args:
        data_dir:  Base directory containing ``train/``, ``valid/``, ``test/``.
        split:     ``"test"``, ``"valid"``, or ``"train"``.

    Returns:
        torch.utils.data.DataLoader (batch_size=None, no collation)
    """
    dataset = MultiScaleBVCFullTrajectoryDataset(
        data_dir=str(Path(data_dir) / split),
        num_scales=num_scales,
        window_size=window_size,
        radius_multiplier=radius_multiplier,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
