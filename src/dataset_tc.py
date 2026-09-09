"""Time-conditioned (TC) frame dataset — history-free, non-autoregressive.

One __getitem__ sample = one (trajectory, random frame t) pair. No sliding
window, no velocity/acceleration derivation. Node features are built from
static/rest geometry + per-trajectory condition only; the target is
displacement-from-rest at the queried frame. See PLAN
(time-conditioned-transolver) for the full scheme this implements.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.utils.data

from src.conditions import CondConfig, parse_conditions, normalize_conditions
from src.dataset import _resolve_traj_dir, traj_name_from_h5


def compute_displacement_stats(traj_h5_paths: list[str]) -> dict:
    """One-pass Welford mean/std of (positions[t] - positions[0]), pooled over
    all frames/nodes/dims across trajectories. Same running-stats update as
    src/dataset.py's compute_global_stats, applied to the displacement-from-rest
    field (which that function has no notion of)."""
    n, mean, M2 = 0, 0.0, 0.0
    for p in traj_h5_paths:
        with h5py.File(p, "r") as fh:
            pos = np.asarray(fh["states/positions"][...], dtype=np.float64)  # (T, N, 3)
        disp = (pos - pos[0]).reshape(-1)
        n_b = disp.size
        if n_b == 0:
            continue
        mean_b = float(disp.mean())
        var_b = float(disp.var())
        delta = mean_b - mean
        n_new = n + n_b
        mean = (n * mean + n_b * mean_b) / n_new
        M2 += var_b * n_b + delta ** 2 * n * n_b / n_new
        n = n_new
    std = (M2 / max(n, 1)) ** 0.5
    return {"mean": mean, "std": std}


def load_or_compute_displacement_stats(train_dirs: list[str], cache_path: Path) -> dict:
    """Load cached displacement stats or recompute from train dirs. Cache key =
    sorted resolved train_dirs; recomputes whenever that changes."""
    key = sorted(str(Path(d).resolve()) for d in train_dirs)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("key") == key:
            print(f"[TCStats] Loaded cached displacement stats from {cache_path}")
            return cached["stats"]

    h5_paths = [_resolve_traj_dir(d)["h5"] for d in train_dirs]
    print(f"[TCStats] Computing displacement stats over {len(train_dirs)} train traj(s) ...")
    stats = compute_displacement_stats(h5_paths)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"key": key, "stats": stats}, indent=2))
    print(f"[TCStats] displacement: mean={stats['mean']:.6f}  std={stats['std']:.6f}")
    return stats


class TCFrameDataset(torch.utils.data.Dataset):
    """Time-conditioned, history-free frame dataset.

    Each __getitem__ returns one (trajectory, frame) sample:
        [0] node_feats  (N, 3 + n_cond)  normalized [reference_coords, cond broadcast]
        [1] t_norm      scalar            frame / (time_ref_frames - 1)
        [2] target      (N, 3)            normalized displacement-from-rest at `frame`
        [3] node_type   (N,)              only present when cfg["data"]["node_type"]=True

    Args:
        cfg requires:
            cfg["data"]["paths"]:           list[str], h5 file paths
            cfg["data"]["metadata_paths"]:  list[str | None]
            cfg["data"]["time_ref_frames"]: int, fixed scored horizon —
                MUST be identical between train and eval configs.
            cfg["data"]["node_type"]:       bool, default False
            cfg["data"]["node_type_field"]: str, default "node_part_label"
            cfg["data"]["barrier_params_list"]: list[dict], per-traj (see
                train.py's _parse_dirs / BVCSlicedDataset's barrier_params_list)
            cfg["condition"]: passed straight to CondConfig(**cfg["condition"])
        stats: {"positions": {"mean","std"}, "displacement": {"mean","std"}} —
            pooled scalar stats from load_or_compute_global_stats(fields=["positions"])
            and load_or_compute_displacement_stats(). None disables normalization.
    """

    POS_KEY = "states/positions"

    def __init__(self, cfg: dict, *, stats: dict | None = None):
        super().__init__()
        data_cfg = cfg["data"]

        paths = data_cfg.get("paths") or data_cfg.get("path")
        if paths is None:
            raise ValueError("TCFrameDataset requires cfg['data']['paths']")
        if isinstance(paths, str):
            paths = [paths]
        self.h5_paths = [Path(p) for p in paths]
        for p in self.h5_paths:
            if not p.exists():
                raise FileNotFoundError(f"H5 file not found: {p}")

        metadata_paths = data_cfg.get("metadata_paths", [])
        if isinstance(metadata_paths, str):
            metadata_paths = [metadata_paths]

        self.time_ref_frames = int(data_cfg["time_ref_frames"])
        if self.time_ref_frames < 2:
            raise ValueError(f"data.time_ref_frames must be >= 2, got {self.time_ref_frames}")
        self.use_node_type = bool(data_cfg.get("node_type", False))
        self.node_type_field = data_cfg.get("node_type_field", "node_part_label")

        self._cond_cfg = CondConfig(**(cfg.get("condition") or {}))
        print(f"[TCFrameDataset] condition: enabled={list(self._cond_cfg.enabled)}, "
              f"n_cond={self._cond_cfg.n_cond()}")

        self._pos_stats = (stats or {}).get("positions")
        self._disp_stats = (stats or {}).get("displacement")

        barrier_params_list = data_cfg.get("barrier_params_list") or []

        self._trajectories: list[dict] = []
        per_traj_n_frames: list[int] = []

        for idx, p in enumerate(self.h5_paths):
            with h5py.File(p, "r") as f:
                pos = f[self.POS_KEY][:].astype(np.float32)  # (T, N, 3)
                node_type_arr = None
                if self.use_node_type:
                    nt_key = f"metadata/{self.node_type_field}"
                    if nt_key not in f:
                        raise KeyError(
                            f"{p}: missing node_type field '{nt_key}'. "
                            f"Available metadata fields: {sorted(f['metadata'].keys())}."
                        )
                    node_type_arr = f[nt_key][:].astype(np.int64)

            T = pos.shape[0]
            n_frames = min(T, self.time_ref_frames)
            if n_frames < 1:
                raise ValueError(f"Trajectory {p} has 0 usable frames")
            per_traj_n_frames.append(n_frames)

            reference_coords = pos[0]  # (N, 3) — frame 0 treated as rest geometry

            dir_name = traj_name_from_h5(p)
            raw_metadata: dict | None = None
            if idx < len(metadata_paths) and metadata_paths[idx]:
                try:
                    with open(metadata_paths[idx]) as mf:
                        raw_metadata = json.load(mf)
                except Exception:
                    pass

            bp = barrier_params_list[idx] if idx < len(barrier_params_list) else {}
            cond_metadata = dict(raw_metadata or {})
            if bp.get("barrier_angle_deg") is not None:
                cond_metadata.setdefault("angle_deg", bp["barrier_angle_deg"])
            if bp.get("speed") is not None:
                cond_metadata.setdefault("speed_kmh", bp["speed"])
            if "barrier_label" in bp:
                cond_metadata.setdefault("barrier_material", bp["barrier_label"])
            if "layers" in bp:
                cond_metadata.setdefault("layer", bp["layers"])
            if "kirigami_thickness" in bp:
                cond_metadata.setdefault("kirigami_thickness", bp["kirigami_thickness"])
            if bp.get("inter_layer_plate_thickness") is not None:
                cond_metadata.setdefault("inter_layer_plate_thickness", bp["inter_layer_plate_thickness"])
            if bp.get("w_beam_thickness") is not None:
                cond_metadata.setdefault("w_beam_thickness", bp["w_beam_thickness"])

            cond_raw = parse_conditions(cond_metadata, dir_name, cfg=self._cond_cfg)
            cond_vec = normalize_conditions(cond_raw, self._cond_cfg)
            print(f"[TCFrameDataset] traj[{idx}] {dir_name} — T={T}, usable_frames={n_frames} | "
                  f"cond_raw={cond_raw} cond={cond_vec}")

            self._trajectories.append({
                "pos_phys":         pos,
                "reference_coords": reference_coords,
                "cond":             cond_vec,
                "node_type":        node_type_arr,
            })

        self._index_map: list[tuple[int, int]] = []
        for ti, n_frames in enumerate(per_traj_n_frames):
            for fi in range(n_frames):
                self._index_map.append((ti, fi))

        print(f"[TCFrameDataset] {len(self._trajectories)} traj(s) | "
              f"{len(self._index_map)} total (traj, frame) samples | "
              f"time_ref_frames={self.time_ref_frames}")

    def _normalize_positions(self, arr: np.ndarray) -> np.ndarray:
        if self._pos_stats is None:
            return arr.astype(np.float32)
        mean, std = self._pos_stats["mean"], max(self._pos_stats["std"], 1e-8)
        return ((arr - mean) / std).astype(np.float32)

    def _normalize_disp(self, arr: np.ndarray) -> np.ndarray:
        if self._disp_stats is None:
            return arr.astype(np.float32)
        mean, std = self._disp_stats["mean"], max(self._disp_stats["std"], 1e-8)
        return ((arr - mean) / std).astype(np.float32)

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int):
        traj_idx, frame = self._index_map[idx]
        traj = self._trajectories[traj_idx]

        reference_coords = traj["reference_coords"]                   # (N, 3) raw mm
        cond_vec = traj["cond"]                                       # (n_cond,)
        N = reference_coords.shape[0]

        ref_norm = self._normalize_positions(reference_coords)        # (N, 3)
        cond_b = np.broadcast_to(cond_vec, (N, cond_vec.shape[0]))    # (N, n_cond)
        node_feats = np.concatenate([ref_norm, cond_b], axis=-1)      # (N, 3+n_cond)

        gt_pos = traj["pos_phys"][frame]                               # (N, 3) raw mm
        disp_raw = gt_pos - reference_coords                           # (N, 3) raw mm
        target = self._normalize_disp(disp_raw)                        # (N, 3)

        t_norm = frame / (self.time_ref_frames - 1)

        base = (
            torch.from_numpy(np.ascontiguousarray(node_feats)).float(),   # (N, 3+n_cond)
            torch.tensor(t_norm, dtype=torch.float32),                    # scalar
            torch.from_numpy(np.ascontiguousarray(target)).float(),       # (N, 3)
        )
        if self.use_node_type:
            nt = traj["node_type"]
            return base + (torch.from_numpy(np.ascontiguousarray(nt)),)
        return base


def build_tc_dataloader(
    cfg: dict,
    dirs: list[str] | str,
    *,
    shuffle: bool,
    batch_size: int,
    stats: dict | None = None,
    barrier_params: list[dict] | None = None,
) -> torch.utils.data.DataLoader:
    """Build a TCFrameDataset DataLoader from a list of trajectory dirs.
    Mirrors src/dataset.py's build_dataloader, swapping in TCFrameDataset."""
    data_cfg = cfg.get("data", {})
    if isinstance(dirs, str):
        dirs = [dirs]
    if not dirs:
        raise ValueError("build_tc_dataloader needs at least one trajectory dir")

    trajectories = [_resolve_traj_dir(d) for d in dirs]
    ds_cfg = {**cfg, "data": {
        **data_cfg,
        "paths":               [t["h5"]       for t in trajectories],
        "metadata_paths":      [t["metadata"] for t in trajectories],
        "barrier_params_list": barrier_params or [],
    }}

    dataset = TCFrameDataset(ds_cfg, stats=stats)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = data_cfg.get("num_workers", 0),
        pin_memory  = data_cfg.get("pin_memory", True),
    )
