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

from src.conditions import CondConfig, parse_conditions, normalize_conditions


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

        # Per-region normalization is only ever populated via from_global_stats
        # (metadata.json has no per-region breakdown) — see set_region_id().
        self.is_per_region: bool = False
        self._region_mean: Dict[str, np.ndarray] = {}
        self._region_std:  Dict[str, np.ndarray] = {}
        self._region_field: Optional[str] = None
        self._region_id: Optional[np.ndarray] = None

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

    def set_region_id(self, region_id: np.ndarray) -> None:
        """Must be called once per loaded trajectory, before any normalize()/
        denormalize() call for that trajectory's data, whenever
        self.is_per_region is True. region_id: (N,) int array in the same
        node order as the arrays passed to normalize/denormalize."""
        self._region_id = np.asarray(region_id, dtype=np.int64)

    def _region_broadcast(self, feature: str, arr_ndim: int):
        """(mean, std) gathered per-node by region, reshaped to broadcast
        against a (N,C) or (T,N,C) array — or None if region data/region_id
        isn't set for this feature, in which case callers fall back to the
        scalar path (this is how loading an old global-only stats file stays
        bit-for-bit identical to before per-region support existed)."""
        if feature not in self._region_mean or self._region_id is None:
            return None
        mean = self._region_mean[feature][self._region_id]   # (N,)
        std  = self._region_std[feature][self._region_id]    # (N,)
        if arr_ndim == 2:
            return mean[:, None], std[:, None]                # (N,1)
        if arr_ndim == 3:
            return mean[None, :, None], std[None, :, None]    # (1,N,1)
        raise ValueError(f"per-region normalize: unsupported ndim {arr_ndim} for '{feature}'")

    def normalize(self, feature: str, arr: np.ndarray) -> np.ndarray:
        region = self._region_broadcast(feature, arr.ndim)
        if region is not None:
            mean, std = region
            return ((arr - mean) / std).astype(np.float32)
        return ((arr - self._mean[feature]) / self._std[feature]).astype(np.float32)

    def normalize_pooled(self, feature: str, arr: np.ndarray) -> np.ndarray:
        """Normalize with the pooled (non-per-region) global mean/std,
        regardless of is_per_region/region_id. For features that
        deliberately never get per-region treatment even when
        per_region_norm=true — currently just the include_position input
        feature (see train.py: 'Position input feature ... uses the pooled
        global mean/std only ... unlike velocity/acceleration'). Also
        sidesteps _region_broadcast's (T,N,C)-only broadcast shape, which
        breaks for rollout.py's node-major (N,T_in,C) position windows."""
        return ((arr - self._mean[feature]) / self._std[feature]).astype(np.float32)

    def denormalize(self, feature: str, arr: np.ndarray) -> np.ndarray:
        region = self._region_broadcast(feature, arr.ndim)
        if region is not None:
            mean, std = region
            return (arr * std + mean).astype(np.float32)
        return (arr * self._std[feature] + self._mean[feature]).astype(np.float32)

    def denormalize_tensor(self, feature: str, t: torch.Tensor) -> torch.Tensor:
        region = self._region_broadcast(feature, t.dim())
        if region is not None:
            mean = torch.tensor(region[0], dtype=t.dtype, device=t.device)
            std  = torch.tensor(region[1], dtype=t.dtype, device=t.device)
            return t * std + mean
        mean = torch.tensor(self._mean[feature], dtype=t.dtype, device=t.device)
        std  = torch.tensor(self._std[feature],  dtype=t.dtype, device=t.device)
        return t * std + mean

    @classmethod
    def from_global_stats(
        cls,
        stats_path: str | Path,
        acc_scale: float | None = None,
    ) -> "NormStats":
        """Build NormStats from a training-run global_stats.json.

        File schema written by load_or_compute_global_stats():
            {"key": [...], "fields": [...], "stats": {field: {"mean": float, "std": float}, ...}}

        Each field's mean/std is a global scalar across all train trajs.
        Broadcasts correctly against any shape (..., C) in normalize/denormalize.
        """
        stats_path = Path(stats_path)
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Global stats not found: {stats_path}. "
                f"Pass --stats-path explicitly or place global_stats.json next to the checkpoint."
            )
        with open(stats_path) as f:
            cached = json.load(f)

        stats = cached["stats"]
        obj = cls.__new__(cls)
        obj._mean = {}
        obj._std  = {}
        obj.is_per_region  = False
        obj._region_mean   = {}
        obj._region_std    = {}
        obj._region_field  = None
        obj._region_id     = None

        for feat, s in stats.items():
            obj._mean[feat] = np.array(s["mean"], dtype=np.float32)
            obj._std[feat]  = np.array(s["std"],  dtype=np.float32)
            if float(obj._std[feat]) < 1e-8:
                print(f"Warning: near-zero std in '{feat}' — clamped to 1.0")
                obj._std[feat] = np.float32(1.0)

            if "region_mean" in s:
                rmean = np.array(s["region_mean"], dtype=np.float32)
                rstd  = np.array(s["region_std"],  dtype=np.float32)
                zero_mask = rstd < 1e-8
                if zero_mask.any():
                    print(f"Warning: near-zero std in '{feat}' regions "
                          f"{np.where(zero_mask)[0]} — clamped to 1.0")
                    rstd[zero_mask] = 1.0
                obj._region_mean[feat] = rmean
                obj._region_std[feat]  = rstd
                obj.is_per_region = True
                obj._region_field = s.get("region_field") or cached.get("region_field")

        obj._acc_scale = float(acc_scale) if acc_scale else None
        if obj._acc_scale is not None:
            print(f"[NormStats] acceleration uses asinh transform, scale={obj._acc_scale:.2e}")
        else:
            print("[NormStats] acceleration uses z-score (global stats)")

        print(f"[NormStats] Loaded global stats from {stats_path}"
              + (f" (per-region mode, region_field={obj._region_field})" if obj.is_per_region else ""))
        for feat, s in stats.items():
            print(f"  {feat}: mean={s['mean']:.6f}  std={s['std']:.6f}")

        return obj


def load_norm_stats(metadata_path: str | Path) -> Optional[NormStats]:
    try:
        return NormStats(metadata_path)
    except FileNotFoundError:
        return None


# ── Global stats (multi-traj) ─────────────────────────────────────────────────

# Default fields match h5 states/ group keys and metadata.json field_stats keys.
_DEFAULT_NORM_FIELDS = ["positions", "velocity", "acceleration"]


def _read_or_derive_field(fh: "h5py.File", field: str) -> np.ndarray:
    """Read states/{field} directly if present, else derive from positions.

    velocity/acceleration are derived by forward difference (dt=1 frame),
    exactly matching BVCSlicedDataset's own np.diff(positions) convention —
    required so global stats match the values actually normalized at train
    time. The new H5 format (dataset/ds/build_dataset.py) only has positions;
    the legacy format stores velocity/acceleration directly and they're used
    as-is when present (their values are tail-padded but compute_global_stats
    only needs aggregate mean/std, so the small padding tail is negligible).
    """
    h5_key = f"states/{field}"
    if h5_key in fh:
        return np.asarray(fh[h5_key][...])
    if field in ("velocity", "acceleration"):
        pos = np.asarray(fh["states/positions"][...]).astype(np.float64)
        vel = np.diff(pos, axis=0)
        return vel if field == "velocity" else np.diff(vel, axis=0)
    raise KeyError(f"missing dataset '{h5_key}' and no derivation rule for field '{field}'")


def _welford_accumulate(s: dict, x: np.ndarray) -> None:
    """Merge one batch x into running Welford accumulator s (in place)."""
    n_b = x.size
    if n_b == 0:
        return
    mean_b = float(x.mean())
    var_b  = float(x.var())
    n_a    = s["n"]
    delta  = mean_b - s["mean"]
    n_new  = n_a + n_b
    s["mean"] = (n_a * s["mean"] + n_b * mean_b) / n_new
    s["M2"]  += var_b * n_b + delta ** 2 * n_a * n_b / n_new
    s["n"]    = n_new


def _derive_field_from_positions(pos: np.ndarray, field: str) -> np.ndarray:
    """velocity/acceleration by forward difference (dt=1 frame), matching
    both _read_or_derive_field and BVCSlicedDataset's own np.diff convention."""
    if field == "positions":
        return pos
    vel = np.diff(np.asarray(pos, dtype=np.float64), axis=0)
    if field == "velocity":
        return vel
    if field == "acceleration":
        return np.diff(vel, axis=0)
    raise KeyError(f"no derivation rule for field '{field}' from live positions")


def _read_traj_fields(p: str, fields: list[str], live_spec: dict | None) -> dict:
    """Read/derive `fields` for one trajectory — from its h5, or (when
    live_spec is given) live-extracted from its raw d3plot sequence, so
    normalization stats are computed from the same source the model
    actually trains on (see dataset/live_source.py). live_spec keys:
    ref_h5 (required), live_t_start/live_t_end/live_n_jobs (optional) —
    mirrors _load_traj_live in the trajectory-loading path above.
    """
    if live_spec is not None:
        from dataset.live_source import extract_live_trajectory
        live = extract_live_trajectory(
            src_dir=Path(p), ref_h5=Path(live_spec["ref_h5"]),
            t_start=live_spec.get("live_t_start"), t_end=live_spec.get("live_t_end"),
            n_jobs=int(live_spec.get("live_n_jobs", 4)),
        )
        return {f: _derive_field_from_positions(live.positions, f) for f in fields}
    with h5py.File(p, "r") as fh:
        return {f: np.asarray(_read_or_derive_field(fh, f)) for f in fields}


def _read_region_id(p: str, region_field: str, live_spec: dict | None) -> np.ndarray:
    """Per-node region_id array — always from the h5 with real /metadata
    (ref_h5 for a live entry, p itself otherwise; a raw d3plot dir has no
    such data)."""
    ref = live_spec["ref_h5"] if live_spec is not None else p
    with h5py.File(ref, "r") as fh:
        rid_key = f"metadata/{region_field}"
        if rid_key not in fh:
            raise KeyError(f"{ref}: missing region field '{rid_key}'")
        return np.asarray(fh[rid_key][...]).astype(np.int64)


def compute_global_stats(
    traj_h5_paths: list[str],
    fields: list[str],
    live_specs: list[dict | None] | None = None,
) -> dict:
    """One-pass Welford mean/std over (frames × nodes × dims) per field.

    Args:
        traj_h5_paths: list of h5 paths (or, for live entries, raw d3plot
                case dirs — see live_specs)
        fields: list of field names matching states/ group keys
                e.g. ["positions", "velocity", "acceleration"]
                velocity/acceleration are derived from positions when the h5
                doesn't store them directly (see _read_or_derive_field).
        live_specs: index-aligned with traj_h5_paths; None per-entry for a
                normal h5 (unchanged behavior), or a dict with ref_h5 (+
                optional live_t_start/live_t_end/live_n_jobs) to read that
                entry live from its raw d3plot sequence instead — see
                _read_traj_fields.

    Returns:
        {field: {"mean": float, "std": float}}
    """
    accum = {f: {"n": 0, "mean": 0.0, "M2": 0.0} for f in fields}
    live_specs = live_specs or [None] * len(traj_h5_paths)
    for p, live_spec in zip(traj_h5_paths, live_specs):
        field_values = _read_traj_fields(p, fields, live_spec)
        for field in fields:
            x = field_values[field].reshape(-1).astype(np.float64)
            _welford_accumulate(accum[field], x)
    return {
        f: {"mean": s["mean"], "std": (s["M2"] / max(s["n"], 1)) ** 0.5}
        for f, s in accum.items()
    }


def compute_region_stats(
    traj_h5_paths: list[str],
    fields: list[str],
    region_field: str = "region_id",
    min_region_nodes: int = 5,
    live_specs: list[dict | None] | None = None,
) -> dict:
    """Per-region + pooled Welford mean/std over (frames × nodes-in-region × dims).

    Splits each field's values by metadata/{region_field} (a per-node int
    label) before pooling, in addition to the usual pooled/global stats, so
    callers can normalize per-node using its own region's mean/std instead of
    one global scalar. Regions with fewer than min_region_nodes nodes fall
    back to that field's pooled mean/std (too few nodes to trust a
    per-region estimate).

    Args:
        traj_h5_paths: list of h5 paths (or, for live entries, raw d3plot
                case dirs — see live_specs)
        fields: list of field names matching states/ group keys, e.g.
                ["positions", "velocity", "acceleration"] — velocity/
                acceleration are derived from positions when the h5 doesn't
                store them directly (see _read_or_derive_field).
        region_field: metadata/ dataset name holding the per-node int label
                (default "region_id", see dataset/constants.py REGION_ID_MAP).
        min_region_nodes: minimum node count for a region to get its own
                stats; smaller regions fall back to the pooled stats.
        live_specs: index-aligned with traj_h5_paths; None per-entry for a
                normal h5 (unchanged behavior), or a dict with ref_h5 (+
                optional live_t_start/live_t_end/live_n_jobs) to read that
                entry live from its raw d3plot sequence instead (region_id
                is still read from ref_h5's /metadata) — see
                _read_traj_fields/_read_region_id.

    Returns:
        {field: {
            "mean": float, "std": float,                 # pooled — same semantics/values as compute_global_stats
            "region_mean": [float, ...], "region_std": [float, ...],
            "region_n_nodes": [int, ...],
            "region_field": region_field,
        }}
    """
    region_accum: dict = {}   # field -> {region_id: {"n","mean","M2"}}
    node_counts: dict = {}    # region_id -> n_nodes (max seen across files)
    live_specs = live_specs or [None] * len(traj_h5_paths)

    for p, live_spec in zip(traj_h5_paths, live_specs):
        region_id = _read_region_id(p, region_field, live_spec)
        n_regions_here = int(region_id.max()) + 1 if region_id.size else 0

        for r in range(n_regions_here):
            n = int((region_id == r).sum())
            if r in node_counts and node_counts[r] != n:
                print(f"[RegionStats] Warning: region {r} node count differs across "
                      f"trajectories ({node_counts[r]} vs {n} in {p}) — using max seen")
            node_counts[r] = max(node_counts.get(r, 0), n)

        field_values = _read_traj_fields(p, fields, live_spec)
        for field in fields:
            x = field_values[field].astype(np.float64)  # (T', N, C)
            accum = region_accum.setdefault(field, {})
            for r in range(n_regions_here):
                mask = region_id == r
                if not mask.any():
                    continue
                xb = x[:, mask, :].reshape(-1)
                n_b = xb.size
                if n_b == 0:
                    continue
                mean_b = float(xb.mean())
                var_b  = float(xb.var())
                s = accum.setdefault(r, {"n": 0, "mean": 0.0, "M2": 0.0})
                n_a   = s["n"]
                delta = mean_b - s["mean"]
                n_new = n_a + n_b
                s["mean"] = (n_a * s["mean"] + n_b * mean_b) / n_new
                s["M2"]  += var_b * n_b + delta ** 2 * n_a * n_b / n_new
                s["n"]    = n_new

    n_regions = (max(node_counts) + 1) if node_counts else 0

    result = {}
    for field in fields:
        accum = region_accum.get(field, {})

        # Merge all per-region accumulators into one pooled accumulator —
        # mathematically identical to compute_global_stats's single-pass pooling.
        pooled = {"n": 0, "mean": 0.0, "M2": 0.0}
        for s in accum.values():
            n_b, mean_b = s["n"], s["mean"]
            if n_b == 0:
                continue
            var_b = s["M2"] / n_b
            n_a   = pooled["n"]
            delta = mean_b - pooled["mean"]
            n_new = n_a + n_b
            pooled["mean"] = (n_a * pooled["mean"] + n_b * mean_b) / n_new
            pooled["M2"]  += var_b * n_b + delta ** 2 * n_a * n_b / n_new
            pooled["n"]    = n_new
        g_mean = pooled["mean"]
        g_std  = (pooled["M2"] / max(pooled["n"], 1)) ** 0.5
        if g_std < 1e-8:
            print(f"[RegionStats] Warning: near-zero pooled std in '{field}' — clamped to 1.0")
            g_std = 1.0

        region_mean = [0.0] * n_regions
        region_std  = [0.0] * n_regions
        region_n    = [0] * n_regions
        for r in range(n_regions):
            n_nodes    = node_counts.get(r, 0)
            region_n[r] = n_nodes
            s = accum.get(r)
            if s is None or s["n"] == 0 or n_nodes < min_region_nodes:
                if n_nodes and n_nodes < min_region_nodes:
                    print(f"[RegionStats] Warning: region {r} in '{field}' has only "
                          f"{n_nodes} node(s) (< min_region_nodes={min_region_nodes}) "
                          f"— falling back to pooled stats")
                region_mean[r] = g_mean
                region_std[r]  = g_std
                continue
            r_mean = s["mean"]
            r_std  = (s["M2"] / max(s["n"], 1)) ** 0.5
            if r_std < 1e-8:
                print(f"[RegionStats] Warning: near-zero std in '{field}' region {r} "
                      f"— clamped to 1.0")
                r_std = 1.0
            region_mean[r] = r_mean
            region_std[r]  = r_std

        result[field] = {
            "mean": g_mean, "std": g_std,
            "region_mean": region_mean, "region_std": region_std,
            "region_n_nodes": region_n,
            "region_field": region_field,
        }
    return result


def load_or_compute_global_stats(
    train_dirs: list[str],
    cache_path: Path,
    fields: list[str],
    region: bool = False,
    region_field: str = "region_id",
    min_region_nodes: int = 5,
    barrier_params: list[dict] | None = None,
) -> dict:
    """Load cached global (or per-region) stats or recompute from train dirs.

    Cache key = sorted resolved train_dirs (live entries suffixed with their
    ref_h5, so the cache invalidates if that changes) + sorted fields + mode
    (+ region_field when region=True). Recomputes whenever any of these
    change, including flipping between global and per-region mode.

    barrier_params: index-aligned with train_dirs (as produced by train.py's
        _parse_dirs). Only "live"/"ref_h5"/"live_t_start"/"live_t_end"/
        "live_n_jobs" are used here — a "live" entry is read straight from
        its raw d3plot sequence via dataset/live_source.py instead of being
        opened as an h5 (train_dirs[i] is a raw d3plot case dir for those
        entries, which _resolve_traj_dir would otherwise reject).
    """
    bps = barrier_params or [{}] * len(train_dirs)
    key = sorted(
        str(Path(d).resolve()) if not bp.get("live")
        else f"{Path(d).resolve()}::live::{bp.get('ref_h5')}"
        for d, bp in zip(train_dirs, bps)
    )
    mode = "per_region" if region else "global"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        cache_ok = (
            cached.get("key") == key
            and cached.get("fields") == sorted(fields)
            and cached.get("mode", "global") == mode
            and (not region or cached.get("region_field") == region_field)
        )
        if cache_ok:
            print(f"[GlobalStats] Loaded cached stats from {cache_path}")
            return cached["stats"]

    h5_paths: list[str] = []
    live_specs: list[dict | None] = []
    for d, bp in zip(train_dirs, bps):
        if bp.get("live"):
            h5_paths.append(str(d))
            live_specs.append({
                "ref_h5":       bp["ref_h5"],
                "live_t_start": bp.get("live_t_start"),
                "live_t_end":   bp.get("live_t_end"),
                "live_n_jobs":  bp.get("live_n_jobs", 4),
            })
        else:
            h5_paths.append(_resolve_traj_dir(d)["h5"])
            live_specs.append(None)

    if region:
        print(f"[GlobalStats] Computing per-region stats over {len(train_dirs)} "
              f"train traj(s) (region_field={region_field}) ...")
        stats = compute_region_stats(h5_paths, fields, region_field=region_field,
                                      min_region_nodes=min_region_nodes, live_specs=live_specs)
    else:
        print(f"[GlobalStats] Computing global stats over {len(train_dirs)} train traj(s) ...")
        stats = compute_global_stats(h5_paths, fields, live_specs=live_specs)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"key": key, "fields": sorted(fields), "mode": mode,
         "region_field": region_field if region else None, "stats": stats},
        indent=2,
    ))
    print(f"[GlobalStats] Stats saved to {cache_path}")
    for field, s in stats.items():
        print(f"  {field}: mean={s['mean']:.6f}  std={s['std']:.6f}")
    return stats


# ── Trajectory dir resolver ───────────────────────────────────────────────────

def _resolve_traj_dir(d: str | Path) -> dict:
    """Resolve a trajectory entry to {h5, metadata} paths. Fails loudly.

    Accepts two conventions:
      - legacy: a directory containing output.h5 (+ optional metadata.json)
      - new (dataset/ds/build_dataset.py output): a direct path to a .h5 file,
        e.g. h5_fps/T_lok_F_shape_barrier_9_3_60km.h5 — no metadata.json sibling.

    metadata.json is always optional: callers (BVCSlicedDataset, NormStats
    fallback, parse_conditions) already treat a missing/None metadata path as
    "derive conditions from the trajectory name instead."
    """
    d = Path(d)
    if d.is_file() and d.suffix == ".h5":
        h5 = d
    elif d.is_dir():
        h5 = d / "output.h5"
        if not h5.is_file():
            raise FileNotFoundError(f"{d}: no output.h5 found in directory")
    else:
        raise FileNotFoundError(f"Trajectory entry not found: {d}")
    meta = h5.parent / "metadata.json"
    return {"h5": str(h5), "metadata": str(meta) if meta.is_file() else None}


def traj_name_from_h5(h5_path: str | Path) -> str:
    """Trajectory name for condition-parsing/logging.

    Legacy convention (<dir>/output.h5): the parent dir name carries the
    trajectory identity (e.g. ".../T_lok_F_shape_barrier_9_3_60km/output.h5").
    New convention (a directly-named .h5 file): the filename stem carries it
    instead (e.g. "h5_fps/T_lok_F_shape_barrier_9_3_60km.h5" — the parent dir
    here is just the sampling-method folder, not the trajectory).
    """
    p = Path(h5_path)
    return p.parent.name if p.name == "output.h5" else p.stem


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


# ── Pluggable per-trajectory array loading (h5 vs. live d3plot) ───────────────

def _load_traj_h5(
    p: Path,
    *,
    pos_key: str,
    alive_key: str,
    use_node_type: bool,
    node_type_field: str,
    per_region_norm: bool,
    region_norm_field: str,
) -> dict:
    """Read one trajectory's per-frame positions/alive + static per-node
    metadata straight from an h5 file's /states and /metadata groups — the
    original (and still default) BVCSlicedDataset trajectory source."""
    with h5py.File(p, "r") as f:
        pos = f[pos_key][:].astype(np.float32)          # (T, N, 3)
        alive = (
            f[alive_key][:].astype(np.float32)          # (T, N) — 1=alive, 0=eroded
            if alive_key in f else None                   # absent in legacy-format h5
        )
        node_type_arr = None
        if use_node_type:
            nt_key = f"metadata/{node_type_field}"
            if nt_key not in f:
                raise KeyError(
                    f"{p}: missing node_type field '{nt_key}'. "
                    f"Available metadata fields: {sorted(f['metadata'].keys())}. "
                    f"For dataset/ds/build_dataset.py output, set "
                    f"data.node_type_field: region_id in the experiment config."
                )
            node_type_arr = f[nt_key][:].astype(np.int64)  # (N,) static

        region_id_arr = None
        if per_region_norm:
            rid_key = f"metadata/{region_norm_field}"
            if rid_key not in f:
                raise KeyError(
                    f"{p}: missing region field '{rid_key}' required for "
                    f"data.per_region_norm=true. Available metadata fields: "
                    f"{sorted(f['metadata'].keys())}."
                )
            region_id_arr = f[rid_key][:].astype(np.int64)  # (N,) static
    return {"pos": pos, "alive": alive, "node_type": node_type_arr, "region_id": region_id_arr}


def _load_traj_live(
    src_dir: Path,
    live_spec: dict,
    *,
    use_node_type: bool,
    node_type_field: str,
    per_region_norm: bool,
    region_norm_field: str,
) -> dict:
    """Read one trajectory's fine (native d3plot dt) positions/alive
    directly from its raw d3plot sequence (see dataset/live_source.py),
    reusing the node subset/connectivity already baked into
    live_spec['ref_h5'] — static per-node metadata (node_type/region_id)
    always comes from ref_h5, never from src_dir, which has no such data.

    live_spec keys: ref_h5 (required), live_t_start/live_t_end (optional,
    default whole case), live_n_jobs (optional, default 4) — see
    train.py's parse_dir_entry for how these are authored in an
    experiment yaml train_dirs/val_dirs entry.
    """
    from dataset.live_source import extract_live_trajectory

    ref_h5 = Path(live_spec["ref_h5"])
    live = extract_live_trajectory(
        src_dir=Path(src_dir), ref_h5=ref_h5,
        t_start=live_spec.get("live_t_start"), t_end=live_spec.get("live_t_end"),
        n_jobs=int(live_spec.get("live_n_jobs", 4)),
    )

    node_type_arr = None
    region_id_arr = None
    with h5py.File(ref_h5, "r") as f:
        if use_node_type:
            nt_key = f"metadata/{node_type_field}"
            if nt_key not in f:
                raise KeyError(
                    f"{ref_h5}: missing node_type field '{nt_key}' (referenced by live "
                    f"entry {src_dir}). Available metadata fields: "
                    f"{sorted(f['metadata'].keys())}."
                )
            node_type_arr = f[nt_key][:].astype(np.int64)
        if per_region_norm:
            rid_key = f"metadata/{region_norm_field}"
            if rid_key not in f:
                raise KeyError(
                    f"{ref_h5}: missing region field '{rid_key}' required for "
                    f"data.per_region_norm=true (referenced by live entry {src_dir})."
                )
            region_id_arr = f[rid_key][:].astype(np.int64)

    return {
        "pos": live.positions,
        "alive": live.node_alive.astype(np.float32),
        "node_type": node_type_arr,
        "region_id": region_id_arr,
    }


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
    Velocity/acceleration are always derived from states/positions by forward
    difference (dt = 1 frame) — the only h5 state field actually required is
    positions; states/velocity, states/acceleration, states/node_alive are
    optional and consumed if present.

    Each __getitem__ returns a window starting at index t:
        - frames [t, t + input_frames):                    velocity input (normalized)
        - frames [t + input_frames, t + input_frames + K): K acceleration targets (normalized)
        - frames [t, t + input_frames):                    raw positions for SDF
        - frames [t + input_frames, t + input_frames + K): raw positions for push-forward SDF
        - frame  t + input_frames - 1:                     raw velocity (for integration)
                                                             AND erosion alive mask

    Returned tuple (8 base elements + 1 optional):
        [0] x_vel       (N, T_in*3)  normalized velocity, flattened
        [1] future_acc  (N, K, 3)    normalized acceleration targets
        [2] input_pos   (N, T_in, 3) raw positions (for SDF)
        [3] future_pos  (N, K, 3)    raw future positions (for push-forward SDF)
        [4] v_last_phys (N, 3)       physical velocity at last input frame
        [5] barrier_angle_deg        scalar
        [6] x_intercept              scalar
        [7] cond        (n_cond,)    normalized condition vector
        [8] alive_mask  (N,)         1.0=alive, 0.0=eroded at the last input frame
                                      (all-ones when the h5 has no node_alive data)
        [9] node_type   (N,)         only present when cfg["data"]["node_type"]=True

    Args:
        cfg requires:
            - cfg["data"]["paths"]:          list[str], h5 file paths (or trajectory dirs)
            - cfg["data"]["metadata_paths"]: list[str | None], metadata.json paths (optional per-entry)
            - cfg["data"]["input_frames"]:   int, default 5
            - cfg["data"]["normalize"]:      bool, default True
            - cfg["data"]["per_region_norm"]:   bool, default False — opt-in per-region
              normalization (one mean/std per metadata/region_norm_field value, instead
              of one global scalar); see load_or_compute_global_stats(region=True).
            - cfg["data"]["region_norm_field"]: str, default "region_id"
            - cfg["train"]["push_forward_k"]: int, default 1
        stats: global (or per-region, when per_region_norm=True) normalization stats
               dict {field: {"mean": float, "std": float, ...}}.
               If None and normalize=True, falls back to first traj's metadata.json.
               CRITICAL: val dataset must receive train stats, not its own metadata stats.
    """

    POS_KEY   = "states/positions"
    ALIVE_KEY = "states/node_alive"   # optional — erosion mask, see dataset/ds/build_dataset.py

    def __init__(self, cfg: dict, *, stats: dict | None = None):
        super().__init__(cfg)
        data_cfg  = cfg["data"]
        train_cfg = cfg.get("train", {})

        # Per-trajectory barrier params injected by build_dataloader
        # List of {"barrier_angle_deg": float, "x_intercept": float}, one per path.
        self._barrier_params: list[dict] = data_cfg.get("barrier_params_list") or []

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
        self.use_node_type   = bool(data_cfg.get("node_type", False))
        self.node_type_field = data_cfg.get("node_type_field", "node_part_label")
        self.per_region_norm   = bool(data_cfg.get("per_region_norm", False))
        self.region_norm_field = data_cfg.get("region_norm_field", "region_id")
        if self.K < 1:
            raise ValueError(f"push_forward_k must be >= 1, got {self.K}")
        self.window_len = self.input_frames + self.K

        # ── Condition config (shared across all trajectories) ─────────────
        self._cond_cfg = CondConfig(**(cfg.get("condition") or {}))
        print(f"[BVCSlicedDataset] condition: enabled={list(self._cond_cfg.enabled)}, "
              f"n_cond={self._cond_cfg.n_cond()}")

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
            # Barrier SDF params: use per-traj values if provided, else
            # defaults. Moved to the top of the loop (was after array
            # loading) because it also carries the live-source dispatch
            # flag/spec (live/ref_h5/live_t_start/live_t_end/live_n_jobs)
            # for this trajectory — see train.py's parse_dir_entry.
            if idx < len(self._barrier_params):
                bp = self._barrier_params[idx]
            else:
                bp = {"barrier_angle_deg": -25.4, "x_intercept": 2056.579}

            load_kwargs = dict(
                use_node_type=self.use_node_type,
                node_type_field=self.node_type_field,
                per_region_norm=self.per_region_norm,
                region_norm_field=self.region_norm_field,
            )
            if bp.get("live"):
                arrs = _load_traj_live(p, bp, **load_kwargs)
            else:
                arrs = _load_traj_h5(p, pos_key=self.POS_KEY, alive_key=self.ALIVE_KEY, **load_kwargs)
            pos, alive = arrs["pos"], arrs["alive"]
            node_type_arr = arrs["node_type"]
            region_id_arr = arrs["region_id"]

            T = pos.shape[0]
            if T < self.window_len:
                raise ValueError(
                    f"Trajectory {p} has only {T} frames, need >= {self.window_len} "
                    f"(input_frames={self.input_frames} + K={self.K})"
                )

            # Kinematics are always derived from positions (forward difference,
            # dt = 1 frame) — never read from states/velocity or
            # states/acceleration, which the new H5 format doesn't have anyway.
            vel_derived     = np.diff(pos, axis=0)          # (T-1, N, 3)
            acc_derived_raw = np.diff(vel_derived, axis=0)  # (T-2, N, 3)
            vel_derived_norm = self._normalize("velocity",     vel_derived,     region_id=region_id_arr)
            acc_derived_norm = self._normalize("acceleration", acc_derived_raw, region_id=region_id_arr)

            n_windows = max(0, T - self.window_len)
            per_traj_windows.append(n_windows)

            dir_name = traj_name_from_h5(p)
            meta_label = ""
            raw_metadata: dict | None = None
            if idx < len(metadata_paths) and metadata_paths[idx]:
                try:
                    with open(metadata_paths[idx]) as mf:
                        raw_metadata = json.load(mf)
                    mc = raw_metadata.get("config", {})
                    speed = mc.get("source", "")
                    meta_label = f"  [{speed.split('/')[-1]}]" if speed else ""
                except Exception:
                    pass

            # barrier_label/layers/kirigami_thickness/inter_layer_plate_thickness/
            # w_beam_thickness have no filename convention (see
            # src/conditions.py's parse_conditions docstring) — they're
            # authored explicitly per data.train_dirs/val_dirs entry in the
            # experiment yaml (train.py's parse_dir_entry) and threaded here
            # via build_dataloader's barrier_params, taking priority over any
            # dir-name regex guess exactly like a metadata.json would.
            cond_metadata = dict(raw_metadata or {})
            cond_metadata.setdefault("angle_deg", bp.get("barrier_angle_deg"))
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

            # Parse and normalize physical conditions for this trajectory.
            # Passing cfg makes inter_layer_plate_thickness/w_beam_thickness
            # raise loudly (rather than silently default to 0) whenever
            # they're enabled, left unset on this entry, AND this isn't a
            # GT/baseline trajectory (which gets gated to 0 regardless).
            cond_raw = parse_conditions(cond_metadata, dir_name, cfg=self._cond_cfg)
            cond_vec = normalize_conditions(cond_raw, self._cond_cfg)
            print(f"[BVCSlicedDataset] traj[{idx}] {dir_name}{meta_label} "
                  f"— T={T}, windows={n_windows} | "
                  f"cond_raw={cond_raw} cond={cond_vec}")

            traj_entry = {
                "pos_phys":          pos,
                "T":                 T,
                "path":              str(p),
                "vel_derived_norm":  vel_derived_norm,
                "vel_derived_phys":  vel_derived,
                "acc_derived_norm":  acc_derived_norm,
                "T_derived":         T - 2,
                "alive":             alive,      # (T, N) float32 1/0, or None if not in this h5
                "barrier_angle_deg": bp["barrier_angle_deg"],
                "x_intercept":       bp["x_intercept"],
                "cond":              cond_vec,    # np.float32 (n_cond,) — constant per traj
                "cond_raw":          cond_raw,    # physical values for logging
            }
            if self.use_node_type:
                traj_entry["node_type"] = node_type_arr  # (N,) int64, static
            self._trajectories.append(traj_entry)

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

    def _normalize(self, field: str, arr: np.ndarray, region_id: np.ndarray | None = None) -> np.ndarray:
        """Apply normalization using whichever stats source is active.

        region_id: (N,) per-node region label, in the same node order as
        arr's node axis (arr is always (T, N, C) at this call site). Only
        used when self.per_region_norm and the active stats dict actually
        carries a per-region breakdown for this field.
        """
        if self._stats_dict is not None:
            s = self._stats_dict.get(field)
            if s is None:
                return arr.astype(np.float32)
            if region_id is not None and "region_mean" in s:
                mean = np.asarray(s["region_mean"], dtype=np.float64)[region_id]  # (N,)
                std  = np.asarray(s["region_std"],  dtype=np.float64)[region_id]  # (N,)
                std  = np.where(std < 1e-8, 1.0, std)
                mean_b = mean[None, :, None]   # (1, N, 1) broadcasts against (T, N, C)
                std_b  = std[None, :, None]
                return ((arr - mean_b) / std_b).astype(np.float32)
            if region_id is not None and "region_mean" not in s:
                print(f"[BVCSlicedDataset] Warning: per_region_norm=True but stats['{field}'] "
                      f"has no region_mean — falling back to global scalar normalization for "
                      f"this field. Was global_stats.json computed with region=True?")
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

        # Erosion mask: alive status at the last input frame (the "current"
        # state the model conditions on). 1.0 = alive, 0.0 = eroded. All-ones
        # when this h5 has no node_alive data (legacy format / no erosion).
        if traj["alive"] is not None:
            alive_mask = traj["alive"][start + T_in - 1]            # (N,) float32
        else:
            alive_mask = np.ones(N, dtype=np.float32)

        base = (
            torch.from_numpy(np.ascontiguousarray(x_vel)),                       # [0] (N, T_in*3)
            torch.from_numpy(np.ascontiguousarray(future_acc)),                  # [1] (N, K, 3)
            torch.from_numpy(np.ascontiguousarray(input_pos)),                   # [2] (N, T_in, 3)
            torch.from_numpy(np.ascontiguousarray(future_pos)),                  # [3] (N, K, 3)
            torch.from_numpy(np.ascontiguousarray(v_last_phys)),                 # [4] (N, 3)
            torch.tensor(traj["barrier_angle_deg"], dtype=torch.float32),        # [5] scalar
            torch.tensor(traj["x_intercept"],       dtype=torch.float32),        # [6] scalar
            torch.from_numpy(np.ascontiguousarray(traj["cond"])),                # [7] (n_cond,)
            torch.from_numpy(np.ascontiguousarray(alive_mask)),                  # [8] (N,) 1=alive 0=eroded
        )
        if self.use_node_type:
            nt = traj["node_type"]                                                # (N,) int64
            return base + (torch.from_numpy(np.ascontiguousarray(nt)),)          # [9] (N,) long
        return base


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
    barrier_params: list[dict] | None = None,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from a list of trajectory dirs.

    Args:
        cfg:             full experiment config dict
        dirs:            one or more trajectory dirs (each must contain output.h5 + metadata.json)
        shuffle:         whether to shuffle windows across all trajs
        batch_size:      batch size
        stats:           global normalization stats from load_or_compute_global_stats.
                         MUST be train stats even for val loader.
        barrier_params:  list of {"barrier_angle_deg": float, "x_intercept": float},
                         one entry per dir.  Defaults to -25.4° / 2056.579 mm if omitted.
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

    # A "live" entry (barrier_params[i]["live"] is truthy) points at a raw
    # d3plot case dir, not an h5/output.h5 — _resolve_traj_dir would reject
    # it, so pass it straight through; BVCSlicedDataset's per-trajectory
    # loop dispatches to _load_traj_live for these instead of h5py.File(p).
    bps_for_resolve = barrier_params or [{}] * len(dirs)
    trajectories = [
        {"h5": str(d), "metadata": None} if bp.get("live") else _resolve_traj_dir(d)
        for d, bp in zip(dirs, bps_for_resolve)
    ]
    ds_cfg = {**cfg, "data": {
        **data_cfg,
        "paths":               [t["h5"]       for t in trajectories],
        "metadata_paths":      [t["metadata"] for t in trajectories],
        "barrier_params_list": barrier_params or [],
    }}

    dataset = dataset_cls(ds_cfg, stats=stats)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = data_cfg.get("num_workers", 0),
        pin_memory  = data_cfg.get("pin_memory", True),
    )
