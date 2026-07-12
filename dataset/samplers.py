"""
Pluggable point samplers — SPEC §5.

All four methods run on the same region definitions/budgets so the
comparison across methods is fair; only `cfg.method` changes between runs.
Each sampler returns *local* indices into the `points` array it was given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SamplerConfig:
    method: str = "fps"           # stride | random | poisson_disk | fps
    n_points: int = 10000
    seed: int = 42
    stride_order: str = "file"  # file | morton | x
    poisson_radius: Optional[float] = None
    enforce_exact_n: bool = True
    density_weighted: bool = False
    density_d0: float = 600.0     # mm, decay length for density-weighted fps


def _morton_encode(coords: np.ndarray) -> np.ndarray:
    """Z-order curve key via bit-interleaving of 21-bit quantized coords."""
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    rng = np.where(mx - mn > 0, mx - mn, 1.0)
    norm = ((coords - mn) / rng * (2**21 - 1)).astype(np.uint64)
    x, y, z = norm[:, 0], norm[:, 1], norm[:, 2]

    def spread(v: np.ndarray) -> np.ndarray:
        v = v & np.uint64(0x1FFFFF)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
        return v

    return spread(x) | (spread(y) << np.uint64(1)) | (spread(z) << np.uint64(2))


def sample_stride(points: np.ndarray, n: int, cfg: SamplerConfig) -> np.ndarray:
    """Sort by `cfg.stride_order`, take every step = N//n.

    `"file"` (original node order) is the deliberately bad baseline that
    exposes naive downsampling; `"morton"` is the fair equal-interval one.
    """
    N = len(points)
    if cfg.stride_order == "file":
        order = np.arange(N)
    elif cfg.stride_order == "x":
        order = np.argsort(points[:, 0])
    else:
        order = np.argsort(_morton_encode(points))
    step = max(1, N // n)
    idx = order[::step]
    if len(idx) > n:
        idx = idx[:n]
    elif len(idx) < n and cfg.enforce_exact_n:
        remaining = np.setdiff1d(order, idx)
        idx = np.concatenate([idx, remaining[: n - len(idx)]])
    return idx


def sample_random(points: np.ndarray, n: int, cfg: SamplerConfig) -> np.ndarray:
    """i.i.d. uniform choice — will produce clusters/voids by chance."""
    N = len(points)
    rng = np.random.default_rng(cfg.seed)
    return rng.choice(N, min(n, N), replace=False)


def sample_poisson_disk(points: np.ndarray, n: int, cfg: SamplerConfig) -> np.ndarray:
    """Dart-throwing: accept a candidate only if no chosen point is within radius r.

    Point count is emergent from r; if `enforce_exact_n`, the result is
    truncated or topped up from the remaining pool to hit exactly n.
    """
    N = len(points)
    rng = np.random.default_rng(cfg.seed)

    if cfg.poisson_radius is None:
        mn, mx = points.min(axis=0), points.max(axis=0)
        vol = np.prod(np.maximum(mx - mn, 1.0))
        r = (vol / n) ** (1.0 / 3.0) * 0.9
    else:
        r = cfg.poisson_radius

    order = rng.permutation(N)
    chosen: list[int] = []
    tree_pts: list[np.ndarray] = []
    tree = None

    for idx in order:
        pt = points[idx]
        if tree is None or len(chosen) == 0:
            chosen.append(idx)
            tree_pts.append(pt)
            if len(chosen) % 500 == 0:
                tree = cKDTree(np.array(tree_pts))
        else:
            nn_dist, _ = tree.query(pt, k=1)
            if nn_dist >= r:
                chosen.append(idx)
                tree_pts.append(pt)
                if len(chosen) % 500 == 0:
                    tree = cKDTree(np.array(tree_pts))

    chosen_arr = np.array(chosen)
    if cfg.enforce_exact_n:
        if len(chosen_arr) > n:
            chosen_arr = chosen_arr[:n]
        elif len(chosen_arr) < n:
            remaining = np.setdiff1d(np.arange(N), chosen_arr)
            chosen_arr = np.concatenate([chosen_arr, remaining[: n - len(chosen_arr)]])
    return chosen_arr


def sample_fps(
    points: np.ndarray,
    n: int,
    cfg: SamplerConfig,
    density: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Farthest-point sampling, optionally density-weighted (PointNet++ convention)."""
    N = len(points)
    n = min(n, N)
    rng = np.random.default_rng(cfg.seed)

    chosen = np.zeros(n, dtype=np.int64)
    chosen[0] = rng.integers(0, N)
    min_dists = np.full(N, np.inf)

    for i in range(1, n):
        last = points[chosen[i - 1]]
        dists = np.sum((points - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        scores = min_dists * density if (cfg.density_weighted and density is not None) else min_dists
        chosen[i] = int(np.argmax(scores))

    return chosen


def run_sampler(
    points: np.ndarray,
    n: int,
    cfg: SamplerConfig,
    density: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dispatch to the configured sampler method."""
    if cfg.method == "stride":
        return sample_stride(points, n, cfg)
    if cfg.method == "random":
        return sample_random(points, n, cfg)
    if cfg.method == "poisson_disk":
        return sample_poisson_disk(points, n, cfg)
    if cfg.method == "fps":
        return sample_fps(points, n, cfg, density)
    raise ValueError(f"Unknown sampler method: {cfg.method!r}")
