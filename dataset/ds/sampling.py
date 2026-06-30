"""
Sampling orchestration — SPEC §4 + §5 combined.

`sample_mesh()` is the single entry point: it takes a parsed MeshData and
returns the final ~100k node selection plus per-node region/segment labels,
respecting:
  - force-keep nodes (§3.4, bypass sampling entirely)
  - per-part allocation within barrier regions (§4.1 / §4.2, full-retain
    floor for rebar/reinforcement, proportional split for the rest)
  - vehicle banding by centerline distance (§4.3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .constants import FULL_RETAIN_FAMILIES, PID_TO_PART_FAMILY
from .kfile_parser import MeshData
from .regions import build_region_masks
from .samplers import SamplerConfig, run_sampler

logger = logging.getLogger(__name__)


@dataclass
class RegionConfig:
    name: str
    sampler: SamplerConfig
    split_by_part: bool = False
    min_per_part: int = 50


# SPEC §8 default config schema, budgets per §4.4
DEFAULT_REGION_CONFIGS = [
    RegionConfig("barrier_fine",   SamplerConfig(n_points=40000, seed=42), split_by_part=True),
    RegionConfig("barrier_coarse", SamplerConfig(n_points=20000, seed=42), split_by_part=True),
    RegionConfig("veh_contact",    SamplerConfig(n_points=10000, seed=42)),
    RegionConfig("veh_near",       SamplerConfig(n_points=18000, seed=42)),
    RegionConfig("veh_far",        SamplerConfig(n_points=12000, seed=42)),
]


def allocate_per_part(
    mesh: MeshData,
    region_mask: np.ndarray,
    n_total: int,
    cfg: SamplerConfig,
    split_by_part: bool = True,
    min_per_part: int = 50,
) -> np.ndarray:
    """Sample a region, splitting the budget across its constituent PIDs.

    Full-retain families (rebar/reinforcement) take all their nodes first;
    the remaining budget is split proportionally (by node count, with a
    min_per_part floor) among the rest and sampled with `cfg.method`.
    """
    region_idx = np.where(region_mask)[0]
    if len(region_idx) == 0:
        return region_idx

    if not split_by_part:
        pts = mesh.coords[region_idx]
        return region_idx[run_sampler(pts, n_total, cfg)]

    pids_in_region = mesh.node_pid[region_idx]
    unique_pids = np.unique(pids_in_region)

    retain_idx: list[np.ndarray] = []
    sample_parts: list[tuple[int, np.ndarray]] = []
    for pid in unique_pids:
        part_family = PID_TO_PART_FAMILY.get(int(pid), "")
        part_global_idx = region_idx[pids_in_region == pid]
        if part_family in FULL_RETAIN_FAMILIES:
            retain_idx.append(part_global_idx)
        else:
            sample_parts.append((pid, part_global_idx))

    n_retained = sum(len(r) for r in retain_idx)
    budget_remaining = max(0, n_total - n_retained)
    total_sample_nodes = sum(len(p[1]) for p in sample_parts)

    sampled_idx: list[np.ndarray] = []
    used = 0
    for i, (pid, part_global_idx) in enumerate(sample_parts):
        if i == len(sample_parts) - 1:
            alloc = budget_remaining - used
        else:
            alloc = max(
                min_per_part,
                int(budget_remaining * len(part_global_idx) / max(total_sample_nodes, 1)),
            )
        alloc = max(min(alloc, len(part_global_idx)), 0)

        pts = mesh.coords[part_global_idx]
        local = run_sampler(pts, alloc, cfg)
        sampled_idx.append(part_global_idx[local])
        used += len(local)

    all_idx = retain_idx + sampled_idx
    return np.concatenate(all_idx) if all_idx else np.array([], dtype=np.int64)


def sample_mesh(
    mesh: MeshData,
    region_configs: list[RegionConfig] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full sampling pipeline.

    Returns
    -------
    sampled_idx   : (~100k,) row indices into mesh arrays
    region_labels : (~100k,) str — region name per sampled node
    segments      : (~100k,) str — part-family label per sampled node
    """
    if region_configs is None:
        region_configs = DEFAULT_REGION_CONFIGS

    masks = build_region_masks(mesh)

    fk_idx = np.where(masks["force_keep"])[0]
    all_sampled: list[np.ndarray] = [fk_idx]
    all_labels: list[np.ndarray] = [np.full(len(fk_idx), "force_keep")]
    all_segments: list[str] = [
        PID_TO_PART_FAMILY.get(int(mesh.node_pid[i]), f"pid_{int(mesh.node_pid[i])}")
        for i in fk_idx
    ]
    already_chosen = set(fk_idx.tolist())

    for rcfg in region_configs:
        region_mask = masks.get(rcfg.name)
        if region_mask is None:
            logger.warning("Region %r not found in masks — skipping", rcfg.name)
            continue

        combined_mask = region_mask.copy()
        if already_chosen:
            combined_mask[np.array(list(already_chosen), dtype=np.int64)] = False

        sampled = allocate_per_part(
            mesh, combined_mask, rcfg.sampler.n_points, rcfg.sampler,
            split_by_part=rcfg.split_by_part, min_per_part=rcfg.min_per_part,
        )
        if len(sampled) == 0:
            logger.warning("Region %r yielded 0 nodes", rcfg.name)
            continue

        all_sampled.append(sampled)
        all_labels.append(np.full(len(sampled), rcfg.name))
        all_segments.extend(
            PID_TO_PART_FAMILY.get(int(mesh.node_pid[i]), f"pid_{int(mesh.node_pid[i])}")
            for i in sampled
        )
        already_chosen.update(sampled.tolist())
        logger.info("Region %r: %d nodes sampled", rcfg.name, len(sampled))

    sampled_idx = np.concatenate(all_sampled)
    labels = np.concatenate(all_labels)
    segments = np.array(all_segments)

    # safety dedup (should already be disjoint by construction)
    _, unique_pos = np.unique(sampled_idx, return_index=True)
    sampled_idx = sampled_idx[unique_pos]
    labels = labels[unique_pos]
    segments = segments[unique_pos]

    logger.info("Total sampled nodes: %d", len(sampled_idx))
    return sampled_idx, labels, segments
