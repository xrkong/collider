# 迁移自: sgnn/transolver/multi_scale_graph.py
# 改动内容: 无改动。仅依赖 torch 和 torch_geometric，无任何项目内部依赖。

"""Hierarchical multi-scale graph construction for the BVC simulator."""

from typing import Dict, Any
import torch
from torch_geometric.nn import radius_graph


class MultiScaleConfig:
    """Configuration for multi-scale mesh parameters."""

    def __init__(self, num_scales: int = 3, window_size: int = 3, radius_multiplier: float = 2.0):
        if num_scales < 2:
            raise ValueError(f"num_scales must be >= 2, got {num_scales}")
        self.num_scales        = num_scales
        self.window_size       = window_size
        self.grid_spacing      = 0.5      # mm — original particle grid spacing (fixed)
        self.radius_multiplier = radius_multiplier
        self.max_neighbors     = 24


class MultiScaleGraph:
    def __init__(self, config: MultiScaleConfig):
        self.config          = config
        self.grid_positions  = None
        self.graph_hierarchy: Dict[int, Dict] = {}

    def create_all_edges(self, grid_positions: torch.Tensor) -> Dict[str, Any]:
        """Build the full multi-scale graph from particle positions.

        Args:
            grid_positions: ``(N, 3)`` positions at the finest scale (CPU tensor).

        Returns:
            Dict with ``graph_hierarchy``, ``grid2mesh_edges``,
            ``mesh2mesh_edges``, ``mesh2grid_edges``.
        """
        if not self.graph_hierarchy:
            self.build_hierarchy(grid_positions)

        g2m, m2g = self._create_grid_mesh_connectivity(grid_positions)

        m2m_parts = []
        for scale in range(1, self.config.num_scales):
            e = self._create_mesh2mesh_edges(scale)
            if e.shape[1] > 0:
                m2m_parts.append(e)
        m2m = (torch.cat(m2m_parts, dim=1) if m2m_parts
               else torch.empty((2, 0), dtype=torch.long))

        return {
            "graph_hierarchy": self.graph_hierarchy,
            "grid2mesh_edges": g2m,
            "mesh2mesh_edges": m2m,
            "mesh2grid_edges": m2g,
        }

    def build_hierarchy(self, grid_positions: torch.Tensor) -> Dict[int, Dict]:
        self.grid_positions = grid_positions
        self.graph_hierarchy[0] = {
            "sampling_indices": torch.arange(len(grid_positions), dtype=torch.long),
            "spacing":          self.config.grid_spacing,
            "num_particles":    len(grid_positions),
        }
        pos, spacing = grid_positions, self.config.grid_spacing
        for scale in range(1, self.config.num_scales):
            pos, spacing, g_idx = self._sample_coarser_scale(pos, spacing, scale)
            self.graph_hierarchy[scale] = {
                "sampling_indices": g_idx,
                "spacing":          spacing,
                "num_particles":    len(pos),
            }
        return self.graph_hierarchy

    def _sample_coarser_scale(self, positions, spacing, scale):
        new_spacing = spacing * self.config.window_size
        sx = torch.sort(torch.unique(positions[:, 0]))[0][:: self.config.window_size]
        sy = torch.sort(torch.unique(positions[:, 1]))[0][:: self.config.window_size]
        mask      = torch.isin(positions[:, 0], sx) & torch.isin(positions[:, 1], sy)
        local_idx = torch.where(mask)[0]
        parent    = self.graph_hierarchy[scale - 1]["sampling_indices"]
        return positions[local_idx], new_spacing, parent[local_idx]

    def _create_grid_mesh_connectivity(self, grid_positions):
        mesh_idx = self.graph_hierarchy[1]["sampling_indices"]
        r        = self.config.radius_multiplier * self.config.grid_spacing
        ei       = radius_graph(grid_positions, r=r, loop=True,
                                max_num_neighbors=self.config.max_neighbors)
        return ei[:, torch.isin(ei[1], mesh_idx)], ei[:, torch.isin(ei[0], mesh_idx)]

    def _create_mesh2mesh_edges(self, scale):
        d      = self.graph_hierarchy[scale]
        r      = d["spacing"] * self.config.radius_multiplier
        s_idx  = d["sampling_indices"]
        ei     = radius_graph(self.grid_positions[s_idx], r=r, loop=True,
                              max_num_neighbors=self.config.max_neighbors)
        return torch.stack([s_idx[ei[0]], s_idx[ei[1]]])
