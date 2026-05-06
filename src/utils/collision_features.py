"""
Per-frame collision feature computation.

Given:
    positions:     (N, 3) all node positions in current frame
    barrier_idx:   (N_b,) indices of barrier nodes
    threshold_mm:  collision distance threshold

Returns:
    collision_feat: (N, 2) — [dist_to_nearest_barrier, is_collision_flag]

Notes:
- Output is for ALL N nodes (car + barrier). Barrier nodes' distance to
  themselves will be 0, which is fine — node_type embedding (if used) lets
  the model distinguish them.
- Computed on whatever device `positions` lives on (CPU in dataloader,
  GPU if you call this on a batch later).
"""

from __future__ import annotations

import torch


def compute_collision_features(
    positions: torch.Tensor,        # (N, 3)
    barrier_idx: torch.Tensor,      # (N_b,) long
    threshold: float = 100.0,       # 100 mm
) -> torch.Tensor:
    """Compute per-node distance to nearest barrier + collision flag.

    Args:
        positions:    (N, 3) coordinates of all nodes for one frame.
        barrier_idx:  (N_b,) indices of barrier nodes within `positions`.
        threshold:    distance below which a node is flagged as colliding.
                      Units must match `positions` (likely mm).

    Returns:
        (N, 2) tensor: [dist_to_nearest_barrier, is_collision_flag (0/1)].
        Same dtype and device as `positions`.
    """
    barrier_pos = positions[barrier_idx]                     # (N_b, 3)

    # cdist returns (N, N_b); min over barrier dim → (N,)
    dist_matrix = torch.cdist(positions, barrier_pos)        # (N, N_b)
    min_dist, _ = dist_matrix.min(dim=1)                     # (N,)

    is_collision = (min_dist < threshold).to(positions.dtype)  # (N,)

    # stack into (N, 2)
    return torch.stack([min_dist, is_collision], dim=-1)


def compute_collision_features_window(
    positions_window: torch.Tensor,  # (T, N, 3)
    barrier_idx: torch.Tensor,       # (N_b,)
    threshold: float = 100.0,
) -> torch.Tensor:
    """Same as above, but for a (T, N, 3) window. Returns (T, N, 2).

    Vectorized: cdist supports batched inputs, so this is one call.
    """
    barrier_pos = positions_window[:, barrier_idx, :]        # (T, N_b, 3)
    dist_matrix = torch.cdist(positions_window, barrier_pos) # (T, N, N_b)
    min_dist, _ = dist_matrix.min(dim=-1)                    # (T, N)
    is_collision = (min_dist < threshold).to(positions_window.dtype)
    return torch.stack([min_dist, is_collision], dim=-1)     # (T, N, 2)
