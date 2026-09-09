"""Time-conditioned (TC) rollout — independent per-frame query, no rollout state.

Unlike src/rollout.py's autoregressive/one-step modes, there is no history
window, no teacher forcing, and no state carried between frames: every frame
is one independent forward pass at its own normalized query time. See PLAN
(time-conditioned-transolver) for the full scheme.

v1 scope: prints per-frame + overall RMSE (physical mm) to console. No GIF
rendering — src/rollout.py's matplotlib pipeline is AR-specific; add later by
reusing its plotting helpers once TC numerics are validated.

Usage:
    python src/rollout_tc.py \
        --checkpoint outputs/checkpoints/wj10_tc/checkpoint-best.safetensors \
        --experiment configs/experiments/wj10_tc.yaml \
        --raw-h5 /path/to/some_case.h5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import models  # noqa: F401
from src.rollout import load_model, load_raw_h5
from src.conditions import CondConfig, parse_conditions, normalize_conditions
from src.dataset import traj_name_from_h5


def _load_stats_json(path: Path, key: str | None = "stats") -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    data = json.loads(path.read_text())
    return data[key] if key else data


@torch.no_grad()
def rollout_tc(
    model: torch.nn.Module,
    cfg: dict,
    raw_h5: str,
    pos_stats: dict,
    disp_stats: dict,
    device: torch.device,
    cond_overrides: dict | None = None,
) -> tuple[np.ndarray, float]:
    """Per-frame independent query over the scored range. Returns (rmse_per_frame, overall_rmse)."""
    data_cfg = cfg["data"]
    use_node_type   = bool(data_cfg.get("node_type", False))
    node_type_field = data_cfg.get("node_type_field", "node_part_label") if use_node_type else None

    raw = load_raw_h5(raw_h5, node_type_field=node_type_field)
    positions = raw["positions"]            # (T, N, 3)
    T, N, _ = positions.shape
    reference_coords = positions[0]         # (N, 3) — frame 0 = rest geometry

    time_ref_frames = int(data_cfg["time_ref_frames"])

    cond_cfg = CondConfig(**(cfg.get("condition") or {}))
    dir_name = traj_name_from_h5(raw_h5)
    cond_raw = parse_conditions(cond_overrides or {}, dir_name, cfg=cond_cfg)
    cond_vec = normalize_conditions(cond_raw, cond_cfg)   # (n_cond,)
    print(f"[RolloutTC] {dir_name} — cond_raw={cond_raw} cond={cond_vec}")

    pos_mean,  pos_std  = pos_stats["mean"],  max(pos_stats["std"], 1e-8)
    disp_mean, disp_std = disp_stats["mean"], max(disp_stats["std"], 1e-8)

    ref_norm   = (reference_coords - pos_mean) / pos_std            # (N, 3)
    cond_b     = np.broadcast_to(cond_vec, (N, cond_vec.shape[0]))  # (N, n_cond)
    node_feats = np.concatenate([ref_norm, cond_b], axis=-1).astype(np.float32)
    node_feats_t = torch.from_numpy(node_feats).to(device)          # (N, 3+n_cond)

    node_type_t = None
    if use_node_type:
        node_type_t = torch.from_numpy(raw["node_type"]).to(device)

    n_frames = min(T, time_ref_frames)
    rmse_per_frame = np.zeros(n_frames, dtype=np.float64)
    for frame in range(n_frames):
        t_norm = frame / (time_ref_frames - 1)
        t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device)

        pred_disp_norm = model(node_feats_t, node_type_t, t_tensor)   # (N, 3)
        pred_disp = pred_disp_norm.cpu().numpy() * disp_std + disp_mean
        pred_pos  = reference_coords + pred_disp

        gt_pos = positions[frame]
        rmse_per_frame[frame] = float(np.sqrt(np.mean((pred_pos - gt_pos) ** 2)))

    overall_rmse = float(np.sqrt(np.mean(rmse_per_frame ** 2)))
    return rmse_per_frame, overall_rmse


def main():
    parser = argparse.ArgumentParser(description="Time-Conditioned Transolver Rollout")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--raw-h5",     required=True)
    parser.add_argument("--stats-path", default=None,
                        help="Path to global_stats.json (defaults to <ckpt_dir>/global_stats.json)")
    parser.add_argument("--disp-stats-path", default=None,
                        help="Path to global_stats_tc_displacement.json "
                             "(defaults to <ckpt_dir>/global_stats_tc_displacement.json)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Condition overrides — mirrors src/rollout.py's flags; only needed when
    # --raw-h5's dir name doesn't already encode these via filename regex
    # (see src/conditions.py's parse_conditions docstring).
    parser.add_argument("--barrier-label", default=None)
    parser.add_argument("--layers", type=float, default=None)
    parser.add_argument("--kirigami-thickness", type=float, default=None)
    parser.add_argument("--inter-layer-plate-thickness", type=float, default=None)
    parser.add_argument("--w-beam-thickness", type=float, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--angle", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, args.experiment, device)

    ckpt_dir = Path(args.checkpoint).parent
    stats_path = Path(args.stats_path) if args.stats_path else ckpt_dir / "global_stats.json"
    disp_stats_path = (
        Path(args.disp_stats_path) if args.disp_stats_path
        else ckpt_dir / "global_stats_tc_displacement.json"
    )
    pos_stats  = _load_stats_json(stats_path)["positions"]
    disp_stats = _load_stats_json(disp_stats_path)

    cond_overrides = {}
    if args.barrier_label is not None:
        cond_overrides["barrier_material"] = args.barrier_label
    if args.layers is not None:
        cond_overrides["layer"] = args.layers
    if args.kirigami_thickness is not None:
        cond_overrides["kirigami_thickness"] = args.kirigami_thickness
    if args.inter_layer_plate_thickness is not None:
        cond_overrides["inter_layer_plate_thickness"] = args.inter_layer_plate_thickness
    if args.w_beam_thickness is not None:
        cond_overrides["w_beam_thickness"] = args.w_beam_thickness
    if args.speed is not None:
        cond_overrides["speed_kmh"] = args.speed
    if args.angle is not None:
        cond_overrides["angle_deg"] = args.angle

    rmse_per_frame, overall_rmse = rollout_tc(
        model, cfg, args.raw_h5, pos_stats, disp_stats, device, cond_overrides
    )

    time_ref_frames = int(cfg["data"]["time_ref_frames"])
    print(f"[RolloutTC] {Path(args.raw_h5).name}")
    for frame, r in enumerate(rmse_per_frame):
        t_norm = frame / (time_ref_frames - 1)
        print(f"  frame={frame:4d}  t_norm={t_norm:.4f}  rmse={r:.4f} mm")
    print(f"[RolloutTC] Overall RMSE: {overall_rmse:.4f} mm")


if __name__ == "__main__":
    main()
