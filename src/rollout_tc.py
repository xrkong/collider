"""Time-conditioned (TC) rollout — independent per-frame query, no rollout state.

Unlike src/rollout.py's autoregressive/one-step modes, there is no history
window, no teacher forcing, and no state carried between frames: every frame
is one independent forward pass at its own normalized query time. See PLAN
(time-conditioned-transolver) for the full scheme.

Prints per-frame + overall RMSE (physical mm) to console; pass --gif to also
render a pred-vs-gt animation via src/rollout.py's render_vis() — that
renderer only consumes two (T,N,3) position arrays + per-node part metadata,
with no AR-specific state, so it's reused as-is (filter_eroded left at its
default False, which sidesteps render_vis's one AR-specific bit — the
INPUT_FRAMES-offset erosion-mask slice — entirely).

Usage:
    python src/rollout_tc.py \
        --checkpoint outputs/checkpoints/wj11_tc/checkpoint-best.safetensors \
        --experiment configs/experiments/wj11_tc.yaml \
        --raw-h5 /path/to/some_case.h5 \
        --gif
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
from src.rollout import load_model, load_raw_h5, render_vis
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
) -> dict:
    """Per-frame independent query over the scored range.

    Returns a dict:
        rmse_per_frame:             (T,) float64 — per-frame position RMSE, all
            nodes, physical mm
        rmse_per_frame_veh_contact: (T,) float64 or None — same, restricted to
            veh_contact nodes (None if the h5 has no region_label metadata) —
            this is the metric src/rollout.py's AR pipeline treats as the
            "real" number, since the all-node average is diluted by the ~90%
            of far-field nodes that barely move.
        rmse_vel: (T-1,) float64 or None — per-frame velocity RMSE, mm/frame
        rmse_acc: (T-2,) float64 or None — per-frame acceleration RMSE, mm/frame^2
            rmse_vel/rmse_acc are finite-differenced from pred_frames (same
            forward-diff convention as everywhere else: vel[i]=pos[i+1]-pos[i],
            see src/rollout.py's module docstring) purely so they land in the
            same physical units as AR's rmse_vel/rmse_acc and can be compared
            head-to-head. CAVEAT: TC was never trained to predict velocity or
            acceleration, and each frame is an independent forward pass with no
            smoothness constraint against its neighbors — unlike AR, which
            integrates step by step and is smooth by construction. A bad
            number here reflects frame-to-frame jitter between independent
            predictions as much as raw inaccuracy, a different failure mode
            than AR's (whose vel/acc error comes from its own one-step physics
            model, not from stitching together unrelated forward passes).
        overall_rmse: float — position, all nodes (sqrt of mean of rmse_per_frame^2)
        pred_frames:  (T, N, 3) float32 — predicted absolute positions
        raw:          dict from load_raw_h5 (positions, velocity, acceleration,
            node_part_id/name, veh_contact_mask, ...)
    """
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

    vc_mask = raw.get("veh_contact_mask")

    n_frames = min(T, time_ref_frames)
    rmse_per_frame    = np.zeros(n_frames, dtype=np.float64)
    rmse_per_frame_vc = np.zeros(n_frames, dtype=np.float64) if vc_mask is not None else None
    pred_frames = np.zeros((n_frames, N, 3), dtype=np.float32)
    for frame in range(n_frames):
        t_norm = frame / (time_ref_frames - 1)
        t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device)

        pred_disp_norm = model(node_feats_t, node_type_t, t_tensor)   # (N, 3)
        pred_disp = pred_disp_norm.cpu().numpy() * disp_std + disp_mean
        pred_pos  = reference_coords + pred_disp
        pred_frames[frame] = pred_pos

        gt_pos = positions[frame]
        rmse_per_frame[frame] = float(np.sqrt(np.mean((pred_pos - gt_pos) ** 2)))
        if vc_mask is not None:
            rmse_per_frame_vc[frame] = float(
                np.sqrt(np.mean((pred_pos[vc_mask] - gt_pos[vc_mask]) ** 2))
            )

    overall_rmse = float(np.sqrt(np.mean(rmse_per_frame ** 2)))

    # Derived kinematics, physical units — see docstring's comparability caveat.
    rmse_vel = rmse_acc = None
    if n_frames >= 2:
        pred_vel = np.diff(pred_frames, axis=0)              # (n_frames-1, N, 3)
        gt_vel   = raw["velocity"][: n_frames - 1]            # (n_frames-1, N, 3)
        rmse_vel = np.sqrt(np.mean((pred_vel - gt_vel) ** 2, axis=(1, 2)))
        if n_frames >= 3:
            pred_acc = np.diff(pred_vel, axis=0)              # (n_frames-2, N, 3)
            gt_acc   = raw["acceleration"][: n_frames - 2]
            rmse_acc = np.sqrt(np.mean((pred_acc - gt_acc) ** 2, axis=(1, 2)))

    return {
        "rmse_per_frame":             rmse_per_frame,
        "rmse_per_frame_veh_contact": rmse_per_frame_vc,
        "rmse_vel":                   rmse_vel,
        "rmse_acc":                   rmse_acc,
        "overall_rmse":               overall_rmse,
        "pred_frames":                pred_frames,
        "raw":                        raw,
    }


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
    parser.add_argument("--output-dir", default=None,
                        help="Where to write the GIF (default: outputs/rollouts_tc/<h5 stem>)")

    # GIF rendering — reuses src/rollout.py's render_vis() as-is.
    parser.add_argument("--gif",            action="store_true",
                        help="Render a pred-vs-gt GIF animation")
    parser.add_argument("--gif-fps",        type=int, default=10)
    parser.add_argument("--gif-max-frames", type=int, default=200)
    parser.add_argument("--gif-name",       default=None,
                        help="GIF filename stem override (default: the h5's trajectory name)")

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

    result = rollout_tc(
        model, cfg, args.raw_h5, pos_stats, disp_stats, device, cond_overrides
    )
    rmse_per_frame = result["rmse_per_frame"]

    time_ref_frames = int(cfg["data"]["time_ref_frames"])
    print(f"[RolloutTC] {Path(args.raw_h5).name}")
    for frame, r in enumerate(rmse_per_frame):
        t_norm = frame / (time_ref_frames - 1)
        print(f"  frame={frame:4d}  t_norm={t_norm:.4f}  rmse={r:.4f} mm")
    print(f"[RolloutTC] Pos RMSE (all nodes)   = {result['overall_rmse']:.4f} mm")
    rmse_vc = result["rmse_per_frame_veh_contact"]
    if rmse_vc is not None:
        print(f"[RolloutTC] Pos RMSE (veh_contact) = {np.sqrt(np.mean(rmse_vc ** 2)):.4f} mm")
    if result["rmse_vel"] is not None:
        print(f"[RolloutTC] Vel RMSE (derived, all nodes) = {result['rmse_vel'].mean():.4f} mm/frame")
    if result["rmse_acc"] is not None:
        print(f"[RolloutTC] Acc RMSE (derived, all nodes) = {result['rmse_acc'].mean():.4f} mm/frame^2 "
              f"(comparable to AR's rmse_acc — see rollout_tc()'s docstring for the caveat)")

    if args.gif:
        h5_stem = traj_name_from_h5(args.raw_h5)
        out_dir = Path(args.output_dir or PROJECT_ROOT / "outputs" / "rollouts_tc" / h5_stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        gif_stem = args.gif_name or h5_stem
        gif_path = str(out_dir / f"{gif_stem}.gif")

        raw = result["raw"]
        gt_frames = raw["positions"][: len(result["pred_frames"])]
        render_result = {
            "pred_frames": result["pred_frames"],
            "gt_frames":   gt_frames,
            "rmse_pos":    rmse_per_frame,
            "mode":        "time_conditioned",
        }
        if result["rmse_per_frame_veh_contact"] is not None:
            render_result["rmse_pos_veh_contact"] = result["rmse_per_frame_veh_contact"]
        render_vis(
            render_result,
            raw,
            out_path          = gif_path,
            fps               = args.gif_fps,
            max_frames        = args.gif_max_frames,
            dpi               = 120,
            group_config_path = "configs/data/required_parts.config",
        )


if __name__ == "__main__":
    main()
