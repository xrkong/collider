"""Rollout visualization — runs one-step and autoregressive inference on raw h5 data.

Reads the original full-trajectory h5 (251 frames, not pre-windowed),
renders left=pred / right=gt GIFs colored by part, saves PKL + console stats.

Conventions (must match the exporter / loader / trainer):
    * Kinematics are forward finite differences with dt = 1 frame:
          vel[i] = pos[i+1] - pos[i],  acc[i] = vel[i+1] - vel[i]
    * Integration is FORWARD EULER (x uses current v, then v updates):
          x_new = x_last + v_last,  v_new = v_last + a        (dt = 1)
    * Units of velocity / acceleration are therefore mm/dt and mm/dt^2
      (per-frame), NOT mm/s. Multiply by 1/dt_seconds (from metadata) only
      when converting to physical units for an external report.
    * The last 1-2 frames of GT velocity/acceleration are padding (forward
      diff has no valid value there), so evaluation stops at T-2.

Usage:
    python src/rollout.py \
        --checkpoint outputs/checkpoints/lc001/checkpoint-best.safetensors \
        --experiment configs/experiments/lc001.yaml \
        --raw-h5 /home/kong/datasets/barrier/h5_fps/T_lok_F_shape_barrier_9_3_60km.h5 \
        --mode both \
        --gif --gif-fps 10 \
        --gif-name 60kph

    # GT-only GIF — no checkpoint/experiment needed
    python src/rollout.py \
        --raw-h5 /data/curtin_ciraee/curtin_xiangrui/data/h5dt_50ns_5fs_mat/T_lok_F_shape_barrier_9_3_100km_plus800kg/output.h5 \
        --mode raw_gt \
        --gif --gif-fps 10 --gif-name 100kph_plus800kg

"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Literal, Optional
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import models  # noqa: F401
from models.registry import build_model
from src.dataset import NormStats, traj_name_from_h5
from src.conditions import CondConfig, parse_conditions, normalize_conditions

try:
    import h5py
    _H5PY = True
except ImportError:
    _H5PY = False
    print("ERROR: h5py required — pip install h5py")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from PIL import Image
    import io
    _VIS = True
except ImportError:
    _VIS = False
    print("Warning: matplotlib/Pillow not installed — GIF will be skipped")

try:
    from safetensors.torch import load_file as _st_load
    _SAFETENSORS = True
except ImportError:
    _SAFETENSORS = False

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    print("Warning: wandb not installed — GIF upload will be skipped")

# ── Constants ─────────────────────────────────────────────────────────────────
# INPUT_FRAMES is overridden from cfg["data"]["input_frames"] in main().
INPUT_FRAMES  = 5
# dt is fixed to 1: kinematics are per-frame forward differences. The physical
# timestep lives in metadata.json (dt_seconds) and is only for unit conversion.
DT            = 1.0
FEATURES      = ["positions", "velocity", "acceleration"]
FEAT_DIMS     = {"positions": 3, "velocity": 3, "acceleration": 3}
FEAT_SLICES   = {
    "positions":    (0,  3),
    "velocity":     (3,  6),
    "acceleration": (6,  9),
}

FONTSIZE=12

# ── W&B meta helpers ──────────────────────────────────────────────────────────

def load_meta(ckpt_dir: "Path | str") -> dict:
    """Load meta.json written by train.py from a checkpoint directory."""
    meta_path = Path(ckpt_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"meta.json not found in {ckpt_dir}. "
            "Ensure train.py wrote it (requires updated train.py)."
        )
    with open(meta_path) as f:
        return json.load(f)


def _find_weights_file(ckpt_dir: Path) -> Path:
    """Find checkpoint weights (.safetensors or .pt) in a directory."""
    for name_stem in ("checkpoint-best", "checkpoint-latest"):
        for ext in (".safetensors", ".pt"):
            p = ckpt_dir / (name_stem + ext)
            if p.exists():
                return p
    for ext in (".safetensors", ".pt"):
        candidates = sorted(ckpt_dir.glob(f"*{ext}"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"No checkpoint weights (.safetensors / .pt) found in {ckpt_dir}")


# ── Model loading ─────────────────────────────────────────────────────────────

def _build_and_load(model_name: str, cfg: dict, weights_path: Path) -> torch.nn.Module:
    model = build_model(model_name, cfg)
    if weights_path.suffix == ".safetensors" and _SAFETENSORS:
        model.load_state_dict(_st_load(str(weights_path)))
    else:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model


def load_model(checkpoint_path: str, experiment_path: str, device: torch.device):
    """Load model from local checkpoint + experiment yaml."""
    from train import load_config

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg        = load_config(experiment_path)
    model_name = cfg["model"]["name"]

    json_path = ckpt_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        print(f"[Rollout] Checkpoint: {ckpt_path.name}")
        print(f"[Rollout] Step:       {meta.get('step', '?')}")
        print(f"[Rollout] Val loss:   {meta.get('val_loss', '?')}")
        print(f"[Rollout] Git commit: {meta.get('git_commit', '?')}")

    model = _build_and_load(model_name, cfg, ckpt_path)
    model.to(device).eval()
    print(f"[Rollout] Loaded '{model_name}' — device={device}")
    return model, cfg


# ── Raw h5 loading ────────────────────────────────────────────────────────────

def _derive_padded_kinematics(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-diff velocity/acceleration (dt=1) at full T length, tail-padded.

    Mirrors exactly what the legacy exporter (dataset/d3plot_to_h5_dt.py)
    stored directly: vel[i]=pos[i+1]-pos[i], acc[i]=vel[i+1]-vel[i], with the
    last 1-2 frames repeating the last valid value (no real finite difference
    exists there). Used when states/velocity or states/acceleration are
    absent from the h5 (the new dataset/ds/build_dataset.py format only
    stores positions) so every downstream index raw_data["velocity"][t] for
    t in [0, T) behaves identically to the legacy stored arrays.
    """
    T = pos.shape[0]
    pos64 = pos.astype(np.float64)
    vel = np.zeros_like(pos64)
    acc = np.zeros_like(pos64)
    n_vel_valid = max(T - 1, 0)
    n_acc_valid = max(T - 2, 0)
    if n_vel_valid > 0:
        vel[:n_vel_valid] = np.diff(pos64, axis=0)
        vel[n_vel_valid:] = vel[n_vel_valid - 1]
    if n_acc_valid > 0:
        acc[:n_acc_valid] = np.diff(vel[:n_vel_valid], axis=0)
        acc[n_acc_valid:] = acc[n_acc_valid - 1]
    return vel.astype(np.float32), acc.astype(np.float32)


def load_raw_h5(h5_path: str, node_type_field: str | None = None,
                 region_norm_field: str | None = None) -> dict:
    """Load full trajectory and part metadata from raw h5.

    Returns dict with:
        positions:    (T, N, 3)  float32
        velocity:     (T, N, 3)  float32   read directly, or derived from
        acceleration: (T, N, 3)  float32   positions (forward diff, tail-padded)
        node_alive:   (T, N)     bool      erosion mask; all-True if absent
        times:        (T,)       float64
        node_part_id:   (N,)     int64
        node_part_name: (N,)     str
        part_ids:       (P,)     int64     unique part IDs (derived if absent)
        part_names:     (P,)     str
    """
    if not _H5PY:
        raise ImportError("h5py required")

    with h5py.File(h5_path, "r") as f:
        positions = f["states/positions"][:].astype(np.float32)

        if "states/velocity" in f and "states/acceleration" in f:
            velocity     = f["states/velocity"][:].astype(np.float32)
            acceleration = f["states/acceleration"][:].astype(np.float32)
        else:
            velocity, acceleration = _derive_padded_kinematics(positions)

        has_node_alive = "states/node_alive" in f
        node_alive = (
            f["states/node_alive"][:].astype(bool) if has_node_alive
            else np.ones(positions.shape[:2], dtype=bool)
        )

        data = {
            "positions":    positions,
            "velocity":     velocity,
            "acceleration": acceleration,
            "node_alive":   node_alive,
            "has_node_alive": has_node_alive,  # False ⇒ node_alive is a stub (all-True), use compute_erosion_mask instead
            "times":        f["states/times"][:],
            "node_part_id":  f["metadata/node_part_id"][:],
            "node_part_name": np.array([
                n.decode("utf-8").strip("\x00") if isinstance(n, bytes) else str(n)
                for n in f["metadata/node_part_name"][:]
            ]),
        }
        if "metadata/part_ids" in f and "metadata/part_names" in f:
            data["part_ids"]   = f["metadata/part_ids"][:]
            data["part_names"] = np.array([
                n.decode("utf-8").strip("\x00") if isinstance(n, bytes) else str(n)
                for n in f["metadata/part_names"][:]
            ])
        else:
            # New format has no compact unique-part-list arrays — derive them
            # from the per-node id/name (used for visualization grouping only).
            _, first_idx = np.unique(data["node_part_id"], return_index=True)
            order = np.sort(first_idx)
            data["part_ids"]   = data["node_part_id"][order]
            data["part_names"] = data["node_part_name"][order]

        if node_type_field is not None:
            nt_key = f"metadata/{node_type_field}"
            if nt_key not in f:
                raise KeyError(
                    f"{h5_path}: missing node_type field '{nt_key}'. "
                    f"Available metadata fields: {sorted(f['metadata'].keys())}. "
                    f"For dataset/ds/build_dataset.py output, set "
                    f"data.node_type_field: region_id in the experiment config."
                )
            data["node_type"] = f[nt_key][:].astype(np.int64)  # (N,)

        if region_norm_field is not None:
            rid_key = f"metadata/{region_norm_field}"
            if rid_key not in f:
                raise KeyError(
                    f"{h5_path}: missing region field '{rid_key}' required for "
                    f"per-region normalization. Available metadata fields: "
                    f"{sorted(f['metadata'].keys())}."
                )
            data["region_id"] = f[rid_key][:].astype(np.int64)  # (N,)

        # Region label (dataset/ds/build_dataset.py only) — used to restrict
        # RMSE to veh_contact, since the global average over all N nodes is
        # diluted by the ~90% of nodes (far barrier, far vehicle) that barely
        # move and is insensitive to how well the model captures the actual
        # collision dynamics. None for legacy-format h5s without region_label.
        if "metadata/region_label" in f:
            region_label = np.array([
                n.decode("utf-8").strip("\x00") if isinstance(n, bytes) else str(n)
                for n in f["metadata/region_label"][:]
            ])
            data["region_label"]      = region_label
            data["veh_contact_mask"]  = (region_label == "veh_contact")
        else:
            data["region_label"]     = None
            data["veh_contact_mask"] = None

    T, N, _ = data["positions"].shape
    P       = len(data["part_ids"])
    n_eroded_final = int((~data["node_alive"][-1]).sum())
    n_vc = int(data["veh_contact_mask"].sum()) if data["veh_contact_mask"] is not None else 0
    print(f"[Data] {T} frames, {N} nodes, {P} parts"
          + (f", {n_eroded_final} eroded by final frame" if n_eroded_final else "")
          + (f", {n_vc} veh_contact nodes" if n_vc else ""))
    return data


# ── Normalization helpers ─────────────────────────────────────────────────────

def normalize_raw(data: dict, norm_stats: NormStats) -> dict:
    """Z-score normalize all features in-place, return new dict."""
    normed = {}
    for feat in FEATURES:
        normed[feat] = norm_stats.normalize(feat, data[feat])   # (T, N, C)
    return normed

def build_velocity_input(normed_v: np.ndarray, t_last: int) -> torch.Tensor:
    """从归一化速度的 [t_last-4 .. t_last] 5 帧构造 (1, N, 15)。
    Layout 与训练 dataset 完全一致: per-node row = [v_t0_xyz, ..., v_t4_xyz]."""
    frames = normed_v[t_last - INPUT_FRAMES + 1: t_last + 1]      # (5, N, 3)
    N = frames.shape[1]
    x = frames.transpose(1, 0, 2).reshape(N, -1)                  # (N, 15)
    return torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)


def build_velocity_input_from_window(window: np.ndarray) -> torch.Tensor:
    """从 (5, N, 3) 归一化速度窗口构造 (1, N, 15)."""
    N = window.shape[1]
    x = window.transpose(1, 0, 2).reshape(N, -1)                  # (N, 15)
    return torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)

def integrate_accel(
    a_pred_norm: torch.Tensor,    # (N, 3)  模型直接输出
    v_last_phys: np.ndarray,      # (N, 3)  物理量 (per-frame)
    x_last_phys: np.ndarray,      # (N, 3)
    dt:          float,
    norm_stats:  NormStats,
):
    """Forward Euler: x uses CURRENT velocity, then v updates.

    Matches the forward-difference (dt = 1) data convention:
        x_new = x_last + v_last * dt      # use current v first
        v_new = v_last + a      * dt      # then update v
    With dt = 1 this exactly inverts the differencing, so feeding GT acc
    reconstructs GT positions. Returns physical (per-frame) (N, 3) arrays.
    """
    a_phys     = norm_stats.denormalize("acceleration", a_pred_norm.cpu().numpy())
    x_new_phys = x_last_phys + v_last_phys * dt      # forward Euler: current v
    v_new_phys = v_last_phys + a_phys     * dt       # then update v
    return a_phys, v_new_phys, x_new_phys


def _pack_pos_only(pos_list):
    """把 list-of-(N,3) 打包成 (T, N, 15),只填位置通道,兼容现有渲染。"""
    T = len(pos_list)
    N = pos_list[0].shape[0]
    arr = np.zeros((T, N, 15), dtype=np.float32)
    for i, p in enumerate(pos_list):
        arr[i, :, 0:3] = p
    return arr

# ── Barrier plate parameters (must match train.py BARRIER_PARAMS) ─────────────
# Source: README "Barrier plate projection on xy plate" table
BARRIER_PARAMS: dict[float, dict[str, float]] = {
    -25.4: {"x_intercept": 2056.579},
    -20.0: {"x_intercept": 2801.525},
    -15.0: {"x_intercept": 4078.004},
}
_DEFAULT_BARRIER_DEG: float = -25.4

# ── Signed Distance Field ─────────────────────────────────────────────────────

def compute_sdf_batch(
    xy:                torch.Tensor,
    barrier_angle_deg: float = _DEFAULT_BARRIER_DEG,
    x_intercept:       float = BARRIER_PARAMS[_DEFAULT_BARRIER_DEG]["x_intercept"],
) -> torch.Tensor:
    """Signed distance (metres) from each point to the barrier line.

    xy: (..., 2)  XY positions in mm
    barrier_angle_deg: impact angle (see BARRIER_PARAMS)
    x_intercept: x-coord (mm) where barrier line crosses y = 0
    """
    device    = xy.device
    anchor    = torch.tensor([x_intercept, 0.0], device=device)
    angle_rad = torch.deg2rad(torch.tensor(barrier_angle_deg, device=device))
    normal_2d = torch.tensor(
        [-torch.sin(angle_rad), torch.cos(angle_rad)], device=device
    )
    diff_2d   = xy - anchor
    return (diff_2d * normal_2d).sum(dim=-1) / 1000.0


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_onestep(model, raw_data, normed, norm_stats, device,
                node_type=None, cond_t=None) -> dict:
    T      = raw_data["positions"].shape[0]
    T_eval = T - 2          # last 2 frames have padded GT vel/acc (forward diff)
    print(f"[One-step] dt = 1 (per-frame), steps = {T_eval - INPUT_FRAMES}")

    normed_v = normed["velocity"]                       # (T, N, 3)
    N        = normed_v.shape[1]
    pred_pos_list, gt_pos_list, rmse_pos_steps = [], [], []
    rmse_vel_steps, rmse_acc_steps = [], []
    rmse_pos_vc_steps = []   # pos RMSE restricted to veh_contact nodes
    pred_acc_list, gt_acc_list = [], []
    pred_acc_norm_list, gt_acc_norm_list = [], []

    # veh_contact-only RMSE: the all-node average is diluted by the ~90% of
    # nodes (far barrier, far vehicle) that barely move, so it's insensitive
    # to how well the model captures the actual collision dynamics.
    vc_mask = raw_data.get("veh_contact_mask")
    if vc_mask is None:
        print("[One-step] WARNING: no region_label in this h5 — "
              "veh_contact RMSE unavailable, falling back to all-node RMSE")
        vc_mask = np.ones(N, dtype=bool)

    # Method A: broadcast cond once, reuse every step (D3). Empty tensor when n_cond=0.
    if cond_t is None:
        cond_t = torch.zeros(0)
    cond_b = cond_t[None, None, :].expand(1, N, -1).to(device)   # (1, N, n_cond)

    for t in range(INPUT_FRAMES, T_eval):
        x_vel_flat  = build_velocity_input(normed_v, t - 1).to(device)  # (1, N, T_in*3)

        input_pos = raw_data["positions"][t - INPUT_FRAMES + 1: t + 1].transpose(1, 0, 2)  # (N, T_in, 3)
        x_sdf = compute_sdf_batch(
            torch.from_numpy(input_pos[..., 0:2])
        ).to(device).unsqueeze(0)                                        # (1, N, T_in)

        x_in = torch.cat([x_vel_flat, x_sdf, cond_b], dim=-1)           # (1, N, T_in*4 + n_cond)

        # Erosion mask: zero out input features for nodes already eroded at
        # the last input frame, before the model sees them (mirrors training).
        alive_t = torch.from_numpy(raw_data["node_alive"][t - 1].astype(np.float32))
        x_in = x_in * alive_t.to(device).view(1, -1, 1)

        a_pred_norm = model(x_in, node_type).squeeze(0)  # (N, 3)

        # 上一帧 GT 速度 / 位置 (物理量) — one-step 模式始终用 GT
        v_last = raw_data["velocity"][t - 1]
        x_last = raw_data["positions"][t - 1]

        a_phys, v_new, x_new = integrate_accel(a_pred_norm, v_last, x_last, DT, norm_stats)

        x_gt = raw_data["positions"][t]
        rmse = float(np.sqrt(np.mean((x_new - x_gt) ** 2)))
        rmse_vc = float(np.sqrt(np.mean((x_new[vc_mask] - x_gt[vc_mask]) ** 2)))

        v_gt = raw_data["velocity"][t]
        a_gt = raw_data["acceleration"][t]
        rmse_vel_steps.append(float(np.sqrt(np.mean((v_new  - v_gt) ** 2))))
        rmse_acc_steps.append(float(np.sqrt(np.mean((a_phys - a_gt) ** 2))))
        pred_acc_list.append(a_phys)
        gt_acc_list.append(a_gt)
        pred_acc_norm_list.append(a_pred_norm.cpu().numpy())
        gt_acc_norm_list.append(norm_stats.normalize("acceleration", a_gt))

        pred_pos_list.append(x_new)
        gt_pos_list.append(x_gt)
        rmse_pos_steps.append(rmse)
        rmse_pos_vc_steps.append(rmse_vc)

        if (t - INPUT_FRAMES + 1) % 50 == 0:
            print(f"  step {t-INPUT_FRAMES+1}/{T_eval-INPUT_FRAMES} | "
                  f"acc_rmse={rmse_acc_steps[-1]:.4f} mm/dt² | "
                  f"pos_rmse_veh_contact={rmse_vc:.2f} mm")

    pred_acc_all = np.stack(pred_acc_list)   # (T_steps, N, 3)
    gt_acc_all   = np.stack(gt_acc_list)
    rmse_acc     = np.array(rmse_acc_steps)
    print(f"[One-step] GT   acc |mean| = {np.abs(gt_acc_all).mean():.4f} mm/dt²")
    print(f"[One-step] Pred acc |mean| = {np.abs(pred_acc_all).mean():.4f} mm/dt²")
    print(f"[One-step] Acc RMSE mean   = {rmse_acc.mean():.4f} mm/dt²")
    print(f"[One-step] Acc RMSE/GT std = {rmse_acc.mean() / gt_acc_all.std():.3f}")
    if norm_stats._acc_scale is not None:
        pred_norm_all = np.stack(pred_acc_norm_list)   # (steps, N, 3)
        gt_norm_all   = np.stack(gt_acc_norm_list)
        rmse_asinh    = np.sqrt(np.mean((pred_norm_all - gt_norm_all) ** 2))
        rmse_physical = np.sqrt(np.mean((pred_acc_all  - gt_acc_all ) ** 2))
        print(f"[One-step] RMSE norm = {rmse_asinh:.6f}")
        print(f"[One-step] RMSE physical    = {rmse_physical:.4f} mm/dt²")
        print(f"[One-step] Amplification    = {rmse_physical / rmse_asinh:.1f}×")

    rmse_pos_vc = np.array(rmse_pos_vc_steps)
    print(f"[One-step] Pos RMSE (all nodes)   = {np.mean(rmse_pos_steps):.2f} mm")
    print(f"[One-step] Pos RMSE (veh_contact) = {rmse_pos_vc.mean():.2f} mm")

    return {
        "pred_frames": _pack_pos_only(pred_pos_list),
        "gt_frames":   _pack_pos_only(gt_pos_list),
        "rmse_pos":    np.array(rmse_pos_steps),
        "rmse_pos_veh_contact": rmse_pos_vc,
        "rmse_vel":    np.array(rmse_vel_steps),
        "rmse_acc":    rmse_acc,
        "mode":        "onestep",
    }

@torch.no_grad()
def run_autoregressive(model, raw_data, normed, norm_stats, device,
                       node_type=None, cond_t=None) -> dict:
    T      = raw_data["positions"].shape[0]
    T_eval = T - 2          # last 2 frames have padded GT vel/acc (forward diff)
    print(f"[Autoregressive] dt = 1 (per-frame), steps = {T_eval - INPUT_FRAMES}")

    pred_pos_list, gt_pos_list, rmse_pos_steps = [], [], []
    rmse_vel_steps, rmse_acc_steps = [], []
    rmse_pos_vc_steps = []   # pos RMSE restricted to veh_contact nodes
    pred_acc_list, gt_acc_list = [], []
    pred_acc_norm_list, gt_acc_norm_list = [], []

    # ── 初始化 ──
    # 速度窗口 (归一化, 模型输入用)
    v_window_norm = normed["velocity"][:INPUT_FRAMES].copy()       # (5, N, 3)
    x_window_phys = raw_data["positions"][:INPUT_FRAMES].copy()
    N = v_window_norm.shape[1]
    # 物理速度 / 位置当前状态
    v_phys = raw_data["velocity"][INPUT_FRAMES - 1].copy()         # (N, 3)
    x_phys = raw_data["positions"][INPUT_FRAMES - 1].copy()        # (N, 3)

    # veh_contact-only RMSE: see run_onestep for rationale.
    vc_mask = raw_data.get("veh_contact_mask")
    if vc_mask is None:
        print("[Autoregressive] WARNING: no region_label in this h5 — "
              "veh_contact RMSE unavailable, falling back to all-node RMSE")
        vc_mask = np.ones(N, dtype=bool)

    # Method A: broadcast cond once, reuse every step (D3). Empty tensor when n_cond=0.
    if cond_t is None:
        cond_t = torch.zeros(0)
    cond_b = cond_t[None, None, :].expand(1, N, -1).to(device)   # (1, N, n_cond)

    for t in range(INPUT_FRAMES, T_eval):
        x_vel_flat  = build_velocity_input_from_window(v_window_norm).to(device)  # (1, N, T_in*3)

        # SDF uses rolling position window
        x_sdf = compute_sdf_batch(
            torch.from_numpy(x_window_phys[..., 0:2].transpose(1, 0, 2)).float()
        ).to(device).unsqueeze(0)                                                   # (1, N, T_in)

        x_in = torch.cat([x_vel_flat, x_sdf, cond_b], dim=-1)                     # (1, N, T_in*4 + n_cond)

        # Erosion mask: zero out input features for nodes already eroded at
        # the last input frame, before the model sees them (mirrors training).
        # Uses the GT erosion timeline since the model doesn't predict erosion.
        alive_t = torch.from_numpy(raw_data["node_alive"][t - 1].astype(np.float32))
        x_in = x_in * alive_t.to(device).view(1, -1, 1)

        a_pred_norm = model(x_in, node_type).squeeze(0)  # (N, 3)

        a_phys_new, v_phys_new, x_phys_new = integrate_accel(
            a_pred_norm, v_phys, x_phys, DT, norm_stats)

        x_gt = raw_data["positions"][t]
        rmse = float(np.sqrt(np.mean((x_phys_new - x_gt) ** 2)))
        rmse_vc = float(np.sqrt(np.mean((x_phys_new[vc_mask] - x_gt[vc_mask]) ** 2)))

        v_gt = raw_data["velocity"][t]
        a_gt = raw_data["acceleration"][t]
        rmse_vel_steps.append(float(np.sqrt(np.mean((v_phys_new - v_gt) ** 2))))
        rmse_acc_steps.append(float(np.sqrt(np.mean((a_phys_new - a_gt) ** 2))))
        pred_acc_list.append(a_phys_new)
        gt_acc_list.append(a_gt)
        pred_acc_norm_list.append(a_pred_norm.cpu().numpy())
        gt_acc_norm_list.append(norm_stats.normalize("acceleration", a_gt))

        pred_pos_list.append(x_phys_new)
        gt_pos_list.append(x_gt)
        rmse_pos_steps.append(rmse)
        rmse_pos_vc_steps.append(rmse_vc)

        # ── 更新状态 ──
        v_phys = v_phys_new
        x_phys = x_phys_new
        # 把新速度归一化后滑窗
        v_new_norm = norm_stats.normalize("velocity", v_phys_new[None])[0]   # (N, 3)
        v_window_norm = np.concatenate(
            [v_window_norm[1:], v_new_norm[None]], axis=0)                   # (5, N, 3)
        # 把预测位置滑窗 (用于 SDF 计算)
        x_window_phys = np.concatenate(
            [x_window_phys[1:], x_phys_new[None]], axis=0)                  # (5, N, 3)

        if (t - INPUT_FRAMES + 1) % 50 == 0:
            print(f"  step {t-INPUT_FRAMES+1}/{T_eval-INPUT_FRAMES} | "
                  f"acc_rmse={rmse_acc_steps[-1]:.4f} mm/dt² | "
                  f"pos_rmse_veh_contact={rmse_vc:.2f} mm")

    pred_acc_all = np.stack(pred_acc_list)   # (T_steps, N, 3)
    gt_acc_all   = np.stack(gt_acc_list)
    rmse_acc     = np.array(rmse_acc_steps)
    print(f"[Autoregressive] GT   acc |mean| = {np.abs(gt_acc_all).mean():.4f} mm/dt²")
    print(f"[Autoregressive] Pred acc |mean| = {np.abs(pred_acc_all).mean():.4f} mm/dt²")
    print(f"[Autoregressive] Acc RMSE mean   = {rmse_acc.mean():.4f} mm/dt²")
    print(f"[Autoregressive] Acc RMSE/GT std = {rmse_acc.mean() / gt_acc_all.std():.3f}")
    if norm_stats._acc_scale is not None:
        pred_norm_all = np.stack(pred_acc_norm_list)   # (steps, N, 3)
        gt_norm_all   = np.stack(gt_acc_norm_list)
        rmse_asinh    = np.sqrt(np.mean((pred_norm_all - gt_norm_all) ** 2))
        rmse_physical = np.sqrt(np.mean((pred_acc_all  - gt_acc_all ) ** 2))
        print(f"[Autoregressive] RMSE asinh-space = {rmse_asinh:.6f}")
        print(f"[Autoregressive] RMSE physical    = {rmse_physical:.4f} mm/dt²")
        print(f"[Autoregressive] Amplification    = {rmse_physical / rmse_asinh:.1f}×")

    rmse_pos_vc = np.array(rmse_pos_vc_steps)
    print(f"[Autoregressive] Pos RMSE (all nodes)   = {np.mean(rmse_pos_steps):.2f} mm")
    print(f"[Autoregressive] Pos RMSE (veh_contact) = {rmse_pos_vc.mean():.2f} mm")

    return {
        "pred_frames": _pack_pos_only(pred_pos_list),
        "gt_frames":   _pack_pos_only(gt_pos_list),
        "rmse_pos":    np.array(rmse_pos_steps),
        "rmse_pos_veh_contact": rmse_pos_vc,
        "rmse_vel":    np.array(rmse_vel_steps),
        "rmse_acc":    rmse_acc,
        "mode":        "autoregressive",
    }

# ── Last-frame baseline ───────────────────────────────────────────────────────

def compute_frozen_baseline(raw_data: dict) -> dict:
    T   = raw_data["positions"].shape[0]
    pos = raw_data["positions"]   # (T, N, 3)

    rmse_onestep, rmse_rollout = [], []
    last = pos[INPUT_FRAMES - 1]
    # frozen baseline
    for t in range(INPUT_FRAMES, T - 2):
        rmse_onestep.append(np.sqrt(np.mean((pos[t - 1] - pos[t]) ** 2)))
        rmse_rollout.append(np.sqrt(np.mean((last      - pos[t]) ** 2)))

    return {
        "rmse_pos_onestep": np.array(rmse_onestep),
        "rmse_pos_rollout": np.array(rmse_rollout),
    }

def compute_baseline(raw_data: dict, dt: float, input_frames: int) -> dict:
    pos = raw_data["positions"]                       # (T, N, 3)
    vel = raw_data["velocity"]                      # (T, N, 3)
    acc = raw_data["acceleration"]                   # (T, N, 3)
    T = pos.shape[0]
    IF = input_frames

    # veh_contact-only RMSE: see run_onestep for rationale. Falls back to all
    # nodes for legacy h5s without region_label, same as the model-side metric.
    vc_mask = raw_data.get("veh_contact_mask")
    if vc_mask is None:
        vc_mask = np.ones(pos.shape[1], dtype=bool)

    def rms(x):  # RMS over nodes & dims for one frame
        return np.sqrt(np.mean(x ** 2))

    def rms_vc(x):  # RMS restricted to veh_contact nodes
        return np.sqrt(np.mean(x[vc_mask] ** 2))

    out = {k: [] for k in [
        "rmse_pos_onestep", "rmse_vel_onestep", "rmse_acc_onestep",
        "rmse_pos_rollout", "rmse_vel_rollout", "rmse_acc_rollout",
        "rmse_pos_onestep_veh_contact", "rmse_pos_rollout_veh_contact",
    ]}

    # rollout: acc=0 → 速度冻结、位置匀速外推
    x0, v0 = pos[IF - 1], vel[IF - 1]

    # stop at T-2: last 1-2 GT vel/acc frames are forward-diff padding
    for t in range(IF, T - 2):
        # ---- acc=0 预测 → 加速度误差就是真值加速度的 RMS（两模式相同）----
        acc_err = rms(acc[t])                          # pred=0
        out["rmse_acc_onestep"].append(acc_err)
        out["rmse_acc_rollout"].append(acc_err)

        # ---- one-step：每步喂 GT 上一帧 (forward Euler, dt=1) ----
        # x_pred = pos[t-1] + vel[t-1]*dt；因 vel 为前向差分，结果≈pos[t]，故 pos_rmse≈0
        v_pred_os = vel[t - 1]                          # a=0 → v 沿用 GT 上一帧
        x_pred_os = pos[t - 1] + v_pred_os * dt
        out["rmse_vel_onestep"].append(rms(v_pred_os - vel[t]))
        out["rmse_pos_onestep"].append(rms(x_pred_os - pos[t]))
        out["rmse_pos_onestep_veh_contact"].append(rms_vc(x_pred_os - pos[t]))

        # ---- rollout：从最后输入帧自我递推 (constant-velocity) ----
        steps = t - (IF - 1)
        x_pred_rl = x0 + v0 * dt * steps                # 匀速外推
        out["rmse_vel_rollout"].append(rms(v0 - vel[t]))     # v 冻结
        out["rmse_pos_rollout"].append(rms(x_pred_rl - pos[t]))
        out["rmse_pos_rollout_veh_contact"].append(rms_vc(x_pred_rl - pos[t]))

    return {k: np.array(v) for k, v in out.items()}

# ── RMSE plot ────────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family":     "DejaVu Serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
_FEAT_NAMES  = ["Position",    "Velocity",    "Acceleration"]
_FEAT_UNITS  = ["mm",          "mm/dt",       "mm/dt²"]
_FEAT_SCALES = [1.0,           1.0,           1.0]    # per-frame units, no conversion
_RMSE_KEYS   = ["rmse_pos",    "rmse_vel",    "rmse_acc"]
_COL_LABELS  = ["One-step",    "Autoregressive"]


def _rmse_axes(axs, row, col, feat, unit, col_label):
    ax = axs[row, col]
    ax.set_title(f"{feat} ({unit}) — {col_label}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"RMSE ({unit})")
    ax.grid(True, alpha=0.3, linestyle=":")
    return ax


def plot_rmse_vs_timestep(onestep: dict | None, autoreg: dict | None, out_dir: Path):
    """3 × 2 grid: rows = pos / vel / acc, cols = one-step / autoregressive."""
    if not _VIS:
        print("Warning: matplotlib/Pillow not available — skipping RMSE plot")
        return

    plt.rcParams.update(_RCPARAMS)
    col_data = [onestep, autoreg]

    fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Rollout RMSE vs Timestep", fontsize=13, fontfamily="DejaVu Serif")

    for row, (feat, unit, scale, rkey) in enumerate(
            zip(_FEAT_NAMES, _FEAT_UNITS, _FEAT_SCALES, _RMSE_KEYS)):
        for col, (result, col_label) in enumerate(zip(col_data, _COL_LABELS)):
            ax = _rmse_axes(axs, row, col, feat, unit, col_label)
            if result is None or rkey not in result:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                continue
            rmse  = result[rkey] * scale
            steps = np.arange(1, len(rmse) + 1)
            ax.plot(steps, rmse, color="#1f77b4", linestyle="-", linewidth=1.5,
                    label="all nodes" if rkey == "rmse_pos" else None)
            # Position row: overlay the veh_contact-only RMSE — the all-node
            # curve above is diluted by far-field nodes that barely move.
            if rkey == "rmse_pos" and "rmse_pos_veh_contact" in result:
                rmse_vc = result["rmse_pos_veh_contact"] * scale
                ax.plot(steps, rmse_vc, color="#d62728", linestyle="-", linewidth=1.5,
                        label="veh_contact")
                ax.legend(fontsize=8)

    out_path = out_dir / "rmse_vs_timestep.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] RMSE plot saved → {out_path}")


def plot_os_ar_veh_contact_rmse(
    onestep: dict | None,
    autoreg: dict | None,
    out_path: "str | Path",
) -> None:
    """Plot one-step vs autoregressive veh_contact pos RMSE on a dual-axis plot.

    One-step error is always far smaller than autoregressive (rollout) error
    — one-step is reset to ground truth every frame, autoregressive
    accumulates drift — so sharing a y-axis flattens the one-step curve to
    near-zero. Left axis = OS scale, right axis = AR scale, so both curves
    are readable on their own terms.

    Falls back to the all-node rmse_pos if rmse_pos_veh_contact isn't in a
    result (legacy pkl from before that field existed, or a legacy-format h5
    without region_label).
    """
    if not _VIS:
        print("Warning: matplotlib/Pillow not available — skipping plot")
        return
    if onestep is None and autoreg is None:
        print("[Plot] Nothing to plot — both onestep and autoreg are None")
        return

    def _series(result):
        if result is None:
            return None
        key = "rmse_pos_veh_contact" if "rmse_pos_veh_contact" in result else "rmse_pos"
        if key == "rmse_pos":
            print("[Plot] WARNING: rmse_pos_veh_contact not in this pkl — "
                  "falling back to all-node rmse_pos")
        return result[key], key

    plt.rcParams.update(_RCPARAMS)
    fig, ax_os = plt.subplots(figsize=(9, 5), constrained_layout=True)

    handles, labels = [], []

    os_series = _series(onestep)
    if os_series is not None:
        rmse_os, key_os = os_series
        steps_os = np.arange(1, len(rmse_os) + 1)
        line_os, = ax_os.plot(steps_os, rmse_os, color="#1f77b4", linewidth=1.6,
                              label=f"one-step ({key_os})")
        ax_os.set_ylabel("OS pos RMSE (mm)", color="#1f77b4")
        ax_os.tick_params(axis="y", labelcolor="#1f77b4")
        handles.append(line_os)
        labels.append(line_os.get_label())
    else:
        ax_os.set_ylabel("OS pos RMSE (mm)")

    ax_os.set_xlabel("Timestep")
    ax_os.grid(True, alpha=0.3, linestyle=":")

    ar_series = _series(autoreg)
    if ar_series is not None:
        rmse_ar, key_ar = ar_series
        steps_ar = np.arange(1, len(rmse_ar) + 1)
        ax_ar = ax_os.twinx()
        line_ar, = ax_ar.plot(steps_ar, rmse_ar, color="#d62728", linewidth=1.6,
                              label=f"autoregressive ({key_ar})")
        ax_ar.set_ylabel("AR pos RMSE (mm)", color="#d62728")
        ax_ar.tick_params(axis="y", labelcolor="#d62728")
        handles.append(line_ar)
        labels.append(line_ar.get_label())

    ax_os.legend(handles, labels, loc="upper left", fontsize=9)
    fig.suptitle("One-step vs Autoregressive — veh_contact pos RMSE", fontsize=13,
                fontfamily="DejaVu Serif")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] OS/AR dual-axis RMSE plot saved → {out_path}")


def plot_multi_rmse(
    experiments: list[tuple[str, "dict | None", "dict | None"]],
    out_path: "str | Path",
):
    """Compare RMSE vs timestep across multiple checkpoints / experiments.

    Args:
        experiments: list of (label, onestep_result, autoreg_result).
                     Either result dict can be None if that mode was not run.
        out_path:    output PNG file path.
    """
    if not _VIS:
        print("Warning: matplotlib/Pillow not available — skipping multi-RMSE plot")
        return

    plt.rcParams.update(_RCPARAMS)

    n      = len(experiments)
    cmap   = plt.get_cmap("tab10", max(n, 1))
    colors = [cmap(i) for i in range(n)]
    lstyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("RMSE vs Timestep — Multi-Experiment Comparison",
                 fontsize=13, fontfamily="DejaVu Serif")

    for row, (feat, unit, scale, rkey) in enumerate(
            zip(_FEAT_NAMES, _FEAT_UNITS, _FEAT_SCALES, _RMSE_KEYS)):
        for col, col_label in enumerate(_COL_LABELS):
            ax = _rmse_axes(axs, row, col, feat, unit, col_label)
            plotted = False
            for i, (label, onestep, autoreg) in enumerate(experiments):
                result = onestep if col == 0 else autoreg
                if result is None or rkey not in result:
                    continue
                rmse  = result[rkey] * scale
                steps = np.arange(1, len(rmse) + 1)
                ax.plot(steps, rmse,
                        color=colors[i],
                        linestyle=lstyles[i % len(lstyles)],
                        linewidth=1.5,
                        label=label)
                plotted = True
            if plotted:
                ax.legend(loc="best")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Multi-experiment RMSE plot saved → {out_path}")


# ── GIF rendering ─────────────────────────────────────────────────────────────
import os
import re


def compute_erosion_mask(
    positions: np.ndarray,
    velocity_threshold: float = 1500.0,
    max_yz_drift: float = 8000.0,
) -> np.ndarray:
    """Return a bool mask (T, N) — False once a node is classified as eroded.

    Two complementary criteria (both use monotone "once eroded, stays eroded"):

    1. Velocity threshold: inter-frame displacement > velocity_threshold mm/frame.
       Catches fast runaways (steel-tube/T-lok barrier weights erode at ~4245 mm/frame).
       Normal crash-zone nodes stay below ~700 mm/frame.

    2. Y/Z cumulative-drift filter: |pos[t,y/z] - pos[0,y/z]| > max_yz_drift.
       Catches slower runaways (~250-550 mm/frame) whose elements erode while
       carrying the crash velocity as free-body motion.  Over 50 frames a node
       moving at 300 mm/frame drifts 15 000 mm — 2× the car's width.  X drift
       is not filtered because the car legitimately travels far in X during impact.

    Both filters are needed: the f-shape offset barrier in this simulation creates
    large lateral forces that eject front-end parts sideways at crash velocity,
    which is indistinguishable from legitimate crash-zone motion by velocity alone.
    """
    T, N, _ = positions.shape
    valid   = np.ones((T, N), dtype=bool)
    ref_yz  = positions[0, :, 1:3].copy()   # (N, 2)  initial Y and Z

    for t in range(1, T):
        # criterion 1 — velocity
        speed = np.linalg.norm(positions[t] - positions[t - 1], axis=1)  # (N,)
        # criterion 2 — cumulative Y/Z drift from initial position
        yz_drift = np.abs(positions[t, :, 1:3] - ref_yz).max(axis=1)    # (N,)
        valid[t] = valid[t - 1] & (speed <= velocity_threshold) & (yz_drift <= max_yz_drift)

    n_eroded = int((~valid[-1]).sum())
    if n_eroded > 0:
        print(f"[ErosionMask] {n_eroded}/{N} nodes masked by frame {T-1} "
              f"(vel>{velocity_threshold:.0f} mm/frame OR Y/Z drift>{max_yz_drift:.0f} mm)")
    return valid  # True = still valid

def _get_grouped_colormap(node_part_id, node_part_name, config_path):
    """根据自定义纯文本格式的 required_parts.config 将节点按组分配颜色"""
    group_dict = {}
    ordered_groups = []  # 记录组名的先后顺序，保证图例美观
    
    if config_path and Path(config_path).exists():
        current_group = "Other"
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 1. 匹配分组标题，例如 "# 3. Front crash structure / load path"
                # 正则解析：匹配以 # 开头，跟着任意空白，再跟着数字和一个点，提取后面的所有文字
                header_match = re.match(r"^#\s*\d+\.\s*(.+)", line)
                if header_match:
                    current_group = header_match.group(1).strip()
                    if current_group not in ordered_groups:
                        ordered_groups.append(current_group)
                    continue
                
                # 2. 忽略其他普通注释和无意义的分割线 (如 # ======)
                if line.startswith('#'):
                    continue
                
                # 3. 记录零件及其所属组别
                part_name = line
                group_dict[part_name] = current_group
    else:
        print(f"Warning: Group config not found at {config_path}. Using default part IDs.")
        return _get_part_colormap(node_part_id)

    # 确保 "Other" 组存在并放在图例最后
    if "Other" not in ordered_groups:
        ordered_groups.append("Other")
        
    # 生成颜色映射表 (根据实际组的数量分配离散颜色)
    cmap = plt.get_cmap("tab20", len(ordered_groups))
    group_to_color = {g: cmap(i) for i, g in enumerate(ordered_groups)}
    
    # 初始化所有节点的颜色数组
    colors = np.zeros((len(node_part_id), 4))
    unique_ids = np.unique(node_part_id)
    
    for pid in unique_ids:
        mask = node_part_id == pid
        if not mask.any(): continue
        
        pname = node_part_name[mask][0]
        assigned_group = None
        
        # 查找所属组别
        # 策略A：精确匹配
        if pname in group_dict:
            assigned_group = group_dict[pname]
        else:
            # 策略B：子串匹配（考虑到仿真软件导出的零件名可能带有 "ID_" 前缀等）
            for cfg_part, g_name in group_dict.items():
                if cfg_part in pname:
                    assigned_group = g_name
                    break
        
        # 如果都没找到，归入 Other
        if assigned_group is None:
            assigned_group = "Other"
            
        colors[mask] = group_to_color[assigned_group]
        
    return colors, group_to_color

def _save_legend_svg(group_to_color, out_path):
    """将图例单独渲染为无背景的 SVG 文件"""
    fig = plt.figure(figsize=(3, len(group_to_color) * 0.3))
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=c, 
               markersize=10, label=g)
        for g, c in group_to_color.items()
    ]
    fig.legend(handles=legend_elements, loc="center", 
               fontsize=10, frameon=False, labelcolor="black")
    plt.axis('off')
    
    # 确保保存为 SVG
    svg_path = str(Path(out_path).with_suffix('.svg'))
    fig.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[Legend] Standalone SVG legend saved → {svg_path}")

def _get_part_colormap(node_part_id: np.ndarray):
    """Build stable part colormap from node_part_id array.

    Returns:
        colors:    (N, 4) RGBA per node, fixed across all frames
        legend_elements: list of Line2D for legend
        unique_ids: sorted unique part IDs
    """
    unique_ids   = np.unique(node_part_id)
    n_parts      = len(unique_ids)
    cmap         = plt.get_cmap("tab20", n_parts)
    id_to_color  = {pid: cmap(i) for i, pid in enumerate(unique_ids)}
    colors       = np.array([id_to_color[pid] for pid in node_part_id])  # (N, 4)
    return colors, id_to_color, unique_ids


def render_vis(
    result:           dict,
    raw_data:         dict,
    out_path:         str,
    fps:              int   = 10,
    max_frames:       int   = 200,
    dpi:              int   = 300,  # 默认提高到 300 保证清晰度
    group_config_path: str  = None, # 传入你的 required_parts.config 路径
    save_png_dir:     str   = None, # 如果传入路径，则额外保存高清 PNG 序列
    filter_eroded:    bool  = False, # 过滤侵蚀节点；False = 关闭过滤
):
    """Render pred (left) vs gt (right) animation, with optional PNG export and grouped coloring."""
    if not _VIS:
        print("Warning: matplotlib/Pillow not available — skipping rendering")
        return

    pred_frames = result["pred_frames"]
    gt_frames   = result["gt_frames"]
    # veh_contact-only RMSE for the on-frame title — the all-node average is
    # diluted by far-field nodes that barely move and reads as ~0.0mm at this
    # precision for most steps; falls back to all-node if unavailable.
    rmse_pos    = result.get("rmse_pos_veh_contact", result["rmse_pos"])
    mode        = result["mode"]
    T           = min(len(pred_frames), max_frames)

    node_part_id   = raw_data["node_part_id"]
    node_part_name = raw_data["node_part_name"]

    pred_pos = pred_frames[:T, :, 0:3]
    gt_pos   = gt_frames[:T,   :, 0:3]

    # Exact raw limits (No padding) to avoid distortion
    all_pos  = np.concatenate([pred_pos, gt_pos], axis=0)
    # x_range  = (all_pos[:, :, 0].min(), all_pos[:, :, 0].max())
    # y_range  = (all_pos[:, :, 1].min(), all_pos[:, :, 1].max())
    # z_range  = (all_pos[:, :, 2].min(), all_pos[:, :, 2].max())
    x_range = (-14000, 20000)
    y_range = (-10000, 8000)
    z_range = (-500, 4000)

    # Visual Setup
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size']   = 12

    # 按各行的数据纵向跨度分配行高,使 equal-aspect 下各行填满格子、消除空白
    x_span = x_range[1] - x_range[0]
    z_span = z_range[1] - z_range[0]      # X-Z 行 (上): ~4500
    y_span = y_range[1] - y_range[0]      # X-Y 行 (下): ~18000

    fig_w     = 12.0
    n_cols    = 2
    col_w     = fig_w / n_cols
    upi       = x_span / col_w             # units per inch (x 方向)
    plot_h    = (z_span + y_span) / upi    # 两排内容真实总高
    fig_h     = plot_h + 1.6               # +1.6 给标题/轴标签留边

    fig, axs = plt.subplots(
        2, 2,
        figsize=(fig_w, fig_h), dpi=dpi,
        gridspec_kw={
            "height_ratios": [z_span, y_span],
            "hspace": 0.18,
            "wspace": 0.18,
        },
    )
    fig.patch.set_facecolor("white")
    

    if group_config_path:
        # 采用按 Config 分组的着色方案
        node_colors , group_to_color = _get_grouped_colormap(node_part_id, node_part_name, group_config_path)
        # 生成独立 SVG 图例
        legend_out = Path(out_path).parent / f"{mode}_legend.svg"
        _save_legend_svg(group_to_color, legend_out)
    else:
        # 回退到原始策略
        node_colors, _, _ = _get_part_colormap(node_part_id)


    # ── Axes Setup (Equal aspect ratio for NO distortion) ──────────────────
    for ax in axs.flat:
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=12)
        ax.set_aspect('equal', adjustable='box')
        
    for i in range(2):
        axs[i, 0].set_xlim(x_range)
        axs[i, 1].set_xlim(x_range)
        if i == 0:  # Top row: X-Z plane
            axs[i, 0].set_ylim(z_range)
            axs[i, 1].set_ylim(z_range)
        else:       # Bottom row: X-Y plane
            axs[i, 0].set_ylim(y_range)
            axs[i, 1].set_ylim(y_range)

    axs[0, 0].set_ylabel("Z", color="black", fontsize=FONTSIZE)
    axs[1, 0].set_ylabel("Y", color="black", fontsize=FONTSIZE)
    axs[1, 0].set_xlabel("X", color="black", fontsize=FONTSIZE)
    axs[1, 1].set_xlabel("X", color="black", fontsize=FONTSIZE)

    # ── Initialize Scatter Plots ───────────────────────────────────────────
    scatters = [[None, None], [None, None]]
    for row in range(2):
        for col in range(2):
            pos = pred_pos[0] if col == 0 else gt_pos[0]
            x_idx, y_idx = 0, (2 if row == 0 else 1)
            
    
            c = node_colors

            sc = axs[row, col].scatter(pos[:, x_idx], pos[:, y_idx], 
                                       c=c, s=0.3, alpha=0.6, linewidths=0)
            scatters[row][col] = sc

    title_pred_xz = axs[0, 0].set_title("", color="black", fontsize=FONTSIZE, pad=3)
    title_gt_xz   = axs[0, 1].set_title("", color="black", fontsize=FONTSIZE, pad=3)
    axs[1, 0].set_title("PRED (X-Y Plane)", color="black", fontsize=FONTSIZE, pad=3)
    axs[1, 1].set_title("GT (X-Y Plane)", color="black", fontsize=FONTSIZE, pad=3)

    # 取消了原有的 fig.axes[0].legend() 避免画面遮挡
    N_nodes = gt_pos.shape[1]
    if filter_eroded:
        # Prefer the real h5 erosion mask (dataset/ds/build_dataset.py's
        # node_alive) for GT when available — only fall back to the
        # position-jump heuristic for legacy h5s that lack it. Predictions
        # never have ground-truth erosion status, so they always use the
        # heuristic.
        if raw_data.get("has_node_alive"):
            erosion_valid_gt = raw_data["node_alive"][INPUT_FRAMES: INPUT_FRAMES + len(gt_pos)]
        else:
            erosion_valid_gt = compute_erosion_mask(gt_pos)
        erosion_valid_pred = compute_erosion_mask(pred_pos)
    else:
        erosion_valid_gt   = np.ones((len(gt_pos),   N_nodes), dtype=bool)
        erosion_valid_pred = np.ones((len(pred_pos), N_nodes), dtype=bool)

    fig.canvas.draw()

    # ── Fast Rendering Loop & PNG export ───────────────────────────────────
    print(f"[Vis] Rendering {T} frames ({mode}) at {dpi} DPI...")
    gif_frames = []

    if save_png_dir:
        Path(save_png_dir).mkdir(parents=True, exist_ok=True)

    for t in range(T):
        title_pred_xz.set_text(f"PRED [{mode}] (X-Z Plane)\nstep={t+1} | pos_rmse_veh_contact={rmse_pos[t]:.2e}mm")
        title_gt_xz.set_text(f"GT (X-Z Plane)\nstep={t+1}")

        for row in range(2):
            for col in range(2):
                raw_pos = pred_pos[t] if col == 0 else gt_pos[t]
                valid_t = erosion_valid_pred[t] if col == 0 else erosion_valid_gt[t]
                pos = raw_pos.copy()
                pos[~valid_t] = np.nan   # eroded nodes → invisible
                x_idx, y_idx = 0, (2 if row == 0 else 1)

                scatters[row][col].set_offsets(np.c_[pos[:, x_idx], pos[:, y_idx]])

        # Update canvas
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_img = Image.fromarray(rgba).convert('RGB')
        gif_frames.append(pil_img)
        
        # 导出逐帧超清 PNG
        if save_png_dir:
            pil_img.save(os.path.join(save_png_dir, f"frame_{t:04d}.png"))

        if (t + 1) % 50 == 0:
            print(f"  rendered {t+1}/{T} frames")

    plt.close(fig)

    # ── Save GIF ───────────────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"[Vis] Saved GIF → {out_path} ({T} frames @ {fps}fps)")
    if save_png_dir:
        print(f"[Vis] Saved PNG sequence → {save_png_dir}")


def render_gt_only(
    raw_data:          dict,
    out_path:          str,
    fps:               int  = 10,
    max_frames:        int  = 200,
    dpi:               int  = 120,
    group_config_path: str  = None,
    save_png_dir:      str  = None,
    filter_eroded:     bool = False,
):
    """Render a GT-only single-column animation (no model required).

    Layout: two rows, one column — top = X-Z plane, bottom = X-Y plane.
    Saves a GIF (and optionally a PNG sequence) identical in style to render_vis
    but without the pred column.
    """
    if not _VIS:
        print("Warning: matplotlib/Pillow not available — skipping rendering")
        return

    T_full = raw_data["positions"].shape[0]
    T      = min(T_full, max_frames)

    node_part_id   = raw_data["node_part_id"]
    node_part_name = raw_data["node_part_name"]
    gt_pos         = raw_data["positions"][:T]   # (T, N, 3)

    x_range = (-14000, 20000)
    y_range = (-15000,  8000)
    z_range =   (-500,  8000)

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size']   = 12

    x_span = x_range[1] - x_range[0]
    z_span = z_range[1] - z_range[0]
    y_span = y_range[1] - y_range[0]

    fig_w  = 6.0                           # single column → half of dual-column
    upi    = x_span / fig_w
    plot_h = (z_span + y_span) / upi
    fig_h  = plot_h + 1.6

    fig, axs = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h), dpi=dpi,
        gridspec_kw={
            "height_ratios": [z_span, y_span],
            "hspace": 0.18,
        },
    )
    fig.patch.set_facecolor("white")

    if group_config_path:
        node_colors, group_to_color = _get_grouped_colormap(
            node_part_id, node_part_name, group_config_path)
        legend_out = Path(out_path).parent / "raw_gt_legend.svg"
        _save_legend_svg(group_to_color, legend_out)
    else:
        node_colors, _, _ = _get_part_colormap(node_part_id)

    for ax in axs:
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=12)
        ax.set_aspect('equal', adjustable='box')

    axs[0].set_xlim(x_range);  axs[0].set_ylim(z_range)
    axs[1].set_xlim(x_range);  axs[1].set_ylim(y_range)
    axs[0].set_ylabel("Z", color="black", fontsize=FONTSIZE)
    axs[1].set_ylabel("Y", color="black", fontsize=FONTSIZE)
    axs[1].set_xlabel("X", color="black", fontsize=FONTSIZE)

    sc_xz = axs[0].scatter(gt_pos[0, :, 0], gt_pos[0, :, 2],
                            c=node_colors, s=0.3, alpha=0.6, linewidths=0)
    sc_xy = axs[1].scatter(gt_pos[0, :, 0], gt_pos[0, :, 1],
                            c=node_colors, s=0.3, alpha=0.6, linewidths=0)

    title_xz = axs[0].set_title("", color="black", fontsize=FONTSIZE, pad=3)
    axs[1].set_title("GT (X-Y Plane)", color="black", fontsize=FONTSIZE, pad=3)

    if filter_eroded:
        # Prefer the real h5 erosion mask when available (see render_vis).
        if raw_data.get("has_node_alive"):
            erosion_valid = raw_data["node_alive"][:T]
        else:
            erosion_valid = compute_erosion_mask(gt_pos)
    else:
        erosion_valid = np.ones(gt_pos.shape[:2], dtype=bool)

    fig.canvas.draw()

    if save_png_dir:
        Path(save_png_dir).mkdir(parents=True, exist_ok=True)

    gif_frames = []
    print(f"[Vis] Rendering {T} GT-only frames at {dpi} DPI...")

    for t in range(T):
        title_xz.set_text(f"GT (X-Z Plane)\nstep={t+1}")
        # mask eroded nodes with NaN so matplotlib skips them
        pos_t = gt_pos[t].copy()
        pos_t[~erosion_valid[t]] = np.nan
        sc_xz.set_offsets(np.c_[pos_t[:, 0], pos_t[:, 2]])
        sc_xy.set_offsets(np.c_[pos_t[:, 0], pos_t[:, 1]])

        fig.canvas.draw()
        rgba    = np.asarray(fig.canvas.buffer_rgba())
        pil_img = Image.fromarray(rgba).convert('RGB')
        gif_frames.append(pil_img)

        if save_png_dir:
            pil_img.save(os.path.join(save_png_dir, f"frame_{t:04d}.png"))

        if (t + 1) % 50 == 0:
            print(f"  rendered {t+1}/{T} frames")

    plt.close(fig)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"[Vis] Saved GT-only GIF → {out_path} ({T} frames @ {fps}fps)")
    if save_png_dir:
        print(f"[Vis] Saved PNG sequence → {save_png_dir}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(onestep: dict | None, autoreg: dict | None, baseline: dict):
    if onestep is not None and autoreg is not None:
        print("\n[Debug] First 10 acc RMSE — one-step vs autoregressive:")
        print(f"  one-step : {onestep['rmse_acc'][-10:]}")
        print(f"  autoreg  : {autoreg['rmse_acc'][-10:]}")
        print(f"  equal    : {np.array_equal(onestep['rmse_acc'][-10:], autoreg['rmse_acc'][-10:])}")

    print("\n" + "=" * 90)
    print("ROLLOUT SUMMARY")
    # pos_rmse(mm) is the all-node average — diluted by the ~90% of nodes (far
    # barrier, far vehicle) that barely move, so it's not very sensitive to
    # model quality. pos_rmse_veh_contact(mm) is the metric that actually
    # tracks how well the collision dynamics are captured.
    print("=" * 90)
    print(f"{'Mode':<20} {'pos_rmse(mm)':>14} {'pos_rmse_veh_contact(mm)':>26} "
          f"{'vel_rmse(mm/dt)':>16} {'acc_rmse(mm/dt²)':>17}")
    print("-" * 90)

    print(f"{'onestep baseline':<20} "
          f"{baseline['rmse_pos_onestep'].mean():>14.3e} "
          f"{baseline['rmse_pos_onestep_veh_contact'].mean():>26.3e} "
          f"{baseline['rmse_vel_onestep'].mean():>16.3f} "
          f"{baseline['rmse_acc_onestep'].mean():>17.3f}")

    if onestep is not None:
        print(f"{'one-step':<20} "
              f"{onestep['rmse_pos'].mean():>14.3e} "
              f"{onestep['rmse_pos_veh_contact'].mean():>26.3e} "
              f"{onestep['rmse_vel'].mean():>16.3f} "
              f"{onestep['rmse_acc'].mean():>17.3f}")

    print(f"{'rollout baseline':<20} "
          f"{baseline['rmse_pos_rollout'].mean():>14.3e} "
          f"{baseline['rmse_pos_rollout_veh_contact'].mean():>26.3e} "
          f"{baseline['rmse_vel_rollout'].mean():>16.3f} "
          f"{baseline['rmse_acc_rollout'].mean():>17.3f}")

    if autoreg is not None:
        print(f"{'autoregressive':<20} "
              f"{autoreg['rmse_pos'].mean():>14.3e} "
              f"{autoreg['rmse_pos_veh_contact'].mean():>26.3e} "
              f"{autoreg['rmse_vel'].mean():>16.3f} "
              f"{autoreg['rmse_acc'].mean():>17.3f}")

    print("=" * 90)


# ── WandB upload ──────────────────────────────────────────────────────────────

class _Tee:
    """Duplicate writes to both the real stdout and an internal StringIO buffer."""
    def __init__(self):
        self._real = sys.stdout
        self._buf  = io.StringIO()
        sys.stdout = self

    def write(self, s: str):
        self._real.write(s)
        self._buf.write(s)

    def flush(self):
        self._real.flush()

    def restore(self) -> str:
        sys.stdout = self._real
        return self._buf.getvalue()


def upload_to_wandb(
    project:   str,
    run_name:  str,
    fps:       int,
    metadata:  dict | None = None,
    gif_paths: list[str] | None = None,
    log_text:  str | None = None,
    metrics:   dict | None = None,
):
    """Upload GIFs, console log, and RMSE metrics to a single WandB run.

    - GIFs appear in the Media panel (viewable in-browser) and as a versioned
      artifact (shareable download URL).
    - Console log is saved as rollout.log inside the same artifact and rendered
      as HTML in the run's Media panel.
    - RMSE summary values are logged as run summary metrics.
    """
    if not _WANDB:
        print("[WandB] wandb not installed — skipping upload")
        return

    run = wandb.init(
        project=project,
        name=run_name,
        job_type="rollout",
        config=metadata or {},
    )

    artifact = wandb.Artifact(name=run_name, type="rollout")

    # ── GIFs ──────────────────────────────────────────────────────────────
    existing_gifs = [p for p in (gif_paths or []) if Path(p).exists()]
    media_dict = {}
    for gif_path in existing_gifs:
        key = f"gif/{Path(gif_path).stem}"
        media_dict[key] = wandb.Video(gif_path, fps=fps, format="gif")
        artifact.add_file(gif_path)

    # ── Console log ───────────────────────────────────────────────────────
    if log_text:
        log_path = Path(wandb.run.dir) / "rollout.log"
        log_path.write_text(log_text)
        artifact.add_file(str(log_path), name="rollout.log")
        # Render as HTML so it's readable in the Files / Media tab
        html_body = log_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        media_dict["log/console"] = wandb.Html(
            f"<pre style='font-family:monospace;font-size:12px;white-space:pre-wrap'>"
            f"{html_body}</pre>"
        )

    if media_dict:
        wandb.log(media_dict)

    # ── RMSE summary metrics ──────────────────────────────────────────────
    if metrics:
        for key, val in metrics.items():
            wandb.run.summary[key] = val

    run.log_artifact(artifact)

    run_id     = run.id
    run_entity = run.entity
    run.finish()

    print(f"[WandB] Uploaded to project '{project}' run '{run_name}'")
    print(f"[WandB] View at: https://wandb.ai/{run_entity}/{project}/runs/{run_id}")


# ── Entry point ───────────────────────────────────────────────────────────────
# NOTE: rollout depends on train.py having written meta.json and logged the
# checkpoint-{exp}:best W&B artifact before this script is run.

def main():
    parser = argparse.ArgumentParser(description="BVC rollout visualization")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Local checkpoint path (fallback when artifact unavailable)")
    parser.add_argument("--ckpt-dir",   default=None,
                        help="Checkpoint directory containing meta.json "
                             "(derived from --checkpoint parent if omitted)")
    parser.add_argument("--experiment", default=None,
                        help="Experiment yaml (auto-derived from meta.json when omitted)")
    parser.add_argument("--raw-h5",     required=True, nargs="+",
                        help="One or more h5 trajectory paths (one per test set)")
    parser.add_argument("--mode",
                        choices=["onestep", "autoregressive", "both", "raw_gt"],
                        default="both")
    parser.add_argument("--plot",         action="store_true",
                        help="Save RMSE vs timestep plot (single test set only)")
    parser.add_argument("--compare-dirs", nargs="+", default=[],
                        metavar="NAME:DIR",
                        help="Compare experiments. Format: 'label:output_dir'")
    parser.add_argument("--gif",          action="store_true",
                        help="Render GIF animations")
    parser.add_argument("--gif-fps",      type=int, default=10)
    parser.add_argument("--gif-max-frames", type=int, default=200)
    parser.add_argument("--no-erosion-filter", action="store_true",
                        help="Disable eroded-node masking in GIF rendering (show all nodes)")
    parser.add_argument("--gif-name",     default=None,
                        help="GIF filename stem override (single test set). "
                             "Ignored when multiple --raw-h5 are given.")
    parser.add_argument("--stats-path",   default=None,
                        help="Path to global_stats.json (defaults to <ckpt_dir>/global_stats.json)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",   default=None,
                        help="Override output directory")
    parser.add_argument("--wandb-project", default=None,
                        help="WandB project override (defaults to meta.json project)")
    parser.add_argument("--wandb-run-name", default=None,
                        help="WandB run name override")
    parser.add_argument("--no-artifact",  action="store_true",
                        help="Load checkpoint from local disk, skip use_artifact (debugging)")
    args = parser.parse_args()

    # ── raw_gt mode: no model, optional GIF upload ────────────────────────
    if args.mode == "raw_gt":
        h5_path  = args.raw_h5[0]
        tee      = _Tee() if args.wandb_project else None
        raw_data = load_raw_h5(h5_path)
        h5_stem  = traj_name_from_h5(h5_path)
        out_dir  = Path(args.output_dir or PROJECT_ROOT / "outputs" / "rollouts" / h5_stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.gif:
            gif_stem = args.gif_name or "raw_gt"
            gif_path = str(out_dir / f"{gif_stem}.gif")
            render_gt_only(
                raw_data,
                out_path          = gif_path,
                fps               = args.gif_fps,
                max_frames        = args.gif_max_frames,
                dpi               = 120,
                group_config_path = "configs/data/required_parts.config",
                save_png_dir      = str(out_dir / f"{gif_stem}_pngs"),
            )
            if args.wandb_project:
                log_text = tee.restore()
                upload_to_wandb(
                    project=args.wandb_project,
                    run_name=args.wandb_run_name or gif_stem,
                    fps=args.gif_fps,
                    metadata={"mode": "raw_gt", "h5": h5_path},
                    gif_paths=[gif_path],
                    log_text=log_text,
                )
        else:
            print("[raw_gt] No --gif flag — nothing to do. Add --gif to render.")
            if tee:
                tee.restore()
        return

    # ── Model-based modes ─────────────────────────────────────────────────

    # Resolve checkpoint directory (needed for meta.json and stats)
    ckpt_dir: Path | None = (
        Path(args.ckpt_dir) if args.ckpt_dir
        else Path(args.checkpoint).parent if args.checkpoint
        else None
    )

    # ── W&B: init with group + use_artifact for checkpoint lineage ────────
    wandb_run       = None
    artifact_ckpt_dir: Path | None = None
    exp_meta: dict | None = None

    if _WANDB and ckpt_dir and not args.no_artifact:
        try:
            exp_meta = load_meta(ckpt_dir)
        except FileNotFoundError as e:
            print(f"[W&B] {e}")

        if exp_meta is not None:
            exp     = exp_meta["experiment"]
            project = args.wandb_project or exp_meta["project"]
            wandb_run = wandb.init(
                project=project,
                group=exp_meta["group"],
                job_type="rollout",
                name=args.wandb_run_name or f"rollout_{exp}_best",
                config={"experiment": exp, "checkpoint": "best"},
            )
            # use_artifact builds the train→checkpoint→rollout lineage in W&B
            try:
                ckpt_artifact = wandb_run.use_artifact(f"checkpoint-{exp}:best")
                artifact_ckpt_dir = Path(ckpt_artifact.download())
                print(f"[W&B] Downloaded checkpoint-{exp}:best → {artifact_ckpt_dir}")
            except Exception as e:
                print(f"[W&B] Warning: artifact download failed ({e}); falling back to local disk")

    # ── Resolve weights path and experiment config ─────────────────────────
    if artifact_ckpt_dir:
        weights_path    = _find_weights_file(artifact_ckpt_dir)
        experiment_path = args.experiment or str(
            PROJECT_ROOT / "configs" / "experiments" / f"{exp_meta['experiment']}.yaml"
        )
    else:
        if args.checkpoint:
            weights_path = Path(args.checkpoint)
        elif ckpt_dir:
            weights_path = _find_weights_file(ckpt_dir)
        else:
            parser.error("--checkpoint or --ckpt-dir is required for model modes "
                         "when wandb artifact is unavailable")
        experiment_path = args.experiment
        if experiment_path is None and exp_meta:
            experiment_path = str(
                PROJECT_ROOT / "configs" / "experiments" / f"{exp_meta['experiment']}.yaml"
            )
        if experiment_path is None:
            parser.error("--experiment is required (or provide --ckpt-dir with meta.json)")

    device     = torch.device(args.device)
    model, cfg = load_model(str(weights_path), experiment_path, device)
    exp_name   = cfg["name"]

    global INPUT_FRAMES
    INPUT_FRAMES = int(cfg["data"].get("input_frames", INPUT_FRAMES))
    print(f"[Rollout] INPUT_FRAMES = {INPUT_FRAMES} (from config)")

    cond_cfg = CondConfig(**(cfg.get("condition") or {}))
    print(f"[Rollout] Condition: enabled={list(cond_cfg.enabled)}, n_cond={cond_cfg.n_cond()}")

    out_dir = Path(args.output_dir or PROJECT_ROOT / "outputs" / "rollouts" / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Shared setup (norm stats, node type) ──────────────────────────────
    use_node_type   = bool(cfg["data"].get("node_type", False))
    node_type_field = cfg["data"].get("node_type_field", None) if use_node_type else None

    stats_path = (
        Path(args.stats_path) if args.stats_path
        else ckpt_dir / "global_stats.json" if ckpt_dir
        else weights_path.parent / "global_stats.json"
    )
    norm_stats = NormStats.from_global_stats(stats_path, acc_scale=cfg["data"].get("acc_scale"))
    # Trust the loaded stats file over cfg for whether per-region normalization
    # was actually used at train time — cfg can drift from what produced the
    # checkpoint, and denormalizing with the wrong scheme silently corrupts
    # physical units.
    if bool(cfg["data"].get("per_region_norm", False)) != norm_stats.is_per_region:
        print(f"[Rollout] Warning: cfg data.per_region_norm="
              f"{bool(cfg['data'].get('per_region_norm', False))} but the loaded stats file "
              f"is_per_region={norm_stats.is_per_region} — using the stats file's mode.")

    # ── W&B Table (one per run, all test sets and modes) ──────────────────
    table: "wandb.Table | None" = None
    if wandb_run:
        table = wandb.Table(columns=[
            "test_set", "traj_id", "mode",
            "pos_rmse_mm", "pos_rmse_veh_contact_mm", "vel_rmse_mm_dt", "acc_rmse_mm_dt2",
            "weight_kg", "speed_kmh", "angle_deg", "concrete_type",
        ])

    all_gif_media: dict = {}

    mse_by_ts: dict[str, list[float]] = {}
    all_pos_rmse: list[float] = []
    # veh_contact-only: the metric that actually tracks collision-dynamics
    # quality (the all-node average above is diluted by far-field nodes).
    mse_vc_by_ts: dict[str, list[float]] = {}
    all_pos_rmse_vc: list[float] = []

    # Track last results for --plot (single-test-set use)
    last_onestep = last_autoreg = last_baseline = None

    _DPI = 120

    # ── Per-test-set loop ─────────────────────────────────────────────────
    for h5_path in args.raw_h5:
        ts_name = traj_name_from_h5(h5_path)
        multi   = len(args.raw_h5) > 1
        print(f"\n[Rollout] === Test set: {ts_name} ===")

        # Parse conditions from dir name for both the table and model input
        try:
            raw_conds = parse_conditions(None, ts_name)
            speed_kmh = float(raw_conds["speed"])
            weight_kg = float(raw_conds["mass"])
            angle_deg = float(raw_conds["angle"])
            cond_vec  = normalize_conditions(raw_conds, cond_cfg)   # (n_cond,) float32
            cond_t    = torch.from_numpy(cond_vec).float()          # passed to inference
            print(f"[Rollout] cond_raw={raw_conds} cond={cond_vec}")
        except Exception as e:
            print(f"[Rollout] Warning: condition parse failed for {ts_name}: {e}")
            speed_kmh = weight_kg = angle_deg = float("nan")
            cond_t = torch.zeros(cond_cfg.n_cond())

        region_norm_field = norm_stats._region_field if norm_stats.is_per_region else None
        raw_data  = load_raw_h5(h5_path, node_type_field=node_type_field,
                                 region_norm_field=region_norm_field)
        node_type = (
            torch.from_numpy(raw_data["node_type"]).to(device) if use_node_type else None
        )
        if norm_stats.is_per_region:
            norm_stats.set_region_id(raw_data["region_id"])
        normed   = normalize_raw(raw_data, norm_stats)
        baseline = compute_baseline(raw_data, dt=DT, input_frames=INPUT_FRAMES)

        # ── Inference ─────────────────────────────────────────────────────
        onestep = autoreg = None
        t0 = time.time()

        pkl_stem = f"{ts_name}_" if multi else ""

        if args.mode in ("onestep", "both"):
            onestep = run_onestep(model, raw_data, normed, norm_stats, device,
                                  node_type=node_type, cond_t=cond_t)
            pkl_path = out_dir / f"{pkl_stem}onestep.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(onestep, f)
            print(f"[Rollout] PKL saved → {pkl_path}")

        if args.mode in ("autoregressive", "both"):
            autoreg = run_autoregressive(model, raw_data, normed, norm_stats, device,
                                         node_type=node_type, cond_t=cond_t)
            pkl_path = out_dir / f"{pkl_stem}autoregressive.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(autoreg, f)
            print(f"[Rollout] PKL saved → {pkl_path}")

        print(f"[Rollout] Inference done in {time.time() - t0:.1f}s")

        # ── GIF rendering ─────────────────────────────────────────────────
        gif_paths: dict[str, str] = {}
        if args.gif:
            base_stem = (args.gif_name if not multi else None) or ts_name

            if onestep is not None:
                gif_stem = base_stem if args.mode != "both" else f"{base_stem}_os"
                gif_path = str(out_dir / f"{gif_stem}.gif")
                render_vis(
                    onestep, raw_data,
                    out_path          = gif_path,
                    fps               = args.gif_fps,
                    max_frames        = args.gif_max_frames,
                    dpi               = _DPI,
                    group_config_path = "configs/data/required_parts.config",
                    save_png_dir      = str(out_dir / f"{gif_stem}_pngs"),
                )
                gif_paths["os"] = gif_path
                if wandb_run:
                    all_gif_media[f"gifs/{ts_name}_os"] = wandb.Video(gif_path, fps=args.gif_fps, format="gif")

            if autoreg is not None:
                gif_stem = base_stem if args.mode != "both" else f"{base_stem}_ar"
                gif_path = str(out_dir / f"{gif_stem}.gif")
                render_vis(
                    autoreg, raw_data,
                    out_path          = gif_path,
                    fps               = args.gif_fps,
                    max_frames        = args.gif_max_frames,
                    dpi               = _DPI,
                    group_config_path = "configs/data/required_parts.config",
                    save_png_dir      = str(out_dir / f"{gif_stem}_pngs"),
                )
                gif_paths["ar"] = gif_path
                if wandb_run:
                    all_gif_media[f"gifs/{ts_name}_ar"] = wandb.Video(gif_path, fps=args.gif_fps, format="gif")

        print_summary(onestep, autoreg, baseline)

        # ── Add rows to W&B Table ─────────────────────────────────────────
        if table is not None:
            if onestep is not None:
                pos_rmse_os    = float(onestep["rmse_pos"].mean())
                pos_rmse_os_vc = float(onestep["rmse_pos_veh_contact"].mean())
                table.add_data(
                    ts_name, ts_name, "os",
                    pos_rmse_os, pos_rmse_os_vc,
                    float(onestep["rmse_vel"].mean()),
                    float(onestep["rmse_acc"].mean()),
                    weight_kg, speed_kmh, angle_deg, "N",
                )
                mse_by_ts.setdefault(ts_name, []).append(pos_rmse_os)
                all_pos_rmse.append(pos_rmse_os)
                mse_vc_by_ts.setdefault(ts_name, []).append(pos_rmse_os_vc)
                all_pos_rmse_vc.append(pos_rmse_os_vc)

            if autoreg is not None:
                pos_rmse_ar    = float(autoreg["rmse_pos"].mean())
                pos_rmse_ar_vc = float(autoreg["rmse_pos_veh_contact"].mean())
                table.add_data(
                    ts_name, ts_name, "ar",
                    pos_rmse_ar, pos_rmse_ar_vc,
                    float(autoreg["rmse_vel"].mean()),
                    float(autoreg["rmse_acc"].mean()),
                    weight_kg, speed_kmh, angle_deg, "N",
                )
                mse_by_ts.setdefault(ts_name, []).append(pos_rmse_ar)
                all_pos_rmse.append(pos_rmse_ar)
                mse_vc_by_ts.setdefault(ts_name, []).append(pos_rmse_ar_vc)
                all_pos_rmse_vc.append(pos_rmse_ar_vc)

            # Zero-acceleration baseline rows (no GIF)
            table.add_data(
                ts_name, ts_name, "zero_os",
                float(baseline["rmse_pos_onestep"].mean()),
                float(baseline["rmse_pos_onestep_veh_contact"].mean()),
                float(baseline["rmse_vel_onestep"].mean()),
                float(baseline["rmse_acc_onestep"].mean()),
                weight_kg, speed_kmh, angle_deg, "N",
            )
            table.add_data(
                ts_name, ts_name, "zero_ar",
                float(baseline["rmse_pos_rollout"].mean()),
                float(baseline["rmse_pos_rollout_veh_contact"].mean()),
                float(baseline["rmse_vel_rollout"].mean()),
                float(baseline["rmse_acc_rollout"].mean()),
                weight_kg, speed_kmh, angle_deg, "N",
            )

        last_onestep = onestep
        last_autoreg = autoreg
        last_baseline = baseline

    # ── Log Table + summary to W&B ────────────────────────────────────────
    if wandb_run:
        wandb_run.log({"rollout_results": table})
        if all_gif_media:
            wandb_run.log(all_gif_media)
        for ts_name, rmse_vals in mse_by_ts.items():
            wandb_run.summary[f"mean_mse/{ts_name}"] = float(np.mean(rmse_vals))
        if all_pos_rmse:
            wandb_run.summary["mean_mse_overall"] = float(np.mean(all_pos_rmse))
        # veh_contact-only — the sensitive metric; the all-node one above is
        # diluted by far-field nodes that barely move.
        for ts_name, rmse_vals in mse_vc_by_ts.items():
            wandb_run.summary[f"mean_mse_veh_contact/{ts_name}"] = float(np.mean(rmse_vals))
        if all_pos_rmse_vc:
            wandb_run.summary["mean_mse_veh_contact_overall"] = float(np.mean(all_pos_rmse_vc))
        wandb_run.finish()
        print("[W&B] Rollout run finished")

    # ── RMSE plot (single test set only) ──────────────────────────────────
    if args.plot and len(args.raw_h5) == 1:
        plot_rmse_vs_timestep(last_onestep, last_autoreg, out_dir)
        plot_os_ar_veh_contact_rmse(
            last_onestep, last_autoreg, out_dir / "os_ar_veh_contact_rmse.png"
        )

    # ── Multi-experiment comparison plot ──────────────────────────────────
    if args.compare_dirs:
        experiments = []
        for entry in args.compare_dirs:
            if ":" not in entry:
                print(f"[Warn] --compare-dirs entry '{entry}' missing label — skipping")
                continue
            label, cdir = entry.split(":", 1)
            cdir = Path(cdir)
            os_pkl  = cdir / "onestep.pkl"
            ar_pkl  = cdir / "autoregressive.pkl"
            os_res  = pickle.load(open(os_pkl,  "rb")) if os_pkl.exists()  else None
            ar_res  = pickle.load(open(ar_pkl,  "rb")) if ar_pkl.exists()  else None
            experiments.append((label, os_res, ar_res))
            print(f"[Compare] loaded '{label}' from {cdir}")
        if experiments:
            plot_multi_rmse(experiments, out_dir / "multi_experiment_rmse.png")


if __name__ == "__main__":
    main()