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
        --checkpoint outputs/checkpoints/dg_001/checkpoint-best.safetensors \
        --experiment configs/experiments/dg_001.yaml \
        --raw-h5 /data/curtin_ciraee/curtin_xiangrui/data/h5dt_50ns_5fs_mat/T_lok_F_shape_barrier_9_3_100km/output.h5 \
        --mode both \
        --gif --gif-fps 10 \
        --gif-name dg001_100kmh

    # GT-only GIF — no checkpoint/experiment needed
    python src/rollout.py \
        --raw-h5 /home/kong/datasets/barrier/h5dt_50ns_5fs_mat/T_lok_F_shape_barrier_9_3_100km/output.h5 \
        --mode raw_gt \
        --gif --gif-fps 10 --gif-name traj_9_3_gt

    python src/rollout.py \
        --checkpoint outputs/checkpoints/sc_021/checkpoint-best.safetensors \
        --experiment configs/experiments/sc_021.yaml \
        --raw-h5 /home/kong/datasets/barrier/h5/T_lok_F_shape_barrier_9_3_100km_50_5/output.h5 \
        --mode both --plot \
        --compare-dirs \
            sc_018:outputs/rollouts/sc_018 \
            sc_019:outputs/rollouts/sc_019 \
            sc_020:outputs/rollouts/sc_020 \
            sc_021:outputs/rollouts/sc_021

"""

from __future__ import annotations

import argparse
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
from src.dataset import NormStats

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

def load_raw_h5(h5_path: str, node_type_field: str | None = None) -> dict:
    """Load full trajectory and part metadata from raw h5.

    Returns dict with:
        positions:    (T, N, 3)  float32
        velocity:     (T, N, 3)  float32
        acceleration: (T, N, 3)  float32
        stress:       (T, N, 6)  float32
        times:        (T,)       float64
        node_part_id:   (N,)     int64
        node_part_name: (N,)     str
        part_ids:       (P,)     int64
        part_names:     (P,)     str
    """
    if not _H5PY:
        raise ImportError("h5py required")

    with h5py.File(h5_path, "r") as f:
        data = {
            "positions":    f["states/positions"][:].astype(np.float32),
            "velocity":     f["states/velocity"][:].astype(np.float32),
            "acceleration": f["states/acceleration"][:].astype(np.float32),
            "times":        f["states/times"][:],
            "node_part_id":  f["metadata/node_part_id"][:],
            "node_part_name": np.array([
                n.decode("utf-8").strip("\x00") if isinstance(n, bytes) else str(n)
                for n in f["metadata/node_part_name"][:]
            ]),
            "part_ids":   f["metadata/part_ids"][:],
            "part_names": np.array([
                n.decode("utf-8").strip("\x00") if isinstance(n, bytes) else str(n)
                for n in f["metadata/part_names"][:]
            ]),
        }
        if node_type_field is not None:
            nt_key = f"metadata/{node_type_field}"
            if nt_key not in f:
                raise KeyError(f"{h5_path}: missing node_type field '{nt_key}'")
            data["node_type"] = f[nt_key][:].astype(np.int64)  # (N,)

    T, N, _ = data["positions"].shape
    P       = len(data["part_ids"])
    print(f"[Data] {T} frames, {N} nodes, {P} parts")
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

# ── Signed Distance Field ────────────────────────────────────────────────────────

def compute_sdf_batch(xy: torch.Tensor, 
                    barrier_angle_deg: float=-25.4, 
                    barrier_anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    sdf: signed distance field
    car_points: (N, T, 2) 
    barrier_angle_deg: 护栏角度 (标量) impace degree -25.4
    barrier_anchor: (3,) 护栏基准点，必须在 GPU 上 xy=(0,2000)

    """
    device = xy.device
    if barrier_anchor is None:
        barrier_anchor = torch.tensor([0.0, 2000.0], device=device)
    else:
        barrier_anchor = barrier_anchor.to(device)

    angle_rad = torch.deg2rad(torch.tensor(barrier_angle_deg, device=device))
    normal_2d = torch.tensor([-torch.sin(angle_rad), torch.cos(angle_rad)], device=device)
    
    diff_2d = xy - barrier_anchor[:2]
    distances = (diff_2d * normal_2d).sum(dim=-1)
    
    return distances / 1000.0


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_onestep(model, raw_data, normed, norm_stats, device, node_type=None) -> dict:
    T      = raw_data["positions"].shape[0]
    T_eval = T - 2          # last 2 frames have padded GT vel/acc (forward diff)
    print(f"[One-step] dt = 1 (per-frame), steps = {T_eval - INPUT_FRAMES}")

    normed_v = normed["velocity"]                       # (T, N, 3)
    pred_pos_list, gt_pos_list, rmse_pos_steps = [], [], []
    rmse_vel_steps, rmse_acc_steps = [], []
    pred_acc_list, gt_acc_list = [], []
    pred_acc_norm_list, gt_acc_norm_list = [], []

    for t in range(INPUT_FRAMES, T_eval):
        x_in        = build_velocity_input(normed_v, t - 1).to(device)

        input_pos = raw_data["positions"][t - INPUT_FRAMES + 1: t + 1].transpose(1, 0, 2) # (N,T,3)

        x_sdf = compute_sdf_batch(torch.from_numpy(input_pos[..., 0:2])).to(device)

        x = torch.cat([x_in, x_sdf.unsqueeze(0) ], dim=-1)
        # x = x_in

        a_pred_norm = model(x, node_type).squeeze(0)  # (N, 3)

        # 上一帧 GT 速度 / 位置 (物理量) — one-step 模式始终用 GT
        v_last = raw_data["velocity"][t - 1]
        x_last = raw_data["positions"][t - 1]

        a_phys, v_new, x_new = integrate_accel(a_pred_norm, v_last, x_last, DT, norm_stats)

        x_gt = raw_data["positions"][t]
        rmse = float(np.sqrt(np.mean((x_new - x_gt) ** 2)))

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

        if (t - INPUT_FRAMES + 1) % 50 == 0:
            print(f"  step {t-INPUT_FRAMES+1}/{T_eval-INPUT_FRAMES} | pos_rmse={rmse:.3f}")

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

    return {
        "pred_frames": _pack_pos_only(pred_pos_list),
        "gt_frames":   _pack_pos_only(gt_pos_list),
        "rmse_pos":    np.array(rmse_pos_steps),
        "rmse_vel":    np.array(rmse_vel_steps),
        "rmse_acc":    rmse_acc,
        "mode":        "onestep",
    }

@torch.no_grad()
def run_autoregressive(model, raw_data, normed, norm_stats, device, node_type=None) -> dict:
    T      = raw_data["positions"].shape[0]
    T_eval = T - 2          # last 2 frames have padded GT vel/acc (forward diff)
    print(f"[Autoregressive] dt = 1 (per-frame), steps = {T_eval - INPUT_FRAMES}")

    pred_pos_list, gt_pos_list, rmse_pos_steps = [], [], []
    rmse_vel_steps, rmse_acc_steps = [], []
    pred_acc_list, gt_acc_list = [], []
    pred_acc_norm_list, gt_acc_norm_list = [], []

    # ── 初始化 ──
    # 速度窗口 (归一化, 模型输入用)
    v_window_norm = normed["velocity"][:INPUT_FRAMES].copy()       # (5, N, 3)
    x_window_phys = raw_data["positions"][:INPUT_FRAMES].copy()  
    # 物理速度 / 位置当前状态
    v_phys = raw_data["velocity"][INPUT_FRAMES - 1].copy()         # (N, 3)
    x_phys = raw_data["positions"][INPUT_FRAMES - 1].copy()        # (N, 3)

    for t in range(INPUT_FRAMES, T_eval):
        x_in        = build_velocity_input_from_window(v_window_norm).to(device) # (1,N,T*C)

        # SDF 用滚动窗口,不再读 raw_data
        x_sdf_in = torch.from_numpy(
            x_window_phys[..., 0:2].transpose(1, 0, 2)              # (N, 5, 2)
        ).float()
        x_sdf = compute_sdf_batch(x_sdf_in).to(device)               # (N, 5)
        x = torch.cat([x_in, x_sdf.unsqueeze(0)], dim=-1)

        a_pred_norm = model(x, node_type).squeeze(0)  # (N, 3)

        a_phys_new, v_phys_new, x_phys_new = integrate_accel(
            a_pred_norm, v_phys, x_phys, DT, norm_stats)

        x_gt = raw_data["positions"][t]
        rmse = float(np.sqrt(np.mean((x_phys_new - x_gt) ** 2)))

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
            print(f"  step {t-INPUT_FRAMES+1}/{T_eval-INPUT_FRAMES} | pos_rmse={rmse:.3f}")

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

    return {
        "pred_frames": _pack_pos_only(pred_pos_list),
        "gt_frames":   _pack_pos_only(gt_pos_list),
        "rmse_pos":    np.array(rmse_pos_steps),
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

    def rms(x):  # RMS over nodes & dims for one frame
        return np.sqrt(np.mean(x ** 2))

    out = {k: [] for k in [
        "rmse_pos_onestep", "rmse_vel_onestep", "rmse_acc_onestep",
        "rmse_pos_rollout", "rmse_vel_rollout", "rmse_acc_rollout",
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

        # ---- rollout：从最后输入帧自我递推 (constant-velocity) ----
        steps = t - (IF - 1)
        x_pred_rl = x0 + v0 * dt * steps                # 匀速外推
        out["rmse_vel_rollout"].append(rms(v0 - vel[t]))     # v 冻结
        out["rmse_pos_rollout"].append(rms(x_pred_rl - pos[t]))

    return {k: np.array(v) for k, v in out.items()}

# ── RMSE plot ────────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family":     "Times New Roman",
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
    fig.suptitle("Rollout RMSE vs Timestep", fontsize=13, fontfamily="Times New Roman")

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
            ax.plot(steps, rmse, color="#1f77b4", linestyle="-", linewidth=1.5)

    out_path = out_dir / "rmse_vs_timestep.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] RMSE plot saved → {out_path}")


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
                 fontsize=13, fontfamily="Times New Roman")

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
):
    """Render pred (left) vs gt (right) animation, with optional PNG export and grouped coloring."""
    if not _VIS:
        print("Warning: matplotlib/Pillow not available — skipping rendering")
        return

    pred_frames = result["pred_frames"]
    gt_frames   = result["gt_frames"]
    rmse_pos    = result["rmse_pos"]
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
    plt.rcParams['font.family'] = 'Times New Roman'
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
    fig.canvas.draw()
    
    # ── Fast Rendering Loop & PNG export ───────────────────────────────────
    print(f"[Vis] Rendering {T} frames ({mode}) at {dpi} DPI...")
    gif_frames = []
    
    if save_png_dir:
        Path(save_png_dir).mkdir(parents=True, exist_ok=True)

    for t in range(T):
        title_pred_xz.set_text(f"PRED [{mode}] (X-Z Plane)\nstep={t+1} | pos_rmse={rmse_pos[t]:.1f}mm")
        title_gt_xz.set_text(f"GT (X-Z Plane)\nstep={t+1}")

        for row in range(2):
            for col in range(2):
                pos = pred_pos[t] if col == 0 else gt_pos[t]
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
    y_range = (-10000,  8000)
    z_range =   (-500,  4000)

    plt.rcParams['font.family'] = 'Times New Roman'
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

    fig.canvas.draw()

    if save_png_dir:
        Path(save_png_dir).mkdir(parents=True, exist_ok=True)

    gif_frames = []
    print(f"[Vis] Rendering {T} GT-only frames at {dpi} DPI...")

    for t in range(T):
        title_xz.set_text(f"GT (X-Z Plane)\nstep={t+1}")
        sc_xz.set_offsets(np.c_[gt_pos[t, :, 0], gt_pos[t, :, 2]])
        sc_xy.set_offsets(np.c_[gt_pos[t, :, 0], gt_pos[t, :, 1]])

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

    print("\n" + "=" * 72)
    print("ROLLOUT SUMMARY")
    print("=" * 72)
    print(f"{'Mode':<20} {'pos_rmse(mm)':>14} {'vel_rmse(mm/dt)':>16} {'acc_rmse(mm/dt²)':>17}")
    print("-" * 72)

    print(f"{'onestep baseline':<20} "
          f"{baseline['rmse_pos_onestep'].mean():>14.3f} "
          f"{baseline['rmse_vel_onestep'].mean():>16.3f} "
          f"{baseline['rmse_acc_onestep'].mean():>17.3f}")

    if onestep is not None:
        print(f"{'one-step':<20} "
              f"{onestep['rmse_pos'].mean():>14.3f} "
              f"{onestep['rmse_vel'].mean():>16.3f} "
              f"{onestep['rmse_acc'].mean():>17.3f}")
    
    print(f"{'rollout baseline':<20} "
          f"{baseline['rmse_pos_rollout'].mean():>14.3f} "
          f"{baseline['rmse_vel_rollout'].mean():>16.3f} "
          f"{baseline['rmse_acc_rollout'].mean():>17.3f}")

    if autoreg is not None:
        print(f"{'autoregressive':<20} "
              f"{autoreg['rmse_pos'].mean():>14.3f} "
              f"{autoreg['rmse_vel'].mean():>16.3f} "
              f"{autoreg['rmse_acc'].mean():>17.3f}")

    print("=" * 72)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BVC rollout visualization")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Local .safetensors checkpoint (not required for --mode raw_gt)")
    parser.add_argument("--experiment",
                        default=None,
                        help="Experiment yaml (not required for --mode raw_gt)")
    parser.add_argument("--raw-h5",      required=True,
                        help="Path to original (non-windowed) h5 trajectory")
    parser.add_argument("--mode",
                        choices=["onestep", "autoregressive", "both", "raw_gt"],
                        default="both")
    parser.add_argument("--plot",         action="store_true",
                        help="Save RMSE vs timestep plot for the current run")
    parser.add_argument("--compare-dirs", nargs="+", default=[],
                        metavar="NAME:DIR",
                        help="Compare multiple experiments. Format: 'label:output_dir' "
                             "where output_dir contains onestep.pkl / autoregressive.pkl")
    parser.add_argument("--gif",         action="store_true",
                        help="Render GIF animations")
    parser.add_argument("--gif-fps",     type=int, default=10)
    parser.add_argument("--gif-max-frames", type=int, default=200,
                        help="Cap frames rendered (for speed)")
    parser.add_argument("--gif-name",    default=None,
                        help="Custom GIF filename stem (no extension), e.g. 'sc026_ar_100kmh'. "
                             "Defaults to mode name (onestep / autoregressive / raw_gt).")
    parser.add_argument("--stats-path",   default=None,
                        help="Path to training-run global_stats.json. "
                             "Defaults to <checkpoint_dir>/global_stats.json.")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",  default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # ── raw_gt mode: skip model entirely ─────────────────────────────────
    if args.mode == "raw_gt":
        raw_data = load_raw_h5(args.raw_h5)
        h5_stem  = Path(args.raw_h5).parent.name  # use parent folder name as exp label
        out_dir  = Path(args.output_dir or
                        PROJECT_ROOT / "outputs" / "rollouts" / h5_stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.gif:
            gif_stem = args.gif_name or "raw_gt"
            render_gt_only(
                raw_data,
                out_path          = str(out_dir / f"{gif_stem}.gif"),
                fps               = args.gif_fps,
                max_frames        = args.gif_max_frames,
                dpi               = 120,
                group_config_path = "configs/data/required_parts.config",
                save_png_dir      = str(out_dir / f"{gif_stem}_pngs"),
            )
        else:
            print("[raw_gt] No --gif flag — nothing to do. Add --gif to render.")
        return

    # ── Model-based modes (onestep / autoregressive / both) ───────────────
    if args.checkpoint is None or args.experiment is None:
        parser.error("--checkpoint and --experiment are required for modes: "
                     "onestep, autoregressive, both")

    device     = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, args.experiment, device)
    exp_name   = cfg["name"]

    # ── INPUT_FRAMES from config (must match training) ────────────────────
    global INPUT_FRAMES
    INPUT_FRAMES = int(cfg["data"].get("input_frames", INPUT_FRAMES))
    print(f"[Rollout] INPUT_FRAMES = {INPUT_FRAMES} (from config)")

    out_dir = Path(args.output_dir or
                   PROJECT_ROOT / "outputs" / "rollouts" / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    use_node_type    = bool(cfg["data"].get("node_type", False))
    node_type_field  = cfg["data"].get("node_type_field", None) if use_node_type else None
    raw_data = load_raw_h5(args.raw_h5, node_type_field=node_type_field)
    node_type = (
        torch.from_numpy(raw_data["node_type"]).to(device) if use_node_type else None
    )

    stats_path = (
        Path(args.stats_path)
        if args.stats_path
        else Path(args.checkpoint).parent / "global_stats.json"
    )
    norm_stats = NormStats.from_global_stats(
        stats_path,
        acc_scale=cfg["data"].get("acc_scale"),
    )
    normed = normalize_raw(raw_data, norm_stats)

    # ── Baseline (dt=1, per-frame; same forward-Euler convention) ─────────
    baseline = compute_baseline(raw_data, dt=DT, input_frames=INPUT_FRAMES)

    # ── Run inference ─────────────────────────────────────────────────────
    onestep = autoreg = None

    t0 = time.time()

    if args.mode in ("onestep", "both"):
        onestep = run_onestep(model, raw_data, normed, norm_stats, device, node_type)
        pkl_path = out_dir / "onestep.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(onestep, f)
        print(f"[Rollout] PKL saved → {pkl_path}")

    if args.mode in ("autoregressive", "both"):
        autoreg = run_autoregressive(model, raw_data, normed, norm_stats, device, node_type)
        pkl_path = out_dir / "autoregressive.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(autoreg, f)
        print(f"[Rollout] PKL saved → {pkl_path}")

    print(f"[Rollout] Inference done in {time.time() - t0:.1f}s")

    # ── GIF ───────────────────────────────────────────────────────────────
    _DPI = 120
    if args.gif:
        if onestep is not None:
            gif_stem = args.gif_name or "onestep"
            render_vis(
                onestep, raw_data,
                out_path          = str(out_dir / f"{gif_stem}.gif"),
                fps               = args.gif_fps,
                max_frames        = args.gif_max_frames,
                dpi               = _DPI,
                group_config_path = "configs/data/required_parts.config",
                save_png_dir      = str(out_dir / f"{gif_stem}_pngs"),
            )
        if autoreg is not None:
            # if both modes run and no custom name, suffix to distinguish them
            if args.gif_name and args.mode == "both":
                gif_stem = f"{args.gif_name}_ar"
            elif args.gif_name:
                gif_stem = args.gif_name
            else:
                gif_stem = "autoregressive"
            render_vis(
                autoreg, raw_data,
                out_path          = str(out_dir / f"{gif_stem}.gif"),
                fps               = args.gif_fps,
                max_frames        = args.gif_max_frames,
                dpi               = _DPI,
                group_config_path = "configs/data/required_parts.config",
                save_png_dir      = str(out_dir / f"{gif_stem}_pngs"),
            )

    # ── RMSE plot ─────────────────────────────────────────────────────────
    if args.plot:
        plot_rmse_vs_timestep(onestep, autoreg, out_dir)

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

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(onestep, autoreg, baseline)


if __name__ == "__main__":
    main()