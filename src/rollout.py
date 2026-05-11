"""Rollout visualization — runs one-step and autoregressive inference on raw h5 data.

Reads the original full-trajectory h5 (251 frames, not pre-windowed),
renders left=pred / right=gt GIFs colored by part, saves PKL + console stats.

Usage:
    python src/rollout.py \
        --checkpoint outputs/checkpoints/sc_003/checkpoint-best.safetensors \
        --experiment configs/experiments/sc_003.yaml \
        --raw-h5 /home/kong/datasets/barrier/h5/T_lok_F_shape_barrier_9_3_100km/output.h5 \
        --mode autoregressive \
        --gif --gif-fps 10 

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
INPUT_FRAMES  = 10
FEATURES      = ["positions", "velocity", "acceleration"]
FEAT_DIMS     = {"positions": 3, "velocity": 3, "acceleration": 3}
FEAT_SLICES   = {
    "positions":    (0,  3),
    "velocity":     (3,  6),
    "acceleration": (6,  9),
}

INPUT_FRAMES   = 10
INPUT_FEATURE  = "velocity"
TARGET_FEATURE = "acceleration"


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

def load_raw_h5(h5_path: str) -> dict:
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
    v_last_phys: np.ndarray,      # (N, 3)  物理量
    x_last_phys: np.ndarray,      # (N, 3)
    dt:          float,
    norm_stats:  NormStats,
):
    """半隐式 Euler: a -> v_new -> x_new. 返回都是物理量 (N, 3)."""
    a_phys     = norm_stats.denormalize("acceleration", a_pred_norm.cpu().numpy())
    v_new_phys = v_last_phys + a_phys * dt
    x_new_phys = x_last_phys + v_new_phys * dt        # 用新速度积分位置
    return a_phys, v_new_phys, x_new_phys


def _pack_pos_only(pos_list):
    """把 list-of-(N,3) 打包成 (T, N, 15),只填位置通道,兼容现有渲染。"""
    T = len(pos_list)
    N = pos_list[0].shape[0]
    arr = np.zeros((T, N, 15), dtype=np.float32)
    for i, p in enumerate(pos_list):
        arr[i, :, 0:3] = p
    return arr


def reconstruct_absolute(pred_residual: torch.Tensor,
                          last_frame_normed: np.ndarray,
                          norm_stats: NormStats) -> dict:
    """Convert normalized residual prediction back to physical-space absolute values.

    pred_absolute (normalized) = pred_residual + last_frame_normed
    then denormalize each feature.

    Args:
        pred_residual:    (N, 15) normalized residual from model
        last_frame_normed: (N, 15) normalized last input frame
        norm_stats:        NormStats for denormalization

    Returns:
        dict of {feat: np.ndarray (N, C)} in physical units
    """
    pred_norm = pred_residual.cpu().numpy() + last_frame_normed   # (N, 15)
    result = {}
    for feat, (s, e) in FEAT_SLICES.items():
        result[feat] = norm_stats.denormalize(feat, pred_norm[:, s:e])
    return result


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
    
    return distances


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_onestep(model, raw_data, normed, norm_stats, device) -> dict:
    T  = raw_data["positions"].shape[0]
    times = raw_data["times"]
    dt_arr = np.diff(times)
    dt_mean = float(dt_arr.mean())
    if dt_arr.std() / max(abs(dt_mean), 1e-12) > 1e-3:
        print(f"[Warn] dt 非均匀: mean={dt_mean:.6g}, std={dt_arr.std():.3g} — 用每步对应 dt")
        uniform_dt = False
    else:
        uniform_dt = True
    print(f"[One-step] dt ≈ {dt_mean:.6g}, steps = {T - INPUT_FRAMES}")

    normed_v = normed["velocity"]                       # (T, N, 3)
    pred_pos_list, gt_pos_list, rmse_pos_steps = [], [], []

    for t in range(INPUT_FRAMES, T):
        x_in        = build_velocity_input(normed_v, t - 1).to(device)

        input_pos = normed["positions"][t - INPUT_FRAMES + 1: t + 1].transpose(1, 0, 2) # (N,T,3)

        x_sdf = compute_sdf_batch(torch.from_numpy(input_pos[..., 0:2])).to(device)

        
        x = torch.cat([x_in, x_sdf.unsqueeze(0) ], dim=-1)

        a_pred_norm = model(x).squeeze(0)            # (N, 3)

        # 上一帧 GT 速度 / 位置 (物理量) — one-step 模式始终用 GT
        v_last = raw_data["velocity"][t - 1]
        x_last = raw_data["positions"][t - 1]
        dt     = dt_mean if uniform_dt else float(times[t] - times[t - 1])

        _, _, x_new = integrate_accel(a_pred_norm, v_last, x_last, dt, norm_stats)

        x_gt = raw_data["positions"][t]
        rmse = float(np.sqrt(np.mean((x_new - x_gt) ** 2)))

        pred_pos_list.append(x_new)
        gt_pos_list.append(x_gt)
        rmse_pos_steps.append(rmse)

        if (t - INPUT_FRAMES + 1) % 50 == 0:
            print(f"  step {t-INPUT_FRAMES+1}/{T-INPUT_FRAMES} | pos_rmse={rmse:.3f}")

    return {
        "pred_frames": _pack_pos_only(pred_pos_list),
        "gt_frames":   _pack_pos_only(gt_pos_list),
        "rmse_pos":    np.array(rmse_pos_steps),
        "mode":        "onestep",
    }

@torch.no_grad()
def run_autoregressive(model, raw_data, normed, norm_stats, device) -> dict:
    T     = raw_data["positions"].shape[0]
    times = raw_data["times"]
    dt_arr = np.diff(times)
    dt_mean = float(dt_arr.mean())
    uniform_dt = (dt_arr.std() / max(abs(dt_mean), 1e-12)) <= 1e-3
    print(f"[Autoregressive] dt ≈ {dt_mean:.6g}, steps = {T - INPUT_FRAMES}")

    pred_pos_list, gt_pos_list, rmse_pos_steps = [], [], []

    # ── 初始化 ──
    # 速度窗口 (归一化, 模型输入用)
    v_window_norm = normed["velocity"][:INPUT_FRAMES].copy()       # (5, N, 3)
    # 物理速度 / 位置当前状态
    v_phys = raw_data["velocity"][INPUT_FRAMES - 1].copy()         # (N, 3)
    x_phys = raw_data["positions"][INPUT_FRAMES - 1].copy()        # (N, 3)

    for t in range(INPUT_FRAMES, T):
        x_in        = build_velocity_input_from_window(v_window_norm).to(device) # (1,N,T*C)

        x_sdf = raw_data["positions"][t - INPUT_FRAMES + 1: t + 1].transpose(1, 0, 2) # (N,T,3)
        x_sdf = torch.from_numpy(x_sdf) # (N,T,3)
        x_sdf = compute_sdf_batch(x_sdf[..., 0:2]).to(device) # (N,T,2)
        
        x = torch.cat([x_in, x_sdf.unsqueeze(0) ], dim=-1)

        a_pred_norm = model(x).squeeze(0)            # (N, 3)

        dt = dt_mean if uniform_dt else float(times[t] - times[t - 1])
        _, v_phys_new, x_phys_new = integrate_accel(
            a_pred_norm, v_phys, x_phys, dt, norm_stats)

        x_gt = raw_data["positions"][t]
        rmse = float(np.sqrt(np.mean((x_phys_new - x_gt) ** 2)))

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

        if (t - INPUT_FRAMES + 1) % 50 == 0:
            print(f"  step {t-INPUT_FRAMES+1}/{T-INPUT_FRAMES} | pos_rmse={rmse:.3f}")

    return {
        "pred_frames": _pack_pos_only(pred_pos_list),
        "gt_frames":   _pack_pos_only(gt_pos_list),
        "rmse_pos":    np.array(rmse_pos_steps),
        "mode":        "autoregressive",
    }

# ── Last-frame baseline ───────────────────────────────────────────────────────

def compute_baseline(raw_data: dict) -> dict:
    """Last-frame copy baseline: predict frame t = frame t-1."""
    T   = raw_data["positions"].shape[0]
    pos = raw_data["positions"]   # (T, N, 3) # (T, N, 6)

    rmse_pos, rmse_vm = [], []
    for t in range(INPUT_FRAMES, T):
        pos_rmse = np.sqrt(np.mean((pos[t - 1] - pos[t]) ** 2))
        rmse_pos.append(pos_rmse)

    return {
        "rmse_pos": np.array(rmse_pos)
    }


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
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=dpi)
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
        ax.tick_params(colors="black", labelsize=6)
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

    axs[0, 0].set_ylabel("Z", color="black", fontsize=8)
    axs[1, 0].set_ylabel("Y", color="black", fontsize=8)
    axs[1, 0].set_xlabel("X", color="black", fontsize=8)
    axs[1, 1].set_xlabel("X", color="black", fontsize=8)

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

    title_pred_xz = axs[0, 0].set_title("", color="black", fontsize=8, pad=3)
    title_gt_xz   = axs[0, 1].set_title("", color="black", fontsize=8, pad=3)
    axs[1, 0].set_title("PRED (X-Y Plane)", color="black", fontsize=8, pad=3)
    axs[1, 1].set_title("GT (X-Y Plane)", color="black", fontsize=8, pad=3)

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

# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(onestep: dict | None, autoreg: dict | None, baseline: dict):
    print("\n" + "=" * 60)
    print("ROLLOUT SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<20} {'mean_pos_rmse':>15} {'mean_vm_rmse':>14}")
    print("-" * 60)

    print(f"{'last-frame baseline':<20} "
          f"{baseline['rmse_pos'].mean():>15.3f} ")

    if onestep is not None:
        print(f"{'one-step':<20} "
              f"{onestep['rmse_pos'].mean():>15.3f} ")

    if autoreg is not None:
        print(f"{'autoregressive':<20} "
              f"{autoreg['rmse_pos'].mean():>15.3f} ")

    print("=" * 60)
    print("(pos_rmse in physical units, vm_rmse in physical units)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BVC rollout visualization")
    parser.add_argument("--checkpoint",  required=True,
                        help="Local .safetensors checkpoint")
    parser.add_argument("--experiment",  required=True,
                        help="Experiment yaml, e.g. configs/experiments/exp_05.yaml")
    parser.add_argument("--raw-h5",      required=True,
                        help="Path to original (non-windowed) h5 trajectory")
    parser.add_argument("--mode",
                        choices=["onestep", "autoregressive", "both"],
                        default="both")
    parser.add_argument("--gif",         action="store_true",
                        help="Render GIF animations")
    parser.add_argument("--gif-fps",     type=int, default=10)
    parser.add_argument("--gif-max-frames", type=int, default=200,
                        help="Cap frames rendered (for speed)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",  default=None,
                        help="Override output directory")
    args = parser.parse_args()

    device     = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, args.experiment, device)
    exp_name   = cfg["name"]

    out_dir = Path(args.output_dir or
                   PROJECT_ROOT / "outputs" / "rollouts" / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    raw_data   = load_raw_h5(args.raw_h5)
    norm_stats = NormStats(cfg["data"]["metadata_path"])
    normed     = normalize_raw(raw_data, norm_stats)

    # ── Baseline ──────────────────────────────────────────────────────────
    baseline = compute_baseline(raw_data)

    # ── Run inference ─────────────────────────────────────────────────────
    onestep = autoreg = None

    t0 = time.time()

    if args.mode in ("onestep", "both"):
        onestep = run_onestep(model, raw_data, normed, norm_stats, device)
        pkl_path = out_dir / "onestep.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(onestep, f)
        print(f"[Rollout] PKL saved → {pkl_path}")

    if args.mode in ("autoregressive", "both"):
        autoreg = run_autoregressive(model, raw_data, normed, norm_stats, device)
        pkl_path = out_dir / "autoregressive.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(autoreg, f)
        print(f"[Rollout] PKL saved → {pkl_path}")

    print(f"[Rollout] Inference done in {time.time() - t0:.1f}s")

    # ── GIF ───────────────────────────────────────────────────────────────
    _DPI = 120
    if args.gif:
        if onestep is not None:
            render_vis(
                onestep, raw_data,
                out_path          = str(out_dir / "onestep.gif"),
                fps               = args.gif_fps,
                max_frames        = args.gif_max_frames,
                dpi               = _DPI,  # 提高 DPI 以获得极高的清晰度
                group_config_path = "configs/data/required_parts.config", # 指向你的配置文件
                save_png_dir      = str(out_dir / "onestep_pngs")    # 生成同名文件夹存放 PNG
            )
        if autoreg is not None:
            render_vis(
                autoreg, raw_data,
                out_path          = str(out_dir / "autoregressive.gif"),
                fps               = args.gif_fps,
                max_frames        = args.gif_max_frames,
                dpi               = _DPI,  # 提高 DPI 以获得极高的清晰度
                group_config_path = "configs/data/required_parts.config", # 指向你的配置文件
                save_png_dir      = str(out_dir / "autoregressive_pngs")    # 生成同名文件夹存放 PNG
            )

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(onestep, autoreg, baseline)


if __name__ == "__main__":
    main()