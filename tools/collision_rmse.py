#!/usr/bin/env python3
"""collision_rmse.py — Collision-aware RMSE diagnostic tool.

Selects nodes with the highest mean XY-plane |acceleration| over the first
N frames of the GT trajectory (collision onset zone), then computes and
visualizes per-timestep RMSE of pos / vel / acc for those nodes vs all nodes.

Pred velocity and acceleration are derived from finite differences of the
stored predicted positions.  This matches the data convention (forward finite
differences, dt = 1 frame) used throughout the project.

Usage:
    python tools/collision_rmse.py \
        --h5  /data/curtin_ciraee/curtin_xiangrui/data/h5dt_50ns_5fs_mat/T_lok_F_shape_barrier_9_3_80km/output.h5 \
        --pkl outputs/rollouts/dg002/onestep.pkl \
        --bbox -2000 2000 0 3000 \
        --out outputs/collision_rmse/dg002.png

        outputs/rollouts/dg013/T_lok_F_shape_barrier_9_3_60km_autoregressive.pkl \

    python tools/collision_rmse.py \
        --h5 /data/curtin_ciraee/curtin_xiangrui/data/h5dt_50ns_10fs_mat/T_lok_F_shape_barrier_9_3_100km_plus800kg/output.h5  \
        --top-k 10000 --warmup-frames 1 \
        --sampling-config configs/data/sampling_config.yaml \
        --out outputs/collision_rmse/100kph800kg_10k_1.png 
        --use-rho
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import yaml as _yaml
    _YAML = True
except ImportError:
    _YAML = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _VIS = True
except ImportError:
    _VIS = False
    print("Warning: matplotlib not installed — plots will be skipped")

try:
    import h5py
except ImportError:
    print("ERROR: h5py required — pip install h5py")
    sys.exit(1)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_h5(h5_path: str) -> dict:
    """Load positions / velocity / acceleration / density from raw h5."""
    with h5py.File(h5_path, "r") as f:
        data = {
            "positions":    f["states/positions"][:].astype(np.float32),     # (T, N, 3)
            "velocity":     f["states/velocity"][:].astype(np.float32),      # (T, N, 3)
            "acceleration": f["states/acceleration"][:].astype(np.float32),  # (T, N, 3)
            "node_part_id": f["metadata/node_part_id"][:],                   # (N,)
            "node_mat_rho": f["metadata/node_mat_rho"][:].astype(np.float32) # (N,) t/mm³
            if "metadata/node_mat_rho" in f else None,
            "node_part_name": np.array([                                       # (N,) str
                n.decode("utf-8").strip("\x00").strip().lower()
                if isinstance(n, bytes) else str(n).strip().lower()
                for n in f["metadata/node_part_name"][:]
            ]),
        }
    T, N, _ = data["positions"].shape
    rho = data["node_mat_rho"]
    rho_info = (f"rho min={rho.min():.3e} max={rho.max():.3e} "
                f"(zero nodes: {(rho == 0).sum()})") if rho is not None else "no rho"
    print(f"[H5]  {T} frames, {N} nodes | {rho_info} → {h5_path}")
    return data


def load_pkl(pkl_path: str) -> dict:
    """Load rollout PKL written by rollout.py."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    T_steps, N = data["pred_frames"].shape[:2]
    print(f"[PKL] mode={data.get('mode','?')}, T_steps={T_steps}, N={N} → {pkl_path}")
    return data


def load_barrier_patterns(yaml_path: str) -> list[str]:
    """Read barrier part-name keywords from sampling_config.yaml."""
    if not _YAML:
        raise ImportError("pyyaml required — pip install pyyaml")
    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f)
    patterns = []
    for section in cfg.values():
        if isinstance(section, dict) and "barrier_parts" in section:
            patterns.extend(section["barrier_parts"])
    return [p.lower() for p in patterns]


def build_group_masks(node_part_names: np.ndarray,
                      barrier_patterns: list[str]) -> dict[str, np.ndarray]:
    """Return boolean masks for 'barrier' and 'car' node groups.

    Matching: a node belongs to 'barrier' if its (lowercased) part name
    contains any barrier pattern as a substring.
    """
    barrier_mask = np.zeros(len(node_part_names), dtype=bool)
    for pat in barrier_patterns:
        barrier_mask |= np.array([pat in n for n in node_part_names])
    car_mask = ~barrier_mask
    print(f"[Groups] barrier={barrier_mask.sum()}  car={car_mask.sum()}  "
          f"total={len(node_part_names)}")
    return {"car": car_mask, "barrier": barrier_mask}


# ── Node selection ────────────────────────────────────────────────────────────

def select_nodes(
    positions_t0:  np.ndarray,          # (N, 3)  GT positions at frame 0
    gt_acc:        np.ndarray,          # (T, N, 3)
    warmup_frames: int,
    top_k:         int,
    bbox:          tuple | None = None, # (xmin, xmax, ymin, ymax) in mm
    node_rho:      np.ndarray | None = None,  # (N,) per-node density
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Select nodes; return (sorted indices, mean_acc, score, label).

    Ranking score:
      - node_rho is None  →  score = mean |acc|  over warmup_frames
      - node_rho given    →  score = mean |acc| × ρ   (nodes with ρ=0 get score=0)

    bbox mode: spatial filter at t=0, top_k ignored.

    Returns mean_acc and score separately so the caller can colour by either.
    """
    acc_mag  = np.linalg.norm(gt_acc[:warmup_frames, :, :], axis=-1)  # (W, N)
    mean_acc = acc_mag.mean(axis=0)                                     # (N,)

    if node_rho is not None:
        score = mean_acc * node_rho          # (N,)  zero where rho=0
        score_name = "|acc|·ρ"
        score_unit = "t/mm²/frame²"          # acc_stored[mm/frame²] × ρ[t/mm³]
    else:
        score = mean_acc
        score_name = "|acc|"
        score_unit = "mm/frame²"             # per-frame 2nd finite diff of position

    if bbox is not None:
        xmin, xmax, ymin, ymax = bbox
        mask    = (
            (positions_t0[:, 0] >= xmin) & (positions_t0[:, 0] <= xmax) &
            (positions_t0[:, 1] >= ymin) & (positions_t0[:, 1] <= ymax)
        )
        top_idx = np.where(mask)[0]
        label   = f"bbox x=[{xmin},{xmax}] y=[{ymin},{ymax}]"
        print(f"[Select] bbox mode | {label}")
        print(f"[Select] {len(top_idx)} nodes selected from {len(positions_t0)} total")
        print(f"[Select] mean {score_name} of selected: {score[top_idx].mean():.4e} {score_unit}")
    else:
        rank    = np.argsort(score)[::-1]
        top_idx = np.sort(rank[:top_k])
        thresh  = score[rank[top_k - 1]]
        label   = f"top-{top_k} by {score_name}"
        print(f"[Select] acc-rank mode | score={score_name} | warmup={warmup_frames} frames")
        print(f"[Select] {score_name} — max={score[rank[0]]:.4e}  "
              f"threshold={thresh:.4e}  global-mean={score.mean():.4e}  {score_unit}")

    return top_idx, mean_acc, score, label


# ── Per-node RMSE ─────────────────────────────────────────────────────────────

def _fd_vel_acc(pos: np.ndarray, dt: float = 1.0):
    """Forward finite differences: vel (T-1, N, 3), acc (T-2, N, 3)."""
    vel = (pos[1:] - pos[:-1]) / dt
    acc = (vel[1:] - vel[:-1]) / dt
    return vel, acc


def compute_per_node_rmse(
    pred_pos: np.ndarray,   # (T_steps, N, 3)  from PKL
    gt_pos:   np.ndarray,   # (T_steps, N, 3)  from PKL gt_frames
    gt_vel:   np.ndarray,   # (T_steps, N, 3)  from H5, already aligned
    gt_acc:   np.ndarray,   # (T_steps, N, 3)  from H5, already aligned
) -> dict:
    """Per-timestep, per-node RMSE for pos / vel / acc.

    Pred vel and acc are derived via finite differences of predicted positions.
    GT vel/acc come directly from the H5 (no differencing needed).

    Returns:
        rmse_pos: (T_steps,   N)
        rmse_vel: (T_steps-1, N)   — one fewer due to first diff
        rmse_acc: (T_steps-2, N)   — two fewer due to second diff
    """
    rmse_pos = np.sqrt(((pred_pos - gt_pos) ** 2).mean(axis=-1))    # (T, N)

    pred_vel, pred_acc = _fd_vel_acc(pred_pos)                        # (T-1, N, 3), (T-2, N, 3)
    Tv, Ta = pred_vel.shape[0], pred_acc.shape[0]

    rmse_vel = np.sqrt(((pred_vel - gt_vel[:Tv]) ** 2).mean(axis=-1))  # (T-1, N)
    rmse_acc = np.sqrt(((pred_acc - gt_acc[:Ta]) ** 2).mean(axis=-1))  # (T-2, N)

    return {"rmse_pos": rmse_pos, "rmse_vel": rmse_vel, "rmse_acc": rmse_acc}


# ── CLI report ────────────────────────────────────────────────────────────────

def print_report(
    per_node:   dict,
    top_idx:    np.ndarray,
    mode:       str,
    group_sels: dict[str, np.ndarray] | None = None,
):
    """Print mean RMSE summary + per-timestep tables.

    group_sels: optional dict {group_name: top_idx_array} — when provided,
    prints one row per group in the summary table and one per-timestep table
    per group.
    """
    rp  = per_node["rmse_pos"]   # (T_steps, N)
    rv  = per_node["rmse_vel"]   # (T_steps-1, N)
    ra  = per_node["rmse_acc"]   # (T_steps-2, N)

    def mean_sub(a, idx):  return float(a[:, idx].mean())
    def fmt(v: float, w: int = 14) -> str:  return f"{v:>{w}.4e}"
    def fmt_row(p, v, a):  return f"  {p:>12.4e}  {v:>14.4e}  {a:>14.4e}"

    W = 80
    print(f"\n{'='*W}")
    print(f"  COLLISION-AWARE RMSE REPORT  [{mode.upper()}]")
    print(f"{'='*W}")
    print(f"  {'Subset':<22} {'pos RMSE (mm)':>14} {'vel RMSE (mm/dt)':>17} {'acc RMSE (mm/dt²)':>18}")
    print(f"  {'-'*(W-2)}")

    all_idx = np.arange(rp.shape[1])
    rows = [("all nodes", all_idx), (f"sel ({len(top_idx)})", top_idx)]
    if group_sels:
        for gname, gidx in group_sels.items():
            rows.append((f"  {gname} ({len(gidx)})", gidx))

    for label, idx in rows:
        p = mean_sub(rp, idx); v = mean_sub(rv, idx); a = mean_sub(ra, idx)
        print(f"  {label:<22}{fmt(p)}{fmt(v,17)}{fmt(a,18)}")

    # ratio rows
    p_all = mean_sub(rp, all_idx)
    v_all = mean_sub(rv, all_idx)
    a_all = mean_sub(ra, all_idx)
    for label, idx in rows[1:]:
        rp_ = mean_sub(rp, idx) / (p_all + 1e-12)
        rv_ = mean_sub(rv, idx) / (v_all + 1e-12)
        ra_ = mean_sub(ra, idx) / (a_all + 1e-12)
        print(f"  {f'ratio {label.strip()}':<22} {rp_:>14.3f}× {rv_:>16.3f}× {ra_:>17.3f}×")
    print(f"{'='*W}")

    T = rp.shape[0]
    stride = max(1, T // 20)

    def _print_ts_table(label, idx):
        print(f"\n  Per-timestep RMSE — {label}:")
        print(f"  {'step':>6}  {'pos (mm)':>12}  {'vel (mm/dt)':>14}  {'acc (mm/dt²)':>14}")
        for t in range(0, T, stride):
            vv = float(rv[t, idx].mean()) if t < rv.shape[0] else float("nan")
            av = float(ra[t, idx].mean()) if t < ra.shape[0] else float("nan")
            print(f"  {t+1:>6}" + fmt_row(float(rp[t, idx].mean()), vv, av))

    _print_ts_table("ALL nodes", all_idx)
    _print_ts_table(f"SEL ({len(top_idx)}) nodes", top_idx)
    if group_sels:
        for gname, gidx in group_sels.items():
            _print_ts_table(f"{gname} top-{len(gidx)}", gidx)


# ── Visualization ─────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family":     "DejaVu Serif",
    "font.size":       10,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


def visualize(
    h5_data:       dict,
    pkl_data:      dict,
    per_node:      dict,
    top_idx:       np.ndarray,
    score:         np.ndarray,   # (N,) ranking score used (acc or acc*rho)
    sel_label:     str,
    warmup_frames: int,
    out_path:      str,
    input_frames:  int = 5,
    group_sels:    dict[str, np.ndarray] | None = None,
):
    """Save multi-panel diagnostic figure.

    Layout (3 rows × 3 cols):
      [0,0] Selection map — XY scatter, top-k coloured by mean |acc_xy|
      [0,1] pos RMSE curves — all nodes vs top-k
      [0,2] vel RMSE curves — all nodes vs top-k
      [1,0] acc RMSE curves — all nodes vs top-k
      [1,1] XY snapshot at  ~0%  of trajectory (GT blue / pred red)
      [1,2] XY snapshot at ~33%  of trajectory
      [2,0] XY snapshot at ~66%  of trajectory
      [2,1] XY snapshot at ~100% of trajectory
      [2,2] Per-node pos-error heat map at final timestep (top-k only)
    """
    if not _VIS:
        print("[Vis] matplotlib not available — skipping")
        return

    plt.rcParams.update(_RCPARAMS)

    N          = h5_data["positions"].shape[1]
    T_steps    = pkl_data["pred_frames"].shape[0]
    mode       = pkl_data.get("mode", "?")
    k          = len(top_idx)

    pred_pos   = pkl_data["pred_frames"][:, :, 0:3]   # (T_steps, N, 3)
    gt_pos_pk  = pkl_data["gt_frames"][:, :, 0:3]     # (T_steps, N, 3)

    rp = per_node["rmse_pos"]   # (T_steps, N)
    rv = per_node["rmse_vel"]   # (T_steps-1, N)
    ra = per_node["rmse_acc"]   # (T_steps-2, N)

    steps_p = np.arange(1, T_steps + 1)
    steps_v = np.arange(1, rv.shape[0] + 1)
    steps_a = np.arange(1, ra.shape[0] + 1)

    rmse_pos_all = rp.mean(axis=1)
    rmse_pos_top = rp[:, top_idx].mean(axis=1)
    rmse_vel_all = rv.mean(axis=1)
    rmse_vel_top = rv[:, top_idx].mean(axis=1)
    rmse_acc_all = ra.mean(axis=1)
    rmse_acc_top = ra[:, top_idx].mean(axis=1)

    # Use t=0 for selection map (matches bbox filter criterion)
    init_frame  = h5_data["positions"][0]  # (N, 3)
    not_top     = np.ones(N, dtype=bool)
    not_top[top_idx] = False

    # Key timestep indices for snapshots
    snap_steps = [0,
                  T_steps // 3,
                  2 * T_steps // 3,
                  T_steps - 1]

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ── [0,0] Selection map ────────────────────────────────────────────────
    _GROUP_COLORS = {"car": "#d62728", "barrier": "#e69f00", "_sel": "#d62728"}
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(init_frame[not_top, 0], init_frame[not_top, 1],
               s=0.2, c="#4878d0", alpha=0.25, linewidths=0, label="other")
    if group_sels:
        for gname, gidx in group_sels.items():
            gc = _GROUP_COLORS.get(gname, "#d62728")
            ax.scatter(init_frame[gidx, 0], init_frame[gidx, 1],
                       s=2.5, c=gc, alpha=0.85, linewidths=0,
                       label=f"{gname} ({len(gidx)})")
    else:
        ax.scatter(init_frame[top_idx, 0], init_frame[top_idx, 1],
                   s=2.5, c="#d62728", alpha=0.85, linewidths=0,
                   label=f"selected ({k})")
    ax.set_title(f"Selected nodes\n{sel_label}")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, markerscale=3)

    # ── [0,1] pos RMSE curves ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps_p, rmse_pos_all, color="#1f77b4", lw=1.5, label="all nodes")
    ax.plot(steps_p, rmse_pos_top, color="#d62728", lw=1.5, ls="--", label=f"sel ({k})")
    if group_sels:
        for gname, gidx in group_sels.items():
            gc = _GROUP_COLORS.get(gname, "#888888")
            ax.plot(steps_p, rp[:, gidx].mean(1), color=gc, lw=1.2,
                    ls=":", label=f"{gname} ({len(gidx)})")
    ax.set_title(f"Position RMSE [{mode}]")
    ax.set_xlabel("Timestep"); ax.set_ylabel("RMSE (mm)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    # ── [0,2] vel RMSE curves ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(steps_v, rmse_vel_all, color="#1f77b4", lw=1.5, label="all nodes")
    ax.plot(steps_v, rmse_vel_top, color="#d62728", lw=1.5, ls="--", label=f"sel ({k})")
    if group_sels:
        for gname, gidx in group_sels.items():
            gc = _GROUP_COLORS.get(gname, "#888888")
            ax.plot(steps_v, rv[:, gidx].mean(1), color=gc, lw=1.2,
                    ls=":", label=f"{gname} ({len(gidx)})")
    ax.set_title(f"Velocity RMSE [{mode}]")
    ax.set_xlabel("Timestep"); ax.set_ylabel("RMSE (mm/dt)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    # ── [1,0] acc RMSE curves ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps_a, rmse_acc_all, color="#1f77b4", lw=1.5, label="all nodes")
    ax.plot(steps_a, rmse_acc_top, color="#d62728", lw=1.5, ls="--", label=f"sel ({k})")
    if group_sels:
        for gname, gidx in group_sels.items():
            gc = _GROUP_COLORS.get(gname, "#888888")
            ax.plot(steps_a, ra[:, gidx].mean(1), color=gc, lw=1.2,
                    ls=":", label=f"{gname} ({len(gidx)})")
    ax.set_title(f"Acceleration RMSE [{mode}]")
    ax.set_xlabel("Timestep"); ax.set_ylabel("RMSE (mm/dt²)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    # ── [1,1], [1,2], [2,0], [2,1] — XY snapshots of top-k nodes ─────────
    snap_axes = [
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    for ax, t_idx in zip(snap_axes, snap_steps):
        gt_xy   = gt_pos_pk[t_idx, top_idx, :2]    # (k, 2)
        pred_xy = pred_pos[t_idx,  top_idx, :2]    # (k, 2)
        ax.scatter(gt_xy[:, 0],   gt_xy[:, 1],
                   s=1.5, c="#1f77b4", alpha=0.65, linewidths=0, label="GT")
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1],
                   s=1.5, c="#d62728", alpha=0.65, linewidths=0, label="Pred")
        pos_err = float(rp[t_idx, top_idx].mean())
        pct     = int(round(t_idx / max(T_steps - 1, 1) * 100))
        ax.set_title(f"t={t_idx+1} (~{pct}%) | pos_rmse={pos_err:.1f} mm")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
        if t_idx == snap_steps[0]:
            ax.legend(markerscale=4, loc="best")

    # ── [2,2] Per-node pos-error heat map at final timestep ───────────────
    ax = fig.add_subplot(gs[2, 2])
    final_t     = T_steps - 1
    node_err    = rp[final_t, top_idx]               # (k,)
    gt_xy_final = gt_pos_pk[final_t, top_idx, :2]   # (k, 2)
    sc2 = ax.scatter(gt_xy_final[:, 0], gt_xy_final[:, 1],
                     s=3.0, c=node_err, cmap="plasma",
                     alpha=0.85, linewidths=0)
    plt.colorbar(sc2, ax=ax, label="pos error (mm)", fraction=0.046, pad=0.04)
    ax.set_title(f"Per-node pos error (GT positions)\nt={final_t+1} (final)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Collision-Aware RMSE | mode={mode} | {k} nodes | {sel_label}",
        fontsize=11, fontweight="bold",
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Vis] Saved → {out_path}")


# ── Combined comparison figure (when multiple PKLs are given) ─────────────────

def visualize_compare(
    all_results: list[tuple[str, dict, dict]],   # (pkl_stem, per_node, pkl_data)
    top_idx:     np.ndarray,
    out_path:    str,
):
    """3×2 RMSE curves comparing multiple PKL files (all nodes and top-k)."""
    if not _VIS or len(all_results) < 2:
        return

    plt.rcParams.update(_RCPARAMS)
    k    = len(top_idx)
    cmap = plt.get_cmap("tab10", len(all_results))

    fig, axs = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(f"Multi-PKL RMSE Comparison | top-{k} nodes highlighted",
                 fontsize=12, fontweight="bold")

    feats  = ["pos",  "vel",  "acc"]
    yunits = ["mm",   "mm/dt", "mm/dt²"]
    rkeys  = ["rmse_pos", "rmse_vel", "rmse_acc"]

    for row, (feat, unit, rkey) in enumerate(zip(feats, yunits, rkeys)):
        ax_all = axs[row, 0]
        ax_top = axs[row, 1]
        ax_all.set_title(f"{feat.capitalize()} RMSE — all nodes ({unit})")
        ax_top.set_title(f"{feat.capitalize()} RMSE — top-{k} nodes ({unit})")
        for ax in (ax_all, ax_top):
            ax.set_xlabel("Timestep"); ax.set_ylabel(f"RMSE ({unit})")
            ax.grid(True, alpha=0.25)

        for i, (stem, per_node, _) in enumerate(all_results):
            arr = per_node[rkey]              # (T, N)
            steps = np.arange(1, arr.shape[0] + 1)
            ax_all.plot(steps, arr.mean(axis=1),
                        color=cmap(i), lw=1.5, label=stem)
            ax_top.plot(steps, arr[:, top_idx].mean(axis=1),
                        color=cmap(i), lw=1.5, label=stem)

        ax_all.legend(); ax_top.legend()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Vis] Comparison plot → {out_path}")


# ── Selection-only visualizer ────────────────────────────────────────────────

def visualize_selection_only(
    h5_data:    dict,
    top_idx:    np.ndarray,
    score:      np.ndarray,
    sel_label:  str,
    out_path:   str,
    group_sels: dict[str, np.ndarray] | None = None,
):
    """Single-panel figure: XY scatter coloured by group (car=red, barrier=orange)."""
    if not _VIS:
        print("[Vis] matplotlib not available — skipping"); return

    plt.rcParams.update(_RCPARAMS)

    _GROUP_COLORS = {"car": "#d62728", "barrier": "#e69f00"}

    pos0    = h5_data["positions"][0]   # (N, 3)
    N       = pos0.shape[0]
    sel_set = set(top_idx.tolist())
    not_top = np.array([i not in sel_set for i in range(N)])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pos0[not_top, 0], pos0[not_top, 1],
               s=0.3, c="#4878d0", alpha=0.25, linewidths=0, label="other")

    if group_sels:
        for gname, gidx in group_sels.items():
            gc = _GROUP_COLORS.get(gname, "#d62728")
            ax.scatter(pos0[gidx, 0], pos0[gidx, 1],
                       s=3.0, c=gc, alpha=0.9, linewidths=0,
                       label=f"{gname} ({len(gidx)})")
    else:
        ax.scatter(pos0[top_idx, 0], pos0[top_idx, 1],
                   s=3.0, c="#d62728", alpha=0.9, linewidths=0,
                   label=f"selected ({len(top_idx)})")

    ax.set_title(f"Node selection at t=0\n{sel_label}", fontsize=12)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, markerscale=4)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Vis] Selection map saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collision-aware RMSE diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h5",            required=True,
                        help="Raw h5 trajectory file path")
    parser.add_argument("--pkl",           required=False, nargs="*", default=[],
                        help="One or more rollout PKL files to analyse. "
                             "Omit to only render the node-selection map.")
    parser.add_argument("--top-k",         type=int, default=500,
                        help="Nodes to select when using acc-rank mode (default: 500, "
                             "ignored when --bbox is given)")
    parser.add_argument("--warmup-frames", type=int, default=10,
                        help="GT frames used to rank nodes by |acc_xy| in acc-rank mode "
                             "(default: 10; also used for vis colouring in bbox mode)")
    parser.add_argument("--bbox",          type=float, nargs=4,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        default=None,
                        help="Select nodes whose XY position at t=0 falls inside this "
                             "bounding box (mm).  Example: --bbox -3000 3000 0 3000.  "
                             "When given, --top-k is ignored.")
    parser.add_argument("--use-rho",        action="store_true",
                        help="Weight ranking score by material density: score = |acc|·ρ. "
                             "Requires node_mat_rho in the h5 metadata.")
    parser.add_argument("--sampling-config", default=None,
                        metavar="YAML",
                        help="Path to sampling_config.yaml.  When given, car and barrier "
                             "nodes are split and each group gets its own top-k selection.")
    parser.add_argument("--input-frames",  type=int, default=5,
                        help="INPUT_FRAMES used during rollout (default: 5)")
    parser.add_argument("--out",           default="outputs/collision_rmse/result.png",
                        help="Output PNG base path (stem is suffixed per PKL)")
    args = parser.parse_args()

    # ── Load GT data ──────────────────────────────────────────────────────
    h5_data = load_h5(args.h5)

    bbox     = tuple(args.bbox) if args.bbox is not None else None
    node_rho = h5_data["node_mat_rho"] if args.use_rho else None
    if args.use_rho and node_rho is None:
        print("[Warn] --use-rho requested but node_mat_rho not found in h5; falling back to |acc|")

    # ── Build car/barrier group masks (optional) ──────────────────────────
    group_masks: dict[str, np.ndarray] | None = None
    if args.sampling_config:
        barrier_pats = load_barrier_patterns(args.sampling_config)
        group_masks  = build_group_masks(h5_data["node_part_name"], barrier_pats)

    # ── Select nodes (overall + per-group when config given) ──────────────
    top_idx, mean_acc, score, sel_label = select_nodes(
        positions_t0  = h5_data["positions"][0],
        gt_acc        = h5_data["acceleration"],
        warmup_frames = args.warmup_frames,
        top_k         = args.top_k,
        bbox          = bbox,
        node_rho      = node_rho,
    )

    # Per-group top-k selections (same top_k applied within each group)
    group_sels: dict[str, np.ndarray] | None = None
    if group_masks:
        group_sels = {}
        for gname, gmask in group_masks.items():
            g_indices = np.where(gmask)[0]
            g_score   = score[g_indices]
            k_g       = min(args.top_k, len(g_indices))
            g_rank    = np.argsort(g_score)[::-1]
            g_top     = np.sort(g_indices[g_rank[:k_g]])
            group_sels[gname] = g_top
            print(f"[Groups] {gname}: {len(g_indices)} nodes → top-{k_g} selected "
                  f"| score max={g_score.max():.4e} threshold={g_score[g_rank[k_g-1]]:.4e}")
        # combined union for overall top_idx when groups are used
        top_idx = np.sort(np.unique(np.concatenate(list(group_sels.values()))))
        sel_label = f"car+barrier top-{args.top_k} each"

    out_base = Path(args.out)

    # ── Selection-only mode (no PKL given) ────────────────────────────────
    if not args.pkl:
        visualize_selection_only(
            h5_data, top_idx, score, sel_label,
            out_path   = str(out_base),
            group_sels = group_sels,
        )
        return

    all_results: list[tuple[str, dict, dict]] = []

    # ── Per-PKL analysis ──────────────────────────────────────────────────
    for pkl_path in args.pkl:
        pkl_data = load_pkl(pkl_path)
        T_steps  = pkl_data["pred_frames"].shape[0]
        mode     = pkl_data.get("mode", "?")
        stem     = Path(pkl_path).stem

        pred_pos = pkl_data["pred_frames"][:, :, 0:3]   # (T_steps, N, 3)
        gt_pos   = pkl_data["gt_frames"][:, :, 0:3]     # (T_steps, N, 3)

        # Align H5 GT vel/acc to the eval window [input_frames, T_h5-2)
        t0 = args.input_frames
        gt_vel = h5_data["velocity"][t0: t0 + T_steps]      # (T_steps, N, 3)
        gt_acc = h5_data["acceleration"][t0: t0 + T_steps]  # (T_steps, N, 3)

        per_node = compute_per_node_rmse(pred_pos, gt_pos, gt_vel, gt_acc)

        print_report(per_node, top_idx, mode, group_sels=group_sels)

        out_path = str(out_base.parent / f"{out_base.stem}_{stem}{out_base.suffix}")
        visualize(
            h5_data, pkl_data, per_node, top_idx, score,
            sel_label     = sel_label,
            warmup_frames = args.warmup_frames,
            out_path      = out_path,
            input_frames  = args.input_frames,
            group_sels    = group_sels,
        )

        all_results.append((stem, per_node, pkl_data))

    # ── Multi-PKL comparison plot ─────────────────────────────────────────
    if len(all_results) >= 2:
        compare_path = str(out_base.parent / f"{out_base.stem}_compare{out_base.suffix}")
        visualize_compare(all_results, top_idx, compare_path)


if __name__ == "__main__":
    main()
