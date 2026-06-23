from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from accelerate import Accelerator
from tqdm import tqdm
import yaml

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT

import models  # triggers auto-import of all registered models
from models.registry import build_model
from src.conditions import CondConfig
from src.dataset import NormStats, build_dataloader, load_or_compute_global_stats, _DEFAULT_NORM_FIELDS
from src.utils.metrics import MetricTracker

# ── Barrier plate parameters (README: "Barrier plate projection on xy plate") ─
# Anchor = (x_intercept, 0): point where the barrier line crosses y = 0.
# Source: physical measurement table; degrees are impact angles.
BARRIER_PARAMS: dict[float, dict[str, float]] = {
    -25.4: {"x_intercept": 2056.579},
    -20.0: {"x_intercept": 2801.525},
    -15.0: {"x_intercept": 4078.004},
}
_DEFAULT_BARRIER_DEG: float = -25.4
_DEFAULT_PROJECT: str = "barrier-vehicle-collision"


def parse_dir_entry(entry: str) -> tuple[str, float]:
    """Parse '<h5_dir>:<barrier_angle_deg>' or plain '<h5_dir>' → (path, deg).

    The degree must match a key in BARRIER_PARAMS.  Omitting it defaults to
    _DEFAULT_BARRIER_DEG (-25.4°).  Uses rsplit so Unix paths with colons work.
    """
    if ":" in entry:
        path, deg_str = entry.rsplit(":", 1)
        try:
            deg = float(deg_str.strip())
        except ValueError:
            raise ValueError(
                f"Could not parse barrier degree from '{entry}'. "
                f"Expected '<path>:<float>', e.g. '/data/foo:-25.4'"
            )
        return path.strip(), deg
    return entry.strip(), _DEFAULT_BARRIER_DEG


def _parse_dirs(raw: list[str]) -> tuple[list[str], list[float]]:
    """Return (clean_paths, barrier_degs) from a list of '<path>[:<deg>]' entries."""
    paths, degs = [], []
    for entry in raw:
        p, deg = parse_dir_entry(entry)
        if deg not in BARRIER_PARAMS:
            raise ValueError(
                f"Barrier angle {deg}° not in BARRIER_PARAMS. "
                f"Known: {sorted(BARRIER_PARAMS.keys())}. "
                f"Add it to BARRIER_PARAMS in train.py if it's a new simulation setup."
            )
        paths.append(p)
        degs.append(deg)
    return paths, degs

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from safetensors.torch import save_file as _st_save
    _SAFETENSORS = True
except ImportError:
    _SAFETENSORS = False
    print("Warning: safetensors not installed — checkpoints will use .pt format")


# ── Git check ─────────────────────────────────────────────────────────────────

def check_git_clean() -> str:
    """Abort if working tree has uncommitted changes; return short commit hash."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if status.returncode != 0:
        print("Warning: not a git repository — skipping dirty check")
        return "unknown"
    dirty = status.stdout.strip()
    if dirty:
        raise RuntimeError(
            "Working tree has uncommitted changes — commit or stash before training.\n"
            f"Dirty files:\n{dirty}"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    ).stdout.strip()
    return commit


# ── Config loading ────────────────────────────────────────────────────────────

def _deep_set(d: dict, dotted_key: str, value):
    """Set a nested dict value using dot notation e.g. ``"train.lr"``."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def load_config(experiment_path: str) -> dict:
    """Load and merge experiment config.

    1. Read experiment yaml.
    2. Read the model params yaml it references.
    3. Apply overrides on top.
    4. Inject experiment name and model name.

    Args:
        experiment_path: Path to ``configs/experiments/*.yaml``.

    Returns:
        Merged config dict with sections: ``model``, ``train``, ``data``, ``wandb``.
    """
    exp_path = Path(experiment_path)
    if not exp_path.is_absolute():
        exp_path = PROJECT_ROOT / exp_path
    with open(exp_path) as f:
        exp = yaml.safe_load(f)

    params_path = Path(exp["model"]["params"])
    if not params_path.is_absolute():
        params_path = PROJECT_ROOT / params_path
    with open(params_path) as f:
        cfg: dict = yaml.safe_load(f)

    for dotted_key, value in (exp.get("overrides") or {}).items():
        _deep_set(cfg, dotted_key, value)

    cfg["name"]        = exp["name"]
    cfg["description"] = exp.get("description", "")
    cfg.setdefault("model", {})["name"] = exp["model"]["name"]

    return cfg


# ── W&B setup ─────────────────────────────────────────────────────────────────

def _write_meta_json(save_dir: Path, cfg: dict) -> dict:
    """Write meta.json to checkpoint dir for train→rollout group linkage; return the meta dict."""
    wandb_cfg = cfg.get("wandb", {})
    project = wandb_cfg.get("project") or os.environ.get("WANDB_PROJECT", _DEFAULT_PROJECT)
    meta = {
        "group":      cfg["name"],
        "project":    project,
        "experiment": cfg["name"],
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def setup_wandb(cfg: dict, git_commit: str, meta: dict):
    """Initialise W&B run. Returns run or None if disabled."""
    wandb_cfg = cfg.get("wandb", {})
    if not _WANDB_AVAILABLE or not wandb_cfg.get("log", True):
        return None

    run_name = wandb_cfg.get("run_name") or cfg["name"]
    return wandb.init(
        project=meta["project"],
        group=meta["group"],
        job_type="train",
        name=run_name,
        config={**cfg, "git_commit": git_commit},
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(model: torch.nn.Module, path: Path, metadata: dict | None = None):
    """Save model weights as safetensors (fallback: .pt). Saves metadata json alongside."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _SAFETENSORS:
        st_path   = path.with_suffix(".safetensors")
        cpu_state = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
        _st_save(cpu_state, str(st_path))
        if metadata:
            with open(st_path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=2)
    else:
        torch.save(model.state_dict(), str(path.with_suffix(".pt")))


# ── W&B Artifact upload ───────────────────────────────────────────────────────

def _log_best_artifact(
    run,
    meta: dict,
    save_dir: Path,
    epoch: int,
    step: int,
    val_loss: float,
    cfg: dict,
):
    """Log best checkpoint as W&B artifact with 'best' and 'epoch_N' aliases.
    No-op when wandb is disabled or unavailable.
    """
    if run is None:
        return
    try:
        artifact = wandb.Artifact(
            name=f"checkpoint-{meta['group']}",
            type="model",
            metadata={
                "epoch":      epoch,
                "step":       step,
                "val_loss":   val_loss,
                "experiment": meta["group"],
                "lr":         cfg["train"].get("lr"),
                "batch_size": cfg["train"].get("batch_size"),
            },
        )
        best_path = save_dir / "checkpoint-best"
        for ext in (".safetensors", ".pt"):
            p = best_path.with_suffix(ext)
            if p.exists():
                artifact.add_file(str(p))
                break
        meta_json = best_path.with_suffix(".json")
        if meta_json.exists():
            artifact.add_file(str(meta_json))
        run.log_artifact(artifact, aliases=[f"epoch_{epoch}", "best"])
        print(f"[Artifact] Uploaded checkpoint-{meta['group']}:best (epoch {epoch})")
    except Exception as e:
        print(f"[Artifact] Warning: upload failed — {e}")

def push_forward_step(
    v_window_norm:   torch.Tensor,    # (B, N, T_in, 3)  normalized velocity window
    a_pred_norm:     torch.Tensor,    # (B, N, 3)         normalized predicted acceleration
    v_phys_last:     torch.Tensor,    # (B, N, 3)         physical velocity at end of window
    acc_mean:        torch.Tensor,    # (3,)
    acc_std:         torch.Tensor,    # (3,)
    vel_mean:        torch.Tensor,    # (3,)
    vel_std:         torch.Tensor,    # (3,)
    dt:              float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One integration step: a_pred -> v_new -> slide window.
    
    Returns:
        v_window_norm_new: (B, N, T_in, 3)  shifted window with v_new appended
        v_phys_new:        (B, N, 3)         new physical velocity (for next step)
    """
    # Denormalize acceleration to physical
    a_phys = a_pred_norm * acc_std + acc_mean        # (B, N, 3)
    # Semi-implicit Euler
    v_phys_new = v_phys_last + a_phys * dt           # (B, N, 3)
    # Re-normalize for next input
    v_new_norm = (v_phys_new - vel_mean) / vel_std   # (B, N, 3)
    # Slide window: drop oldest frame, append new
    v_window_norm_new = torch.cat(
        [v_window_norm[:, :, 1:, :], v_new_norm.unsqueeze(2)],
        dim=2,
    )  # (B, N, T_in, 3)
    return v_window_norm_new, v_phys_new


def build_sdf_window(
    pos_window:       torch.Tensor,  # (B, N, T_in, 3)  physical positions
    barrier_angle_deg: float = _DEFAULT_BARRIER_DEG,
    x_intercept:       float = BARRIER_PARAMS[_DEFAULT_BARRIER_DEG]["x_intercept"],
) -> torch.Tensor:
    """Compute SDF for every frame in the position window.

    Returns: (B, N, T_in)  SDF per node per frame
    """
    return compute_sdf_batch(pos_window[..., :2], barrier_angle_deg, x_intercept)


# ── Loss functions ────────────────────────────────────────────────────────────
def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Per-sample relative L2, averaged over batch.

    pred, target: (B, N, D)  — same shape
    Returns: scalar
    def relative_l2(pred, target, eps=1e-6):
    return ((pred - target)**2 / (target**2 + eps)).mean()
    """

    squared_diff = (pred - target) ** 2
    denominator = (target ** 2) + eps
    
    return (squared_diff / denominator).mean()

def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Acceleration-only MSE loss.

    pred, target: (B, N, D_acc)
    """
    # loss_criterion = torch.nn.L1Loss(reduction='none')
    # loss_per_var = loss_criterion(pred, target).mean(dim=0)
    # loss = loss_per_var.mean()
    # loss = torch.nn.functional.mse_loss(pred, target)
    loss = F.smooth_l1_loss(pred, target, beta=1.0)
    return loss, {"loss_huber": loss.item()}



def compute_sdf_batch(
    xy:                torch.Tensor,
    barrier_angle_deg: float = _DEFAULT_BARRIER_DEG,
    x_intercept:       float = BARRIER_PARAMS[_DEFAULT_BARRIER_DEG]["x_intercept"],
) -> torch.Tensor:
    """Signed distance (metres) from each point to the barrier line.

    xy: (..., 2)  XY positions in mm, any leading batch dims
    barrier_angle_deg: impact angle in degrees (see BARRIER_PARAMS)
    x_intercept: x-coordinate (mm) where the barrier line crosses y = 0
                 — use BARRIER_PARAMS[deg]["x_intercept"] for each simulation.
    """
    device    = xy.device
    anchor    = torch.tensor([x_intercept, 0.0], device=device)
    angle_rad = torch.deg2rad(torch.tensor(barrier_angle_deg, device=device))
    normal_2d = torch.tensor(
        [-torch.sin(angle_rad), torch.cos(angle_rad)], device=device
    )
    diff_2d   = xy - anchor
    return (diff_2d * normal_2d).sum(dim=-1) / 1000.0


# ── Validation loop ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_validation(model, val_loader, device, use_node_type: bool = False, accelerator=None) -> dict:
    model.eval()
    total, n_batches = 0.0, 0
    for batch in val_loader:
        x_vel      = batch[0].to(device)
        future_acc = batch[1].to(device)
        input_pos  = batch[2].to(device)
        # batch[5]=barrier_angle_deg, batch[6]=x_intercept
        # batch[7]=cond (n_cond,), batch[8]=node_type (when use_node_type)
        barrier_angle_deg = float(batch[5][0].item())
        x_intercept       = float(batch[6][0].item())
        cond      = batch[7].to(device)                      # (B, n_cond)
        node_type = batch[8].to(device) if use_node_type else None

        # Validation: one-step only (k=0)
        B, N, _ = x_vel.shape
        T_in    = input_pos.shape[2]

        x_sdf = compute_sdf_batch(input_pos[..., :2], barrier_angle_deg, x_intercept)
        x_in  = torch.cat([x_vel, x_sdf], dim=-1)

        # Method A: broadcast cond to (B, N, n_cond) and concat last (D6)
        if cond.shape[-1] > 0:
            cond_b = cond[:, None, :].expand(-1, N, -1)
            x_in = torch.cat([x_in, cond_b], dim=-1)

        autocast_ctx = accelerator.autocast() if accelerator is not None else torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with autocast_ctx:
            pred = model(x_in, node_type)
            target = future_acc[:, :, 0, :]                  # first step target
            loss, _ = compute_loss(pred, target)
        total += loss.item()
        n_batches += 1

    if accelerator is not None:
        total_t   = torch.tensor(total,     device=device)
        n_t       = torch.tensor(n_batches, device=device)
        total_t   = accelerator.reduce(total_t, reduction="sum")
        n_t       = accelerator.reduce(n_t,     reduction="sum")
        return {"loss": (total_t / n_t.clamp(min=1)).item()}
    return {"loss": total / max(n_batches, 1)}

# ── Main training loop ────────────────────────────────────────────────────────
def train(cfg: dict, git_commit: str = "unknown"):
    """TransolverNet training loop (velocity → acceleration, relative L2).

    Data flow:
        BVCDataset  →  (x: B,N,D_vel)  →  TransolverNet  →  (pred: B,N,D_acc)
                       (y: B,N,D_acc)  →  relative L2 loss
    """
    accelerator = Accelerator(mixed_precision="bf16")
    device      = accelerator.device
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]

    # Consistency assertion: data.node_type and model embedding must be co-enabled
    use_node_type   = bool(data_cfg.get("node_type", False))
    has_node_emb    = model_cfg.get("num_node_types", 0) > 0 and model_cfg.get("type_emb_dim", 0) > 0
    has_partial_emb = model_cfg.get("num_node_types", 0) > 0 or model_cfg.get("type_emb_dim", 0) > 0
    if use_node_type and not has_node_emb:
        raise ValueError(
            "data.node_type=true requires model.num_node_types > 0 and model.type_emb_dim > 0"
        )
    if not use_node_type and has_partial_emb:
        raise ValueError(
            "data.node_type=false but model.num_node_types or model.type_emb_dim is configured — "
            "set both to 0 or enable data.node_type"
        )

    accelerator.print(f"[Train] Device: {device}  |  num_processes: {accelerator.num_processes}")

    # ── Condition config ─────────────────────────────────────────────────
    cond_cfg = CondConfig(**(cfg.get("condition") or {}))
    n_cond   = cond_cfg.n_cond()
    T_in     = int(data_cfg.get("input_frames", 5))
    base_features = T_in * 4   # T_in * (3 vel + 1 sdf)
    assert model_cfg["nnode_in_features"] == base_features + n_cond, (
        f"nnode_in_features={model_cfg['nnode_in_features']} != "
        f"{base_features} + n_cond={n_cond} = {base_features + n_cond}"
    )
    print(f"[Train] Condition: enabled={list(cond_cfg.enabled)}, n_cond={n_cond}, "
          f"nnode_in_features={model_cfg['nnode_in_features']}")

    # ── Build model ───────────────────────────────────────────────────────
    model = build_model(model_cfg["name"], cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model '{model_cfg['name']}' — {n_params:,} trainable params")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    # ── Data ──────────────────────────────────────────────────────────────
    # Parse '<path>[:<barrier_angle_deg>]' entries; degree defaults to -25.4°
    train_dirs, train_degs = _parse_dirs(data_cfg["train_dirs"])
    val_dirs,   val_degs   = _parse_dirs(data_cfg["val_dirs"])

    # Log barrier params for this run
    unique_degs = sorted(set(train_degs + val_degs))
    for deg in unique_degs:
        xi = BARRIER_PARAMS[deg]["x_intercept"]
        print(f"[Train] Barrier angle {deg:+.1f}°  x-intercept = {xi:.3f} mm")

    # Fail fast if train and val dirs overlap
    train_resolved = {str(Path(d).resolve()) for d in train_dirs}
    val_resolved   = {str(Path(d).resolve()) for d in val_dirs}
    overlap = train_resolved & val_resolved
    if overlap:
        raise ValueError(f"Val dirs overlap with train dirs: {overlap}")

    # Compute or load global normalization stats (train trajs only)
    run_output_dir = PROJECT_ROOT / "outputs" / "checkpoints" / cfg["name"]
    norm_fields = data_cfg.get("norm_fields", _DEFAULT_NORM_FIELDS)
    train_stats = load_or_compute_global_stats(
        train_dirs  = train_dirs,
        cache_path  = run_output_dir / "global_stats.json",
        fields      = norm_fields,
    )

    # Build per-trajectory barrier param dicts for the dataset
    train_barrier = [
        {"barrier_angle_deg": d, "x_intercept": BARRIER_PARAMS[d]["x_intercept"]}
        for d in train_degs
    ]
    val_barrier = [
        {"barrier_angle_deg": d, "x_intercept": BARRIER_PARAMS[d]["x_intercept"]}
        for d in val_degs
    ]

    # Both loaders share the same train stats (critical: val must NOT use its own stats)
    train_loader = build_dataloader(
        cfg, train_dirs,
        shuffle          = True,
        batch_size       = train_cfg.get("batch_size", 1),
        stats            = train_stats,
        barrier_params   = train_barrier,
    )
    val_loader = build_dataloader(
        cfg, val_dirs,
        shuffle          = False,
        batch_size       = train_cfg.get("val_batch_size", 1),
        stats            = train_stats,
        barrier_params   = val_barrier,
    )

    # Distribute loaders across ranks; len(train_loader) now reflects per-rank count
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    # ── Push-forward & noise config ──────────────────────────────────────
    push_K     = int(train_cfg.get("push_forward_k", 1))
    noise_std  = float(train_cfg.get("noise_std", 0.0))
    dt         = float(data_cfg.get("dt", 1))
    print(f"[Train] push_forward_k = {push_K}, noise_std = {noise_std}, dt = {dt}")
    
    # ── Pre-load normalization stats as GPU tensors (for in-graph denorm/renorm) ──
    # Uses global train stats (scalars); broadcasts correctly against (B, N, 3).
    acc_mean = torch.tensor(train_stats["acceleration"]["mean"], dtype=torch.float32, device=device)
    acc_std  = torch.tensor(train_stats["acceleration"]["std"],  dtype=torch.float32, device=device)
    vel_mean = torch.tensor(train_stats["velocity"]["mean"],     dtype=torch.float32, device=device)
    vel_std  = torch.tensor(train_stats["velocity"]["std"],      dtype=torch.float32, device=device)
    
    # 流式累加，不要堆全部 target 到内存
    sum_abs = 0.0
    sum_sq  = 0.0
    n_active_01 = 0
    n_active_05 = 0
    n_total = 0
    for batch in train_loader:
        future_acc = batch[1]              # 5- or 6-tuple
        target = future_acc[:, :, 0, :]   # k=0
        abs_t = target.abs()
        sum_abs     += abs_t.sum().item()
        sum_sq      += (target ** 2).sum().item()
        n_active_01 += (abs_t > 0.1).sum().item()
        n_active_05 += (abs_t > 0.5).sum().item()
        n_total     += target.numel()

    print(f"mean |a|:             {sum_abs / n_total:.6f}   # 期望 ≈ 0.23")
    print(f"rms  |a|:             {(sum_sq / n_total) ** 0.5:.6f}")
    print(f"active ratio (>0.1):  {n_active_01 / n_total:.4%}")
    print(f"active ratio (>0.5):  {n_active_05 / n_total:.4%}")

    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    accum_steps = int(train_cfg.get("accum_steps", 1))   
    print(f"[Train] accum_steps = {accum_steps} | effective batch = {train_cfg['batch_size'] * accum_steps}")

    n_epochs        = int(train_cfg["n_epochs"])
    steps_per_epoch = len(train_loader)

    lr     = float(train_cfg.get("lr", 1e-3))
    min_lr = float(train_cfg.get("min_lr", lr))

    # Cosine decay from lr → min_lr over n_epochs epochs (stepped once per epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs * accelerator.num_processes,
        eta_min=min_lr,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # ── Checkpoint state ──────────────────────────────────────────────────
    save_dir      = PROJECT_ROOT / "outputs" / "checkpoints" / cfg["name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "checkpoint_manifest.json"
    keep_top_k    = int(train_cfg.get("keep_top_k", 3))

    if accelerator.is_main_process:
        exp_meta      = _write_meta_json(save_dir, cfg)
        ckpt_history  = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        best_val_loss = min((r["val_loss"] for r in ckpt_history), default=float("inf"))
    else:
        exp_meta      = {}
        ckpt_history  = []
        best_val_loss = float("inf")

    # ── W&B ──────────────────────────────────────────────────────────────
    # rollout.py reads meta.json from the checkpoint dir to join the same group
    wandb_run = setup_wandb(cfg, git_commit, exp_meta) if accelerator.is_main_process else None

    val_every = int(train_cfg.get("val_every_epochs", 1))

    print(f"[Train] Starting — {n_epochs} epochs | {steps_per_epoch} steps/epoch | "
          f"batch={train_cfg['batch_size']} | lr={lr}→{min_lr} (cosine) | val_every={val_every}")

    step = 0
    model.train()

    try:
        for epoch in range(n_epochs):
            epoch_loss_sum = 0.0
            epoch_batches  = 0
            model.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch",
                        dynamic_ncols=True, leave=True, disable=not accelerator.is_main_process)
            for batch_idx, batch in enumerate(pbar):
                # ── Unpack batch (8 or 9 tensors from BVCSlicedDataset) ──
                # [0] x_vel:             (B, N, T_in*3)   normalized velocity, flattened
                # [1] future_acc:        (B, N, K, 3)     K-step normalized acceleration targets
                # [2] input_pos:         (B, N, T_in, 3)  raw input positions
                # [3] future_pos:        (B, N, K, 3)     raw future positions (for SDF rolling)
                # [4] v_last_phys:       (B, N, 3)        physical velocity at last input frame
                # [5] barrier_angle_deg: (B,)             per-trajectory barrier angle (degrees)
                # [6] x_intercept:       (B,)             barrier x-intercept (mm)
                # [7] cond:              (B, n_cond)      normalized condition vector
                # [8] node_type:         (B, N)           per-node int label (only when use_node_type)
                x_vel       = batch[0].to(device)
                future_acc  = batch[1].to(device)
                input_pos   = batch[2].to(device)
                future_pos  = batch[3].to(device)
                v_last_phys = batch[4].to(device)
                barrier_angle_deg = float(batch[5][0].item())
                x_intercept       = float(batch[6][0].item())
                cond        = batch[7].to(device)                   # (B, n_cond)
                node_type   = batch[8].to(device) if use_node_type else None

                B, N, _ = x_vel.shape
                T_in    = input_pos.shape[2]

                # Reshape x_vel back to (B, N, T_in, 3) for window manipulation
                v_window_norm = x_vel.view(B, N, T_in, 3)        # (B, N, T_in, 3)
                pos_window    = input_pos                          # (B, N, T_in, 3)
                v_phys_curr   = v_last_phys                        # (B, N, 3)

                # Method A: broadcast cond once; reuse unchanged across all unroll steps (D3)
                cond_b = cond[:, None, :].expand(-1, N, -1)       # (B, N, n_cond) — view, no copy

                # ── Push-forward K-step training loop ─────────────────────
                total_loss = 0.0
                step_losses = []   # for per-step logging

                for k in range(push_K):
                    # Noise injection: add to velocity window only on input
                    if noise_std > 0:
                        v_window_input = v_window_norm + torch.randn_like(v_window_norm) * noise_std
                    else:
                        v_window_input = v_window_norm

                    # Build model input: flatten T_in dim into channels, concat SDF, concat cond (D6)
                    x_vel_flat = v_window_input.reshape(B, N, -1)         # (B, N, T_in*3)
                    x_sdf      = build_sdf_window(                        # (B, N, T_in)
                        pos_window, barrier_angle_deg, x_intercept
                    )
                    x_in       = torch.cat([x_vel_flat, x_sdf, cond_b], dim=-1)  # (B, N, T_in*4+n_cond)
                    assert x_in.shape[-1] == model_cfg["nnode_in_features"], \
                        f"x_in dim {x_in.shape[-1]} != nnode_in_features {model_cfg['nnode_in_features']}"

                    # Forward
                    with accelerator.autocast():
                        a_pred = model(x_in, node_type)                       # (B, N, 3)

                        # Loss for this step — only nodes within threshold of barrier
                        target_k  = future_acc[:, :, k, :]                    # (B, N, 3)
                        loss_k, _ = compute_loss(a_pred, target_k)
                        total_loss = total_loss + loss_k
                        step_losses.append(loss_k.item())

                    # If not last step, prepare next iteration
                    if k < push_K - 1:
                        v_window_norm, v_phys_curr = push_forward_step(
                            v_window_norm, a_pred, v_phys_curr,
                            acc_mean, acc_std, vel_mean, vel_std, dt,
                        )
                        # Slide position window: use GT future position for SDF
                        new_pos = future_pos[:, :, k:k+1, :]               # (B, N, 1, 3)
                        pos_window = torch.cat(
                            [pos_window[:, :, 1:, :], new_pos],
                            dim=2,
                        )                                                  # (B, N, T_in, 3)

                loss = total_loss / push_K

                # ── Backward ──────────────────────────────────────────────
                # optimizer.zero_grad()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # optimizer.step()
                # scheduler.step()
                accelerator.backward(loss / accum_steps)

                is_accum_boundary = ((batch_idx + 1) % accum_steps == 0) \
                                    or (batch_idx + 1 == len(train_loader))
                if is_accum_boundary:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                step           += 1
                epoch_batches  += 1
                epoch_loss_sum += loss.item()
                lr_now          = scheduler.get_last_lr()[0]

                # ── Step log ──────────────────────────────────────────────
                wandb_log = {
                    "train/loss": loss.item(),
                    "lr":         lr_now,
                }
                for k, lk in enumerate(step_losses):
                    wandb_log[f"train/loss_step{k}"] = lk

                pbar.set_postfix(loss=f"{loss.item() if hasattr(loss, 'item') else float(loss):.4f}", lr=f"{lr_now:.2e}")

                if wandb_run:
                    wandb_run.log(wandb_log, step=step)

            # ── End of epoch: log train loss; validate every val_every epochs ──
            epoch_avg_loss = epoch_loss_sum / max(epoch_batches, 1)

            epoch_log = {"epoch": epoch + 1, "train/epoch_loss": epoch_avg_loss}

            if (epoch + 1) % val_every == 0:
                val_metrics = run_validation(model, val_loader, device, use_node_type, accelerator=accelerator)
                val_loss    = val_metrics["loss"]

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    meta_payload = {
                        "epoch":      epoch + 1,
                        "step":       step,
                        "val_loss":   val_loss,
                        "git_commit": git_commit,
                        "experiment": cfg["name"],
                    }

                    _save_checkpoint(unwrapped, save_dir / "checkpoint-latest", meta_payload)

                    ckpt_name = f"model-epoch-{epoch+1:04d}"
                    _save_checkpoint(unwrapped, save_dir / ckpt_name, meta_payload)

                    ckpt_history.append({"epoch": epoch + 1, "step": step, "val_loss": val_loss, "file": ckpt_name})
                    ckpt_history.sort(key=lambda r: r["val_loss"])
                    for stale in ckpt_history[keep_top_k:]:
                        for ext in (".safetensors", ".pt", ".json"):
                            p = save_dir / f"{stale['file']}{ext}"
                            if p.exists():
                                p.unlink()
                    ckpt_history = ckpt_history[:keep_top_k]
                    manifest_path.write_text(json.dumps(ckpt_history, indent=2))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_checkpoint(unwrapped, save_dir / "checkpoint-best", meta_payload)
                        tick = "✓ NEW BEST"
                        _log_best_artifact(wandb_run, exp_meta, save_dir, epoch + 1, step, val_loss, cfg)
                    else:
                        tick = ""

                    print(f"[Val]   Epoch {epoch+1}/{n_epochs} | "
                          f"train_loss={epoch_avg_loss:.5f} | val_loss={val_loss:.5f} | "
                          f"best={best_val_loss:.5f} {tick}")

                epoch_log.update({f"val/{k}": v for k, v in val_metrics.items()})

            scheduler.step()  # advance LR once per epoch; min_lr reached at epoch n_epochs

            if wandb_run:
                wandb_run.log(epoch_log, step=step)

    except KeyboardInterrupt:
        print("[Train] Interrupted by user")
    
    accelerator.wait_for_everyone()
    accelerator.print(f"[Train] Done — best val_loss: {best_val_loss:.5f}")
    return best_val_loss

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BVC TransolverNet Training")
    parser.add_argument("--experiment",      required=True,
                        help="Path to configs/experiments/*.yaml")
    parser.add_argument("--skip-git-check", action="store_true",
                        help="Skip git dirty check (debugging only)")
    args = parser.parse_args()

    git_commit = "unknown"
    if not args.skip_git_check:
        git_commit = check_git_clean()
        print(f"[Git] Clean — commit {git_commit}")

    cfg = load_config(args.experiment)
    print(f"[Config] Loaded experiment: {cfg['name']}")

    try:
        train(cfg, git_commit)
    finally:
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
            print("[W&B] Run finished")


if __name__ == "__main__":
    main()