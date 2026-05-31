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
from tqdm import tqdm
import yaml

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT

import models  # triggers auto-import of all registered models
from models.registry import build_model
from src.dataset import NormStats, build_dataloader, load_or_compute_global_stats, _DEFAULT_NORM_FIELDS
from src.utils.metrics import MetricTracker

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

def setup_wandb(cfg: dict, git_commit: str):
    """Initialise W&B run. Returns run or None if disabled."""
    wandb_cfg = cfg.get("wandb", {})
    if not _WANDB_AVAILABLE or not wandb_cfg.get("log", True):
        return None

    project = wandb_cfg.get("project") or os.environ.get("WANDB_PROJECT", "my-project")
    return wandb.init(
        project=project,
        name=cfg["name"],
        config={**cfg, "git_commit": git_commit},
        reinit=True,
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

def upload_artifact(cfg: dict, run, git_commit: str, val_loss: float):
    """Upload best checkpoint + config to W&B Artifacts."""
    if run is None:
        return

    model_name = cfg["model"]["name"]
    artifact   = wandb.Artifact(
        name=model_name,
        type="model",
        metadata={
            "git_commit":  git_commit,
            "experiment":  cfg["name"],
            "val_loss":    val_loss,
            "description": cfg.get("description", ""),
            "model_name":  cfg["model"]["name"],
        },
    )

    save_dir  = PROJECT_ROOT / "outputs" / "checkpoints" / cfg["name"]
    best_path = save_dir / "checkpoint-best"
    for ext in (".safetensors", ".pt"):
        p = best_path.with_suffix(ext)
        if p.exists():
            artifact.add_file(str(p))
            break

    meta_json = best_path.with_suffix(".json")
    if meta_json.exists():
        artifact.add_file(str(meta_json))

    run.log_artifact(artifact)
    print(f"[Artifact] Uploaded '{model_name}' to W&B Artifacts")

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
    pos_window: torch.Tensor,    # (B, N, T_in, 3)  physical positions
) -> torch.Tensor:
    """Compute SDF for every frame in the position window.
    
    Returns: (B, N, T_in)  SDF per node per frame
    """
    return compute_sdf_batch(pos_window[..., :2])  


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



def compute_sdf_batch(xy: torch.Tensor, 
                    barrier_angle_deg: float=-25.4, 
                    barrier_anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    sdf: signed distance field
    car_points: (B, N, 2) 整个 Batch 的车辆点云，必须已经在 GPU 上
    barrier_angle_deg: 护栏角度 (标量) impace degree -25.4
    barrier_anchor: (2,) 护栏基准点，必须在 GPU 上 xy=(0,2000)
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


# ── Validation loop ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_validation(model, val_loader, device) -> dict:
    model.eval()
    total, n_batches = 0.0, 0
    for batch in val_loader:
        x_vel, future_acc, input_pos, future_pos, v_last_phys = batch
        x_vel      = x_vel.to(device)
        future_acc = future_acc.to(device)
        input_pos  = input_pos.to(device)
        
        # Validation: one-step only (k=0)
        B, N, _ = x_vel.shape
        T_in    = input_pos.shape[2]
        
        x_sdf = compute_sdf_batch(input_pos[..., :2])   # (B, N, T_in)
        x_in  = torch.cat([x_vel, x_sdf], dim=-1)
        # x_in = x_vel  # ablation: no SDF
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x_in)
            target = future_acc[:, :, 0, :]                  # first step target
            loss, _ = compute_loss(pred, target)
        total += loss.item()
        n_batches += 1
    return {"loss": total / max(n_batches, 1)}

# ── Main training loop ────────────────────────────────────────────────────────
def train(cfg: dict, git_commit: str = "unknown"):
    """TransolverNet training loop (velocity → acceleration, relative L2).

    Data flow:
        BVCDataset  →  (x: B,N,D_vel)  →  TransolverNet  →  (pred: B,N,D_acc)
                       (y: B,N,D_acc)  →  relative L2 loss
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]

    print(f"[Train] Device: {device}")

    # ── Build model ───────────────────────────────────────────────────────
    model = build_model(model_cfg["name"], cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model '{model_cfg['name']}' — {n_params:,} trainable params")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    # ── Data ──────────────────────────────────────────────────────────────
    train_dirs = data_cfg["train_dirs"]
    val_dirs   = data_cfg["val_dirs"]

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

    # Both loaders share the same train stats (critical: val must NOT use its own stats)
    train_loader = build_dataloader(
        cfg, train_dirs,
        shuffle    = True,
        batch_size = train_cfg.get("batch_size", 1),
        stats      = train_stats,
    )
    val_loader = build_dataloader(
        cfg, val_dirs,
        shuffle    = False,
        batch_size = train_cfg.get("val_batch_size", 1),
        stats      = train_stats,
    )

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
        _, future_acc, _, _, _ = batch    # 5 元组
        target = future_acc[:, :, 0, :]    # k=0
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
    accum_steps = int(train_cfg.get("accum_steps", 1))   # <-- 新增
    print(f"[Train] accum_steps = {accum_steps} | effective batch = {train_cfg['batch_size'] * accum_steps}")

    n_epochs        = int(train_cfg["n_epochs"])
    steps_per_epoch = len(train_loader)
    opt_steps_per_epoch = (steps_per_epoch + accum_steps - 1) // accum_steps   # ceil
    total_steps     = n_epochs * opt_steps_per_epoch

    lr     = float(train_cfg.get("lr", 1e-3))
    min_lr = float(train_cfg.get("min_lr", lr))
    div_factor = 25

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.01,
        div_factor=div_factor,
        final_div_factor=(lr / div_factor) / min_lr,
    )

    # ── Checkpoint state ──────────────────────────────────────────────────
    save_dir      = PROJECT_ROOT / "outputs" / "checkpoints" / cfg["name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "checkpoint_manifest.json"
    ckpt_history  = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    best_val_loss = min((r["val_loss"] for r in ckpt_history), default=float("inf"))
    keep_top_k    = int(train_cfg.get("keep_top_k", 3))

    # ── W&B ──────────────────────────────────────────────────────────────
    wandb_run = setup_wandb(cfg, git_commit)

    val_every = int(train_cfg.get("val_every_epochs", 1))

    print(f"[Train] Starting — {n_epochs} epochs | {steps_per_epoch} steps/epoch | "
          f"batch={train_cfg['batch_size']} | lr={train_cfg['lr']} | val_every={val_every}")

    step = 0
    model.train()

    try:
        for epoch in range(n_epochs):
            epoch_loss_sum = 0.0
            epoch_batches  = 0
            model.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch",
                        dynamic_ncols=True, leave=True)
            for batch_idx, batch in enumerate(pbar):
                # ── Unpack batch (5 tensors from new BVCSlicedDataset) ────
                x_vel, future_acc, input_pos, future_pos, v_last_phys = batch
                # x_vel:       (B, N, T_in*3)   normalized velocity, flattened
                # future_acc:  (B, N, K, 3)     K-step normalized acceleration targets
                # input_pos:   (B, N, T_in, 3)  raw input positions
                # future_pos:  (B, N, K, 3)     raw future positions (for SDF rolling)
                # v_last_phys: (B, N, 3)        physical velocity at last input frame

                x_vel       = x_vel.to(device)
                future_acc  = future_acc.to(device)
                input_pos   = input_pos.to(device)
                future_pos  = future_pos.to(device)
                v_last_phys = v_last_phys.to(device)

                B, N, _ = x_vel.shape
                T_in    = input_pos.shape[2]

                # Reshape x_vel back to (B, N, T_in, 3) for window manipulation
                v_window_norm = x_vel.view(B, N, T_in, 3)        # (B, N, T_in, 3)
                pos_window    = input_pos                          # (B, N, T_in, 3)
                v_phys_curr   = v_last_phys                        # (B, N, 3)

                # ── Push-forward K-step training loop ─────────────────────
                total_loss = 0.0
                step_losses = []   # for per-step logging

                for k in range(push_K):
                    # Noise injection: add to velocity window only on input
                    if noise_std > 0:
                        v_window_input = v_window_norm + torch.randn_like(v_window_norm) * noise_std
                    else:
                        v_window_input = v_window_norm

                    # Build model input: flatten T_in dim into channels, concat SDF
                    x_vel_flat = v_window_input.reshape(B, N, -1)         # (B, N, T_in*3)
                    x_sdf      = build_sdf_window(pos_window)             # (B, N, T_in)
                    # sdf_threshold = 50.0 / 1000.0  # 50mm in SDF units (metres)

                    x_in       = torch.cat([x_vel_flat, x_sdf], dim=-1)   # (B, N, T_in*4)

                    # Forward
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        a_pred = model(x_in)                                  # (B, N, 3)

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
                (loss / accum_steps).backward()                            # <-- 缩放 loss
    
                is_accum_boundary = ((batch_idx + 1) % accum_steps == 0) \
                                    or (batch_idx + 1 == len(train_loader))
                if is_accum_boundary:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    scheduler.step()
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
                val_metrics = run_validation(model, val_loader, device)
                val_loss    = val_metrics["loss"]

                meta_payload = {
                    "epoch":      epoch + 1,
                    "step":       step,
                    "val_loss":   val_loss,
                    "git_commit": git_commit,
                    "experiment": cfg["name"],
                }

                _save_checkpoint(model, save_dir / "checkpoint-latest", meta_payload)

                ckpt_name = f"model-epoch-{epoch+1:04d}"
                _save_checkpoint(model, save_dir / ckpt_name, meta_payload)

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
                    _save_checkpoint(model, save_dir / "checkpoint-best", meta_payload)
                    tick = "✓ NEW BEST"
                else:
                    tick = ""

                print(f"[Val]   Epoch {epoch+1}/{n_epochs} | "
                      f"train_loss={epoch_avg_loss:.5f} | val_loss={val_loss:.5f} | "
                      f"best={best_val_loss:.5f} {tick}")

                epoch_log.update({f"val/{k}": v for k, v in val_metrics.items()})

            if wandb_run:
                wandb_run.log(epoch_log, step=step)
    
    except KeyboardInterrupt:
        print("[Train] Interrupted by user")
    
    print(f"[Train] Done — best val_loss: {best_val_loss:.5f}")
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
        best_val_loss = train(cfg, git_commit)
        wandb_run = wandb.run if _WANDB_AVAILABLE else None
        if wandb_run:
            upload_artifact(cfg, wandb_run, git_commit, best_val_loss)
    finally:
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
            print("[W&B] Run finished")


if __name__ == "__main__":
    main()