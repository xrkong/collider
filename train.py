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
import yaml

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT

import models  # triggers auto-import of all registered models
from models.registry import build_model
from src.dataset import NormStats, build_dataloader
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
    """Acceleration-only relative L2 loss.

    pred, target: (B, N, D_acc)
    """
    loss_criterion = torch.nn.MSELoss(reduction='none')
    loss_per_var = loss_criterion(pred, target).mean(dim=0)
    loss = loss_per_var.mean()
    return loss, {"loss_MSE": loss.item}

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
    
    return distances


# ── Validation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(model, val_loader, device) -> dict:
    model.eval()
    total, n_batches = 0.0, 0
    for x, y, x_pos in val_loader:
        x, y, x_pos = x.to(device), y.to(device), x_pos.to(device)   # x: (B,N,T*C) vel, y: (B,N,3), x_pos: (B,N,3)
        x_sdf = compute_sdf_batch(x_pos[...,:2]) # 
        x_sdf = x_sdf.transpose(1, 2) # (B,T,N) -> (B,N,T)

        x = torch.cat([x, x_sdf], dim=-1)

        pred = model(x)
        loss, _ = compute_loss(pred, y)
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
    train_loader = build_dataloader(cfg, split="train")
    val_loader   = build_dataloader(cfg, split="valid")
    print(f"[Train] train={len(train_loader.dataset)} windows, "
          f"val={len(val_loader.dataset)} windows")
    
    # all_targets = []
    # for batch in train_loader:
    #     _, y, _ = batch
    #     all_targets.append(y.flatten())
    # all_targets = torch.cat(all_targets)

    # print(f"target 统计:")
    # print(f"  绝对值最小: {all_targets.abs().min():.6f}")
    # print(f"  绝对值中位数: {all_targets.abs().median():.6f}")
    # print(f"  |target| < 1e-3 占比: {(all_targets.abs() < 1e-3).float().mean():.4%}")
    # print(f"  |target| < 1e-2 占比: {(all_targets.abs() < 1e-2).float().mean():.4%}")
    # print(f"  |target| < 1e-1 占比: {(all_targets.abs() < 1e-1).float().mean():.4%}")

    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(train_cfg.get("lr", 1e-3)),
        total_steps=int(train_cfg["ntraining_steps"]),
        final_div_factor=1000.,
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

    nsteps = int(train_cfg["ntraining_steps"])
    nsave  = int(train_cfg["nsave_steps"])

    print(f"[Train] Starting — {nsteps} steps | "
          f"batch={train_cfg['batch_size']} | lr={train_cfg['lr']} | "
          f"save_every={nsave}")

    step = 0
    model.train()

    try:
        while step < nsteps:
            for x, y, x_pos in train_loader:
                if step >= nsteps:
                    break
                
                x, y, x_pos = x.to(device), y.to(device), x_pos.to(device)   # x: (B,N,T*C) vel, y: (B,N,3), x_pos: (B,N,3)
                x_sdf = compute_sdf_batch(x_pos[...,:2]) # (B, N)
                x_sdf = x_sdf.transpose(1, 2) # (B,T,N) -> (B,N,T)

                x = torch.cat([x, x_sdf], dim=-1)

                # ── Forward ───────────────────────────────────────────────
                pred = model(x)                      # (B, N, 4)

                # ── Loss (Relative L2 on acceleration) ────────────────────
                loss, _ = compute_loss(pred, y)

                # ── Backward ──────────────────────────────────────────────
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                step  += 1
                lr_now = scheduler.get_last_lr()[0]

                # ── Step log ──────────────────────────────────────────────
                wandb_log = {
                    "train/loss":            loss.item(),
                    "lr":                    lr_now,
                }

                if step % 10 == 0:
                    print(f"[Train] Step {step}/{nsteps} | "
                          f"loss={loss.item():.5f} | lr={lr_now:.2e}")

                # ── Validation + checkpoint ───────────────────────────────
                if step % nsave == 0:
                    val_metrics = run_validation(model, val_loader, device)
                    val_loss = val_metrics["loss"]

                    meta_payload = {
                        "step":       step,
                        "val_loss":   val_loss,
                        "git_commit": git_commit,
                        "experiment": cfg["name"],
                    }

                    # Always save latest
                    _save_checkpoint(model, save_dir / "checkpoint-latest", meta_payload)

                    # Save per-step checkpoint
                    step_name = f"model-step-{step:06d}"
                    _save_checkpoint(model, save_dir / step_name, meta_payload)

                    # Prune to top-K by val_loss
                    ckpt_history.append({"step": step, "val_loss": val_loss, "file": step_name})
                    ckpt_history.sort(key=lambda r: r["val_loss"])
                    for stale in ckpt_history[keep_top_k:]:
                        for ext in (".safetensors", ".pt", ".json"):
                            p = save_dir / f"{stale['file']}{ext}"
                            if p.exists():
                                p.unlink()
                    ckpt_history = ckpt_history[:keep_top_k]
                    manifest_path.write_text(json.dumps(ckpt_history, indent=2))

                    # Save best
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_checkpoint(model, save_dir / "checkpoint-best", meta_payload)
                        tick = "✓ NEW BEST"
                    else:
                        tick = ""

                    print(f"[Val]   Step {step} | "
                          f"val_loss={val_loss:.5f} | "
                          f"best={best_val_loss:.5f} {tick}")

                    wandb_log.update({f"val/{k}": v for k, v in val_metrics.items()})
                    model.train()

                if wandb_run:
                    wandb_run.log(wandb_log, step=step)

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