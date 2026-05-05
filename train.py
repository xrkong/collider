from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.parsers.expat import model

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

# Target vector layout (15 dims):
#   positions    [0:3]
#   velocity     [3:6]
#   acceleration [6:9]
#   stress       [9:15]   ← Voigt: sxx, syy, szz, sxy, syz, sxz


def _von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Von Mises scalar from 6-component Voigt stress [..., 6].

    Args:
        stress: Tensor of shape (..., 6) in order [sxx, syy, szz, sxy, syz, sxz].

    Returns:
        Tensor of shape (...,) with Von Mises stress values.
    """
    s = stress
    return torch.sqrt(0.5 * (
        (s[..., 0] - s[..., 1]) ** 2
        + (s[..., 1] - s[..., 2]) ** 2
        + (s[..., 2] - s[..., 0]) ** 2
        + 6.0 * (s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2)
    ) + 1e-12)

_POS_S, _POS_E = 0, 3
_VEL_S, _VEL_E = 3, 6
_ACC_S, _ACC_E = 6, 9
_STR_S, _STR_E = 9, 15

def compute_loss(
    pred, target, norm_stats,
    loss_w_pos:    float = 1.0,
    loss_w_vel:    float = 1.0,
    loss_w_acc:    float = 1.0,
    loss_w_stress: float = 1.0,
    loss_w_vm:     float = 1.0,
    device:        torch.device = torch.device("cpu"),
    ):
    loss_pos    = F.mse_loss(pred[..., _POS_S:_POS_E], target[..., _POS_S:_POS_E])
    loss_vel    = F.mse_loss(pred[..., _VEL_S:_VEL_E], target[..., _VEL_S:_VEL_E])
    loss_acc    = F.mse_loss(pred[..., _ACC_S:_ACC_E], target[..., _ACC_S:_ACC_E])

    pred_stress   = pred[...,   _STR_S:_STR_E]
    target_stress = target[..., _STR_S:_STR_E]
    loss_stress = F.mse_loss(pred_stress, target_stress)

    # VM 在 normalized stress 空间算（数值稳定，不破坏梯度）
    loss_vm = F.mse_loss(_von_mises(pred_stress), _von_mises(target_stress))

    total = (loss_w_pos    * loss_pos
           + loss_w_vel    * loss_vel
           + loss_w_acc    * loss_acc
           + loss_w_stress * loss_stress
           + loss_w_vm     * loss_vm)

    # 仅用于监控：denorm 后的 VM 量纲（MPa/Pa），方便看物理意义
    with torch.no_grad():
        pred_vm_phys   = _von_mises(norm_stats.denormalize_tensor("stress", pred_stress))
        target_vm_phys = _von_mises(norm_stats.denormalize_tensor("stress", target_stress))

    log = {
        "loss_pos":    loss_pos.item(),
        "loss_vel":    loss_vel.item(),
        "loss_acc":    loss_acc.item(),
        "loss_stress": loss_stress.item(),
        "loss_vm":     loss_vm.item(),
        "vm_pred_mean_phys":   pred_vm_phys.mean().item(),
        "vm_target_mean_phys": target_vm_phys.mean().item(),
    }
    return total, log

# ── Validation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(
    model, val_loader, norm_stats,
    loss_w_pos, loss_w_vel, loss_w_acc, loss_w_stress, loss_w_vm,
    device,
    ) -> dict:
    tracker = MetricTracker()

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        pred_residual = model(x)
        last_frame    = x[..., -15:]
        pred = pred_residual + last_frame

        loss, log = compute_loss(
            pred, y, norm_stats,
            loss_w_pos=loss_w_pos,
            loss_w_vel=loss_w_vel,
            loss_w_acc=loss_w_acc,
            loss_w_stress=loss_w_stress,
            loss_w_vm=loss_w_vm,
            device=device,
        )
        bs = x.size(0)
        tracker.update("loss",        loss.item(),         n=bs)
        tracker.update("loss_pos",    log["loss_pos"],     n=bs)
        tracker.update("loss_vel",    log["loss_vel"],     n=bs)
        tracker.update("loss_acc",    log["loss_acc"],     n=bs)
        tracker.update("loss_stress", log["loss_stress"],  n=bs)
        tracker.update("loss_vm",     log["loss_vm"],      n=bs)

    return tracker.compute()

# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, git_commit: str = "unknown"):
    """TransolverNet training loop.

    Data flow:
        BVCDataset  →  (x: B,N,75)  →  TransolverNet  →  (pred: B,N,15)
                       (y: B,N,15)  →  loss (MSE + Von Mises)

    Returns:
        best_val_loss (float)
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    print(f"[Train] Device: {device}")

    # ── Normalization stats (for Von Mises denorm) ────────────────────────
    meta_path  = cfg["data"]["metadata_path"]
    norm_stats = NormStats(meta_path)

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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("scheduler_step_size", 5000)),
        gamma=float(train_cfg.get("scheduler_gamma", 0.8)),
    )

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader = build_dataloader(cfg, split="train")
    val_loader   = build_dataloader(cfg, split="valid")
    print(f"[Train] train={len(train_loader.dataset)} windows, "
          f"val={len(val_loader.dataset)} windows")

    # ── Loss weights ──────────────────────────────────────────────────────
    #   loss_weight_position: 1.0
    #   loss_weight_strain: 0.0
    loss_w_pos    = float(train_cfg.get("loss_weight_position",     1.0))
    loss_w_vel    = float(train_cfg.get("loss_weight_velocity",     0.0))
    loss_w_acc    = float(train_cfg.get("loss_weight_acceleration", 0.0))
    loss_w_stress = float(train_cfg.get("loss_weight_stress",       1.0))
    loss_w_vm     = float(train_cfg.get("loss_weight_vm",           1.0))

    grad_clip  = float(train_cfg.get("grad_clip",          1.0))

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
            for x, y in train_loader:
                if step >= nsteps:
                    break

                x, y = x.to(device), y.to(device)   # (B, N, 75), (B, N, 15)

                # ── Forward ───────────────────────────────────────────────
                pred = model(x)                       # (B, N, 15)

                # ── Loss ──────────────────────────────────────────────────
                # loss, loss_log = compute_loss(
                #     pred, y, norm_stats,
                #     loss_w_mse=loss_w_mse,
                #     loss_w_vm=loss_w_vm,
                #     device=device,
                # )

                loss, loss_log = compute_loss(
                    pred, y, norm_stats,
                    loss_w_pos=loss_w_pos,
                    loss_w_vel=loss_w_vel,
                    loss_w_acc=loss_w_acc,
                    loss_w_stress=loss_w_stress,
                    loss_w_vm=loss_w_vm,
                    device=device,
                )

                # ── Backward ──────────────────────────────────────────────
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                step    += 1
                lr_now   = scheduler.get_last_lr()[0]

                # ── Step log ──────────────────────────────────────────────
                wandb_log = {
                    "train/loss":        loss.item(),
                    "train/loss_pos":    loss_log["loss_pos"],
                    "train/loss_vel":    loss_log["loss_vel"],
                    "train/loss_acc":    loss_log["loss_acc"],
                    "train/loss_stress": loss_log["loss_stress"],
                    "train/loss_vm":     loss_log["loss_vm"],
                    "train/vm_pred_phys":   loss_log["vm_pred_mean_phys"],
                    "train/vm_target_phys": loss_log["vm_target_mean_phys"],
                    "lr":                lr_now,
                }

                if step % 10 == 0:
                    print(f"[Train] Step {step}/{nsteps} | "
                          f"loss={loss.item():.5f} | "
                          f"pos={loss_log['loss_pos']:.5f} | "
                        #   f"vel={loss_log['loss_vel']:.5f} | "
                        #   f"acc={loss_log['loss_acc']:.5f} | "
                        #   f"stress={loss_log['loss_stress']:.5f} | "
                        #   f"vm={loss_log['loss_vm']:.5f} | "
                          f"lr={lr_now:.2e}")

                # ── Validation + checkpoint ───────────────────────────────
                if step % nsave == 0:
                    model.eval()

                    val_metrics = run_validation(
                        model, val_loader, norm_stats,
                        loss_w_pos, loss_w_vel, loss_w_acc, loss_w_stress, loss_w_vm,
                        device,
                    )
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
                        f"pos={val_metrics['loss_pos']:.5f} | "
                        f"stress={val_metrics['loss_stress']:.5f} | "
                        f"vm={val_metrics['loss_vm']:.5f} | "
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