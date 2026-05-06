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
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def load_config(experiment_path: str) -> dict:
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
def relative_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-element relative L2.

    pred, target: (B, N, D)
    weight:       (B, N) or None — per-node weight (e.g. collision flag scaling)
    Returns scalar.
    """
    squared_diff = (pred - target) ** 2
    denominator  = (target ** 2) + eps
    per_elem     = squared_diff / denominator   # (B, N, D)

    if weight is None:
        return per_elem.mean()

    # weight: (B, N) → (B, N, 1) to broadcast over D
    w = weight.unsqueeze(-1)
    # Weighted mean: sum(w * per_elem) / sum(w * D)  (D = per_elem.shape[-1])
    numer = (w * per_elem).sum()
    denom = w.sum() * per_elem.shape[-1] + 1e-8
    return numer / denom


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_flag: torch.Tensor | None = None,
    collision_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Acceleration-only relative L2 loss with optional collision weighting.

    Args:
        pred:             (B, N, D_acc)
        target:           (B, N, D_acc)
        target_flag:      (B, N) 0/1 — 1 means colliding at target frame
        collision_weight: extra weight applied to colliding nodes
                          (1.0 = no weighting; e.g. 5.0 = colliding nodes
                          contribute 5× to the loss)

    Returns:
        (scalar_loss, metric_dict)
    """
    metrics: dict = {}

    # Always log the unweighted loss for comparability
    base_loss = relative_l2_loss(pred, target)
    metrics["loss_acc_relL2_unweighted"] = base_loss.item()

    if target_flag is None or collision_weight == 1.0:
        return base_loss, {**metrics, "loss_acc_relL2": base_loss.item()}

    # weight = 1 + (collision_weight - 1) * flag
    #   non-colliding nodes: weight = 1
    #   colliding nodes:     weight = collision_weight
    weight = 1.0 + (collision_weight - 1.0) * target_flag.float()
    weighted_loss = relative_l2_loss(pred, target, weight=weight)

    # Diagnostic: log the loss restricted to colliding nodes only
    with torch.no_grad():
        if target_flag.sum() > 0:
            colliding_loss = relative_l2_loss(pred, target, weight=target_flag.float())
            metrics["loss_acc_relL2_colliding"] = colliding_loss.item()
        non_coll = 1.0 - target_flag.float()
        if non_coll.sum() > 0:
            non_coll_loss = relative_l2_loss(pred, target, weight=non_coll)
            metrics["loss_acc_relL2_non_colliding"] = non_coll_loss.item()
        metrics["fraction_colliding"] = float(target_flag.float().mean().item())

    metrics["loss_acc_relL2"] = weighted_loss.item()
    return weighted_loss, metrics


# ── Validation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(model, val_loader, device, collision_weight: float = 1.0) -> dict:
    model.eval()
    total_unweighted = 0.0
    total_colliding = 0.0
    total_non_coll  = 0.0
    n_batches = 0
    n_coll_batches = 0
    n_non_coll_batches = 0

    for batch in val_loader:
        # Backward-compatible: accept (x, y) or (x, y, flag)
        if len(batch) == 3:
            x, y, flag = batch
            flag = flag.to(device)
        else:
            x, y = batch
            flag = None

        x, y = x.to(device), y.to(device)
        pred = model(x)
        _, metrics = compute_loss(pred, y, flag, collision_weight)

        total_unweighted += metrics.get("loss_acc_relL2_unweighted", 0.0)
        if "loss_acc_relL2_colliding" in metrics:
            total_colliding += metrics["loss_acc_relL2_colliding"]
            n_coll_batches += 1
        if "loss_acc_relL2_non_colliding" in metrics:
            total_non_coll += metrics["loss_acc_relL2_non_colliding"]
            n_non_coll_batches += 1
        n_batches += 1

    out = {"loss": total_unweighted / max(n_batches, 1)}
    if n_coll_batches > 0:
        out["loss_colliding"] = total_colliding / n_coll_batches
    if n_non_coll_batches > 0:
        out["loss_non_colliding"] = total_non_coll / n_non_coll_batches
    return out


# ── Main training loop ────────────────────────────────────────────────────────
def train(cfg: dict, git_commit: str = "unknown"):
    """TransolverNet training loop (velocity + collision → acceleration).

    Data flow:
        BVCDataset  →  (x: B, N, 25)         5 frames × [vel(3) + dist(1) + flag(1)]
                       (y: B, N, 3)          acceleration target
                       (flag: B, N)          collision flag at target frame
                    →  TransolverNet         (B, N, 3)
                    →  relative L2 loss      (optionally collision-weighted)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg.get("scheduler_step_size", 10000)),
        eta_min=float(train_cfg.get("min_lr", 1e-6)),
    )

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader = build_dataloader(cfg, split="train")
    val_loader   = build_dataloader(cfg, split="valid")
    print(f"[Train] train={len(train_loader.dataset)} windows, "
          f"val={len(val_loader.dataset)} windows")

    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    collision_weight = float(train_cfg.get("collision_weight", 1.0))
    if collision_weight != 1.0:
        print(f"[Train] Collision-weighted loss enabled: weight={collision_weight}")

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
            for batch in train_loader:
                if step >= nsteps:
                    break

                # Backward-compatible unpack
                if len(batch) == 3:
                    x, y, flag = batch
                    flag = flag.to(device)
                else:
                    x, y = batch
                    flag = None

                x, y = x.to(device), y.to(device)

                # ── Forward ───────────────────────────────────────────────
                pred = model(x)                           # (B, N, 3)

                # ── Loss ──────────────────────────────────────────────────
                loss, loss_metrics = compute_loss(pred, y, flag, collision_weight)

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
                # forward all per-step diagnostics
                for k, v in loss_metrics.items():
                    wandb_log[f"train/{k}"] = v

                if step % 10 == 0:
                    extras = ""
                    if "loss_acc_relL2_colliding" in loss_metrics:
                        extras = (f" | coll={loss_metrics['loss_acc_relL2_colliding']:.4f}"
                                  f" | non_coll={loss_metrics['loss_acc_relL2_non_colliding']:.4f}"
                                  f" | f_coll={loss_metrics['fraction_colliding']:.3f}")
                    print(f"[Train] Step {step}/{nsteps} | "
                          f"loss={loss.item():.5f} | lr={lr_now:.2e}{extras}")

                # ── Validation + checkpoint ───────────────────────────────
                if step % nsave == 0:
                    val_metrics = run_validation(model, val_loader, device, collision_weight)
                    val_loss = val_metrics["loss"]

                    meta_payload = {
                        "step":       step,
                        "val_loss":   val_loss,
                        "git_commit": git_commit,
                        "experiment": cfg["name"],
                    }

                    _save_checkpoint(model, save_dir / "checkpoint-latest", meta_payload)

                    step_name = f"model-step-{step:06d}"
                    _save_checkpoint(model, save_dir / step_name, meta_payload)

                    ckpt_history.append({"step": step, "val_loss": val_loss, "file": step_name})
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

                    coll_str = ""
                    if "loss_colliding" in val_metrics:
                        coll_str = (f" | coll={val_metrics['loss_colliding']:.5f}"
                                    f" | non_coll={val_metrics['loss_non_colliding']:.5f}")
                    print(f"[Val]   Step {step} | "
                          f"val_loss={val_loss:.5f} | "
                          f"best={best_val_loss:.5f}{coll_str} {tick}")

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