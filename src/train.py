# 迁移自: sgnn/transolver/train.py
# 改动内容:
#   - 添加 git dirty check、W&B artifact 上传、safetensors checkpoint
#   - 加载配置改为 experiment yaml → params yaml → overrides 合并
#   - 原有 BVC loss 计算、optimizer 步骤、checkpoint pruning 逻辑保留不变
#   - 原有训练循环保留在 _bvc_train_loop() 中，未修改

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── Project root on path so we can import models / src ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent  # used only for git operations

import models  # triggers auto-import of all registered models
from models.registry import build_model
from src.dataset import BVCWindowDataset, BVCFullTrajectoryDataset
from src.utils.metrics import MetricTracker

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
    _SAFETENSORS = True
except ImportError:
    _SAFETENSORS = False
    print("Warning: safetensors not installed — checkpoints will use .pt format")


# ── Git check ─────────────────────────────────────────────────────────────────

def check_git_clean() -> str:
    """Abort if the working tree has uncommitted changes; return commit hash.

    Raises:
        RuntimeError: Lists dirty files and refuses to start training.
    """
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
    """Set a nested dict key using dot notation, e.g. ``"train.lr"``."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def load_config(experiment_path: str) -> dict:
    """Load and merge experiment config.

    1. Read experiment yaml.
    2. Read the params yaml referenced by ``experiment.model.params``.
    3. Apply ``experiment.overrides`` on top of params.
    4. Inject ``experiment.name`` and ``experiment.model.name``.

    Args:
        experiment_path: Path to a ``configs/experiments/*.yaml`` file.

    Returns:
        Merged config dict with sections ``model``, ``train``, ``data``, ``wandb``.
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

    # Apply overrides
    for dotted_key, value in (exp.get("overrides") or {}).items():
        _deep_set(cfg, dotted_key, value)

    # Inject experiment-level fields
    cfg["name"]        = exp["name"]
    cfg["description"] = exp.get("description", "")
    cfg.setdefault("model", {})["name"] = exp["model"]["name"]

    return cfg


# ── W&B setup ─────────────────────────────────────────────────────────────────

def setup_wandb(cfg: dict, git_commit: str):
    """Initialise a W&B run.

    Args:
        cfg:        Full merged config.
        git_commit: Short commit hash to log.

    Returns:
        wandb.run or None if W&B is disabled / unavailable.
    """
    wandb_cfg = cfg.get("wandb", {})
    if not _WANDB_AVAILABLE or not wandb_cfg.get("log", True):
        return None

    project = wandb_cfg.get("project") or os.environ.get("WANDB_PROJECT", "my-project")
    run = wandb.init(
        project=project,
        name=cfg["name"],
        config={**cfg, "git_commit": git_commit},
        reinit=True,
    )
    return run


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(model: torch.nn.Module, path: Path, metadata: dict | None = None):
    """Save model weights in safetensors format (fallback: .pt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _SAFETENSORS:
        st_path = path.with_suffix(".safetensors")
        cpu_state = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
        _st_save(cpu_state, str(st_path))
        if metadata:
            with open(st_path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=2)
    else:
        torch.save(model.state_dict(), str(path.with_suffix(".pt")))


def _ckpt_path(cfg: dict, suffix: str) -> Path:
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    return ckpt_dir / f"{cfg['name']}_{suffix}"


# ── W&B Artifact upload ───────────────────────────────────────────────────────

def upload_artifact(cfg: dict, run, git_commit: str, val_loss: float):
    """Create and log a W&B model artifact.

    Args:
        cfg:        Full config.
        run:        Active wandb run (or None).
        git_commit: Commit hash for provenance.
        val_loss:   Best validation loss to store as artifact metadata.
    """
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
        },
    )

    best_path = _ckpt_path(cfg, "best")
    for ext in (".safetensors", ".pt"):
        p = best_path.with_suffix(ext)
        if p.exists():
            artifact.add_file(str(p))
            break

    params_json = best_path.with_suffix(".json")
    if params_json.exists():
        artifact.add_file(str(params_json))

    run.log_artifact(artifact)
    print(f"[Artifact] Uploaded '{model_name}' to W&B Artifacts")


# ── BVC-specific helpers ──────────────────────────────────────────────────────

def _von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Von Mises scalar from 6-component Voigt notation [sxx,syy,szz,sxy,syz,sxz]."""
    s = stress
    return torch.sqrt(0.5 * (
        (s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2
        + 6.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    ) + 1e-12)


def _build_normalization_stats(metadata: dict, noise_std: float, device) -> dict:
    """Compute normalisation tensors for MultiScaleSimulator from metadata.json."""
    gs        = metadata["global_stats"]
    disp_mean = np.array(gs["displacement_mean"], dtype=np.float32)
    disp_std  = np.array(gs["displacement_std"],  dtype=np.float32)
    acc_mean  = np.array(gs["acceleration_mean"], dtype=np.float32)
    acc_std   = np.array(gs["acceleration_std"],  dtype=np.float32)

    if metadata.get("config", {}).get("normalised", False):
        pos_std   = np.array(metadata["normalization_stats"]["positions"]["std"], np.float32)
        disp_mean /= pos_std; disp_std /= pos_std
        acc_mean  /= pos_std; acc_std  /= pos_std

    return {
        "velocity": {
            "mean": torch.FloatTensor(disp_mean).to(device),
            "std":  torch.sqrt(torch.FloatTensor(disp_std) ** 2 + noise_std ** 2).to(device),
        },
        "acceleration": {
            "mean": torch.FloatTensor(acc_mean).to(device),
            "std":  torch.sqrt(torch.FloatTensor(acc_std)  ** 2 + noise_std ** 2).to(device),
        },
    }


def _collate_bvc(batch: list) -> dict:
    """Collate BVC window samples into a batched dict."""
    out = {"context": {}, "prediction": {}, "meta": [], "graph": None}
    for sample in batch:
        out["meta"].append(sample["meta"])
        for feat, t in sample["context"].items():
            out["context"].setdefault(feat, []).append(t)
        for feat, t in sample["prediction"].items():
            out["prediction"].setdefault(feat, []).append(t)
        if "graph" in sample and out["graph"] is None:
            out["graph"] = sample["graph"]
    out["context"]    = {k: torch.stack(v) for k, v in out["context"].items()}
    out["prediction"] = {k: torch.stack(v) for k, v in out["prediction"].items()}
    return out


# ── Main training loop (BVC / MultiScaleSimulator) ───────────────────────────

def train(cfg: dict, git_commit: str = "unknown"):
    """Full BVC training loop.

    Preserves original loss / optimizer / scheduler / checkpoint-pruning logic
    Original training logic preserved. Adds git check, W&B logging, and safetensors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Load metadata ─────────────────────────────────────────────────────
    meta_path = Path(cfg["data"]["metadata_path"])
    with open(meta_path) as f:
        metadata = json.load(f)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    noise_std = float(train_cfg.get("noise_std", 0.0))

    # ── Augment model cfg with runtime fields ─────────────────────────────
    input_seq   = int(model_cfg["input_sequence_length"])
    dim         = int(model_cfg["dim"])
    n_types     = int(metadata.get("num_particle_types", 1))
    nnode_in    = (input_seq - 1) * dim + 1
    if n_types > 1:
        nnode_in += int(model_cfg["particle_type_embedding_size"])

    model_cfg["nnode_in"]          = nnode_in
    model_cfg["nparticle_types"]   = n_types
    model_cfg["normalization_stats"] = _build_normalization_stats(metadata, noise_std, device)
    model_cfg["device"]            = str(device)

    # ── Build model ───────────────────────────────────────────────────────
    simulator = build_model(model_cfg["name"], cfg)
    simulator.to(device)
    print(f"[Train] Model '{model_cfg['name']}' built — nnode_in={nnode_in}, dim={dim}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(simulator.parameters(), lr=float(train_cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("scheduler_step_size", 5000)),
        gamma=float(train_cfg.get("scheduler_gamma", 0.8)),
    )

    # ── Data ──────────────────────────────────────────────────────────────
    from src.data_loader import get_multi_scale_bvc_data_loader_by_windows
    from src.evaluator import validate_multi_scale_simulator
    from src.utils import noise_utils

    data_loader = get_multi_scale_bvc_data_loader_by_windows(
        data_dir=cfg["data"]["base_path"],
        context_length=input_seq,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_scales=int(model_cfg["num_scales"]),
        window_size=int(model_cfg["window_size"]),
        radius_multiplier=float(model_cfg["radius_multiplier"]),
        split="train",
    )

    # ── Checkpoint state ──────────────────────────────────────────────────
    save_dir   = PROJECT_ROOT / "outputs" / "checkpoints" / cfg["name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "checkpoint_manifest.json"
    ckpt_history  = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    best_val_loss = min((r["val_loss"] for r in ckpt_history), default=float("inf"))
    keep_top_k    = int(train_cfg.get("keep_top_k", 3))

    # ── W&B ──────────────────────────────────────────────────────────────
    wandb_run = setup_wandb(cfg, git_commit)

    nsteps          = int(train_cfg["ntraining_steps"])
    nsave           = int(train_cfg["nsave_steps"])
    loss_w_pos      = float(train_cfg.get("loss_weight_position", 1.0))
    loss_w_strain   = float(train_cfg.get("loss_weight_strain", 0.0))
    grad_clip       = float(train_cfg.get("grad_clip", 1.0))
    inference_mode  = train_cfg.get("inference_mode", "autoregressive")

    step = 0
    simulator.train()

    print(f"[Train] Starting — {nsteps} steps, batch={train_cfg['batch_size']}, "
          f"lr={train_cfg['lr']}, save_every={nsave}")

    # ── First batch graph init ────────────────────────────────────────────
    first_batch = next(iter(data_loader))
    simulator.set_static_graph(first_batch["graph"])

    not_reached_nsteps = True
    try:
        while not_reached_nsteps:
            for data_sample in data_loader:
                log: dict = {}

                position    = data_sample["context"]["positions"].to(device)
                B, T, N, D  = position.shape
                next_pos    = data_sample["prediction"]["positions"].to(device)
                next_strain = data_sample["prediction"]["stress"].to(device)

                position    = position.reshape(-1, T, D)
                next_pos    = next_pos.reshape(-1, D)
                next_strain = next_strain.reshape(-1, 6)
                particle_type = torch.ones(position.shape[0], dtype=torch.int64, device=device)
                n_per_example = torch.full((B,), N, dtype=torch.int64, device=device)

                simulator.set_static_graph(data_sample["graph"])

                sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
                    position, noise_std_last_step=noise_std
                ).to(device)

                pred_acc, _, pred_strain = simulator.predict_accelerations(
                    next_positions=next_pos,
                    position_sequence_noise=sampled_noise,
                    position_sequence=position,
                    nparticles_per_example=n_per_example,
                    particle_types=particle_type,
                )

                noisy_seq    = position + sampled_noise
                pred_next    = simulator._decoder_postprocessor(pred_acc, noisy_seq)
                target_next  = next_pos + sampled_noise[:, -1]

                loss_pos    = F.mse_loss(pred_next, target_next, reduction="none").mean(dim=-1)
                target_vm   = _von_mises(next_strain)
                loss_strain = (pred_strain - target_vm) ** 2

                loss = (loss_w_pos * loss_pos + loss_w_strain * loss_strain).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(simulator.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                lr_now = scheduler.get_last_lr()[0]

                step += 1

                log["train/loss"]          = loss.item()
                log["train/pos_rmse"]      = loss_pos.mean().sqrt().item()
                log["train/loss_position"] = loss_pos.mean().item()
                log["train/loss_strain"]   = loss_strain.mean().item()
                log["lr"]                  = lr_now

                if step % 10 == 0:
                    print(f"[Train] Step {step}/{nsteps} | "
                          f"loss={loss.item():.5f} | pos_rmse={loss_pos.mean().sqrt().item():.5f} | "
                          f"lr={lr_now:.2e}")

                # ── Validation + checkpoint ───────────────────────────────
                if step % nsave == 0:
                    simulator.eval()

                    val_metrics = validate_multi_scale_simulator(
                        simulator=simulator,
                        data_path=str(Path(cfg["data"]["base_path"]) / "valid"),
                        metadata=metadata,
                        device=str(device),
                        input_sequence_length=input_seq,
                        inference_mode=inference_mode,
                        num_scales=int(model_cfg["num_scales"]),
                        window_size=int(model_cfg["window_size"]),
                        radius_multiplier=float(model_cfg["radius_multiplier"]),
                    )
                    val_loss = float(val_metrics.get("val/loss_total", 1e9))

                    model_fname  = f"model-step-{step:06d}"
                    meta_payload = {
                        "step": step, "val_loss": val_loss, "git_commit": git_commit,
                        "experiment": cfg["name"],
                    }

                    # Save latest (always)
                    _save_checkpoint(simulator, save_dir / "checkpoint-latest", meta_payload)

                    # Save per-step checkpoint
                    _save_checkpoint(simulator, save_dir / model_fname, meta_payload)

                    # Prune to top-K
                    ckpt_history.append({"step": step, "val_loss": val_loss, "file": model_fname})
                    ckpt_history.sort(key=lambda r: r["val_loss"])
                    for stale in ckpt_history[keep_top_k:]:
                        for ext in (".safetensors", ".pt", ".json"):
                            p = save_dir / f"{stale['file']}{ext}"
                            if p.exists(): p.unlink()
                    ckpt_history = ckpt_history[:keep_top_k]
                    manifest_path.write_text(json.dumps(ckpt_history, indent=2))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_checkpoint(simulator, save_dir / "checkpoint-best", meta_payload)
                        tick = "✓ NEW BEST"
                    else:
                        tick = ""

                    print(f"[Val] Step {step} | val_loss={val_loss:.5f} | best={best_val_loss:.5f} {tick}")

                    log["val/loss"] = val_loss
                    if wandb_run:
                        wandb_run.log(log, step=step)

                    simulator.train()
                else:
                    if wandb_run:
                        wandb_run.log(log, step=step)

                if step >= nsteps:
                    not_reached_nsteps = False
                    break

    except KeyboardInterrupt:
        print("[Train] Interrupted by user")

    print(f"[Train] Done. Best val_loss: {best_val_loss:.5f}")
    return best_val_loss


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BVC MLOps Training")
    parser.add_argument("--experiment", required=True,
                        help="Path to configs/experiments/*.yaml")
    parser.add_argument("--skip-git-check", action="store_true",
                        help="Skip git dirty check (for debugging only)")
    args = parser.parse_args()

    git_commit = "unknown"
    if not args.skip_git_check:
        git_commit = check_git_clean()
        print(f"[Git] Clean — commit {git_commit}")

    cfg = load_config(args.experiment)
    print(f"[Config] Loaded: {cfg['name']}")

    wandb_run = None
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
