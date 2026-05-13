"""Offline evaluation — velocity → acceleration, with sparse-signal diagnostics.

Loads a trained TransolverNet checkpoint and runs on the test set. Reports:
  - L1 loss (matches train.py), with zero-prediction baseline & skill score
  - Normalized & physical-space RMSE for acceleration
  - Active vs quiet node partition (diagnoses sparse-signal failure modes)

Matches the current train.py:
  - Input:  velocity-only, no SDF (B, N, T_in*3)
  - Output: acceleration only (B, N, 3), predicted directly (NOT residual)
  - Target: future_acc[:, :, 0, :]  (k=0, one-step)
  - Loss:   L1 in normalized space

Usage:
    # From W&B artifact
    python src/evaluate.py \\
        --artifact "my-project/transolver_net:best" \\
        --output-dir outputs/eval/exp_001

    # From local checkpoint (no W&B needed)
    python src/evaluate.py \\
        --checkpoint outputs/checkpoints/exp_001/checkpoint-best.safetensors \\
        --experiment configs/experiments/exp_001.yaml \\
        --output-dir outputs/eval/exp_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import models  # noqa: F401 — fills registry
from models.registry import build_model
from src.dataset import NormStats, build_dataloader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from safetensors.torch import load_file as _st_load
    _SAFETENSORS = True
except ImportError:
    _SAFETENSORS = False


# ── Shared: build model and load weights ──────────────────────────────────────

def _build_and_load(model_name: str, cfg: dict, weights_path: Path) -> torch.nn.Module:
    """Build model from registry and load weights file."""
    model = build_model(model_name, cfg)
    if weights_path.suffix == ".safetensors" and _SAFETENSORS:
        model.load_state_dict(_st_load(str(weights_path)))
    else:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model


# ── Loading path A: W&B artifact ─────────────────────────────────────────────

def load_model_from_artifact(artifact_str: str):
    """Download model artifact from W&B and rebuild it.

    Returns:
        Tuple[nn.Module, dict]: (model, cfg)
    """
    if not _WANDB_AVAILABLE:
        raise ImportError("wandb is required: pip install wandb")

    run      = wandb.init(job_type="eval")
    artifact = run.use_artifact(artifact_str, type="model")
    art_dir  = Path(artifact.download())

    print(f"[Eval] Artifact:   {artifact_str}")
    print(f"[Eval] Git commit: {artifact.metadata.get('git_commit', 'unknown')}")
    print(f"[Eval] Val loss:   {artifact.metadata.get('val_loss', 'unknown')}")

    meta_files = list(art_dir.glob("*.json"))
    cfg: dict  = json.loads(meta_files[0].read_text()) if meta_files else {}

    weights = list(art_dir.glob("*.safetensors")) + list(art_dir.glob("*.pt"))
    if not weights:
        raise FileNotFoundError(f"No weights file in artifact at {art_dir}")

    model_name = artifact.metadata.get("model_name") or cfg.get("model", {}).get("name")
    if not model_name:
        raise ValueError("Cannot determine model name from artifact metadata.")

    model = _build_and_load(model_name, cfg, weights[0])
    print(f"[Eval] Loaded '{model_name}' from W&B artifact")
    return model, cfg


# ── Loading path B: local checkpoint ─────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, experiment_path: str):
    """Load model from a local weights file + experiment yaml. No W&B needed.

    Args:
        checkpoint_path: Path to .safetensors or .pt file.
        experiment_path: Experiment yaml used during training.

    Returns:
        Tuple[nn.Module, dict]: (model, cfg)
    """
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
        print(f"[Eval] Checkpoint: {ckpt_path.name}")
        print(f"[Eval] Step:       {meta.get('step', 'unknown')}")
        print(f"[Eval] Val loss:   {meta.get('val_loss', 'unknown')}")
        print(f"[Eval] Git commit: {meta.get('git_commit', 'unknown')}")
    else:
        print(f"[Eval] Checkpoint: {ckpt_path.name}")

    model = _build_and_load(model_name, cfg, ckpt_path)
    print(f"[Eval] Loaded '{model_name}' from local checkpoint")
    return model, cfg


# ── Evaluation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model:      torch.nn.Module,
    cfg:        dict,
    output_dir: str,
    device:     torch.device,
):
    """Run full test-set evaluation and save metrics.json.

    Reports the training loss (L1) plus diagnostic metrics for sparse-signal data:
    zero-prediction baseline, MSE/L1 skill scores, physical-space RMSE,
    and active vs quiet node partition (to detect "model collapsed to zero" failures).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Normalization stats (for physical-space RMSE) ─────────────────────
    norm_stats = NormStats(cfg["data"]["metadata_path"])
    acc_mean = torch.from_numpy(norm_stats._mean["acceleration"]).to(device).float()  # (3,)
    acc_std  = torch.from_numpy(norm_stats._std ["acceleration"]).to(device).float()  # (3,)

    # ── Active-node threshold (normalized space) ──────────────────────────
    # Used for the sparse-signal diagnostic. Override via cfg["eval"]["active_threshold"].
    eval_cfg = cfg.get("eval", {}) or {}
    active_thresh = float(eval_cfg.get("active_threshold", 0.1))

    model.to(device).eval()

    test_loader = build_dataloader(cfg, split="test")
    print(f"[Eval] Test set:         {len(test_loader.dataset)} windows")
    print(f"[Eval] Active threshold: |a_norm| > {active_thresh}")

    # ── Sum-based accumulators (more accurate than batch-mean) ────────────
    n_elem    = 0
    n_windows = 0
    sum_abs_err     = 0.0   # → L1 model
    sum_abs_target  = 0.0   # → L1 zero-prediction baseline (== mean |target|)
    sum_sq_err      = 0.0   # → MSE/RMSE in normalized space
    sum_sq_target   = 0.0   # → MSE skill score
    sum_sq_err_phys = 0.0   # → physical RMSE (m/s²)

    # Active/quiet partition (per-node; broadcast to all 3 accel components)
    sum_sq_err_active = 0.0
    sum_sq_err_quiet  = 0.0
    n_elem_active = 0
    n_elem_quiet  = 0

    for batch in test_loader:
        # Match BVCDataset 5-tensor format used by train.py
        x_vel, future_acc, input_pos, future_pos, v_last_phys = batch
        x_vel      = x_vel.to(device)
        future_acc = future_acc.to(device)

        # Velocity-only input (matches current train.py; SDF is commented out there)
        x_in = x_vel
        pred   = model(x_in)              # (B, N, 3) — direct acceleration prediction
        target = future_acc[:, :, 0, :]   # (B, N, 3) — k=0 (one-step), matches val

        # ── Element-wise errors ──────────────────────────────────────────
        err     = pred - target
        sq_err  = err ** 2
        abs_err = err.abs()

        sum_abs_err     += abs_err.sum().item()
        sum_abs_target  += target.abs().sum().item()
        sum_sq_err      += sq_err.sum().item()
        sum_sq_target   += (target ** 2).sum().item()

        # ── Physical-space error (denormalize: x_phys = x_norm * std + mean) ──
        pred_phys   = pred   * acc_std + acc_mean
        target_phys = target * acc_std + acc_mean
        sum_sq_err_phys += ((pred_phys - target_phys) ** 2).sum().item()

        # ── Active vs quiet partition ────────────────────────────────────
        # A node is "active" if any acceleration component exceeds threshold.
        # This separates the few impact-zone nodes from the many near-zero ones.
        node_mag    = target.abs().max(dim=-1).values             # (B, N)
        active_mask = (node_mag > active_thresh)                  # (B, N)
        active_3d   = active_mask.unsqueeze(-1).expand_as(sq_err) # (B, N, 3)
        quiet_3d    = ~active_3d

        sum_sq_err_active += sq_err[active_3d].sum().item()
        sum_sq_err_quiet  += sq_err[quiet_3d ].sum().item()
        n_elem_active     += int(active_3d.sum().item())
        n_elem_quiet      += int(quiet_3d .sum().item())

        n_elem    += target.numel()
        n_windows += x_vel.size(0)

    # ── Aggregate metrics ─────────────────────────────────────────────────
    eps = 1e-12
    l1_model = sum_abs_err    / n_elem
    l1_zero  = sum_abs_target / n_elem
    mse_norm = sum_sq_err     / n_elem
    mse_zero = sum_sq_target  / n_elem

    metrics = {
        # ── Training-matching loss & baselines ──
        "test/loss_l1":                l1_model,
        "test/loss_l1_zero_baseline":  l1_zero,                                    # = mean(|target|)
        "test/skill_l1_vs_zero":       1.0 - l1_model / max(l1_zero, eps),         # >0 means beats zero pred

        # ── RMSE (normalized & physical) ──
        "test/rmse_norm":              mse_norm ** 0.5,
        "test/rmse_phys":              (sum_sq_err_phys / n_elem) ** 0.5,          # m/s² (or whatever physical unit)
        "test/skill_mse_vs_zero":      1.0 - mse_norm / max(mse_zero, eps),

        # ── Sparse-signal diagnostic ──
        "test/rmse_norm_active":       (sum_sq_err_active / max(n_elem_active, 1)) ** 0.5,
        "test/rmse_norm_quiet":        (sum_sq_err_quiet  / max(n_elem_quiet,  1)) ** 0.5,
        "test/active_node_ratio":      n_elem_active / max(n_elem_active + n_elem_quiet, 1),

        # ── Meta ──
        "n_windows":                   n_windows,
        "active_threshold":            active_thresh,
    }

    # ── Save & print ──────────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[Eval] Test results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:36s} {v:.6f}")
        else:
            print(f"  {k:36s} {v}")
    print(f"\n[Eval] Saved → {metrics_path}")

    # ── Interpretive hints ────────────────────────────────────────────────
    skill_mse = metrics["test/skill_mse_vs_zero"]
    skill_l1  = metrics["test/skill_l1_vs_zero"]
    rmse_a    = metrics["test/rmse_norm_active"]
    rmse_q    = metrics["test/rmse_norm_quiet"]

    print("\n[Eval] Diagnosis:")
    if skill_mse <= 0 or skill_l1 <= 0:
        print(f"  ⚠ skill_mse={skill_mse:+.4f}, skill_l1={skill_l1:+.4f} — "
              f"model is NOT beating zero prediction. Loss/data design issue.")
    elif skill_mse < 0.1:
        print(f"  ⚠ skill_mse={skill_mse:.4f} — model only marginally beats zero baseline.")
    else:
        print(f"  ✓ skill_mse={skill_mse:.4f}, skill_l1={skill_l1:.4f}")

    if rmse_q > 0 and rmse_a / max(rmse_q, eps) > 3.0:
        print(f"  ⚠ rmse_active ({rmse_a:.4f}) >> rmse_quiet ({rmse_q:.4f}) — "
              f"model collapsed to predicting near-zero on impact nodes too. "
              f"Sparse-signal failure mode.")

    if _WANDB_AVAILABLE and wandb.run:
        wandb.log(metrics)

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BVC TransolverNet evaluation")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--artifact",
                     help='W&B artifact, e.g. "my-project/transolver_net:best"')
    src.add_argument("--checkpoint",
                     help="Local .safetensors file, e.g. "
                          "outputs/checkpoints/exp_001/checkpoint-best.safetensors")

    parser.add_argument("--experiment",
                        help="Required with --checkpoint: "
                             "configs/experiments/exp_001.yaml")
    parser.add_argument("--output-dir", default="outputs/eval/")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint and not args.experiment:
        parser.error("--experiment is required when using --checkpoint")

    device = torch.device(args.device)

    if args.artifact:
        model, cfg = load_model_from_artifact(args.artifact)
    else:
        model, cfg = load_model_from_checkpoint(args.checkpoint, args.experiment)

    run_evaluation(model, cfg, args.output_dir, device)

    if _WANDB_AVAILABLE and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()