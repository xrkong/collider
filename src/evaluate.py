"""Offline evaluation script — loads a model and runs on test set.

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
import torch.nn.functional as F

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

_STRESS_START = 9
_STRESS_END   = 15


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
                         e.g. ``outputs/checkpoints/exp_001/checkpoint-best.safetensors``
        experiment_path: Experiment yaml used during training.
                         e.g. ``configs/experiments/exp_001.yaml``

    Returns:
        Tuple[nn.Module, dict]: (model, cfg)
    """
    from train import load_config

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg        = load_config(experiment_path)
    model_name = cfg["model"]["name"]

    # Print sidecar metadata if available
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


# ── Von Mises ─────────────────────────────────────────────────────────────────

def _von_mises(stress: torch.Tensor) -> torch.Tensor:
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

# ── Evaluation loop ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_evaluation(
    model:      torch.nn.Module,
    cfg:        dict,
    output_dir: str,
    device:     torch.device,
):
    """Run full test-set evaluation and save metrics.json.

    Reports per-quantity losses (consistent with training), physical-space
    RMSE for position & stress, and a last-frame-copy baseline for context.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_stats  = NormStats(cfg["data"]["metadata_path"])
    model.to(device).eval()

    test_loader = build_dataloader(cfg, split="test")
    print(f"[Eval] Test set: {len(test_loader.dataset)} windows")

    # ── Loss weights (read from cfg, same as training) ────────────────────
    train_cfg = cfg["train"]
    loss_w_pos    = float(train_cfg.get("loss_weight_position",     1.0))
    loss_w_vel    = float(train_cfg.get("loss_weight_velocity",     0.0))
    loss_w_acc    = float(train_cfg.get("loss_weight_acceleration", 0.0))
    loss_w_stress = float(train_cfg.get("loss_weight_stress",       1.0))
    loss_w_vm     = float(train_cfg.get("loss_weight_vm",           1.0))

    # ── Sample-weighted accumulators (more accurate than batch-mean) ──────
    # We accumulate sum-of-squared-errors and total element count, then
    # take sqrt at the end for true RMSE. Loss values use sample-count weighting.
    n_samples = 0          # total windows seen (B summed over batches)
    sums = {
        # weighted MSE losses (× B per batch)
        "loss":        0.0,
        "loss_pos":    0.0,
        "loss_vel":    0.0,
        "loss_acc":    0.0,
        "loss_stress": 0.0,
        "loss_vm":     0.0,
        # baseline (last-frame copy, no model)
        "baseline_loss_pos":    0.0,
        "baseline_loss_stress": 0.0,
    }
    # squared-error sums for true physical-space RMSE
    sse_pos_phys     = 0.0
    sse_stress_phys  = 0.0
    sse_vm_phys      = 0.0
    n_elem_pos       = 0
    n_elem_stress    = 0
    n_elem_vm        = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        bs   = x.size(0)

        # ── Model forward + residual reconstruction (same as training) ────
        pred_residual = model(x)
        last_frame    = x[..., -15:]
        pred          = pred_residual + last_frame

        # ── Per-quantity losses (normalized space) ────────────────────────
        loss_pos    = F.mse_loss(pred[..., _POS_S:_POS_E], y[..., _POS_S:_POS_E]).item()
        loss_vel    = F.mse_loss(pred[..., _VEL_S:_VEL_E], y[..., _VEL_S:_VEL_E]).item()
        loss_acc    = F.mse_loss(pred[..., _ACC_S:_ACC_E], y[..., _ACC_S:_ACC_E]).item()

        pred_stress = pred[..., _STR_S:_STR_E]
        true_stress = y[...,    _STR_S:_STR_E]
        loss_stress = F.mse_loss(pred_stress, true_stress).item()
        loss_vm     = F.mse_loss(_von_mises(pred_stress), _von_mises(true_stress)).item()

        loss_total  = (loss_w_pos    * loss_pos
                     + loss_w_vel    * loss_vel
                     + loss_w_acc    * loss_acc
                     + loss_w_stress * loss_stress
                     + loss_w_vm     * loss_vm)

        # ── Last-frame-copy baseline (pred = last_frame, residual = 0) ────
        baseline_loss_pos    = F.mse_loss(last_frame[..., _POS_S:_POS_E],
                                          y[...,         _POS_S:_POS_E]).item()
        baseline_loss_stress = F.mse_loss(last_frame[..., _STR_S:_STR_E],
                                          y[...,         _STR_S:_STR_E]).item()

        # ── Physical-space errors (denormalized) ──────────────────────────
        # Note: key must match metadata — adjust "position" / "positions" to your NormStats.
        pred_pos_phys   = norm_stats.denormalize_tensor("positions", pred[..., _POS_S:_POS_E])
        true_pos_phys   = norm_stats.denormalize_tensor("positions", y[...,    _POS_S:_POS_E])
        pred_str_phys   = norm_stats.denormalize_tensor("stress",   pred_stress)
        true_str_phys   = norm_stats.denormalize_tensor("stress",   true_stress)

        sse_pos_phys    += ((pred_pos_phys - true_pos_phys) ** 2).sum().item()
        sse_stress_phys += ((pred_str_phys - true_str_phys) ** 2).sum().item()
        n_elem_pos      += pred_pos_phys.numel()
        n_elem_stress   += pred_str_phys.numel()

        # VM in physical space
        pred_vm_phys    = _von_mises(pred_str_phys)
        true_vm_phys    = _von_mises(true_str_phys)
        sse_vm_phys    += ((pred_vm_phys - true_vm_phys) ** 2).sum().item()
        n_elem_vm      += pred_vm_phys.numel()

        # ── Accumulate (weighted by batch size) ───────────────────────────
        sums["loss"]                  += loss_total            * bs
        sums["loss_pos"]              += loss_pos              * bs
        sums["loss_vel"]              += loss_vel              * bs
        sums["loss_acc"]              += loss_acc              * bs
        sums["loss_stress"]           += loss_stress           * bs
        sums["loss_vm"]               += loss_vm               * bs
        sums["baseline_loss_pos"]     += baseline_loss_pos     * bs
        sums["baseline_loss_stress"]  += baseline_loss_stress  * bs
        n_samples                     += bs

    # ── Final metrics ─────────────────────────────────────────────────────
    metrics = {f"test/{k}": v / n_samples for k, v in sums.items()}

    # True element-wise RMSE in physical units (m, Pa, etc.)
    metrics["test/pos_rmse_phys"]    = (sse_pos_phys    / n_elem_pos)    ** 0.5
    metrics["test/stress_rmse_phys"] = (sse_stress_phys / n_elem_stress) ** 0.5
    metrics["test/vm_rmse_phys"]     = (sse_vm_phys     / n_elem_vm)     ** 0.5

    # How much better is the model vs. just copying the last frame?
    eps = 1e-12
    metrics["test/pos_skill_score"] = (
        1.0 - metrics["test/loss_pos"] / (metrics["test/baseline_loss_pos"] + eps)
    )
    metrics["test/stress_skill_score"] = (
        1.0 - metrics["test/loss_stress"] / (metrics["test/baseline_loss_stress"] + eps)
    )

    metrics["n_windows"] = len(test_loader.dataset)

    # ── Save & print ──────────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[Eval] Test results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print(f"[Eval] Saved → {metrics_path}")

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