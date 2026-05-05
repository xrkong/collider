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
    _WANDB = True
except ImportError:
    _WANDB = False

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
    if not _WANDB:
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


# ── Evaluation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model:      torch.nn.Module,
    cfg:        dict,
    output_dir: str,
    device:     torch.device,
):
    """Run full test-set evaluation and save metrics.json."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_stats  = NormStats(cfg["data"]["metadata_path"])
    model.to(device).eval()

    test_loader = build_dataloader(cfg, split="test")
    print(f"[Eval] Test set: {len(test_loader.dataset)} windows")

    total_loss = total_mse = total_vm = 0.0
    total_pos_rmse = total_stress_rmse = 0.0
    n_batches  = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss_mse  = F.mse_loss(pred, y).item()
        pred_vm   = _von_mises(pred[..., _STRESS_START:_STRESS_END])
        target_vm = _von_mises(y[...,   _STRESS_START:_STRESS_END])
        loss_vm   = F.mse_loss(pred_vm, target_vm).item()

        pred_pos_raw      = norm_stats.denormalize_tensor("positions", pred[..., 0:3])
        target_pos_raw    = norm_stats.denormalize_tensor("positions", y[..., 0:3])
        pos_rmse          = torch.sqrt(F.mse_loss(pred_pos_raw, target_pos_raw)).item()

        pred_stress_raw   = norm_stats.denormalize_tensor("stress", pred[..., _STRESS_START:_STRESS_END])
        target_stress_raw = norm_stats.denormalize_tensor("stress", y[...,   _STRESS_START:_STRESS_END])
        stress_rmse       = torch.sqrt(F.mse_loss(pred_stress_raw, target_stress_raw)).item()

        total_loss        += loss_mse + loss_vm
        total_mse         += loss_mse
        total_vm          += loss_vm
        total_pos_rmse    += pos_rmse
        total_stress_rmse += stress_rmse
        n_batches         += 1

    metrics = {
        "test/loss":        total_loss        / n_batches,
        "test/loss_mse":    total_mse         / n_batches,
        "test/loss_vm":     total_vm          / n_batches,
        "test/pos_rmse":    total_pos_rmse    / n_batches,
        "test/stress_rmse": total_stress_rmse / n_batches,
        "n_windows":        len(test_loader.dataset),
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[Eval] Test results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"[Eval] Saved → {metrics_path}")

    if _WANDB and wandb.run:
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

    if _WANDB and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()