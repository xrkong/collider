"""Autoregressive rollout — runs long-horizon inference on test data.

Usage:
    # From W&B artifact
    python src/rollout.py \\
        --artifact "my-project/transolver_net:best" \\
        --input /scratch/datasets/dataset_v1/test/test_data.h5

    # From local checkpoint (no W&B needed)
    python src/rollout.py \\
        --checkpoint outputs/checkpoints/exp_001/checkpoint-best.safetensors \\
        --experiment configs/experiments/exp_001.yaml \\
        --input /scratch/datasets/dataset_v1/test/test_data.h5

    # List all available W&B versions
    python src/rollout.py --artifact "transolver_net" --list-versions
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import models  # noqa: F401 — fills registry
from models.registry import build_model
from src.dataset import NormStats

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
_INPUT_FRAMES = 5


# ── Shared: build model and load weights ──────────────────────────────────────

def _build_and_load(model_name: str, cfg: dict, weights_path: Path) -> torch.nn.Module:
    model = build_model(model_name, cfg)
    if weights_path.suffix == ".safetensors" and _SAFETENSORS:
        model.load_state_dict(_st_load(str(weights_path)))
    else:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model


# ── Loading path A: W&B artifact ─────────────────────────────────────────────

def load_model_for_inference(artifact_str: str, device: torch.device):
    """Download versioned artifact from W&B and reconstruct the model.

    Returns:
        Tuple[nn.Module, dict]: model in eval mode, config dict.
    """
    if not _WANDB:
        raise ImportError("wandb is required: pip install wandb")

    run      = wandb.init(job_type="inference")
    artifact = run.use_artifact(artifact_str, type="model")
    art_dir  = Path(artifact.download())

    print(f"[Rollout] Artifact:   {artifact_str}")
    print(f"[Rollout] Version:    {artifact.version}")
    print(f"[Rollout] Git commit: {artifact.metadata.get('git_commit', 'unknown')}")
    print(f"[Rollout] Val loss:   {artifact.metadata.get('val_loss', 'unknown')}")

    meta_files = list(art_dir.glob("*.json"))
    cfg: dict  = json.loads(meta_files[0].read_text()) if meta_files else {}

    weights = list(art_dir.glob("*.safetensors")) + list(art_dir.glob("*.pt"))
    if not weights:
        raise FileNotFoundError(f"No weights file in artifact at {art_dir}")

    model_name = artifact.metadata.get("model_name") or cfg.get("model", {}).get("name")
    if not model_name:
        raise ValueError("Cannot determine model name from artifact metadata.")

    model = _build_and_load(model_name, cfg, weights[0])
    model.to(device).eval()
    print(f"[Rollout] Loaded '{model_name}' from W&B artifact")
    return model, cfg


# ── Loading path B: local checkpoint ─────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, experiment_path: str, device: torch.device):
    """Load model from a local weights file + experiment yaml. No W&B needed.

    Args:
        checkpoint_path: e.g. ``outputs/checkpoints/exp_001/checkpoint-best.safetensors``
        experiment_path: e.g. ``configs/experiments/exp_001.yaml``
        device:          Target device.

    Returns:
        Tuple[nn.Module, dict]: model in eval mode, config dict.
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
        print(f"[Rollout] Checkpoint: {ckpt_path.name}")
        print(f"[Rollout] Step:       {meta.get('step', 'unknown')}")
        print(f"[Rollout] Val loss:   {meta.get('val_loss', 'unknown')}")
        print(f"[Rollout] Git commit: {meta.get('git_commit', 'unknown')}")
    else:
        print(f"[Rollout] Checkpoint: {ckpt_path.name}")

    model = _build_and_load(model_name, cfg, ckpt_path)
    model.to(device).eval()
    print(f"[Rollout] Loaded '{model_name}' from local checkpoint")
    return model, cfg


def list_artifact_versions(artifact_name: str):
    """Print all versions and aliases for a model artifact."""
    if not _WANDB:
        raise ImportError("wandb is required: pip install wandb")
    api = wandb.Api()
    print(f"\nVersions for '{artifact_name}':")
    for v in api.artifact_versions("model", artifact_name):
        aliases = ", ".join(v.aliases) or "(none)"
        print(f"  {v.version:6s}  aliases=[{aliases:20s}]  "
              f"val_loss={v.metadata.get('val_loss', 'n/a')}")


# ── Von Mises ─────────────────────────────────────────────────────────────────

def _von_mises(stress: torch.Tensor) -> torch.Tensor:
    s = stress
    return torch.sqrt(0.5 * (
        (s[..., 0] - s[..., 1]) ** 2
        + (s[..., 1] - s[..., 2]) ** 2
        + (s[..., 2] - s[..., 0]) ** 2
        + 6.0 * (s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2)
    ) + 1e-12)


# ── Autoregressive rollout ────────────────────────────────────────────────────

@torch.no_grad()
def run_autoregressive_rollout(
    model:       torch.nn.Module,
    window_data: dict,
    norm_stats:  NormStats,
    device:      torch.device,
) -> dict:
    """Autoregressive rollout on one 6-frame window.

    Seed: frames 0-4 → predict frame 5 → slide window → predict frame 6 ...
    For 6-frame windows this is 1 prediction step.

    Args:
        model:       TransolverNet in eval mode.
        window_data: Dict from BVCFullTrajectoryDataset.__getitem__.
        norm_stats:  For physical-space metric computation.
        device:      Torch device.

    Returns:
        Dict with pred/gt arrays and per-step RMSE.
    """
    FEATURES = ["positions", "velocity", "acceleration", "stress"]

    frames = torch.cat([
        window_data[feat].to(device) for feat in FEATURES
    ], dim=-1)                                          # (6, N, 15)

    T, N, _        = frames.shape
    current_window = frames[:_INPUT_FRAMES]             # (5, N, 15)

    pred_frames       = []
    gt_frames         = [frames[t].cpu() for t in range(_INPUT_FRAMES, T)]
    rmse_pos_steps    = []
    rmse_stress_steps = []
    rmse_vm_steps     = []

    for step in range(T - _INPUT_FRAMES):
        x    = current_window.reshape(N, -1).unsqueeze(0)   # (1, N, 75)
        pred = model(x).squeeze(0)                           # (N, 15)
        gt   = frames[_INPUT_FRAMES + step]                  # (N, 15)

        # Physical-space RMSE
        pred_pos_raw      = norm_stats.denormalize_tensor("positions", pred[..., 0:3])
        gt_pos_raw        = norm_stats.denormalize_tensor("positions", gt[..., 0:3])
        rmse_pos          = torch.sqrt(F.mse_loss(pred_pos_raw, gt_pos_raw)).item()

        pred_stress_raw   = norm_stats.denormalize_tensor("stress", pred[..., _STRESS_START:_STRESS_END])
        gt_stress_raw     = norm_stats.denormalize_tensor("stress", gt[..., _STRESS_START:_STRESS_END])
        rmse_stress       = torch.sqrt(F.mse_loss(pred_stress_raw, gt_stress_raw)).item()

        pred_vm   = _von_mises(pred_stress_raw)
        gt_vm     = _von_mises(gt_stress_raw)
        rmse_vm   = torch.sqrt(F.mse_loss(pred_vm, gt_vm)).item()

        rmse_pos_steps.append(rmse_pos)
        rmse_stress_steps.append(rmse_stress)
        rmse_vm_steps.append(rmse_vm)
        pred_frames.append(pred.cpu())

        # Slide window: drop oldest frame, append prediction
        current_window = torch.cat(
            [current_window[1:], pred.unsqueeze(0)], dim=0)  # (5, N, 15)

    return {
        "pred_frames":  torch.stack(pred_frames).numpy(),    # (steps, N, 15)
        "gt_frames":    torch.stack(gt_frames).numpy(),
        "rmse_pos":     np.array(rmse_pos_steps),
        "rmse_stress":  np.array(rmse_stress_steps),
        "rmse_vm":      np.array(rmse_vm_steps),
    }


# ── Main inference ────────────────────────────────────────────────────────────

def run_inference(
    model:       torch.nn.Module,
    cfg:         dict,
    input_path:  str,
    output_path: str,
    device:      torch.device,
):
    """Run rollout over all windows in a test h5 file."""
    from src.dataset import BVCFullTrajectoryDataset

    norm_stats = NormStats(cfg["data"]["metadata_path"])

    test_cfg = {**cfg, "data": {**cfg["data"], "path": input_path}}
    dataset  = BVCFullTrajectoryDataset(test_cfg)
    print(f"[Rollout] {len(dataset)} windows in {input_path}")

    all_results      = []
    mean_pos_rmse    = []
    mean_stress_rmse = []
    mean_vm_rmse     = []

    t0 = time.time()
    for i, window in enumerate(dataset):
        result = run_autoregressive_rollout(model, window, norm_stats, device)
        all_results.append({
            "window_name": window["meta"]["window_name"],
            "pred_frames": result["pred_frames"],
            "gt_frames":   result["gt_frames"],
            "rmse_pos":    result["rmse_pos"],
            "rmse_stress": result["rmse_stress"],
            "rmse_vm":     result["rmse_vm"],
        })
        mean_pos_rmse.append(result["rmse_pos"].mean())
        mean_stress_rmse.append(result["rmse_stress"].mean())
        mean_vm_rmse.append(result["rmse_vm"].mean())

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[Rollout] {i+1}/{len(dataset)} | "
                  f"pos_rmse={mean_pos_rmse[-1]:.5f} | "
                  f"stress_rmse={mean_stress_rmse[-1]:.5f} | "
                  f"vm_rmse={mean_vm_rmse[-1]:.5f}")

    elapsed = time.time() - t0
    summary = {
        "mean_pos_rmse":    float(np.mean(mean_pos_rmse)),
        "mean_stress_rmse": float(np.mean(mean_stress_rmse)),
        "mean_vm_rmse":     float(np.mean(mean_vm_rmse)),
        "n_windows":        len(dataset),
        "elapsed_s":        elapsed,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"summary": summary, "windows": all_results}, f)

    print(f"\n[Rollout] Done in {elapsed:.1f}s")
    print(f"[Rollout] mean_pos_rmse={summary['mean_pos_rmse']:.5f} | "
          f"mean_vm_rmse={summary['mean_vm_rmse']:.5f}")
    print(f"[Rollout] Saved → {output_path}")

    if _WANDB and wandb.run:
        wandb.log({f"rollout/{k}": v for k, v in summary.items()})

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BVC TransolverNet rollout")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--artifact",
                     help='W&B artifact, e.g. "transolver_net:best"')
    src.add_argument("--checkpoint",
                     help="Local .safetensors file")

    parser.add_argument("--experiment",
                        help="Required with --checkpoint: experiment yaml path")
    parser.add_argument("--input",         default=None, help="Test h5 file path")
    parser.add_argument("--output",        default=None, help="Output .pkl path")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-versions", action="store_true",
                        help="List all W&B artifact versions (--artifact only)")
    args = parser.parse_args()

    if args.list_versions:
        list_artifact_versions(args.artifact.split(":")[0])
        return

    if not args.input:
        parser.error("--input is required")

    if args.checkpoint and not args.experiment:
        parser.error("--experiment is required when using --checkpoint")

    device = torch.device(args.device)

    if args.artifact:
        model, cfg = load_model_for_inference(args.artifact, device)
    else:
        model, cfg = load_model_from_checkpoint(args.checkpoint, args.experiment, device)

    ver_tag  = (args.artifact or args.checkpoint).replace("/", "_").replace(":", "_")
    out_path = args.output or str(
        PROJECT_ROOT / "outputs" / "rollouts" / f"rollout_{ver_tag}.pkl"
    )

    run_inference(model, cfg, args.input, out_path, device)

    if _WANDB and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()