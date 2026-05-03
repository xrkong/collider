"""Offline evaluation script — loads a model from W&B Artifacts and runs on test data.

Usage:
    python src/evaluate.py \\
        --artifact "barrier-vehicle-collision/multi_scale_simulator:best" \\
        --data dataset/data_processed/test \\
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
from src.utils.metrics import compute_metrics
from src.utils.visualization import plot_confusion_matrix, log_figures_to_wandb

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


# ── Artifact loading ──────────────────────────────────────────────────────────

def load_model_from_artifact(artifact_str: str):
    """Download a model artifact from W&B and rebuild it.

    Args:
        artifact_str: e.g. ``"my-project/multi_scale_simulator:v3"``
                      or ``"multi_scale_simulator:best"``.

    Returns:
        Tuple[nn.Module, dict]: ``(model, cfg)``
    """
    if not _WANDB:
        raise ImportError("wandb is required for artifact loading: pip install wandb")

    run = wandb.init(job_type="eval")
    artifact = run.use_artifact(artifact_str, type="model")
    artifact_dir = Path(artifact.download())

    # Find weights file
    weights_path = next(
        (artifact_dir / f for f in artifact.files()
         if f.endswith(".safetensors") or f.endswith(".pt")),
        None,
    )
    if weights_path is None:
        # Try by glob
        candidates = list(artifact_dir.glob("*.safetensors")) + list(artifact_dir.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No weights file in artifact at {artifact_dir}")
        weights_path = candidates[0]

    # Find metadata json
    meta_candidates = list(artifact_dir.glob("*.json"))
    cfg: dict = {}
    if meta_candidates:
        with open(meta_candidates[0]) as f:
            cfg = json.load(f)

    model_name = artifact.metadata.get("model_name") or cfg.get("model", {}).get("name")
    if model_name is None:
        raise ValueError("Cannot determine model name from artifact metadata")

    model = build_model(model_name, cfg)

    if weights_path.suffix == ".safetensors" and _SAFETENSORS:
        model.load_state_dict(_st_load(str(weights_path)), strict=False)
    else:
        raw = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        model.load_state_dict(raw.get("model_state", raw), strict=False)

    print(f"[Eval] Loaded '{model_name}' from artifact '{artifact_str}'")
    print(f"[Eval] Git commit: {artifact.metadata.get('git_commit', 'unknown')}")
    return model, cfg


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation(
    model: torch.nn.Module,
    data_path: str,
    output_dir: str,
    device: str = "cpu",
):
    """Run full test-set evaluation.

    Args:
        model:      Trained model (will be set to eval mode).
        data_path:  Path to test split directory.
        output_dir: Directory to save metrics.json and figures.
        device:     PyTorch device string.
    """
    from src.data_loader import MultiScaleBVCFullTrajectoryDataset
    from src.evaluator import evaluate_multi_scale_rollout

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device)
    model.to(device_t).eval()

    dataset = MultiScaleBVCFullTrajectoryDataset(data_dir=data_path)

    all_pos_rmse: list[float] = []
    all_strain_rmse: list[float] = []

    with torch.no_grad():
        for i, traj in enumerate(dataset):
            model.set_static_graph(traj["graph"])
            positions = traj["data"]["positions"].to(device_t)
            T, N, D   = positions.shape
            input_seq = getattr(model, "_input_seq", 5)
            nsteps    = T - input_seq

            particle_type = torch.ones(N, dtype=torch.int64, device=device_t)
            n_per_example = torch.full((T,), N, dtype=torch.int64, device=device_t)
            strains_dummy = positions.reshape(-1, D)

            result = evaluate_multi_scale_rollout(
                simulator=model,
                positions=positions,
                particle_type=particle_type,
                n_particles_per_example=n_per_example,
                strains=strains_dummy,
                nsteps=nsteps,
                dim=D,
                device=str(device_t),
                input_sequence_length=input_seq,
                inference_mode="autoregressive",
            )
            all_pos_rmse.append(float(result["rmse_position"][-1]))
            all_strain_rmse.append(float(result["rmse_strain"][-1]))
            print(f"[Eval] Traj {i+1}/{len(dataset)} — pos_rmse={all_pos_rmse[-1]:.5f}")

    import numpy as np
    metrics = {
        "mean_pos_rmse":    float(np.mean(all_pos_rmse)),
        "mean_strain_rmse": float(np.mean(all_strain_rmse)),
        "n_trajectories":   len(all_pos_rmse),
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[Eval] Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"[Eval] Metrics saved → {metrics_path}")
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BVC offline evaluation")
    parser.add_argument("--artifact",   required=True,
                        help='W&B artifact string, e.g. "project/model-name:best"')
    parser.add_argument("--data",       required=True, help="Test data directory")
    parser.add_argument("--output-dir", default="outputs/eval/",
                        help="Where to save metrics and figures")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, cfg = load_model_from_artifact(args.artifact)
    run_evaluation(model, args.data, args.output_dir, args.device)

    if _WANDB and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
