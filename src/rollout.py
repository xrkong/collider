# 回滚到指定版本:
#   python src/rollout.py --artifact "multi_scale_simulator:v2" --input /path/to/raw.h5
#
# 列出所有可用版本:
#   python src/rollout.py --artifact "multi_scale_simulator" --list-versions

"""Inference / deployment script — loads a versioned artifact from W&B and runs rollout."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py

import models  # noqa: F401 — fills registry
from models.registry import build_model


# ── Raw HDF5 I/O (load_raw_h5 / apply_normalization inlined here) ────────────

def _load_raw_h5(h5_path: str) -> dict:
    """Load all fields from a raw HDF5 file produced by d3plot_to_h5.py."""
    with h5py.File(h5_path, "r") as f:
        data = {
            "positions":     f["states/positions"][:].astype(np.float32),
            "velocity":      f["states/velocity"][:].astype(np.float32),
            "acceleration":  f["states/acceleration"][:].astype(np.float32),
            "stress":        f["states/stress"][:].astype(np.float32),
            "times":         f["states/times"][:],
            "ref_positions": f["metadata/ref_positions"][:].astype(np.float32),
            "node_part_name": np.array(
                [n.decode("utf-8") if isinstance(n, bytes) else str(n)
                 for n in f["metadata/node_part_name"][:]],
                dtype=object,
            ),
        }
    T, N, _ = data["positions"].shape
    print(f"  Loaded {T} frames, {N} nodes from {h5_path}")
    return data


def _apply_normalization(data: dict, metadata: dict) -> dict:
    """Z-score normalise raw fields using metadata normalization_stats."""
    if not metadata.get("config", {}).get("normalised", False):
        return data
    ns  = metadata.get("normalization_stats", {})
    out = dict(data)
    for field in ("positions", "velocity", "acceleration", "stress"):
        if field in ns:
            m = np.array(ns[field]["mean"], dtype=np.float32)
            s = np.array(ns[field]["std"],  dtype=np.float32)
            out[field] = (data[field] - m) / s
    return out

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

def load_model_for_inference(artifact_str: str, device: torch.device):
    """Download a versioned artifact from W&B and reconstruct the model.

    Args:
        artifact_str: e.g. ``"multi_scale_simulator:v3"`` or ``"model-name:best"``.
        device:       Target compute device.

    Returns:
        Tuple[nn.Module, dict]: model in eval mode, and config dict.
    """
    if not _WANDB:
        raise ImportError("wandb required: pip install wandb")

    run      = wandb.init(job_type="inference")
    artifact = run.use_artifact(artifact_str, type="model")
    art_dir  = Path(artifact.download())

    print(f"[Rollout] Artifact:   {artifact_str}")
    print(f"[Rollout] Version:    {artifact.version}")
    print(f"[Rollout] Git commit: {artifact.metadata.get('git_commit', 'unknown')}")
    print(f"[Rollout] Val loss:   {artifact.metadata.get('val_loss', 'unknown')}")

    meta_files = list(art_dir.glob("*.json"))
    cfg: dict  = json.loads(meta_files[0].read_text()) if meta_files else {}

    weights = (list(art_dir.glob("*.safetensors")) + list(art_dir.glob("*.pt")))
    if not weights:
        raise FileNotFoundError(f"No weights file in artifact at {art_dir}")

    model_name = artifact.metadata.get("model_name") or cfg.get("model", {}).get("name")
    if not model_name:
        raise ValueError("Cannot determine model name from artifact metadata")

    model = build_model(model_name, cfg)
    w = weights[0]
    if w.suffix == ".safetensors" and _SAFETENSORS:
        model.load_state_dict(_st_load(str(w)), strict=False)
    else:
        raw = torch.load(str(w), map_location="cpu", weights_only=False)
        model.load_state_dict(raw.get("model_state", raw), strict=False)

    model.to(device).eval()
    return model, cfg


def list_artifact_versions(artifact_name: str):
    """Print all versions and aliases for a model artifact.

    Args:
        artifact_name: Without version suffix, e.g. ``"multi_scale_simulator"``.
    """
    if not _WANDB:
        raise ImportError("wandb required: pip install wandb")
    api = wandb.Api()
    print(f"\nVersions for '{artifact_name}':")
    for v in api.artifact_versions("model", artifact_name):
        aliases = ", ".join(v.aliases) or "(none)"
        print(f"  {v.version:5s}  aliases=[{aliases}]  created={v.created_at}")


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, input_path: str, output_path: str, device: torch.device, cfg: dict):
    """Run autoregressive rollout on a raw HDF5 file; save as .pkl.

    Args:
        model:       Loaded model (eval mode).
        input_path:  Raw HDF5 with ``/states/positions`` layout.
        output_path: Destination ``.pkl`` path.
        device:      Compute device.
        cfg:         Merged config dict.
    """
    from src.evaluator import evaluate_multi_scale_rollout
    from src.data_loader import build_static_multi_scale_graph

    model_cfg = cfg.get("model", cfg)
    data_cfg  = cfg.get("data", {})
    meta_path = Path(data_cfg.get("metadata_path", "dataset/data_processed/metadata.json"))
    with open(meta_path) as f:
        metadata = json.load(f)

    raw    = _load_raw_h5(input_path)
    normed = _apply_normalization(raw, metadata)

    positions = torch.from_numpy(normed["positions"]).to(device)
    T, N, D   = positions.shape
    input_seq = int(model_cfg.get("input_sequence_length", 5))
    nsteps    = T - input_seq

    particle_type = torch.zeros(N, dtype=torch.long, device=device)
    n_per_example = torch.tensor([N], dtype=torch.long, device=device)

    print(f"[Rollout] {T} frames, {N} nodes, {nsteps} steps to predict")
    graph = build_static_multi_scale_graph(
        initial_positions=positions[0],
        num_scales=int(model_cfg.get("num_scales", 2)),
        window_size=int(model_cfg.get("window_size", 6)),
        radius_multiplier=float(model_cfg.get("radius_multiplier", 2.0)),
    )
    model.set_static_graph(graph)

    t0 = time.time()
    with torch.no_grad():
        output = evaluate_multi_scale_rollout(
            simulator=model, positions=positions,
            particle_type=particle_type, n_particles_per_example=n_per_example,
            strains=positions.reshape(-1, D),
            nsteps=nsteps, dim=D, device=str(device),
            input_sequence_length=input_seq, inference_mode="autoregressive",
        )
    elapsed = time.time() - t0

    output.update({
        "run_time": elapsed, "metadata": metadata,
        "case_name": Path(input_path).stem, "node_part_name": raw["node_part_name"],
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"[Rollout] {elapsed:.1f}s — final pos_rmse={output['rmse_position'][-1]:.5f}")
    print(f"[Rollout] Saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BVC inference / rollout")
    parser.add_argument("--artifact",      required=True,
                        help='W&B artifact, e.g. "multi_scale_simulator:v3"')
    parser.add_argument("--input",         default=None, help="Raw HDF5 input file")
    parser.add_argument("--output",        default=None, help="Output .pkl path")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-versions", action="store_true",
                        help="List all artifact versions for the given model name")
    args = parser.parse_args()

    if args.list_versions:
        list_artifact_versions(args.artifact.split(":")[0])
        return

    if not args.input:
        parser.error("--input is required")

    device     = torch.device(args.device)
    model, cfg = load_model_for_inference(args.artifact, device)

    case_name = Path(args.input).stem
    ver_tag   = args.artifact.replace("/", "_").replace(":", "_")
    out_path  = args.output or str(
        PROJECT_ROOT / "outputs" / "rollouts" / f"{case_name}_{ver_tag}.pkl"
    )

    run_inference(model, args.input, out_path, device, cfg)

    if _WANDB and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
