# 迁移自: sgnn/transolver/multi_scale_evaluate.py
# 改动内容:
#   - 导入 MultiScaleBVCFullTrajectoryDataset 改为 src.data_loader
#   - 其余验证/rollout 逻辑保留不变

from __future__ import annotations

import time
from typing import Dict, Any

import numpy as np
import torch

from src.data_loader import MultiScaleBVCFullTrajectoryDataset


def validate_multi_scale_simulator(
    simulator,
    data_path: str,
    metadata: Dict[str, Any],
    device: str,
    input_sequence_length: int = 5,
    inference_mode: str = "autoregressive",
    num_scales: int = 3,
    window_size: int = 3,
    radius_multiplier: float = 2.0,
) -> Dict[str, float]:
    """Validate the simulator on BVC validation trajectories.

    Args:
        simulator:             Trained MultiScaleSimulator.
        data_path:             Path to the validation split directory.
        metadata:              Dataset metadata dict.
        device:                PyTorch device string.
        input_sequence_length: Context frames.
        inference_mode:        ``"autoregressive"`` or ``"onestep"``.
        num_scales:            Graph hierarchy levels.
        window_size:           Spatial sub-sampling stride.
        radius_multiplier:     Connectivity radius multiplier.

    Returns:
        Dict of metric names → float values.
    """
    print(f"  [Val] data={data_path}  mode={inference_mode}  scales={num_scales}")

    dataset = MultiScaleBVCFullTrajectoryDataset(
        data_dir=data_path,
        num_scales=num_scales,
        window_size=window_size,
        radius_multiplier=radius_multiplier,
    )

    simulator.eval()
    simulator.to(device)

    total_loss, pos_losses, strain_losses, onestep_losses, times = [], [], [], [], []

    with torch.no_grad():
        for i, traj in enumerate(dataset):
            print(f"  [Val] example {i + 1}/{len(dataset)} ...", end="\r")
            simulator.set_static_graph(traj["graph"])

            positions = traj["data"]["positions"].to(device)
            T, N, D   = positions.shape
            nsteps    = T - input_sequence_length

            particle_type = torch.ones(N, dtype=torch.int64, device=device)
            n_per_example = torch.full((T,), N, dtype=torch.int64, device=device)
            strains_dummy = positions.reshape(-1, D)

            t0 = time.time()
            result = evaluate_multi_scale_rollout(
                simulator=simulator,
                positions=positions,
                particle_type=particle_type,
                n_particles_per_example=n_per_example,
                strains=strains_dummy,
                nsteps=nsteps,
                dim=D,
                device=device,
                input_sequence_length=input_sequence_length,
                inference_mode=inference_mode,
            )
            times.append(time.time() - t0)

            pos_losses.append(result["rmse_position"][-1])
            strain_losses.append(result["rmse_strain"][-1])
            onestep_losses.append(result["rmse_position"][0] + result["rmse_strain"][0])
            total_loss.append(pos_losses[-1] + strain_losses[-1])

    print()
    metrics = {
        "val/loss_total":    float(np.mean(total_loss)),
        "val/loss_position": float(np.mean(pos_losses)),
        "val/loss_strain":   float(np.mean(strain_losses)),
        "val/loss_oneStep":  float(np.mean(onestep_losses)),
        "val/mean_time":     float(np.mean(times)),
    }
    print(f"  [Val] total={metrics['val/loss_total']:.5f}  "
          f"pos={metrics['val/loss_position']:.5f}  "
          f"strain={metrics['val/loss_strain']:.5f}")
    return metrics


def evaluate_multi_scale_rollout(
    simulator,
    positions: torch.Tensor,
    particle_type: torch.Tensor,
    n_particles_per_example: torch.Tensor,
    strains: torch.Tensor,
    nsteps: int,
    dim: int,
    device: str,
    input_sequence_length: int,
    inference_mode: str = "autoregressive",
) -> Dict[str, Any]:
    """Run an autoregressive or one-step rollout and return metrics + arrays.

    Args:
        simulator:               Trained MultiScaleSimulator.
        positions:               Ground-truth positions ``(T, N, dim)``.
        particle_type:           Particle type indices ``(N,)``.
        n_particles_per_example: Unused (kept for API compatibility).
        strains:                 Unused (kept for API compatibility).
        nsteps:                  Number of prediction steps.
        dim:                     Spatial dimension (3 for BVC).
        device:                  PyTorch device string.
        input_sequence_length:   Context frames.
        inference_mode:          ``"autoregressive"`` or ``"onestep"``.

    Returns:
        Dict with ``predicted_rollout``, ``ground_truth_rollout``,
        ``rmse_position``, ``rmse_strain``, etc.
    """
    # Data loader gives (T, N, dim); simulator expects (N, T, dim)
    positions = positions.permute(1, 0, 2).contiguous()  # (N, T, dim)

    current_pos   = positions[:, :input_sequence_length].clone()
    pred_positions, pred_strains, rmse_pos, rmse_strain = [], [], [], []

    for step in range(nsteps):
        target = positions[:, input_sequence_length + step]

        next_pos, next_strain = simulator.predict_positions(
            current_positions=current_pos,
            nparticles_per_example=n_particles_per_example,
            particle_types=particle_type,
        )

        err     = torch.norm(next_pos - target, dim=-1)
        rmse_pos.append(torch.sqrt(torch.mean(err ** 2)).item())
        rmse_strain.append(rmse_pos[-1])  # placeholder until strain is predicted

        pred_positions.append(next_pos.cpu().numpy())
        pred_strains.append(next_strain.cpu().numpy())

        new_frame   = next_pos if inference_mode == "autoregressive" else target
        current_pos = torch.cat([current_pos[:, 1:], new_frame.unsqueeze(1)], dim=1)

    pred_arr = np.array(pred_positions)   # (nsteps, N, dim)
    gt_arr   = positions[:, input_sequence_length:input_sequence_length + nsteps].cpu().numpy()

    return {
        "initial_positions":    positions[:, :input_sequence_length].cpu().numpy().transpose(1, 0, 2),
        "predicted_rollout":    pred_arr,
        "ground_truth_rollout": gt_arr.transpose(1, 0, 2),
        "predicted_strain":     np.array(pred_strains),
        "particle_types":       particle_type.cpu().numpy(),
        "rmse_position":        np.array(rmse_pos),
        "rmse_strain":          np.array(rmse_strain),
        "run_time":             0.0,
        "inference_mode":       inference_mode,
    }
