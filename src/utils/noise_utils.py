# 迁移自: sgnn/noise_utils.py
# 改动内容: 移除对 sgnn.single_scale.learned_simulator.time_diff 的循环依赖，内联实现

import torch


def _time_diff(position_sequence: torch.Tensor) -> torch.Tensor:
    return (position_sequence[:, 1:] - position_sequence[:, :-1]).contiguous()


def get_random_walk_noise_for_position_sequence(
    position_sequence: torch.Tensor,
    noise_std_last_step: float,
) -> torch.Tensor:
    """Random-walk noise in velocity space integrated back to position space.

    Args:
        position_sequence:   Shape ``(N, T, dim)``.
        noise_std_last_step: Std of the noise at the final velocity step.

    Returns:
        Noise tensor with same shape as ``position_sequence``.
    """
    velocity_sequence = _time_diff(position_sequence)
    num_velocities    = velocity_sequence.shape[1]
    vel_noise         = torch.randn_like(velocity_sequence) * (noise_std_last_step / num_velocities ** 0.5)
    vel_noise         = torch.cumsum(vel_noise, dim=1)
    pos_noise         = torch.cat([torch.zeros_like(vel_noise[:, :1]),
                                   torch.cumsum(vel_noise, dim=1)], dim=1)
    return pos_noise
