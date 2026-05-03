# 迁移自: sgnn/transolver/multi_scale_simulator.py
# 改动内容:
#   - 添加 @register("multi_scale_simulator")
#   - __init__ 改为接收 cfg 对象；原有构建逻辑不变，移入 _build_from_cfg()
#   - 导入路径改为 project 内部路径
#   - 原有 predict_accelerations / predict_positions / save / load 等方法保留不变

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any

from models.registry import register
from models.multi_scale_gnn import MultiScaleGNN
from models.multi_scale_graph import MultiScaleConfig


@register("multi_scale_simulator")
class MultiScaleSimulator(nn.Module):
    """Multi-scale Transolver simulator for particle-based crash simulations.

    Registry key: ``"multi_scale_simulator"``

    Required cfg keys (all under ``model``):
        dim, hidden_dim, layers, nmlp_layers, num_heads, dropout, mlp_ratio,
        block_act, slice_num, num_scales, window_size, radius_multiplier,
        input_sequence_length, particle_type_embedding_size,
        nnode_in (computed at runtime), nparticle_types (from metadata),
        normalization_stats (dict of tensors, set at runtime).

    Runtime fields are added to cfg by train.py before calling build_model().
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg.get("model", cfg)
        self._build_from_cfg(m)

    def _build_from_cfg(self, m):
        self._kinematic_dimensions = m["dim"]
        self._normalization_stats  = m["normalization_stats"]   # set at runtime
        self._nparticle_types      = m.get("nparticle_types", 1)
        self._num_scales           = m["num_scales"]
        self._window_size          = m["window_size"]
        self._device               = m.get("device", "cpu")

        self._particle_type_embedding = nn.Embedding(
            self._nparticle_types, m["particle_type_embedding_size"]
        )
        self._multi_scale_config = MultiScaleConfig(
            num_scales=m["num_scales"],
            window_size=m["window_size"],
            radius_multiplier=m["radius_multiplier"],
        )

        # Build MultiScaleGNN via its own constructor (pass subset of cfg)
        gnn_cfg = {
            "model": {
                "nnode_in_features":  m["nnode_in"],
                "nnode_out_features": m["dim"] + 1,
                "hidden_dim":  m["hidden_dim"],
                "layers":      m["layers"],
                "nmlp_layers": m.get("nmlp_layers", 2),
                "num_heads":   m.get("num_heads", 8),
                "dropout":     m.get("dropout", 0.0),
                "mlp_ratio":   m.get("mlp_ratio", 1),
                "block_act":   m.get("block_act", "gelu"),
                "slice_num":   m.get("slice_num", 64),
            }
        }
        self._multi_scale_gnn = MultiScaleGNN(gnn_cfg)
        self._static_graph_data = None

    # ── Public API (unchanged from original) ──────────────────────────────

    def forward(self):
        pass

    def set_static_graph(self, graph_data: Dict[str, Any]):
        self._static_graph_data = graph_data

    def get_static_graph_data(self) -> Optional[Dict[str, Any]]:
        return self._static_graph_data

    def predict_positions(self, current_positions, nparticles_per_example, particle_types):
        node_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types)
        pred = self._multi_scale_gnn(node_features)
        if pred.dim() == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        next_positions = self._decoder_postprocessor(
            pred[..., :self._kinematic_dimensions], current_positions)
        return next_positions, pred[..., -1]

    def predict_accelerations(self, next_positions, position_sequence_noise,
                              position_sequence, nparticles_per_example, particle_types):
        noisy = position_sequence + position_sequence_noise
        node_features = self._encoder_preprocessor(noisy, nparticles_per_example, particle_types)
        pred = self._multi_scale_gnn(node_features)
        if pred.dim() == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        pred_acc = pred[..., :self._kinematic_dimensions]
        pred_strain = pred[..., -1]
        next_pos_adj = next_positions + position_sequence_noise[:, -1]
        target_acc = self._inverse_decoder_postprocessor(next_pos_adj, noisy)
        return pred_acc, target_acc, pred_strain

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = raw.get("model_state", raw)
        self.load_state_dict(state_dict)

    # ── Private helpers (unchanged) ───────────────────────────────────────

    def _encoder_preprocessor(self, position_sequence, nparticles_per_example, particle_types):
        del nparticles_per_example
        position_sequence = position_sequence.contiguous()
        nparticles = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1].contiguous()
        velocity_sequence = self._time_diff(position_sequence).contiguous()

        vel_stats = self._normalization_stats["velocity"]
        norm_vel  = ((velocity_sequence - vel_stats["mean"]) / vel_stats["std"]).contiguous()
        node_features = [norm_vel.reshape(nparticles, -1)]

        grid_radius = (self._multi_scale_config.grid_spacing
                       * self._multi_scale_config.radius_multiplier)
        wall_dist = torch.clamp(most_recent_position[:, 0:1] + 2.0,
                                min=0.0, max=grid_radius) / grid_radius
        node_features.append(wall_dist)

        if self._nparticle_types > 1:
            node_features.append(self._particle_type_embedding(particle_types))

        return torch.cat(node_features, dim=-1)

    def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
        acc_stats = self._normalization_stats["acceleration"]
        acceleration = normalized_acceleration * acc_stats["std"] + acc_stats["mean"]
        most_recent = position_sequence[:, -1]
        prev_velocity = most_recent - position_sequence[:, -2]
        return most_recent + prev_velocity + acceleration

    def _inverse_decoder_postprocessor(self, next_position, position_sequence):
        prev = position_sequence[:, -1]
        prev_vel = prev - position_sequence[:, -2]
        acc = (next_position - prev) - prev_vel
        acc_stats = self._normalization_stats["acceleration"]
        return (acc - acc_stats["mean"]) / acc_stats["std"]

    def _time_diff(self, position_sequence):
        return (position_sequence[:, 1:] - position_sequence[:, :-1]).contiguous()
