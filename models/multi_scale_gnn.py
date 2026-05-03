# 迁移自: sgnn/transolver/multi_scale_gnn.py
# 改动内容:
#   - 导入路径改为 models.blocks.transolver
#   - MultiScaleGNN 添加 @register("multi_scale_gnn")
#   - 原有逻辑保留在 MultiScaleGNN 和 TemporalMultiScaleGNN 中，未修改

import torch
import torch.nn as nn

from models.registry import register
from models.blocks.transolver import Transolver_block


def build_mlp(
    input_size: int,
    hidden_layer_sizes: list[int],
    output_size: int | None = None,
    output_activation: type[nn.Module] = nn.Identity,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size is not None:
        layer_sizes.append(output_size)
    nlayers = len(layer_sizes) - 1
    acts = [activation] * nlayers
    acts[-1] = output_activation
    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module(f"NN-{i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module(f"Act-{i}", acts[i]())
    return mlp


@register("multi_scale_gnn")
class MultiScaleGNN(nn.Module):
    """MLP encoder → Transolver blocks → MLP decoder (single-frame input).

    Registry key: ``"multi_scale_gnn"``

    cfg keys used (all under ``model``):
        nnode_in_features, nnode_out_features, hidden_dim, layers,
        nmlp_layers, num_heads, dropout, mlp_ratio, block_act, slice_num.
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg.get("model", cfg)

        nnode_in  = m["nnode_in_features"]
        nnode_out = m["nnode_out_features"]
        latent    = m["hidden_dim"]
        nsteps    = m["layers"]
        nmlp      = m.get("nmlp_layers", 2)
        heads     = m.get("num_heads", 8)
        dropout   = m.get("dropout", 0.0)
        mlp_ratio = m.get("mlp_ratio", 1)
        block_act = m.get("block_act", "gelu")
        slice_num = m.get("slice_num", 64)

        self._init_network(nnode_in, nnode_out, latent, nsteps, nmlp,
                           heads, dropout, mlp_ratio, block_act, slice_num)

    def _init_network(self, nnode_in, nnode_out, latent_dim, nmessage_passing_steps,
                      nmlp_layers, num_heads, dropout, mlp_ratio, block_act, slice_num):
        if latent_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({latent_dim}) must be divisible by num_heads ({num_heads})")

        self.input_proj = nn.Sequential(
            build_mlp(nnode_in, [latent_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.blocks = nn.ModuleList([
            Transolver_block(
                num_heads=num_heads, hidden_dim=latent_dim, dropout=dropout,
                act=block_act, mlp_ratio=mlp_ratio, last_layer=False,
                out_dim=nnode_out, slice_num=slice_num,
            )
            for _ in range(nmessage_passing_steps)
        ])
        self.output_head = build_mlp(latent_dim, [latent_dim] * nmlp_layers, nnode_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Supports [N, C] or [B, N, C]."""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        tokens = self.input_proj(x)
        for block in self.blocks:
            tokens = block(tokens)
        out = self.output_head(tokens)
        return out.squeeze(0) if squeeze else out


class TemporalMultiScaleGNN(nn.Module):
    """Temporal attention over T frames → spatial Transolver.

    Not registered — experimental; wire up explicitly when needed.
    """

    def __init__(self, nnode_in_features, output_dims, latent_dim,
                 nmessage_passing_steps, nmlp_layers, num_frames=5,
                 num_heads=8, dropout=0.0, mlp_ratio=1, block_act="gelu",
                 slice_num=64, temporal_layers=2, temporal_heads=4, temporal_dropout=0.1):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})")
        if latent_dim % temporal_heads != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by temporal_heads ({temporal_heads})")

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.output_dims = output_dims

        self.frame_encoder = nn.Sequential(
            build_mlp(nnode_in_features, [latent_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.time_embedding = nn.Embedding(num_frames, latent_dim)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=temporal_heads,
            dim_feedforward=latent_dim * 2, dropout=temporal_dropout,
            batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers, norm=nn.LayerNorm(latent_dim)
        )
        self.spatial_blocks = nn.ModuleList([
            Transolver_block(num_heads=num_heads, hidden_dim=latent_dim, dropout=dropout,
                             act=block_act, mlp_ratio=mlp_ratio, last_layer=False,
                             out_dim=None, slice_num=slice_num)
            for _ in range(nmessage_passing_steps)
        ])
        self.output_heads = nn.ModuleDict({
            field: build_mlp(latent_dim, [latent_dim] * nmlp_layers, dim)
            for field, dim in output_dims.items()
        })
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.normal_(self.time_embedding.weight, std=0.02)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        squeeze = frames.dim() == 3
        if squeeze: frames = frames.unsqueeze(0)
        B, T, N, C = frames.shape
        if T != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {T}")
        h = self.frame_encoder(frames.reshape(B * T * N, C)).reshape(B, T, N, self.latent_dim)
        t_emb = self.time_embedding(torch.arange(T, device=frames.device))
        h = h + t_emb[None, :, None, :]
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.latent_dim)
        h = self.temporal_encoder(h)[:, -1, :].reshape(B, N, self.latent_dim)
        for block in self.spatial_blocks:
            h = block(h)
        outputs = {field: head(h) for field, head in self.output_heads.items()}
        if squeeze:
            outputs = {k: v.squeeze(0) for k, v in outputs.items()}
        return outputs
