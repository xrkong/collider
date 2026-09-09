# 迁移自: sgnn/transolver/multi_scale_gnn.py
# 改动内容:
#   - 导入路径改为 models.blocks.transolver
#   - MultiScaleGNN 添加 @register("multi_scale_gnn")
#   - 原有逻辑保留在 MultiScaleGNN 和 TemporalMultiScaleGNN 中，未修改

import math

import torch
import torch.nn as nn

from models.registry import register
from models.blocks.transolver import Transolver_block
from models.blocks.Transolver_plus import Transolver_plus_block


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of a per-example scalar time in [0, 1]. timesteps: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.reshape(-1, 1).float() * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


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


@register("transolverplus_net")
class TransolverplusNet(nn.Module):
    """MLP encoder → Transolver blocks → MLP decoder (single-frame input).

    Registry key: ``"transolverplus_net"``

    cfg keys used (all under ``model``):
        nnode_in_features, nnode_out_features, hidden_dim, layers,
        num_heads, dropout, mlp_ratio, block_act, slice_num.
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg.get("model", cfg)

        nnode_in  = m["nnode_in_features"]
        nnode_out = m["nnode_out_features"]
        latent    = m["hidden_dim"]
        layers    = m["layers"]
        heads     = m.get("num_heads", 8)
        dropout   = m.get("dropout", 0.0)
        mlp_ratio = m.get("mlp_ratio", 1)
        block_act = m.get("block_act", "gelu")
        slice_num = m.get("slice_num", 64)

        self.num_node_types = m.get("num_node_types", 0)
        type_emb_dim        = m.get("type_emb_dim", 0)
        if self.num_node_types > 0:
            assert type_emb_dim > 0, "type_emb_dim must be > 0 when num_node_types > 0"
            self.type_embed = nn.Embedding(self.num_node_types, type_emb_dim)
        else:
            self.type_embed = None
            type_emb_dim = 0

        self._init_network(nnode_in + type_emb_dim, nnode_out, latent, layers,
                           heads, dropout, mlp_ratio, block_act, slice_num)

        self.time_conditioned = bool(m.get("time_conditioned", False))
        self.time_fc = nn.Sequential(
            nn.Linear(latent, latent), nn.SiLU(), nn.Linear(latent, latent)
        ) if self.time_conditioned else None

    def _init_network(self, nnode_in, nnode_out, latent_dim, layers,
                      num_heads, dropout, mlp_ratio, block_act, slice_num):
        if latent_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({latent_dim}) must be divisible by num_heads ({num_heads})")

        self.input_proj = nn.Linear(nnode_in, latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(layers):
            is_last = (i == layers - 1)  
            self.blocks.append(
                Transolver_plus_block(
                    num_heads=num_heads, 
                    hidden_dim=latent_dim, 
                    dropout=dropout,
                    act=block_act, 
                    mlp_ratio=mlp_ratio, 
                    last_layer=is_last,  
                    out_dim=nnode_out, 
                    slice_num=slice_num,
                )
            )
        # self.output_proj = nn.Linear(latent_dim, nnode_out)

    def forward(self, x: torch.Tensor, node_type: torch.Tensor | None = None,
                t: torch.Tensor | None = None) -> torch.Tensor:
        """Supports [N, C] or [B, N, C]. Pass node_type (N,) or (B,N) when type_embed is active.
        Pass t (scalar or (B,), normalized query time) when time_conditioned is active."""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        if self.type_embed is not None:
            assert node_type is not None, "node_type must be provided when type_embed is enabled"
            nt = node_type.long()
            if nt.dim() == 1:
                nt = nt.unsqueeze(0).expand(x.shape[0], -1)  # (B, N)
            emb = self.type_embed(nt)                        # (B, N, type_emb_dim)
            x = torch.cat([x, emb], dim=-1)
        tokens = self.input_proj(x)
        if self.time_conditioned:
            assert t is not None, "t must be provided when model.time_conditioned=true"
            tt = t.to(tokens.dtype)
            if tt.dim() == 0:
                tt = tt.unsqueeze(0)
            time_emb = self.time_fc(timestep_embedding(tt, tokens.shape[-1]))  # (B, H)
            tokens = tokens + time_emb.unsqueeze(1)  # broadcast over N
        for block in self.blocks:
            tokens = block(tokens)
        # out = self.output_proj(tokens)
        out = tokens
        return out.squeeze(0) if squeeze else out

