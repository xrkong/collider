# 迁移自: sgnn/transolver/multi_scale_gnn.py
# 改动内容:
#   - 导入路径改为 models.blocks.transolver
#   - MultiScaleGNN 添加 @register("multi_scale_gnn")
#   - 原有逻辑保留在 MultiScaleGNN 和 TemporalMultiScaleGNN 中，未修改

import torch
import torch.nn as nn

from models.registry import register
from models.blocks.transolver import Transolver_block
from models.blocks.Transolver_plus import Transolver_plus_block


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


@register("transolver_net")
class TransolverNet(nn.Module):
    """MLP encoder → Transolver blocks → MLP decoder (single-frame input).

    Registry key: ``"transolver_net"``

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

        self._init_network(nnode_in, nnode_out, latent, layers, 
                           heads, dropout, mlp_ratio, block_act, slice_num)

    def _init_network(self, nnode_in, nnode_out, latent_dim, layers,
                      num_heads, dropout, mlp_ratio, block_act, slice_num):
        if latent_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({latent_dim}) must be divisible by num_heads ({num_heads})")

        self.input_proj = nn.Linear(nnode_in, latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(layers):
            is_last = (i == layers - 1)  
            self.blocks.append(
                Transolver_block(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Supports [N, C] or [B, N, C]."""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        tokens = self.input_proj(x)
        for block in self.blocks:
            tokens = block(tokens)
        # out = self.output_proj(tokens)
        out = tokens
        return out.squeeze(0) if squeeze else out

