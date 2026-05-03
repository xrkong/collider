# 新建模型时复制此文件，修改:
#   1. 类名 (ModelTemplate → YourModel)
#   2. @register("model_template") 中的字符串 key
#   3. __init__ 中从 cfg 读取的参数
#   4. forward 中的网络逻辑

import torch
import torch.nn as nn

from models.registry import register


def _get(cfg, key: str, default=None):
    """Read from dict or OmegaConf DictConfig transparently."""
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return default


@register("model_template")
class ModelTemplate(nn.Module):
    """Minimal template demonstrating the cfg-based __init__ pattern."""

    def __init__(self, cfg):
        super().__init__()

        # ── Read model params from cfg ────────────────────────────────────
        # cfg may be a plain dict or OmegaConf DictConfig.
        # Access via cfg["key"] works for both.
        model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg.get("model", cfg)

        hidden_dim  = _get(model_cfg, "hidden_dim",  256)
        num_layers  = _get(model_cfg, "num_layers",  4)
        dropout     = _get(model_cfg, "dropout",     0.1)
        num_classes = _get(model_cfg, "num_classes", 10)

        # ── Network (placeholder with nn.Linear) ─────────────────────────
        layers = []
        in_dim = hidden_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor of shape (B, hidden_dim).
        Returns:
            Logits of shape (B, num_classes).
        """
        return self.net(x)
