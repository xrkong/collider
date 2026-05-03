from __future__ import annotations
from typing import Any

MODEL_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator that registers an nn.Module class under *name*.

    Usage::

        @register("my_model")
        class MyModel(nn.Module):
            def __init__(self, cfg): ...
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise KeyError(
                f"Model '{name}' is already registered by {MODEL_REGISTRY[name]}"
            )
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, cfg: Any):
    """Instantiate a registered model by name.

    Args:
        name: Registry key, e.g. ``"multi_scale_simulator"``.
        cfg:  Config passed directly to the model ``__init__``.
              Accepts plain ``dict`` or OmegaConf ``DictConfig``.

    Raises:
        KeyError: If *name* is not registered; lists all available names.
    """
    if name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY)
        raise KeyError(
            f"Model '{name}' not found in registry.\n"
            f"Available: {available}\n"
            f"Ensure the model module is imported before calling build_model()."
        )
    return MODEL_REGISTRY[name](cfg)


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(MODEL_REGISTRY)
