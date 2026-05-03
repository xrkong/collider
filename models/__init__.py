"""Auto-import every model_*.py file and BVC-specific model files.

Any .py file in this directory (except registry.py and __init__.py) that
contains @register(...) calls will be imported here, which fills MODEL_REGISTRY
at package-import time.
"""
import importlib
import pkgutil
from pathlib import Path

# Files to skip
_SKIP = {"__init__", "registry"}

for _info in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if _info.name not in _SKIP:
        importlib.import_module(f"models.{_info.name}")
