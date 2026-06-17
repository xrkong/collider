"""Shared condition representation. Mechanism-agnostic.
Method A/B/D/F all consume normalize_conditions(); only the injection differs.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np

# Canonical ordering. Channels appear in this order; disabled ones are skipped.
SCALAR_ORDER = ("speed", "mass", "angle")


@dataclass
class CondConfig:
    enabled: tuple[str, ...] = ("speed", "mass")      # D4 default
    ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "speed": (60.0, 100.0),
        "mass":  (0.0, 1000.0),
        "angle": (0.0, 30.0),    # magnitude; sign dropped in parse_conditions (D5)
    })
    material_vocab: tuple[str, ...] = ("F",)
    use_material: bool = False                         # D4: off while single type

    def n_cond(self) -> int:
        n = sum(1 for k in SCALAR_ORDER if k in self.enabled)
        if self.use_material and len(self.material_vocab) > 1:
            n += len(self.material_vocab)
        return n


def parse_conditions(metadata: dict | None, dir_name: str) -> dict:
    """Return raw physical values: {speed, mass, angle, material}.

    Dir naming convention: T_lok_F_shape_barrier_9_3_{speed}km[_plus{mass}kg]
    Metadata keys are checked first; dir-name regex is the fallback.
    Mass defaults to 0.0 when not encoded in the dir name (no 'plus...kg' suffix).
    """
    md = metadata or {}
    out: dict = {}
    # Speed in km/h
    out["speed"] = _get(md, ["speed_kmh", "speed", "v"], dir_name,
                        r"[_-](\d+)km(?:[_.]|$)")
    # Added mass in kg — absent suffix means 0 kg
    out["mass"] = _get(md, ["mass_kg", "added_mass", "m"], dir_name,
                       r"plus(\d+)kg", required=False, default=0.0)
    # Barrier orientation angle — take magnitude (D5); single-sided range (0, 30)
    out["angle"] = abs(_get(md, ["angle_deg", "orientation_deg", "angle"], dir_name,
                            r"(?:a|ang|angle)[_-]?(-?\d+(?:\.\d+)?)",
                            required=False, default=25.4))
    out["material"] = md.get("barrier_material", md.get("material", "F"))
    return out


def normalize_conditions(raw: dict, cfg: CondConfig) -> np.ndarray:
    """Physical dict -> float32 vector in canonical order. Length == cfg.n_cond()."""
    vec: list[float] = []
    for key in SCALAR_ORDER:
        if key not in cfg.enabled:
            continue
        lo, hi = cfg.ranges[key]
        x = float(raw[key])
        vec.append(2.0 * (x - lo) / (hi - lo) - 1.0)
    if cfg.use_material and len(cfg.material_vocab) > 1:
        oh = [0.0] * len(cfg.material_vocab)
        oh[list(cfg.material_vocab).index(raw["material"])] = 1.0
        vec.extend(oh)
    out = np.asarray(vec, dtype=np.float32)
    assert out.shape[0] == cfg.n_cond(), \
        f"cond shape {out.shape[0]} != n_cond={cfg.n_cond()}"
    return out


def _get(md, keys, dir_name, pattern, required=True, default=None):
    for k in keys:
        if k in md and md[k] is not None:
            return float(md[k])
    m = re.search(pattern, dir_name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    if required:
        raise KeyError(
            f"condition not found in metadata {list(md)} or dir '{dir_name}' "
            f"via pattern {pattern!r}"
        )
    return default
