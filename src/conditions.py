"""Shared condition representation. Mechanism-agnostic.
Method A/B/D/F all consume normalize_conditions(); only the injection differs.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np

# Canonical ordering. Channels appear in this order; disabled ones are skipped.
# "layer"/"thickness" describe a barrier's added material stack (e.g. a rubber
# layer on top of the base T-lok design) — only meaningful for barrier designs
# that actually vary them, see the gating note on GATED_BY_MATERIAL below.
SCALAR_ORDER = ("speed", "mass", "angle", "layer", "thickness")

# layer/thickness are only physically meaningful for barrier designs that
# actually have a variable material stack — for the GT/baseline design
# (material_vocab[0] by convention) normalize_conditions() forces them to a
# fixed neutral 0.0 regardless of their raw value, so the model sees a
# consistent "not applicable" sentinel rather than whatever default/leftover
# raw value happened to be supplied for a GT trajectory.
GATED_BY_MATERIAL = ("layer", "thickness")


@dataclass
class CondConfig:
    enabled: tuple[str, ...] = ("speed", "mass")      # D4 default
    ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "speed":     (60.0, 100.0),
        "mass":      (0.0, 1000.0),
        "angle":     (0.0, 30.0),    # magnitude; sign dropped in parse_conditions (D5)
        "layer":     (0.0, 5.0),     # number of added material layers
        "thickness": (0.0, 30.0),    # mm, added layer thickness
    })
    material_vocab: tuple[str, ...] = ("F",)
    use_material: bool = False                         # D4: off while single type

    def n_cond(self) -> int:
        n = sum(1 for k in SCALAR_ORDER if k in self.enabled)
        if self.use_material and len(self.material_vocab) > 1:
            n += len(self.material_vocab)
        return n


def parse_conditions(metadata: dict | None, dir_name: str) -> dict:
    """Return raw physical values: {speed, mass, angle, layer, thickness, material}.

    Dir naming convention: T_lok_F_shape_barrier_9_3_{speed}km[_plus{mass}kg]
    Metadata keys are checked first; dir-name regex is the fallback.
    Mass defaults to 0.0 when not encoded in the dir name (no 'plus...kg' suffix).
    Speed defaults to 100.0 km/h when not encoded in the dir name — some
    trajectories (e.g. New_Road_Barrier_*) are single-speed runs with no
    '{speed}km' suffix.

    layer/thickness/material have no dir-name naming convention (no filename
    in the corpus encodes them) — they're expected to come from `metadata`,
    populated per-trajectory from the experiment yaml's data.train_dirs/
    val_dirs entries (see train.py's parse_dir_entry / src/dataset.py's
    per-trajectory metadata merge), not guessed from the path.
    """
    md = metadata or {}
    out: dict = {}
    # Speed in km/h
    out["speed"] = _get(md, ["speed_kmh", "speed", "v"], dir_name,
                        r"[_-](\d+)km(?:[_.]|$)", required=False, default=100.0)
    # Added mass in kg — absent suffix means 0 kg
    out["mass"] = _get(md, ["mass_kg", "added_mass", "m"], dir_name,
                       r"plus(\d+)kg", required=False, default=0.0)
    # Barrier orientation angle — take magnitude (D5); single-sided range (0, 30)
    out["angle"] = abs(_get(md, ["angle_deg", "orientation_deg", "angle"], dir_name,
                            r"(?:a|ang|angle)[_-]?(-?\d+(?:\.\d+)?)",
                            required=False, default=25.4))
    # Added-layer count / thickness — metadata-only in practice (see docstring)
    out["layer"] = _get(md, ["layer", "layers", "n_layers"], dir_name,
                        r"(\d+)[_-]?layers?", required=False, default=0.0)
    out["thickness"] = _get(md, ["thickness", "thickness_mm"], dir_name,
                            r"thickness[_-]?(\d+(?:\.\d+)?)", required=False, default=0.0)
    out["material"] = md.get("barrier_material", md.get("material", "F"))
    return out


def normalize_conditions(raw: dict, cfg: CondConfig) -> np.ndarray:
    """Physical dict -> float32 vector in canonical order. Length == cfg.n_cond()."""
    vec: list[float] = []
    gated_positions: list[int] = []   # indices in `vec` holding GATED_BY_MATERIAL values
    for key in SCALAR_ORDER:
        if key not in cfg.enabled:
            continue
        lo, hi = cfg.ranges[key]
        x = float(raw[key])
        vec.append(2.0 * (x - lo) / (hi - lo) - 1.0)
        if key in GATED_BY_MATERIAL:
            gated_positions.append(len(vec) - 1)

    # Barrier-type gating: layer/thickness only mean something for barrier
    # designs that actually vary them. By convention material_vocab[0] is the
    # GT/baseline design — force gated channels to a fixed neutral 0.0 for it,
    # regardless of whatever raw layer/thickness value was supplied.
    if gated_positions and cfg.use_material and len(cfg.material_vocab) > 0:
        if raw.get("material") == cfg.material_vocab[0]:
            for i in gated_positions:
                vec[i] = 0.0

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
