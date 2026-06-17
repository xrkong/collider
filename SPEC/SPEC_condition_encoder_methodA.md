# SPEC — Condition Encoder (shared) + Method A: broadcast concat

**Status:** ready for agent implementation
**Owner:** Ray
**Scope of this SPEC:** build the *shared* condition-representation layer (parse → normalize → plumb through the dataset) and wire it into the model via **Method A (broadcast concat)** only.
**Explicitly NOT in this SPEC:** FiLM (B), AdaLN (D), slice-token injection (F), per-node barrier material, and any change to SDF angle handling. Those are separate SPECs that will *reuse* the representation layer built here.

---

## 0. Goal in one sentence

Make the four physical conditions (speed, added mass, barrier orientation angle, barrier material) *visible* to the model by parsing them per trajectory, normalizing them with declared physical ranges, returning them as a per-sample vector, and concatenating that vector onto every node's input feature.

Today the model sees only node-level velocity history + a hardcoded SDF. It must reverse-engineer the regime from dynamics alone. After this change the regime is an explicit input.

---

## 1. Design decisions (read before coding)

These are deliberate. Do not "optimize" them away.

- **D1 — Normalize conditions with *declared physical ranges*, NOT data-computed stats.**
  With only 3 discrete levels per axis, a data-computed mean/std is degenerate and split-dependent (the trajectory-level split means train may not even contain all levels). A fixed `(lo, hi)` declared in config maps each scalar to `[-1, 1]`, is invariant to which trajectories are in the split, and is reproducible across experiments as the factorial grows. We want speed=90 (unseen) to land cleanly at +0.5, between 80 (0.0) and 100 (+1.0).
  Ranges are set to the **tested envelope** (speed 60–100, mass 0–1000, angle 0–30), which gives maximum `[-1,1]` resolution across the interpolation region we actually care about. Tradeoff to be aware of: speed has **no extrapolation headroom** — 60 and 100 sit exactly at ∓1, so any future speed outside [60,100] lands outside [-1,1]. That is fine as long as we don't intend to evaluate beyond the trained envelope; widen the range later if we do.

- **D2 — Separate *representation* from *injection*.**
  A new module `src/conditions.py` owns parsing + normalization + canonical ordering. It is mechanism-agnostic. Method A's only job is to broadcast + concat the vector this module produces. B/D/F will later consume the *same* vector and add their own injection. No condition-parsing logic lives in `train.py` or the model.

- **D3 — Condition is constant within a trajectory.**
  Speed/mass/angle/material do not change over the rollout. The vector is computed once per trajectory and is identical for every window and every push-forward unroll step. The K-step unroll loop must reuse it unchanged (see 4.4).

- **D4 — Drop zero-variance channels by default.**
  Current 9-trajectory set varies only `speed × mass`; barrier orientation angle is fixed (~25.4°), barrier is F-shape only. Including a constant channel just feeds the model a useless dimension. Therefore the **default enabled set is `["speed", "mass"]` → n_cond = 2.** Angle and material are parsed and supported but disabled until they actually vary. (When a 2nd barrier type or a varying angle enters the factorial, flip them on in config — no code change.)

- **D5 — Angle is barrier orientation; it enters via the SDF, the scalar is redundant-by-design.**
  This angle is *not* an impact heading — it's the orientation of the barrier centerline projected onto the xy-plane. Its sign is meaningless (it only describes a direction within a narrow 15–25° window), so it is taken as `abs()` **before** normalization and mapped from a single-sided range `(0, 30)`. Because barrier geometry is exactly what the SDF encodes, this orientation is already present in `x_sdf`; the scalar channel is therefore optional and OFF by default.
  IMPORTANT: `compute_sdf_batch` currently hardcodes `barrier_angle_deg=-25.4`. If the current data already varies barrier orientation across trajectories, the SDF is **wrong right now** (all trajectories distanced against one fixed orientation). This must be confirmed and, if so, fixed first (read orientation per-trajectory from metadata). See §6.

- **D6 — Append condition channels at the END of the feature vector.**
  Keep input dims `[0:20]` exactly as they are so any feature-index-specific logic downstream is untouched. Condition occupies `[20 : 20+n_cond]`.

---

## 2. Where the four conditions live (current → target)

| Condition | Type | Current location | After this SPEC |
|---|---|---|---|
| vehicle speed | continuous scalar | dir name only | parsed → normalized → `cond[0]` |
| added mass | continuous scalar | dir name only | parsed → normalized → `cond[1]` |
| barrier orientation angle | continuous scalar (single-sided) | hardcoded in SDF | parsed (abs), supported, **OFF by default** (in SDF) |
| barrier material | categorical | dir name only | parsed, supported, **OFF by default** (single type) |

Declared physical ranges (config defaults; set to the tested envelope — see D1):

| Condition | lo | hi | dataset values | note |
|---|---|---|---|---|
| speed (km/h) | 60 | 100 | 60, 80, 100 | envelope exactly; no headroom by choice |
| mass added (kg) | 0 | 1000 | 0, 400, 800 | headroom above 800 |
| angle (deg, magnitude) | 0 | 30 | ~25.4 (abs) | sign dropped via `abs()` before norm |
| material | N/A | N/A | F | single type for now; no one-hot needed |

Normalization (all enabled scalars): `x_norm = 2 * (x - lo) / (hi - lo) - 1`. For angle, `x` is `abs(raw_angle)`.
For this case, the varible include speed and mass, set angle and material to default values (angle=25.4, material="F"), and run with the default config (`enabled: [speed, mass]`).


---

## 3. File-by-file change list

1. **NEW** `src/conditions.py` — representation layer (parse + normalize + ordering).
2. **EDIT** config YAML — add a `condition:` block; make `nnode_in_features` derived.
3. **EDIT** `src/dataset.py` — call the parser at load time, store on the trajectory entry, return `cond` from `__getitem__`.
4. **EDIT** `train.py` — broadcast `cond` to `(B, N, n_cond)` and concat onto `x_in`; reuse unchanged across unroll steps.

No change to the model class is required for A: the input projection is built from `nnode_in_features`, so a wider input + the bumped config value is sufficient. (Verify — see §7 assumption A5.)

---

## 4. Exact interfaces & edits

### 4.1 NEW `src/conditions.py`

```python
"""Shared condition representation. Mechanism-agnostic.
Method A/B/D/F all consume build_condition_vector(); only the injection differs.
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
        "angle": (0.0, 30.0),     # magnitude; sign dropped in parse_conditions (D5)
    })
    material_vocab: tuple[str, ...] = ("F",)          # one-hot only if len > 1
    use_material: bool = False                        # D4: off while single type

    def n_cond(self) -> int:
        n = sum(1 for k in SCALAR_ORDER if k in self.enabled)
        if self.use_material and len(self.material_vocab) > 1:
            n += len(self.material_vocab)
        return n


def parse_conditions(metadata: dict | None, dir_name: str) -> dict:
    """Return raw PHYSICAL values: {speed, mass, angle, material}.
    Prefer metadata.json; fall back to regex on dir name. Raise on missing
    required fields rather than silently defaulting (fail loud)."""
    out: dict = {}
    md = metadata or {}
    out["speed"]    = _get(md, ["speed_kmh", "speed", "v"],   dir_name, r"(?:v|speed)[_-]?(\d+(?:\.\d+)?)")
    out["mass"]     = _get(md, ["mass_kg", "added_mass", "m"], dir_name, r"(?:m|mass)[_-]?(\d+(?:\.\d+)?)")
    # angle: barrier orientation, sign is meaningless (single-sided) -> take magnitude (D5)
    out["angle"]    = abs(_get(md, ["angle_deg", "orientation_deg", "angle"], dir_name,
                              r"(?:a|ang|angle)[_-]?(-?\d+(?:\.\d+)?)", required=False, default=25.4))
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
        oh[cfg.material_vocab.index(raw["material"])] = 1.0
        vec.extend(oh)
    out = np.asarray(vec, dtype=np.float32)
    assert out.shape[0] == cfg.n_cond(), (out.shape, cfg.n_cond())
    return out


def _get(md, keys, dir_name, pattern, required=True, default=None):
    for k in keys:
        if k in md and md[k] is not None:
            return float(md[k])
    m = re.search(pattern, dir_name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    if required:
        raise KeyError(f"condition not found in metadata {list(md)} or dir '{dir_name}' via {pattern!r}")
    return default
```

> Adjust the metadata key candidate lists and the dir-name regexes to the real schema/naming once known (assumptions A2, A3).

### 4.2 Config YAML

```yaml
condition:
  enabled: [speed, mass]          # D4
  ranges:
    speed: [60.0, 100.0]
    mass:  [0.0, 1000.0]
    angle: [0.0, 30.0]            # magnitude; abs() applied before norm
  material_vocab: [F]
  use_material: false

model:
  # nnode_in_features must equal 20 + condition.n_cond().
  # Prefer DERIVING it in code (see 4.4) rather than hardcoding.
  nnode_in_features: 22           # 20 + 2 ; assert at startup
```

### 4.3 `src/dataset.py`

At trajectory load time (~ where each `train_dir` / `val_dir` is registered, near line 375):

```python
from src.conditions import CondConfig, parse_conditions, normalize_conditions

cond_cfg = CondConfig(**cfg.condition)   # built once, shared across trajectories
raw = parse_conditions(metadata, dir_name=os.path.basename(traj_dir))
traj_entry["cond"] = normalize_conditions(raw, cond_cfg)   # np.float32 (n_cond,)
traj_entry["cond_raw"] = raw                               # keep for logging / sanity
```

In `__getitem__`, add `cond` to the returned sample (constant for every window of the trajectory):

```python
sample["cond"] = torch.from_numpy(traj_entry["cond"])      # (n_cond,) float32
```

Collation: default stacking yields `cond` of shape `(B, n_cond)`. If a custom `collate_fn` exists, ensure it passes `cond` through with a simple `torch.stack`.

### 4.4 `train.py` forward (the broadcast + concat — the actual Method A)

Around the current `x_in` assembly (~ lines 493–501):

```python
# existing
x_in = torch.cat([x_vel_flat, x_sdf], dim=-1)        # (B, N, 20)

# Method A:
cond = batch["cond"]                                  # (B, n_cond)
cond_b = cond[:, None, :].expand(-1, x_in.shape[1], -1)   # (B, N, n_cond) — view, no copy
x_in = torch.cat([x_in, cond_b], dim=-1)              # (B, N, 20 + n_cond)  (D6: cond last)
```

Derive / assert the feature width once at setup:

```python
n_cond = cond_cfg.n_cond()
assert model_cfg.nnode_in_features == 20 + n_cond, \
    f"nnode_in_features={model_cfg.nnode_in_features} != 20 + n_cond={20 + n_cond}"
```

**K-step push-forward unroll (D3, critical):** the velocity window is rebuilt from predictions each unroll step, but `cond_b` is constant. Build `cond_b` once before the loop and re-concat it onto each step's freshly-assembled `x_in`. Do NOT recompute or vary it per step.

```python
cond_b = batch["cond"][:, None, :].expand(-1, N, -1)
for k in range(K):
    x_in_k = torch.cat([x_vel_flat_k, x_sdf, cond_b], dim=-1)   # same cond_b each step
    pred_k = model(x_in_k, ...)
    ...
```

---

## 5. Verification checklist (run in order, stop on first failure)

1. **Shape guard.** Assert `x_in.shape[-1] == model_cfg.nnode_in_features == 20 + n_cond`. Add as a runtime assert, not just a print.
2. **Zero-cond regression.** Set `condition.enabled: []` → `n_cond = 0`, `nnode_in_features: 20`. Training must be byte-for-byte equivalent to the current pipeline (no `cond` channels). Guards against accidental coupling.
3. **Single-trajectory overfit.** Re-run the existing overfit sanity with `enabled: [speed, mass]`. Adding 2 constant channels to a single trajectory must not degrade the overfit (loss curve should match baseline within noise). Confirms the wiring is inert when condition doesn't vary.
4. **Two-trajectory separation (the real test).** Same speed, mass ∈ {+0, +800} kg, two trajectories. Train short (`runtime=0.5`). Compare per-trajectory val RMSE **with vs without** cond:
   - Without cond: the two rollouts should be near-identical (mass invisible) → high error on at least one.
   - With cond: rollouts should diverge and per-trajectory RMSE should separate.
   Log both trajectories' RMSE to W&B and eyeball the rollout divergence. This is the diagnostic that justifies the whole change.
5. **Normalization sanity.** Print `cond_raw` and normalized `cond` for each trajectory at load. Confirm under the declared ranges: speed 60→−1.0, 80→0.0, 100→+1.0 (range [60,100]); mass 0→−1.0, 400→−0.2, 800→+0.6 (range [0,1000]). If angle is enabled: `abs(25.4)` under [0,30] → +0.693. (Spot-check the arithmetic.)
6. **No leakage into stats.** Confirm `global_stats.json` (velocity/accel norm) is untouched — condition normalization is fully separate and config-declared.

---

## 6. Out of scope — tracked for later

- **B / D / F injection** — consume the same `cond` vector; separate SPECs.
- **Per-node barrier material** — broadcast onto barrier nodes only, not global; do when a 2nd barrier type exists.
- **`compute_sdf_batch` hardcoded orientation (CONFIRM FIRST).** It hardcodes `barrier_angle_deg=-25.4`. If barrier orientation already varies across the current trajectories, the SDF is wrong *now* (every trajectory distanced against one fixed orientation) — highest priority, fix before trusting any current results. If orientation is genuinely fixed across all current data, it's a latent bug to fix before orientation joins the factorial. Own SPEC; the abs()/range work here does not touch the SDF.
- **(Optional) physically-derived condition.** MASH defines impact severity `IS = ½·M·V²·sin²θ`. A single derived, normalized severity channel may carry more signal per dimension than raw scalars. Worth an ablation later (A + IS vs A alone); not now (one-variable-at-a-time).

---

## 7. Assumptions to verify against the real code (DO THIS FIRST)

The author has not seen the current source. Before editing, confirm each; if any is false, adjust the edit, not the design.

- **A1.** `train.py` assembles `x_in` from `x_vel_flat` and `x_sdf` near lines 493–501, final dim 20. (Variable names may differ — match them.)
- **A2.** `metadata.json` exists per trajectory dir and contains speed/mass/angle/material under *some* keys. Capture the real key names → update `parse_conditions` candidate lists.
- **A3.** Trajectory dir naming encodes speed/mass (fallback regex). Capture the real pattern (e.g. `v80_m400_...` vs `80kmh_400kg`) → update regexes.
- **A4.** The batch dict reaches the forward as `batch[...]` and the dataset returns a dict-style sample. If it's a tuple/namedtuple, thread `cond` through accordingly.
- **A5.** The model builds its input projection from `nnode_in_features` (config-driven), so bumping the config widens the first Linear automatically with no model-class edit. If the input width is hardcoded elsewhere, update that single site too.
- **A6.** A custom `collate_fn`, if present, must pass `cond` through (`torch.stack`). Default collation handles it automatically.

---

## 8. Definition of done

- `src/conditions.py` exists; `n_cond()` and `normalize_conditions` covered by the §5.5 spot-check.
- Dataset returns `cond (B, n_cond)`; forward concatenates it last; `nnode_in_features` asserted.
- §5 checklist passes through step 4, with the two-trajectory divergence logged to W&B.
- Zero-cond regression (§5.2) confirms the path is inert when disabled.
- No change to global stats, SDF, or model class beyond the input width.
