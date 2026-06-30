# SPEC — Mesh Downsampling & Field Reconstruction Pipeline for a Transolver Crash Surrogate

**Version:** 1.0
**Target implementer:** coding agent
**Language/stack:** Python 3.10+, NumPy, SciPy (`cKDTree`, sparse), optional `lasso-python` for d3plot. No LS-DYNA software dependency.

---

## 0. Objective

Build a pipeline that:

1. **Downsamples** a fixed FE mesh (~1.77M nodes) to a **100,000-node** point set, using a configurable, region-aware sampling scheme.
2. Builds a **time-invariant sparse reconstruction matrix `W`** (≈1.77M × 100k) that maps the sampled **displacement** field back to the full-field displacement at any time step.
3. Provides a **closed-loop validation** harness (FE-truth only, no surrogate needed) that measures how well the sampled set + `W` reconstruct physically meaningful barrier responses.

The 100k set is the input/output node set for a downstream Transolver autoregressive surrogate (out of scope for this spec — but the sampling and `W` must serve it).

**Why these design choices (do not "optimize away"):**
- The mesh has **no element erosion** (confirmed): node count is constant across all time steps ⇒ the sampled set and `W` are fixed once, reused for every time step and every rollout step.
- `W` is built on the **reference (t=0) geometry** and acts on the **displacement field**, NOT absolute coordinates. This is what makes `W` time-invariant. See §6.

---

## 1. Inputs

### 1.1 k-file (primary geometry source)
- Path: `car_and_barriers.k` (~262 MB, ~3.58M lines).
- Provides: **t=0 reference coordinates** (`*NODE`) and **element connectivity** (PID → node membership).
- Total nodes: **1,776,987**.
- This file alone is sufficient to compute the sampled set and build `W`.

### 1.2 d3plot (time-series source — needed only for validation/training)
- Provides per-time-step nodal **displacement** (or absolute coordinates; derive displacement as `x(t) − X_ref`).
- ~500 output states over ~1 s physical time.
- The k-file does NOT contain any t>0 data. Treat d3plot access behind an interface (`DisplacementSource`) so the sampling/W build runs without it.
- Recommended reader: `lasso-python` (`from lasso.dyna import D3plot, ArrayType`). `qd-cae-python` is deprecated.

### 1.3 Unit system
- `[ton, mm, s]`. Length mm, mass tonne, force N, stress MPa, **acceleration mm/s²** (divide by 9806.65 for g). Relevant only if the agent computes acceleration-derived metrics.

---

## 2. ⚠️ CRITICAL PARSING GOTCHAS (read before writing the parser)

The k-file uses **MIXED formatting within the same keyword block**. Naive `str.split()` will silently corrupt barrier data.

| Section | Format | Parse rule |
|---|---|---|
| `*NODE` lines | space-delimited `nid x y z tc rc` | `split()` OK; take first 4 fields |
| `*ELEMENT_SOLID` — **vehicle** PIDs | space-delimited | `split()` OK |
| `*ELEMENT_SOLID` — **barrier** PIDs | **fixed-width 8-char, NO spaces** (digits run together, e.g. `10000001100000011004361...`) | MUST slice every 8 chars |
| `*ELEMENT_SHELL` | space-delimited `eid pid n1..n4` | `split()` OK |
| `*ELEMENT_BEAM` — barrier rebar/reinforcement | **fixed-width 8-char** (`eid pid n1 n2 n3 ...`) | slice fixed width for first fields |
| `*MAT_RIGID` data line | fixed-width, sci-notation glued to int (e.g. `10000017.89000E-9`) | slice fixed width if used |

**Robust field parser (use everywhere):**
```
def parse_fields(line, n_expected):
    parts = line.split()
    if len(parts) >= n_expected:
        return parts                      # space-delimited
    # fall back to fixed 8-char width
    return [line[i:i+8].strip() for i in range(0, len(line.rstrip()), 8)]
```
Validate by sanity-checking parsed PIDs fall in known ranges (§3.1). If a "PID" comes out as an absurd 15-digit number, the fixed-width fallback didn't trigger — fail loudly.

Also: a trailing `0` in fixed node lists is **padding**, not node id 0 — filter it out. Node ids start at 1.

---

## 3. Mesh facts (extracted, authoritative — use directly)

### 3.1 PID ranges
| Region | PID range |
|---|---|
| Ground (rigid) | 1000001 |
| **Vehicle** (Dodge Ram) | 2000000 – 8999999 |
| **Sensors / dummy** | 9000xxx |
| **Barrier** (NJ concrete) | 10000001 – 10000024 |

### 3.2 Barrier component PIDs — split by FINE vs COARSE segment
Suffix convention: `-1/-2` = fine (impact) segment; `-c1/-c2` = coarse (far) segment.

| Component | FINE-segment PIDs | COARSE-segment PIDs |
|---|---|---|
| concrete | 10000001, 10000007 | 10000013, 10000019 |
| anchor rebar (T+S) | 10000002, 10000008, 10000003, 10000009 | 10000014, 10000020, 10000015, 10000021 |
| reinforcement | 10000004, 10000010 | 10000016, 10000022 |
| steel_tube | 10000005, 10000011 | 10000017, 10000023 |
| t_lok | 10000006, 10000012 | 10000018, 10000024 |

### 3.3 Node counts per component (from k-file; `before` numbers)
| Component | fine | coarse | total |
|---|---:|---:|---:|
| concrete | 568,536 | 195,456 | 763,992 |
| steel_tube | 22,080 | 66,240 | 88,320 |
| t_lok | 13,920 | 41,760 | 55,680 |
| anchor rebar | 872 | 2,616 | 3,488 |
| reinforcement | 996 | 2,988 | 3,984 |
| **Vehicle total** | | | 853,962 |

### 3.4 Force-keep nodes (bypass sampling entirely — always retained)
| Part | PID(s) | nodes |
|---|---|---:|
| Dummy_Beams | 9000002 | 8 |
| lap_strap | 9000003 | 8 |
| shoulder_strap | 9000004 | 6 |
| accelerometers (CG / engine / L-rear / R-rear) | 9000100–9000103 | 4×8 = 32 |
| **Total force-keep** | | **54** |

---

## 4. Region partitioning (fixed; computed on t=0 geometry)

Six sampling regions. Region membership is computed **once** on reference coords and never changes.

### 4.1 Barrier-fine region — target **40,000**, sampled **per-part** (`split_by_part=True`)
Members = all FINE-segment PIDs (§3.2). Per-part allocation (final `after` targets):

| Part | before | after | rule |
|---|---:|---:|---|
| anchor rebar | 872 | 872 | full retain |
| reinforcement | 996 | 996 | full retain |
| t_lok (fine) | 13,920 | ~5,000 | FPS |
| steel_tube (fine) | 22,080 | ~4,000 | FPS |
| concrete_fine | 568,536 | ~29,132 | FPS (remainder) |

### 4.2 Barrier-coarse region — target **20,000**, sampled per-part
| Part | before | after | rule |
|---|---:|---:|---|
| anchor rebar | 2,616 | 2,616 | full retain |
| reinforcement | 2,988 | 2,988 | full retain |
| t_lok (coarse) | 41,760 | ~4,000 | FPS |
| steel_tube (coarse) | 66,240 | ~3,000 | FPS |
| concrete_coarse | 195,456 | ~7,396 | FPS (remainder) |

> NOTE: rebar/reinforcement are **different parts** in fine vs coarse — do NOT double-count. The two segments together = 3,488 rebar + 3,984 reinforcement total.

### 4.3 Vehicle regions — target **40,000 total**, banded by distance to barrier centerline
Distance metric = **perpendicular distance to the barrier centerline** in the xy-plane.

Centerline: `y = −0.474835·x + 976.536`  ⇒  standard form `a·x + b·y + c = 0` with
`a = 0.474835, b = 1.0, c = −976.536`, `denom = hypot(a,b) = 1.1069`.
```
d_centerline(x, y) = |a*x + b*y + c| / denom
```
(Validated: corr 0.991 vs true 3D nearest-distance to barrier — this analytic metric is the approved one.)

| Band | condition | before | after | rule |
|---|---|---:|---:|---|
| contact front | d < 500 mm | 16,195 | ~10,000 | FPS |
| near field | 500 ≤ d < 1000 | ~91,000 | ~18,000 | FPS |
| far field | d ≥ 1000 | ~746,000 | ~12,000 | FPS |

> Vehicle membership = PID in 2000000–8999999. Exclude the 54 force-keep nodes from band counts (negligible).

### 4.4 Budget check
40,000 (fine) + 20,000 (coarse) + 40,000 (vehicle) = **100,000**. Force-keep 54 are inside the vehicle/sensor budget; subtract from the nearest vehicle band or accept the +54 rounding.

---

## 5. Samplers (pluggable; 4 methods for comparison study)

Common interface:
```
def sample(points: np.ndarray[N,3], n: int, cfg: SamplerConfig,
           density: np.ndarray[N] | None = None) -> np.ndarray[idx]
```
Returns local indices into `points`. Must be deterministic given `cfg.seed`.

| method | algorithm | notes |
|---|---|---|
| `stride` | sort by `cfg.stride_order`, take every `step = N//n`. | `stride_order ∈ {"file","morton","x"}`. `"file"` = original node order (the **bad baseline** to expose). `"morton"` = Z-order on coords (fair equal-interval baseline). |
| `random` | `rng.choice(N, n, replace=False)` | i.i.d. uniform; will produce clusters/voids. |
| `poisson_disk` | dart-throwing on the point cloud: accept a candidate only if no already-chosen point lies within radius `r`. | `r = cfg.poisson_radius` or auto-estimate from target `n`. Point count is emergent; if `enforce_exact_n`, relax `r` and top-up (or truncate) to hit exactly `n`. |
| `fps` | farthest-point sampling: seed, then iteratively add the point maximizing min-distance to chosen set. | If `density` given and `density_weighted`, scale candidate distances by density so high-density (near-centerline) regions are preferred. Matches PointNet++ convention. |

All four must run on the **same region definitions and budgets** so the comparison is fair. Only `cfg.method` changes between runs.

**Per-part allocation** (when `split_by_part=True`): distribute the region's `n_points` across its parts by **node-count proportion**, with a **`min_per_part` floor** (small parts like rebar/reinforcement must not vanish; here they're full-retain anyway, but the floor protects the general case). Parts marked full-retain take all their nodes first; remaining budget is split proportionally among the rest.

---

## 6. Reconstruction matrix `W`  (the core deliverable)

### 6.1 Semantics — non-negotiable
- Built on **reference (t=0) coordinates** `X_ref` → **time-invariant**, built ONCE.
- Acts on the **displacement** field, not absolute coordinates:
  ```
  disp_full(t) ≈ W @ disp_sampled(t)
  x_full(t)    = X_ref + disp_full(t)
  ```
- Shape: sparse `(1,776,987 × 100,000)`. Each retained sampled node maps to itself (identity row). Each dropped node is a convex/weighted combination of nearby sampled nodes.
- Recommended: predict & reconstruct **displacement increments** Δdisp rather than absolute displacement (better conditioned, avoids identity-map degeneracy in the surrogate). `W` is identical either way.

### 6.2 Construction
For each dropped node `i` (reference coord `X_ref[i]`):
1. Find its `k` nearest **sampled** neighbors in `X_ref` (`cKDTree`, `k = cfg.knn`, default 4–8).
2. Compute weights: inverse-distance / RBF / barycentric (configurable). Normalize to sum 1.
3. Write row `i` of `W` with those `k` weights.

Sampled nodes get an identity row.

### 6.3 Segment-awareness — MANDATORY
- **Do NOT interpolate across T_lok seams.** Adjacent barrier segments have **discontinuous** relative displacement at the connection (segments slide/open). Interpolating across a seam smears a true jump and corrupts working-width estimates.
- Implement as **block-diagonal `W` per barrier segment**: a dropped node's kNN search is restricted to sampled nodes **within the same segment** (and same component family where appropriate). Provide a `segment_id` per node (derive from PID: fine vs coarse, plus per-physical-segment if available).
- Vehicle and barrier are separate blocks; never mix.

### 6.4 Rigid-segment option
- Coarse far segments and vehicle `*MAT_RIGID` parts move ~rigidly (6 DOF). For these, an optional **rigid-body-fit reconstruction** is more accurate than linear kNN: fit translation+rotation from the segment's sampled points, apply to all dropped nodes in that segment. Expose as `reconstruction_mode ∈ {"interp","rigid"}` per region. Default `"interp"`; allow `"rigid"` for coarse-far / vehicle-far.

### 6.5 No-erosion guarantee
Node count is constant ⇒ `W` never needs rebuilding mid-trajectory. (If erosion were present, `W` would break in eroded zones — not our case, but assert node-count constancy if d3plot is read.)

---

## 7. Closed-loop validation harness (FE-truth only)

Purpose: quantify information lost by `sampling + W`, **before** any surrogate training. No surrogate involved.

Pipeline per validation trajectory:
```
for t in states:
    disp_true_full      = DisplacementSource[t]            # 1.77M
    disp_sampled        = disp_true_full[sampled_idx]      # 100k
    disp_recon_full     = W @ disp_sampled                 # 1.77M
    error_field(t)      = disp_recon_full − disp_true_full
    metrics(t)          = physical_metrics(disp_recon_full, disp_true_full)
```

**Physical metrics to compute (these matter, not just L2):**
- **Dynamic deflection** `D_m`: max displacement of the barrier impact face. (Primary.)
- **Working width** `W_m`: max lateral extent of barrier (or vehicle) from undeformed face.
- **Permanent set**: residual barrier displacement at end of trajectory.
- **Region-weighted L2**: report L2 separately for impact zone vs rigid zone (do NOT report a single global L2 — rigid regions dilute it).
- **RMSE in mm** for the above (physical units, interpretable) in addition to relative L2.

Acceptance target: reconstruction error on `D_m` / working width within a small tolerance (e.g. <5%) of full-res FE. The agent should expose these as a report; the human tunes sampling params (densities, `knn`, `density_d0`) to minimize them.

> This harness is also the objective for the sampling-method comparison (§5): run each of stride/random/poisson/fps through it and compare reconstruction error. Expected ordering: file-stride worst → random → poisson/fps best.

---

## 8. Config schema

```yaml
global:
  total_budget: 100000
  centerline: [0.474835, 1.0, -976.536]   # a, b, c
  knn: 6
  weight_mode: inverse_distance           # inverse_distance | rbf | barycentric
  force_keep_pids: [9000002, 9000003, 9000004, 9000100, 9000101, 9000102, 9000103]
  tlok_seam_protect: true

regions:
  - name: barrier_fine
    members: fine_segment            # resolves to §3.2 fine PIDs
    split_by_part: true
    min_per_part: 50
    full_retain_parts: [rebar, reinforcement]
    reconstruction_mode: interp
    sampler: {method: fps, n_points: 40000, seed: 42, density_weighted: false}

  - name: barrier_coarse
    members: coarse_segment
    split_by_part: true
    min_per_part: 50
    full_retain_parts: [rebar, reinforcement]
    reconstruction_mode: interp       # or rigid for far segments
    sampler: {method: fps, n_points: 20000, seed: 42}

  - name: veh_contact
    members: vehicle
    band: {metric: centerline, lo: 0, hi: 500}
    sampler: {method: fps, n_points: 10000, seed: 42}

  - name: veh_near
    members: vehicle
    band: {metric: centerline, lo: 500, hi: 1000}
    sampler: {method: fps, n_points: 18000, seed: 42}

  - name: veh_far
    members: vehicle
    band: {metric: centerline, lo: 1000, hi: .inf}
    reconstruction_mode: interp       # candidate for rigid
    sampler: {method: fps, n_points: 12000, seed: 42}

sampler_defaults:
  stride_order: morton                # file | morton | x
  poisson_radius: null                # auto from n_points
  enforce_exact_n: true
  density_d0: 600.0                   # mm, density-decay length for density_weighted fps
```

Switching the whole comparison study = override `sampler.method` for all regions.

---

## 9. Outputs

1. `sampled_node_ids.npy` — int array, exactly ~100,000 ids (incl. the 54 force-keep), sorted, deduplicated.
2. `W.npz` — SciPy sparse CSR, shape `(1776987, 100000)`, column order aligned to `sampled_node_ids`.
3. `node_ref_coords.npy` — `X_ref` for all nodes (so `x_full = X_ref + W@disp`).
4. `region_assignment.npy` / sidecar — per-sampled-node region + segment id (for debugging & for the surrogate's positional features).
5. `validation_report.json/md` — per-metric reconstruction errors (§7), per sampling method if comparison run.

---

## 10. Acceptance criteria

- [ ] Parser recovers **exactly 1,776,987** nodes and the §3.3 per-component counts (within ±0 for rebar/reinforcement, exact for concrete). If barrier components come back as 0, the fixed-width fallback failed — must fix.
- [ ] Sampled set size = total_budget ± (force-keep rounding); all 54 force-keep nodes present.
- [ ] `W` rows each sum to 1.0 (±1e-6); sampled-node rows are identity; **no row mixes across a T_lok seam or across barrier/vehicle blocks**.
- [ ] `W` reproduces a **rigid-body translation** exactly (feed uniform displacement → reconstruct → zero error) — basic correctness test.
- [ ] Closed-loop `D_m` / working-width reconstruction error reported; tunable below target tolerance.
- [ ] Runs sampling + `W` build from the **k-file alone** (no d3plot). d3plot only required for §7 validation, behind `DisplacementSource`.

---

## 11. Implementation order (suggested)

1. Mesh parser (§2–3) + acceptance count check.
2. Region partition + centerline banding (§4).
3. Samplers (§5), start with `fps`; add others for the study.
4. Sampling orchestration: per-part allocation, force-keep, budget reconcile.
5. `W` builder (§6): segment-aware kNN weights; rigid-body unit test.
6. `DisplacementSource` (lasso-python d3plot reader) + closed-loop validation (§7).
7. Sampling-method comparison driver + report.

---

## Appendix A — key constants
```
TOTAL_NODES        = 1_776_987
CENTERLINE         = (0.474835, 1.0, -976.536)   # a,b,c ; d = |a x + b y + c| / 1.1069
G_IN_MM_S2         = 9806.65
FORCE_KEEP_PIDS    = [9000002, 9000003, 9000004, 9000100, 9000101, 9000102, 9000103]
BARRIER_PID_MIN    = 10_000_000
VEHICLE_PID_RANGE  = (2_000_000, 8_999_999)
FINE_CONCRETE_PIDS   = [10000001, 10000007]
COARSE_CONCRETE_PIDS = [10000013, 10000019]
# (full per-component PID lists in §3.2)
```
