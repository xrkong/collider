# Refactor Prompt: `d3plot_to_h5_dt.py` — Replace Node Sampling with YAML Config

## Task

Refactor `d3plot_to_h5_dt.py` to replace the current node sampling logic with a new YAML-based config system that applies **different sampling strategies** to different part groups.

---

## Context

### Current behavior (to be replaced)

- CLI arg `--required-config` points to a flat `.config` text file (one part-name pattern per line, comments with `#`)
- CLI arg `--node-stride N` applies **uniform** stride decimation across ALL selected nodes (Step 1b, lines ~517–526)
- `_load_patterns()` reads the flat config file

### New behavior

Node selection becomes **two-layer with different sampling strategies**:

**Layer 1 — collision_zone**: high-priority parts. All nodes kept unconditionally. No stride.

**Layer 2 — required_parts**: remaining structural parts. Uniform stride applied (`--node-stride` still exists but only applies to this layer).

The union of Layer 1 (all nodes) + Layer 2 (strided nodes), deduplicated, becomes `sel_node_idx`.

---

## New YAML Config Structure

```yaml
# Layer 1: collision zone — ALL nodes kept, no stride
collision_zone:
  car_contact_parts:
    - frontface
    - bumper
    - bumpersteel
    - bumpercover
    - bumperhousing
    - railfront
    - noserail
    - frontrail

  barrier_parts:
    - concrete_fine_mesh
    - anchor rebar_t-1
    - anchor rebar_s-1
    - reinforcement-1
    - steel tube-1
    - t lok-1
    - anchor rebar_t-2
    - anchor rebar_s-2
    - reinforcement-2
    - steel tube-2
    - t lok-2

# Layer 2: required parts — stride sampled
required_parts:
  front_crash_structure:
    - frontface
    - bumper
    - bumpersteel
    - bumpercover
    - bumperhousing
    - hood
    - hoodinner
    - fender
    - windshield
    - railfront
    - noserail
    - frontrail
    - railfrontplate
    - railfrontplaterear
    - raillargefront
    - raillargeinnerfront
    - raillargefrontouter
    - railmid
    - xmemberfront
    - xmembermidfront
    - xmemberrearmiddlebottom
    - framexmemberplates
    - radiator
    - radiatorframe
    - radiatorframebottom
    - radiatorframebrkttop
    - radiatorside
    - radiatorsolid
    - radiatornullshell
    - condensor

  occupant_cage:
    - firewall
    - firewalltray
    - firewallsupport
    - ipbeamfirewallbrkt
    - floor
    - floorfront
    - rearfloor
    - floorxmember
    - floorlongitudesupport
    - floorsupport
    - floorbottomplate
    - rocker
    - rockerinner
    - rockerfront
    - rockerinternal
    - rockerfrontinnersuppport
    - apillar
    - bpillar
    - cpillar
    - pillar
    - cabrail
    - cabrailupper
    - cabraillower
    - cabraillatitudebar
    - cabrailconnection
    - cabrailpanel
    - cabrailframeconnector
    - roof
    - roofrail
    - roofrailfront
    - roofrailrear
    - roofrailxmember
    - sidepanel
    - backwall
    - cabpanel

  doors:
    - doorfront
    - doorrear
    - doorouter
    - doorinner
    - frontdoorinner
    - doorfrontbar
    - doorrearlowerbar
    - doorrearuppersupport
    - doorfrtlongitudesuprt
    - doorfrontlonguppersprt
    - doorrearlatitudesuprt
    - windowframe
    - windowguide
    - lockplate
    - hinge

  interior:
    - ipbeam
    - dash
    - dashcover
    - dashfan
    - dashscreen
    - dashcenter
    - dashbottom
    - dashcompartment
    - glovecompartment
    - airbag
    - airbagbrkt
    - airbagbkt
    - airbagcover
    - steering
    - steeringwheel
    - steeringcolmn
    - steeringcolumn
    - steeringrack

  seats:
    - seat
    - seatdriver
    - seadriver
    - backseat
    - seatback
    - seatfoam
    - foam
    - driverseat
    - passengerseat
    - rearseat
    - headrest
    - seatbelt
    - seatdriverouterrail
    - seatdriverinnerrail

  dummy_and_sensors:
    - dummy
    - dummy_beams
    - lap_strap
    - shoulder_strap
    - accelerometer
    - accelerometers

  powertrain:
    - engine
    - engineoilpan
    - enginemount
    - enginemountrubber
    - transmission
    - transmissionoilpan
    - transmissionmount
    - battery
    - batterymount
    - fusebox
    - brake
    - brakebooster
    - brakeboostermnt
    - brakefluidcontainer

  suspension:
    - aarm
    - upperarm
    - lowerarm
    - spindle
    - diskfront
    - upright
    - tire
    - rim
    - swaybar
    - shockhousingfront
    - shocksupport
    - suspensionbracketinner
```

The YAML file path replaces `--required-config`. The CLI arg should be renamed to `--sampling-config`.

---

## Specific Code Changes Required

### 1. Replace `_load_patterns()` with a YAML loader

```python
def _load_sampling_config(path: Path) -> tuple[list[str], list[str]]:
    """
    Returns:
        collision_patterns : flat list of all patterns from collision_zone.*
        required_patterns  : flat list of all patterns from required_parts.*
    """
```

Flatten all nested lists under `collision_zone` (both `car_contact_parts` and
`barrier_parts`) into one list. Flatten all nested lists under `required_parts`
into another list.

---

### 2. Rewrite Step 1b — two-layer node selection

Replace the current uniform stride block (lines ~517–526) with the following logic:

```python
# Layer 1: collision_zone — keep ALL nodes, no stride
collision_mask = _build_selected_part_mask(part_names, collision_patterns)
# (use existing _select_elements_and_nodes with collision_mask to get node indices)
collision_node_set = set(collision_node_idx.tolist())

# Layer 2: required_parts — stride sampled, then exclude collision nodes
required_mask = _build_selected_part_mask(part_names, required_patterns)
# (use existing _select_elements_and_nodes with required_mask to get node indices)
required_strided = required_node_idx[::args.node_stride]
required_new = required_strided[~np.isin(required_strided, list(collision_node_set))]

# Union: collision (all) + required (strided, deduplicated)
sel_node_idx = np.union1d(collision_node_idx, required_new)
```

Log the breakdown:
```
Collision-zone nodes (all)     : XXXX
Required-parts nodes (strided) : XXXX  (stride=N)
  of which new (not in collision zone): XXXX
Total nodes                    : XXXX
```

---

### 3. Update HDF5 metadata

- Keep `mg.attrs["node_stride"]` — it still applies to Layer 2.
- Add `mg.attrs["node_selection"] = "collision_zone:all + required_parts:strided"`.
- Keep existing `barrier_idx` / `frontface_idx` datasets unchanged — they are
  still needed downstream. Update the source patterns to come from
  `collision_zone.barrier_parts` and `collision_zone.car_contact_parts` respectively.
- In `metadata["config"]`, replace `required_config` with `sampling_config`,
  and add `collision_zone_n_nodes` and `required_parts_n_nodes` for diagnostics.

---

### 4. Update CLI args

| Action | Arg |
|--------|-----|
| Remove | `--required-config` |
| Rename | `--required-config` → `--sampling-config` |
| Keep   | `--node-stride` (now applies to Layer 2 only) |
| Keep   | `--src`, `--tmp`, `--out`, `--frame-stride`, `--frame-limit`, `--frame-scale` |

---

### 5. Update module docstring

Update the Pipeline section:

```
Step 1a – Part filtering (two-layer)
    Layer 1 (collision_zone):  all nodes from car_contact_parts and barrier_parts
                               are kept unconditionally.
    Layer 2 (required_parts):  nodes from all other structural parts are kept
                               every --node-stride-th entry.
    The union of both layers (deduplicated) forms the final node set.

Step 1b – (removed; stride now applied per-layer inside Step 1a)
```

Update the example CLI invocation to use `--sampling-config`.

---

## Dependency

Add `import yaml` at the top. `pyyaml` is already available in the conda environment.

---

## Do NOT Change

- `_select_elements_and_nodes()`
- `_build_selected_part_mask()`
- `_compute_node_stress()`
- `_diff_velocity_acceleration()`
- `_scan_state_times()`, `_select_frames()`
- `_create_h5_datasets()`
- `FieldStats`
- All HDF5 field extraction logic (Pass 2/2)
- Finite-difference kinematics
- All stress / velocity / acceleration computation
