"""
Mesh constants from SPEC_sampling_reconstruction.md §3 and Appendix A.

All PID values come directly from the spec's part-ID tables; nothing here is
inferred from data, so treat this file as the single source of truth for
"which PID belongs to which physical part".
"""

from __future__ import annotations

TOTAL_NODES = 1_776_987

# Barrier centerline (xy-plane), standard form a*x + b*y + c = 0
CENTERLINE = (0.474835, 1.0, -976.536)   # a, b, c
CENTERLINE_DENOM = 1.1069                # hypot(a, b)

G_IN_MM_S2 = 9806.65                     # 1 g in mm/s^2, for unit conversions

# §3.4 force-keep nodes — bypass sampling, always retained
FORCE_KEEP_PIDS = [9000002, 9000003, 9000004, 9000100, 9000101, 9000102, 9000103]

# §3.1 PID ranges
BARRIER_PID_MIN = 10_000_000
VEHICLE_PID_RANGE = (2_000_000, 8_999_999)

# §3.2 barrier component PIDs, split fine (impact) vs coarse (far) segment
FINE_CONCRETE_PIDS   = [10000001, 10000007]
COARSE_CONCRETE_PIDS = [10000013, 10000019]
FINE_REBAR_PIDS       = [10000002, 10000008, 10000003, 10000009]
COARSE_REBAR_PIDS     = [10000014, 10000020, 10000015, 10000021]
FINE_REINF_PIDS       = [10000004, 10000010]
COARSE_REINF_PIDS     = [10000016, 10000022]
FINE_STEEL_TUBE_PIDS   = [10000005, 10000011]
COARSE_STEEL_TUBE_PIDS = [10000017, 10000023]
FINE_TLOK_PIDS   = [10000006, 10000012]
COARSE_TLOK_PIDS = [10000018, 10000024]

# New_Road_Barrier design (car_and_new_barrier.k) — a different physical
# barrier from the T-lok/F-shape one above, with its own PID scheme
# (10100xxx / 10200xxx / 10300xxx, instead of T-lok's flat 10000001-24).
# Its 10000001-10000005 (locking bar, reinforcements, rings,
# Concrete_fine_mesh, locking_plate) already fall inside FINE_PIDS above —
# coincidence of numeric range, not a real correspondence to the T-lok parts
# those PIDs name. These are the PIDs that don't:
NEW_BARRIER_COARSE_CONCRETE_PIDS = [10100015]   # "Concrete_coarse_mesh"
# No fine/coarse naming exists for these (sandwich-panel skins/cores,
# wave-beam/Z-column hardware) — bucketed fine rather than silently dropped
# from every region mask (was landing in none, and yielding 0-node regions
# downstream).
NEW_BARRIER_FINE_MISC_PIDS = [
    10200016, 10200017, 10200018, 10200019, 10200020,   # panel skins/cores
    10200021, 10200022, 10200023, 10200024,
    10300034, 10300035, 10300036,                       # wave beam / Z column
]

FINE_PIDS = set(
    FINE_CONCRETE_PIDS + FINE_REBAR_PIDS + FINE_REINF_PIDS +
    FINE_STEEL_TUBE_PIDS + FINE_TLOK_PIDS + NEW_BARRIER_FINE_MISC_PIDS
)
COARSE_PIDS = set(
    COARSE_CONCRETE_PIDS + COARSE_REBAR_PIDS + COARSE_REINF_PIDS +
    COARSE_STEEL_TUBE_PIDS + COARSE_TLOK_PIDS + NEW_BARRIER_COARSE_CONCRETE_PIDS
)

# PID → part-family label, used for §6.3 seam-aware grouping and per-part
# sampling allocation (§4.1 / §4.2)
PID_TO_PART_FAMILY: dict[int, str] = {}
for _pid in FINE_CONCRETE_PIDS:    PID_TO_PART_FAMILY[_pid] = "fine_concrete"
for _pid in FINE_REBAR_PIDS:       PID_TO_PART_FAMILY[_pid] = "fine_rebar"
for _pid in FINE_REINF_PIDS:       PID_TO_PART_FAMILY[_pid] = "fine_reinf"
for _pid in FINE_STEEL_TUBE_PIDS:  PID_TO_PART_FAMILY[_pid] = "fine_steel_tube"
for _pid in FINE_TLOK_PIDS:        PID_TO_PART_FAMILY[_pid] = "fine_tlok"
for _pid in COARSE_CONCRETE_PIDS:    PID_TO_PART_FAMILY[_pid] = "coarse_concrete"
for _pid in COARSE_REBAR_PIDS:       PID_TO_PART_FAMILY[_pid] = "coarse_rebar"
for _pid in COARSE_REINF_PIDS:       PID_TO_PART_FAMILY[_pid] = "coarse_reinf"
for _pid in COARSE_STEEL_TUBE_PIDS:  PID_TO_PART_FAMILY[_pid] = "coarse_steel_tube"
for _pid in COARSE_TLOK_PIDS:        PID_TO_PART_FAMILY[_pid] = "coarse_tlok"
for _pid in NEW_BARRIER_COARSE_CONCRETE_PIDS: PID_TO_PART_FAMILY[_pid] = "coarse_concrete"
for _pid in NEW_BARRIER_FINE_MISC_PIDS:       PID_TO_PART_FAMILY[_pid] = "fine_new_barrier_misc"
del _pid

# Part families that are always fully retained (never sub-sampled), §4.1/§4.2
FULL_RETAIN_FAMILIES = {"fine_rebar", "fine_reinf", "coarse_rebar", "coarse_reinf"}

# Region label → compact integer id, stored in the output HDF5 as "region_id"
REGION_ID_MAP: dict[str, int] = {
    "force_keep":     0,
    "barrier_fine":   1,
    "barrier_coarse": 2,
    "veh_contact":    3,
    "veh_near":       4,
    "veh_far":        5,
}
REGION_ID_LEGEND = (
    "0=force_keep 1=barrier_fine 2=barrier_coarse "
    "3=veh_contact 4=veh_near 5=veh_far"
)
