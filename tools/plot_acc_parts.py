"""
Plot mean acceleration (X / Y / Z) vs timestep for each of the 10 part categories
defined in configs/data/required_parts.config.

Output: 10 figures × 3 subplots (x/y/z) = 30 axes total.
Each subplot shows mean ± std across all nodes in that category.

Usage:
  python3 tools/plot_acc_parts.py path/to/output.h5
  python3 tools/plot_acc_parts.py path/to/output.h5 --out-dir results/acc_parts/
  python3 tools/plot_acc_parts.py path/to/output.h5 --out-dir results/acc_parts/ --fmt pdf
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py

# ── 10 part categories (name → list of substring patterns) ──────────────────
CATEGORIES = {
    "1_Barrier": [
        "concrete_fine_mesh",
    ],
    "2_FrontCrash": [
        "frontface", "bumper", "bumpersteel", "bumpercover", "bumperhousing",
        "hood", "hoodinner", "fender", "windshield",
        "railfront", "noserail", "frontrail", "railfrontplate",
        "railfrontplaterear", "raillargefront", "raillargeinnerfront",
        "raillargefrontouter", "railmid",
        "xmemberfront", "xmembermidfront", "xmemberrearmiddlebottom",
        "framexmemberplates",
        "radiator", "radiatorframe", "radiatorframebottom",
        "radiatorframebrkttop", "radiatorside", "radiatorsolid",
        "radiatornullshell", "condensor",
    ],
    "3_OccupantCage": [
        "firewall", "firewalltray", "firewallsupport", "ipbeamfirewallbrkt",
        "floor", "floorfront", "rearfloor", "floorxmember",
        "floorsupport", "floorbottomplate",
        "rocker", "rockerinner", "rockerfront", "rockerinternal",
        "rockerfrontinnersuppport",
        "apillar", "bpillar", "cpillar", "pillar",
        "cabrail", "cabrailupper", "cabraillower", "cabraillatitudebar",
        "cabrailconnection", "cabrailpanel", "cabrailframeconnector",
        "roof", "roofrail", "roofrailfront", "roofrailrear", "roofrailxmember",
        "sidepanel", "backwall", "cabpanel",
    ],
    "4_Doors": [
        "doorfront", "doorrear", "doorouter", "doorinner", "frontdoorinner",
        "doorfrontbar", "doorrearlowerbar", "doorrearuppersupport",
        "doorfrtlongitudesuprt", "doorfrontlonguppersprt",
        "doorrearlatitudesuprt",
        "windowframe", "windowguide", "lockplate", "hinge",
    ],
    "5_Interior": [
        "ipbeam", "dash", "dashcover", "dashfan", "dashscreen", "dashcenter",
        "dashbottom", "dashcompartment", "glovecompartment",
        "airbag", "airbagbrkt", "airbagbkt", "airbagcover",
        "steering", "steeringwheel", "steeringcolmn", "steeringcolumn",
        "steeringrack",
    ],
    "6_Seats": [
        "seat", "seatdriver", "seadriver", "backseat", "seatback",
        "seatfoam", "foam", "driverseat", "passengerseat", "rearseat",
        "headrest", "seatbelt", "seatdriverouterrail", "seatdriverinnerrail",
    ],
    "7_Dummy": [
        "dummy", "dummy_beams", "lap_strap", "shoulder_strap",
        "accelerometer", "accelerometers",
    ],
    "8_Powertrain": [
        "engine", "engineoilpan", "enginemount", "enginemountrubber",
        "transmission", "transmissionoilpan", "transmissionmount",
        "battery", "batterymount", "fusebox",
        "brake", "brakebooster", "brakeboostermnt", "brakefluidcontainer",
    ],
    "9_Suspension": [
        "aarm", "upperarm", "lowerarm", "spindle", "diskfront", "upright",
        "tire", "rim", "swaybar", "shockhousingfront", "shocksupport",
        "suspensionbracketinner",
    ],
    # floorlongitudesupport is in the config but not in a named category above;
    # add to OccupantCage implicitly via "floor" pattern — already covered.
}

COMPONENT_LABELS = ["X", "Y", "Z"]
COLORS = ["#e05c5c", "#4a90d9", "#4caf7d"]   # red / blue / green per component


def load_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        times         = f["states/times"][:]           # (T,)
        acc           = f["states/acceleration"][:]    # (T, N, 3)
        node_names    = np.array([n.decode().lower()
                                  for n in f["metadata/node_part_name"][:]])
    return times, acc, node_names


def assign_nodes(node_names):
    """Return dict: category_key → boolean mask over nodes."""
    masks = {}
    for cat_key, patterns in CATEGORIES.items():
        mask = np.zeros(len(node_names), dtype=bool)
        for p in patterns:
            mask |= np.char.find(node_names, p) >= 0
        masks[cat_key] = mask
        n = mask.sum()
        print(f"  {cat_key:20s}: {n:5d} nodes")
    return masks


def plot_category(cat_key, mask, times, acc, out_dir, fmt):
    """One figure with 3 subplots (X / Y / Z) for a single category."""
    nodes = acc[:, mask, :]          # (T, n_nodes, 3)
    n_nodes = nodes.shape[1]

    mean = nodes.mean(axis=1)        # (T, 3)
    std  = nodes.std(axis=1)         # (T, 3)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"Category: {cat_key}  ({n_nodes} nodes)", fontsize=13, fontweight="bold")

    for ci, (ax, label, color) in enumerate(zip(axes, COMPONENT_LABELS, COLORS)):
        ax.plot(times, mean[:, ci], color=color, linewidth=1.2, label="mean")
        ax.fill_between(times,
                        mean[:, ci] - std[:, ci],
                        mean[:, ci] + std[:, ci],
                        color=color, alpha=0.20, label="±1 std")
        ax.set_ylabel(f"Acc {label} (mm/s²)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 4))

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{cat_key}.{fmt}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  → saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean acc X/Y/Z vs timestep for each of the 10 part categories.")
    parser.add_argument("h5_path", help="Path to output.h5")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help="Directory to save figures (default: show interactively)")
    parser.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"],
                        help="Output format (default: png)")
    args = parser.parse_args()

    print(f"Loading {args.h5_path} ...")
    times, acc, node_names = load_h5(args.h5_path)
    print(f"  timesteps={len(times)}, nodes={acc.shape[1]}")

    print("\nAssigning nodes to categories:")
    masks = assign_nodes(node_names)

    print("\nPlotting...")
    for cat_key in CATEGORIES:
        if masks[cat_key].sum() == 0:
            print(f"  WARNING: no nodes matched for {cat_key}, skipping.")
            continue
        plot_category(cat_key, masks[cat_key], times, acc, args.out_dir, args.fmt)

    print("\nDone.")


if __name__ == "__main__":
    main()
