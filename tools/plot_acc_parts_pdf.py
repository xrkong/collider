"""
Plot acceleration magnitude per-part vs timestep for each of the 10 part categories.
All parts shown as thin semi-transparent lines; top-5 by peak magnitude highlighted.
Output: single multi-page PDF.

Usage:
  python3 tools/plot_acc_parts_pdf.py path/to/output.h5
  python3 tools/plot_acc_parts_pdf.py path/to/output.h5 --out acc_parts.pdf
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import h5py

TOP_N = 5

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
}


def load_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        times      = f["states/times"][:]           # (T,)
        acc        = f["states/acceleration"][:]    # (T, N, 3)
        node_names = np.array([n.decode().lower()
                                for n in f["metadata/node_part_name"][:]])
    return times, acc, node_names


def compute_part_curves(cat_key, patterns, times, acc, node_names):
    """Returns dict: clean_part_name -> acc_magnitude curve (T,)."""
    cat_mask = np.zeros(len(node_names), dtype=bool)
    for p in patterns:
        cat_mask |= np.char.find(node_names, p) >= 0

    unique_parts = np.unique(node_names[cat_mask])
    curves = {}
    for part in unique_parts:
        idx = np.where(node_names == part)[0]
        mag = np.linalg.norm(acc[:, idx, :], axis=-1)  # (T, n_nodes)
        curves[part] = mag.mean(axis=1)                 # (T,)
    return curves


def strip_prefix(name):
    """Remove leading numeric/short prefix: '715_ob_doorfrontbar' -> 'doorfrontbar'."""
    parts = name.split("_")
    out = []
    for p in parts:
        if p.isdigit() or p in ("ob", "bw", "int", "sp", "ss", "fr", "rr"):
            continue
        out.append(p)
    return "_".join(out) if out else name


def plot_category_page(cat_key, curves, times, fig):
    ax = fig.add_subplot(1, 1, 1)

    n_parts = len(curves)
    cmap = cm.get_cmap("tab20" if n_parts <= 20 else "turbo", n_parts)

    part_names = list(curves.keys())
    peak_vals  = np.array([c.max() for c in curves.values()])
    order      = np.argsort(peak_vals)           # ascending — background first

    top_idx = set(order[-TOP_N:])

    # draw background lines first, then top lines on top
    for rank, i in enumerate(order):
        name  = part_names[i]
        curve = curves[name]
        color = cmap(rank / max(n_parts - 1, 1))
        is_top = i in top_idx
        ax.plot(times, curve,
                color=color,
                linewidth=1.6 if is_top else 0.6,
                alpha=0.9  if is_top else 0.25,
                zorder=2   if is_top else 1)

    # annotate top-N with name at their peak
    legend_handles = []
    for rank_within_top, i in enumerate(order[-TOP_N:]):
        name  = part_names[i]
        curve = curves[name]
        color = cmap(list(order).index(i) / max(n_parts - 1, 1))
        label = strip_prefix(name)
        t_peak = times[np.argmax(curve)]
        ax.annotate(label,
                    xy=(t_peak, curve.max()),
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=6, color=color, fontweight="bold",
                    ha="center", clip_on=True)
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.8, label=label))

    ax.legend(handles=legend_handles, title=f"Top-{TOP_N} by peak",
              fontsize=7, title_fontsize=7,
              loc="upper right", framealpha=0.85)

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Acc magnitude (mm/s²)", fontsize=9)
    ax.set_title(
        f"{cat_key}  —  {n_parts} parts  |  per-part mean magnitude",
        fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(times[0], times[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Per-part acc magnitude PDF for all 10 categories.")
    parser.add_argument("h5_path", help="Path to output.h5")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output PDF path (default: <h5_dir>/acc_parts.pdf)")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(args.h5_path)),
                                "acc_parts.pdf")

    print(f"Loading {args.h5_path} ...")
    times, acc, node_names = load_h5(args.h5_path)
    print(f"  timesteps={len(times)}, nodes={acc.shape[1]}")

    with PdfPages(args.out) as pdf:
        for cat_key, patterns in CATEGORIES.items():
            curves = compute_part_curves(cat_key, patterns, times, acc, node_names)
            if not curves:
                print(f"  SKIP {cat_key}: no nodes matched.")
                continue

            print(f"  {cat_key}: {len(curves)} parts")
            fig = plt.figure(figsize=(14, 5))
            plot_category_page(cat_key, curves, times, fig)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
