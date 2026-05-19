"""
Plot acceleration vs timestep for selected nodes from an HDF5 simulation file.

Usage examples:
  # Specify node indices explicitly (0-indexed within the file's node array)
  python tools/plot_acc_nodes.py path/to/output.h5 --nodes 0 5 10 100 200

  # Use frontface strip (predefined)
  python tools/plot_acc_nodes.py path/to/output.h5 --strip frontface

  # Use barrier strip
  python tools/plot_acc_nodes.py path/to/output.h5 --strip barrier

  # Specify by global node IDs (as stored in metadata/node_global_idx)
  python tools/plot_acc_nodes.py path/to/output.h5 --global-ids 5701 5751 5801

  # Plot a specific component instead of magnitude (x, y, z, or mag)
  python tools/plot_acc_nodes.py path/to/output.h5 --nodes 0 1 2 --component x
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py


MAX_NODES = 12


def load_h5(h5_path: str):
    f = h5py.File(h5_path, "r")
    times = f["states/times"][:]                    # (T,)
    acc   = f["states/acceleration"][:]             # (T, N, 3)
    node_global_idx = f["metadata/node_global_idx"][:]  # (N,)
    frontface_idx   = f["metadata/frontface_idx"][:]    # node-array indices for front face
    barrier_idx     = f["metadata/barrier_idx"][:]      # node-array indices for barrier
    f.close()
    return times, acc, node_global_idx, frontface_idx, barrier_idx


def resolve_nodes(args, node_global_idx, frontface_idx, barrier_idx):
    """Return array-local node indices (≤ MAX_NODES)."""
    N = len(node_global_idx)

    if args.strip == "frontface":
        indices = frontface_idx
        label = "frontface strip"
    elif args.strip == "barrier":
        indices = barrier_idx
        label = "barrier strip"
    elif args.global_ids:
        gid_map = {gid: i for i, gid in enumerate(node_global_idx)}
        indices = []
        for gid in args.global_ids:
            if gid not in gid_map:
                print(f"WARNING: global id {gid} not found in file, skipping.")
            else:
                indices.append(gid_map[gid])
        indices = np.array(indices)
        label = f"global ids {args.global_ids}"
    else:
        indices = np.array(args.nodes)
        bad = indices[(indices < 0) | (indices >= N)]
        if len(bad):
            sys.exit(f"ERROR: node indices out of range [0, {N-1}]: {bad.tolist()}")
        label = f"nodes {args.nodes}"

    if len(indices) > MAX_NODES:
        print(f"WARNING: {len(indices)} nodes selected; truncating to first {MAX_NODES}.")
        indices = indices[:MAX_NODES]

    return indices, label


def compute_values(acc, node_indices, component):
    """Extract per-node values over time. Shape: (T, n_nodes)"""
    comp_map = {"x": 0, "y": 1, "z": 2}
    if component == "mag":
        vals = np.linalg.norm(acc[:, node_indices, :], axis=-1)  # (T, n)
        ylabel = "Acceleration magnitude (mm/s²)"
    else:
        c = comp_map[component]
        vals = acc[:, node_indices, c]                            # (T, n)
        ylabel = f"Acceleration {component.upper()} (mm/s²)"
    return vals, ylabel


def plot(times, vals, node_indices, node_global_idx, label, ylabel, component, out_path):
    n = vals.shape[1]
    colors = cm.tab20(np.linspace(0, 1, max(n, 1)))

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, ni in enumerate(node_indices):
        gid = node_global_idx[ni]
        ax.plot(times, vals[:, i], color=colors[i], linewidth=1.2,
                label=f"node[{ni}] gid={gid}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    comp_str = f" [{component}]" if component != "mag" else " [magnitude]"
    ax.set_title(f"Acceleration{comp_str} over time  —  {label}\n{n} node(s) | {len(times)} timesteps")
    ax.legend(fontsize=8, ncol=max(1, n // 6 + 1), loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot acceleration vs timestep for selected nodes.")
    parser.add_argument("h5_path", help="Path to the HDF5 simulation file (output.h5)")

    node_group = parser.add_mutually_exclusive_group(required=True)
    node_group.add_argument("--nodes", type=int, nargs="+",
                            metavar="IDX", help=f"Array-local node indices (up to {MAX_NODES})")
    node_group.add_argument("--global-ids", type=int, nargs="+",
                            metavar="GID", help="Global node IDs (from metadata/node_global_idx)")
    node_group.add_argument("--strip", choices=["frontface", "barrier"],
                            help="Use a predefined node strip")

    parser.add_argument("--component", choices=["mag", "x", "y", "z"], default="mag",
                        help="Which acceleration component to plot (default: mag)")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Save figure to this path instead of showing interactively")

    args = parser.parse_args()

    print(f"Loading {args.h5_path} ...")
    times, acc, node_global_idx, frontface_idx, barrier_idx = load_h5(args.h5_path)
    print(f"  timesteps={len(times)}, nodes={acc.shape[1]}")

    node_indices, label = resolve_nodes(args, node_global_idx, frontface_idx, barrier_idx)
    print(f"  plotting {len(node_indices)} node(s): {node_indices.tolist()}")

    vals, ylabel = compute_values(acc, node_indices, args.component)
    plot(times, vals, node_indices, node_global_idx, label, ylabel, args.component, args.out)


if __name__ == "__main__":
    main()
