"""
plot_os_ar_rmse.py – Plot one-step vs autoregressive veh_contact pos RMSE
from existing rollout .pkl files, on a dual-axis plot (no model/checkpoint
needed — just the pkl files src/rollout.py already saved).

Left y-axis = one-step (OS) scale, right y-axis = autoregressive (AR) scale.
They're on separate axes because OS error (reset to ground truth every
frame) is always far smaller than AR error (accumulates drift) — sharing one
axis would flatten the OS curve to near-zero.

Uses rmse_pos_veh_contact when present in the pkl (the sensitive metric —
the all-node average is diluted by far-field nodes that barely move);
falls back to all-node rmse_pos for older pkls saved before that field
existed.

Usage:
  python3 tools/plot_os_ar_rmse.py \
      --os outputs/rollouts/lc001/onestep.pkl \
      --ar outputs/rollouts/lc001/autoregressive.pkl \
      --out outputs/rollouts/lc001/os_ar_veh_contact_rmse.png
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pickle

from src.rollout import plot_os_ar_veh_contact_rmse


def _load(path: str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"pkl not found: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--os", dest="os_pkl", default=None, help="Path to onestep.pkl")
    parser.add_argument("--ar", dest="ar_pkl", default=None, help="Path to autoregressive.pkl")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    if args.os_pkl is None and args.ar_pkl is None:
        parser.error("provide at least one of --os / --ar")

    onestep = _load(args.os_pkl)
    autoreg = _load(args.ar_pkl)

    plot_os_ar_veh_contact_rmse(onestep, autoreg, args.out)


if __name__ == "__main__":
    main()
