"""Simulate a CT scan of a saved head and save the HU volume with provenance.

Usage: ``python scripts/run_ct.py --head head.h5 --config configs/ct_default.yaml
--seed 0 --out ct.h5``. Prints the cost estimate before running.
"""

from __future__ import annotations

import argparse
import sys

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.seeding import SeedTree
from neurosim.ct.config import CtConfig
from neurosim.ct.io import save_ct
from neurosim.ct.simulate import estimate_cost, simulate_ct
from neurosim.head.io import load_head

_BYTES_PER_GIB = 2**30


def main(argv: list[str]) -> int:
    """Run one CT simulation from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--head", required=True, help="head bundle written by the head package")
    parser.add_argument("--config", required=True, help="CT config YAML (see configs/)")
    parser.add_argument("--seed", type=int, required=True, help="root seed of the run")
    parser.add_argument("--out", required=True, help="output HDF5 file")
    args = parser.parse_args(argv)

    cfg = from_dict(CtConfig, load_yaml(args.config))
    head = load_head(args.head)
    cost = estimate_cost(head.grid, cfg)
    print(
        f"CT scan: {cost['n_slices']:.0f} slices, {cost['rays']:.3g} rays, "
        f"{cost['samples']:.3g} ray samples, "
        f"about {cost['memory_bytes'] / _BYTES_PER_GIB:.2f} GiB and {cost['time_s']:.0f} s"
    )
    ct = simulate_ct(head, cfg, SeedTree(args.seed))
    save_ct(args.out, ct, cfg, args.seed)
    print(f"saved {args.out}: HU volume of shape {ct.hu.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
