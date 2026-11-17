"""Generate one ground-truth head and save it with provenance.

Usage: ``python scripts/generate_head.py --config configs/head_default.yaml
--seed 0 --out head.h5``. Prints the memory and time estimate before generating.
"""

from __future__ import annotations

import argparse
import sys

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.seeding import SeedTree
from neurosim.head.config import HeadConfig
from neurosim.head.generate import estimate_memory_bytes, estimate_runtime_seconds, generate_head
from neurosim.head.io import save_head

_BYTES_PER_GIB = 2**30


def main(argv: list[str]) -> int:
    """Generate and save one head from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="head config YAML (see configs/)")
    parser.add_argument("--seed", type=int, required=True, help="root seed of the run")
    parser.add_argument("--out", required=True, help="output HDF5 file")
    args = parser.parse_args(argv)

    cfg = from_dict(HeadConfig, load_yaml(args.config))
    print(
        f"head ({cfg.grid.region}, {cfg.grid.spacing * 1e3:g} mm voxels): about "
        f"{estimate_memory_bytes(cfg) / _BYTES_PER_GIB:.2f} GiB peak memory and "
        f"{estimate_runtime_seconds(cfg):.0f} s"
    )
    head = generate_head(cfg, SeedTree(args.seed))
    save_head(args.out, head, cfg, args.seed)
    print(f"saved {args.out}: head volume of shape {head.grid.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
