"""Generate the clean CBV images the diffusion prior is trained on.

Usage: ``python scripts/make_prior_dataset.py --config configs/prior_default.yaml
--seed 0 --out outputs/prior/dataset.h5``. Uses the ``data`` section of the
config. Prints the time and memory estimate before generating.
"""

from __future__ import annotations

import argparse
import sys
import time

from neurosim.common.config import from_dict, load_yaml, to_dict
from neurosim.common.io import provenance, save_bundle
from neurosim.common.seeding import SeedTree
from neurosim.common.units import bytes_to_gib
from neurosim.recon.cbv_images import (
    estimate_memory_bytes,
    estimate_runtime_seconds,
    sample_cbv_images,
)
from neurosim.recon.train import PriorConfig


def main(argv: list[str]) -> int:
    """Generate and save the data set from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="prior config YAML (see configs/)")
    parser.add_argument("--seed", type=int, required=True, help="root seed of the data set")
    parser.add_argument("--out", required=True, help="output HDF5 bundle")
    args = parser.parse_args(argv)

    cfg = from_dict(PriorConfig, load_yaml(args.config)).data
    n_images = cfg.n_heads * cfg.planes_per_head
    print(
        f"prior data set: {cfg.n_heads} heads x {cfg.planes_per_head} planes = {n_images} "
        f"images of {cfg.pixel_grid.n} x {cfg.pixel_grid.n} pixels; about "
        f"{estimate_runtime_seconds(cfg):.0f} s and "
        f"{bytes_to_gib(estimate_memory_bytes(cfg)):.2f} GiB peak memory"
    )
    start = time.perf_counter()
    arrays = sample_cbv_images(cfg, SeedTree(args.seed))
    attrs = provenance(cfg, args.seed)
    attrs["pixel_grid"] = to_dict(cfg.pixel_grid)
    save_bundle(args.out, arrays, attrs)
    print(
        f"saved {args.out}: {n_images} images ({int(arrays['heldout'].sum())} held out, "
        f"{int(arrays['active'].sum())} active) in {time.perf_counter() - start:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
