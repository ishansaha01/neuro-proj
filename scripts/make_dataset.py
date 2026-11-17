"""Generate a paired dataset: one run bundle per scene of a sweep, plus an index.

Usage: ``python scripts/make_dataset.py --config configs/sweep_demo.yaml --seed 0
--out outputs/demo_sweep [--scenes 0 3] [--keep-existing]``. Prints the cost estimate
before running; ``--estimate-only`` stops there. Running it again on the same directory
resumes: scenes whose complete bundle was produced by the current git commit are skipped.
A bundle that another commit produced is generated again, so that a dataset does not mix
code versions without saying so; ``--keep-existing`` keeps such bundles instead, and the
index then shows their commit.

The directory holds ``index.json`` (file, seed, varied parameters and producing git commit
of every scene) and ``scene_<nnnn>.h5`` (`neurosim.data.io.save_run`); read it with
`neurosim.data.sweep.load_dataset`.
"""

from __future__ import annotations

import argparse
import logging
import sys

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.seeding import SeedTree
from neurosim.common.units import bytes_to_gib
from neurosim.data.scene import estimate_cost
from neurosim.data.sweep import SweepConfig, run_sweep, scene_config


def main(argv: list[str]) -> int:
    """Generate a dataset from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="sweep config YAML (configs/sweep_*.yaml)")
    parser.add_argument("--seed", type=int, required=True, help="root seed of the sweep")
    parser.add_argument("--out", help="dataset directory (required unless --estimate-only)")
    parser.add_argument(
        "--scenes", type=int, nargs="+", help="scene indices to generate (default: all)"
    )
    parser.add_argument(
        "--estimate-only", action="store_true", help="print the cost estimate and stop"
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="keep complete bundles produced by another git commit instead of generating them "
        "again; index.json records the commit of every scene",
    )
    args = parser.parse_args(argv)
    if args.out is None and not args.estimate_only:
        parser.error("--out is required unless --estimate-only is given")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = from_dict(SweepConfig, load_yaml(args.config))
    indices = tuple(range(cfg.n_scenes)) if args.scenes is None else tuple(args.scenes)
    tree = SeedTree(args.seed)
    costs = [estimate_cost(scene_config(cfg, tree, index)[0]) for index in indices]
    print(
        f"sweep of {len(indices)} scenes, seed {args.seed}: about "
        f"{sum(cost['time_s'] for cost in costs):.0f} s single-threaded, "
        f"{bytes_to_gib(max(cost['memory_bytes'] for cost in costs)):.2f} GiB peak memory, "
        f"{bytes_to_gib(sum(cost['bundle_bytes'] for cost in costs)):.2f} GiB on disk"
    )
    if args.estimate_only:
        return 0

    index = run_sweep(cfg, args.seed, args.out, indices, keep_existing=args.keep_existing)
    print(f"dataset in {args.out}: {len(indices)} of {index['n_scenes']} scenes generated or kept")
    other_commit = [
        index["scenes"][scene]["index"]
        for scene in indices
        if index["scenes"][scene]["git_commit"] != index["git_commit"]
    ]
    if other_commit:
        print(f"kept from another git commit (see index.json): scenes {other_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
