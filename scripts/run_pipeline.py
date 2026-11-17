"""Run the whole simulator once (one config, one seed) and save the run with provenance.

Usage: ``python scripts/run_pipeline.py --config configs/run_demo.yaml --seed 0
--out outputs/demo/run.h5 [--save-media]``. Prints the cost estimate before running;
``--estimate-only`` stops there.

The run bundle is written by `neurosim.data.io.save_run`. With ``--save-media`` the true
plane medium (skull-effect switches applied) and the CT-derived plane medium are written
next to it (`neurosim.head.io.save_plane_medium`, under the names of
`neurosim.data.io.media_paths`); ``scripts/plot_run.py`` draws them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.io import provenance
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig, trial_fidelity
from neurosim.data.io import media_paths, save_run
from neurosim.data.scene import build_scene_with_extras, estimate_cost
from neurosim.data.trials import simulate_trials
from neurosim.head.io import save_plane_medium

_BYTES_PER_GIB = 2**30


def describe_cost(cost: dict[str, Any]) -> str:
    """One paragraph that states the cost estimate of `data.scene.estimate_cost`."""
    bank, ct_prior = cost["bank"], cost["ct_prior"]
    return (
        f"estimated cost: about {cost['time_s']:.0f} s single-threaded, "
        f"{cost['memory_bytes'] / _BYTES_PER_GIB:.2f} GiB peak memory, "
        f"{cost['bundle_bytes'] / _BYTES_PER_GIB:.2f} GiB on disk\n"
        f"  head {cost['head']['time_s']:.0f} s; CT prior {ct_prior['time_s']:.0f} s "
        f"({ct_prior['n_lateral']:.0f} x {ct_prior['n_depth']:.0f} pixels per slice); "
        f"{cost['n_backend_banks']} banks of {bank['run_time_s']:.0f} s "
        f"({bank['n_simulations']} simulations of {bank['grid_points']} grid points x "
        f"{bank['time_steps']} time steps) and {cost['n_ray_banks']} ray banks of "
        f"{cost['bank_ray']['run_time_s']:.0f} s; {cost['n_operator_passes']} operator passes "
        f"of {cost['operator']['time_s']:.0f} s; {cost['n_images']} images x 4 beamformers of "
        f"{cost['trial_image']['time_s']:.2g} s"
    )


def main(argv: list[str]) -> int:
    """Run the pipeline from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="run config YAML (see configs/run_*.yaml)")
    parser.add_argument("--seed", type=int, required=True, help="root seed of the run")
    parser.add_argument("--out", help="output HDF5 file (required unless --estimate-only)")
    parser.add_argument(
        "--save-media", action="store_true", help="also save the true and CT-derived plane media"
    )
    parser.add_argument(
        "--estimate-only", action="store_true", help="print the cost estimate and stop"
    )
    args = parser.parse_args(argv)
    if args.out is None and not args.estimate_only:
        parser.error("--out is required unless --estimate-only is given")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = from_dict(RunConfig, load_yaml(args.config))
    print(f"run ({cfg.backend.backend} backend, {trial_fidelity(cfg)} fidelity), seed {args.seed}")
    print(describe_cost(estimate_cost(cfg)))
    if args.estimate_only:
        return 0

    tree = SeedTree(args.seed)
    scene, extras = build_scene_with_extras(cfg, tree)
    trials = simulate_trials(scene, cfg, tree, extras)
    save_run(args.out, scene, trials, cfg, args.seed)
    print(
        f"saved {args.out}: {trials.time.size} images of shape {scene.pixel_grid.shape}; "
        f"R^2 of CT-derived against true skull sound speed {scene.meta['ct_sound_speed_r2']:.2f}"
    )
    if args.save_media:
        attrs = provenance(cfg, args.seed)
        for path, medium in zip(media_paths(args.out), (extras.medium, extras.medium_ct)):
            save_plane_medium(path, medium, attrs)
            print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
