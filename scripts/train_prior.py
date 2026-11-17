"""Train the diffusion prior on a data set made by ``make_prior_dataset.py``.

Usage: ``python scripts/train_prior.py --config configs/prior_default.yaml --data
outputs/prior/dataset.h5 --seed 0 --out outputs/prior/prior.pt``. Prints the
measured time per step and the estimated total before training, then the
held-out loss at a few noise levels as training goes.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from neurosim.common.config import config_hash, from_dict, load_yaml
from neurosim.common.io import load_bundle
from neurosim.common.seeding import SeedTree
from neurosim.recon.checkpoint import DATASET_PROVENANCE_KEYS, save_prior
from neurosim.recon.seeding import TRAINING_STREAM, TrainingDraw, torch_generator
from neurosim.recon.train import PriorConfig, measure_step_seconds, train_prior
from neurosim.recon.unet import build_unet, count_parameters

_SECONDS_PER_MINUTE = 60.0


def _print_record(record: dict[str, Any]) -> None:
    levels = "  ".join(
        f"x0 mse @ alpha_bar {level}: {values['x0_mse']:.5f}"
        for level, values in record["heldout"].items()
    )
    print(f"step {record['step']:6d}  train v-loss {record['train_loss']:.4f}  held-out {levels}")


def main(argv: list[str]) -> int:
    """Train and save a prior from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="prior config YAML (see configs/)")
    parser.add_argument("--data", required=True, help="data set bundle (make_prior_dataset.py)")
    parser.add_argument("--seed", type=int, required=True, help="root seed of the training run")
    parser.add_argument("--out", required=True, help="output checkpoint file")
    args = parser.parse_args(argv)

    cfg = from_dict(PriorConfig, load_yaml(args.config))
    arrays, attrs = load_bundle(args.data)
    if attrs["config_hash"] != config_hash(cfg.data):
        print(
            f"{args.data} was not generated with the data section of {args.config} "
            "(config hashes differ); regenerate it with scripts/make_prior_dataset.py",
            file=sys.stderr,
        )
        return 2
    train_images = arrays["images"][~arrays["heldout"]]
    heldout_images = arrays["images"][arrays["heldout"]]

    tree = SeedTree(args.seed)
    n_parameters = count_parameters(
        build_unet(cfg.unet, torch_generator(tree, TRAINING_STREAM, TrainingDraw.WEIGHT_INIT))
    )
    step_seconds = measure_step_seconds(cfg, tree)
    total_minutes = step_seconds * cfg.training.n_steps / _SECONDS_PER_MINUTE
    limit = cfg.training.max_wall_minutes
    print(
        f"prior: {n_parameters / 1e6:.2f} M parameters, {train_images.shape[0]} training and "
        f"{heldout_images.shape[0]} held-out images, {cfg.training.num_threads} thread(s); "
        f"about {step_seconds:.3f} s per step, {total_minutes:.1f} min for "
        f"{cfg.training.n_steps} steps"
        + (f" (stops after {limit:g} min at the latest)" if limit is not None else "")
    )
    result = train_prior(cfg, train_images, heldout_images, tree, on_eval=_print_record)
    save_prior(
        args.out, result, cfg, args.seed, {key: attrs[key] for key in DATASET_PROVENANCE_KEYS}
    )
    print(
        f"saved {args.out}: {result.steps_completed} of {cfg.training.n_steps} steps in "
        f"{result.wall_seconds / _SECONDS_PER_MINUTE:.1f} min"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
