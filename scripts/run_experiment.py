"""Run the reconstruction experiment on a dataset and write its results.

Usage: ``python scripts/run_experiment.py --dataset outputs/demo_sweep --prior
outputs/prior/prior.pt --config configs/recon_default.yaml --seed 0 --out
outputs/experiment``. Prints a time estimate first; ``--estimate-only`` stops there.

Writes ``results.json`` (configuration, per-scene metrics, summary), ``results.md``
(the summary table) and ``examples.npz`` (truth and every method's image for one rest
and one task trial of the first scene) into ``--out``; ``scripts/plot_experiment.py``
turns them into a figure.

``--make-synthetic`` first writes the synthetic phase-screen stand-in dataset of the
config's ``synthetic`` section into ``--dataset`` (if it has no index yet); its results
are labelled synthetic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.io import provenance
from neurosim.data.sweep import INDEX_NAME, load_dataset
from neurosim.recon.checkpoint import load_prior
from neurosim.recon.config import ReconConfig
from neurosim.recon.experiment import estimate_seconds, markdown_table, run_experiment, summarise
from neurosim.recon.seeding import configure_torch
from neurosim.recon.synthetic import write_synthetic_dataset


def main(argv: list[str]) -> int:
    """Run the experiment from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="dataset directory (data.sweep layout)")
    parser.add_argument("--prior", required=True, help="trained prior checkpoint (.pt)")
    parser.add_argument("--config", required=True, help="reconstruction config YAML")
    parser.add_argument("--seed", type=int, required=True, help="root seed of the experiment")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--scenes", type=int, help="use only the first SCENES scenes")
    parser.add_argument("--estimate-only", action="store_true", help="print the estimate and stop")
    parser.add_argument(
        "--make-synthetic",
        action="store_true",
        help="write the config's synthetic stand-in dataset into --dataset first",
    )
    args = parser.parse_args(argv)

    cfg = from_dict(ReconConfig, load_yaml(args.config))
    dataset_dir = Path(args.dataset)
    if not Path(args.prior).is_file():
        print(
            f"error: no trained prior at {args.prior}; create it with scripts/train_prior.py",
            file=sys.stderr,
        )
        return 2
    if args.make_synthetic and not (dataset_dir / INDEX_NAME).exists():
        if cfg.synthetic is None:
            parser.error("--make-synthetic needs a `synthetic` section in the config")
        print(f"writing {cfg.synthetic.n_scenes} synthetic phase-screen scenes to {dataset_dir}")
        write_synthetic_dataset(cfg.synthetic, args.seed, dataset_dir)
    index = json.loads((dataset_dir / INDEX_NAME).read_text(encoding="utf-8"))
    synthetic = bool(index.get("synthetic", False))
    n_scenes = min(index["n_scenes"], args.scenes or index["n_scenes"])

    configure_torch(cfg.num_threads)
    first_scene, first_trials = next(iter(load_dataset(dataset_dir)))
    estimate = estimate_seconds(first_scene, len(first_trials.label), n_scenes, cfg)
    del first_scene, first_trials
    print(
        f"{n_scenes} scenes{' (SYNTHETIC phase-screen skulls)' if synthetic else ''}: "
        f"estimated {estimate['total'] / 60.0:.1f} min single-threaded ("
        + ", ".join(f"{name} {value:.0f} s" for name, value in estimate.items() if name != "total")
        + "), under 1.5 GB"
    )
    if args.estimate_only:
        return 0

    prior = load_prior(args.prior)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    scenes, examples = [], None
    for record, scene_examples in run_experiment(
        islice(load_dataset(dataset_dir), n_scenes), prior, cfg, args.seed
    ):
        scenes.append(record)
        examples = examples or scene_examples
        print(f"scene {record['index']} done, {time.perf_counter() - start:.0f} s elapsed")
    wall_seconds = time.perf_counter() - start

    summary = summarise(scenes)
    results = {
        **provenance(cfg, args.seed),
        "dataset": str(dataset_dir),
        "synthetic_dataset": synthetic,
        "prior": str(args.prior),
        "prior_provenance": {
            key: prior.provenance.get(key) for key in ("config_hash", "root_seed", "git_commit")
        },
        "wall_seconds": wall_seconds,
        "estimated_seconds": estimate,
        "scenes": scenes,
        "summary": summary,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    title = "# Reconstruction experiment"
    if synthetic:
        title += " (SYNTHETIC phase-screen dataset)"
    table = (
        f"{title}\n\nDataset `{dataset_dir}`, {len(scenes)} scenes, seed {args.seed}; mean ± "
        f"standard deviation over scenes. Run time {wall_seconds / 60.0:.1f} min.\n\n"
        + markdown_table(summary)
    )
    (out_dir / "results.md").write_text(table, encoding="utf-8")
    np.savez(out_dir / "examples.npz", **examples)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
