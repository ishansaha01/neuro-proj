"""Compute a Green's bank from a saved plane medium and a job config, and save it with provenance.

Usage: ``python scripts/compute_bank.py --medium plane.h5 --config bank.yaml --out bank.h5``.
Prints the cost estimate before running; ``--estimate-only`` stops there.

The plane medium is a bundle written by `neurosim.head.io.save_plane_medium`; the bank is
written with `neurosim.fus.io.save_bank`. The job config is the YAML form of `BankJob` below.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

import numpy as np

from neurosim.common.config import field, from_dict, load_yaml
from neurosim.common.grid import Grid2D
from neurosim.common.io import load_bundle, provenance
from neurosim.common.units import bytes_to_gib
from neurosim.fus.backends import BackendConfig, compute_bank, estimate_cost
from neurosim.fus.config import RoiConfig
from neurosim.fus.io import save_bank
from neurosim.fus.types import Transducer
from neurosim.head.io import plane_medium_from_bundle


@dataclass(frozen=True)
class FrequencyConfig:
    """Uniformly spaced frequency list of the bank."""

    first: float = field(gt=0.0, doc="Lowest frequency [Hz].")
    last: float = field(gt=0.0, doc="Highest frequency [Hz].")
    count: int = field(ge=1, doc="Number of frequencies.")


@dataclass(frozen=True)
class BankJob:
    """One bank computation: array, ROI, frequencies and backend."""

    transducer: Transducer = field(doc="The array (see `neurosim.fus.types.Transducer`).")
    roi: RoiConfig = field(
        doc="Coarse grid the bank is stored on (`neurosim.fus.config.RoiConfig`; keys that "
        "are left out take its defaults)."
    )
    frequencies: FrequencyConfig = field(doc="Frequency list.")
    backend: BackendConfig = field(default_factory=BackendConfig, doc="Backend and its settings.")


def main(argv: list[str]) -> int:
    """Run one bank computation from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--medium", required=True, help="plane-medium bundle (HDF5)")
    parser.add_argument("--config", required=True, help="bank job config (YAML form of BankJob)")
    parser.add_argument("--out", help="output HDF5 file (required unless --estimate-only)")
    parser.add_argument(
        "--estimate-only", action="store_true", help="print the cost estimate and stop"
    )
    args = parser.parse_args(argv)
    if args.out is None and not args.estimate_only:
        parser.error("--out is required unless --estimate-only is given")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    job = from_dict(BankJob, load_yaml(args.config))
    medium_arrays, medium_attrs = load_bundle(args.medium)
    medium = plane_medium_from_bundle(medium_arrays, medium_attrs)
    roi = Grid2D.from_extent(job.roi.x_range, job.roi.z_range, job.roi.spacing)
    freqs = np.linspace(job.frequencies.first, job.frequencies.last, job.frequencies.count)

    cost = estimate_cost(medium, job.transducer, roi, freqs, job.backend)
    print(
        f"Green's bank ({job.backend.backend}): {cost['n_simulations']} simulations of "
        f"{cost['grid_points']} grid points x {cost['time_steps']} time steps, "
        f"about {bytes_to_gib(cost['memory_bytes']):.2f} GiB and {cost['run_time_s']:.0f} s"
    )
    if args.estimate_only:
        return 0

    bank = compute_bank(medium, job.transducer, roi, freqs, job.backend)
    # The bank computation draws no random numbers; the seed is that of the medium.
    save_bank(args.out, bank, provenance(job, medium_attrs["root_seed"]))
    print(f"saved {args.out}: B of shape {bank.B.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
