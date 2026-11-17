"""Simulate one fUS power Doppler acquisition and save it with provenance.

Usage: ``python scripts/run_fus.py --config configs/fus_default.yaml (--bank bank.h5 | --water)
--cbv plane.h5 --seed 0 --out fus.h5``. Prints the cost estimate before running;
``--estimate-only`` stops there.

Inputs:

- ``--bank``: a Green's-bank bundle written by `neurosim.fus.io.save_bank` (for example by
  ``scripts/compute_bank.py``). Its array must be the array of the config. ``--water`` uses
  the homogeneous reference medium on the config's ROI and frequency grid instead.
- ``--model-bank``: the medium the beamformer assumes, in the same format. Default: the
  homogeneous medium (conventional delay-and-sum).
- ``--cbv``: a plane-medium bundle written by `neurosim.head.io.save_plane_medium`. Its
  ``cbv_baseline`` and ``blood_velocity`` maps are sampled bilinearly at the pixel centres;
  the pixel grid is the CBV grid of the simulation.

Output bundle: ``power_doppler`` (nz, nx) float32 and, at scatterer fidelity, ``frames``
(n_frames, nz, nx) complex64; attributes: provenance, the pixel grid, the bank's backend and
the number of singular components removed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D, sample_plane_map
from neurosim.common.io import provenance, save_bundle
from neurosim.common.seeding import SeedTree
from neurosim.common.units import bytes_to_gib
from neurosim.fus.acquire import acquire_frames, estimate_cost, statistical_power_doppler
from neurosim.fus.beamform import make_beamformer
from neurosim.fus.config import ImagingConfig, ImagingSetup, imaging_setup
from neurosim.fus.doppler import choose_svd_cutoff, power_doppler, svd_clutter_filter
from neurosim.fus.io import load_bank
from neurosim.fus.noise import electronic_noise_variance
from neurosim.fus.operators import pd_operator, psf_operator
from neurosim.fus.scatterers import blood_scatterers, tissue_amplitude, tissue_scatterers
from neurosim.fus.speckle import speckle_frames
from neurosim.fus.types import GreensBank
from neurosim.head.io import load_plane_medium


def load_bank_for(path: str | os.PathLike[str], setup: ImagingSetup) -> GreensBank:
    """Read a Green's-bank bundle and check that it is for the array of the config."""
    bank = load_bank(path)
    if bank.transducer != setup.transducer:
        raise ValueError(
            f"{path}: the bank is for {bank.transducer}, the config has {setup.transducer}"
        )
    return bank


def load_cbv(path: str | os.PathLike[str], grid: Grid2D) -> tuple[np.ndarray, np.ndarray]:
    """Read the CBV and blood velocity maps of a plane medium and sample them on `grid`.

    Returns `(cbv, velocity)`: float64, shapes (nz, nx) and (2, nz, nx) [fraction, m/s].
    """
    medium = load_plane_medium(path)
    points = grid.points()
    cbv = sample_plane_map(medium.cbv_baseline, medium.grid, points)
    velocity = sample_plane_map(medium.blood_velocity, medium.grid, points)
    return cbv.astype(np.float64), velocity.astype(np.float64)


def main(argv: list[str]) -> int:
    """Run one fUS acquisition from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="imaging config YAML (see configs/)")
    medium = parser.add_mutually_exclusive_group(required=True)
    medium.add_argument("--bank", help="Green's-bank bundle of the medium (HDF5)")
    medium.add_argument("--water", action="store_true", help="homogeneous reference medium")
    parser.add_argument("--model-bank", help="bank the beamformer assumes (default: homogeneous)")
    parser.add_argument("--cbv", required=True, help="plane-medium bundle with the CBV maps (HDF5)")
    parser.add_argument("--seed", type=int, required=True, help="root seed of the run")
    parser.add_argument("--out", help="output HDF5 file (required unless --estimate-only)")
    parser.add_argument(
        "--estimate-only", action="store_true", help="print the cost estimate and stop"
    )
    args = parser.parse_args(argv)
    if args.out is None and not args.estimate_only:
        parser.error("--out is required unless --estimate-only is given")

    cfg = from_dict(ImagingConfig, load_yaml(args.config))
    setup = imaging_setup(cfg)
    source_grid = setup.pixel_grid
    moving_tissue = cfg.noise.tissue_clutter and cfg.noise.tissue_motion
    n_scatterers = cfg.scatterers.n_blood + (cfg.scatterers.n_tissue if moving_tissue else 0)
    statistical = cfg.fidelity == "statistical"
    n_items = source_grid.shape[0] * source_grid.shape[1] if statistical else n_scatterers
    cost = estimate_cost(cfg, n_items)
    print(
        f"fUS ({cfg.fidelity}): {cost['n_frames']:.0f} frames, {cost['flops']:.3g} flops, "
        f"about {bytes_to_gib(cost['memory_bytes']):.2f} GiB and {cost['time_s']:.0f} s"
    )
    if args.estimate_only:
        return 0

    tree = SeedTree(args.seed)
    if args.water:
        bank = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)
    else:
        bank = load_bank_for(args.bank, setup)
    if args.model_bank:
        model_bank = load_bank_for(args.model_bank, setup)
    else:
        model_bank = GreensBank.homogeneous(bank.transducer, bank.roi, bank.freqs, bank.c_ref)
    cbv, velocity = load_cbv(args.cbv, source_grid)

    arrays = {}
    attrs = provenance(cfg, args.seed)
    attrs.update(
        grid_shape=list(setup.pixel_grid.shape),
        grid_spacing=setup.pixel_grid.spacing,
        grid_origin=list(setup.pixel_grid.origin),
        backend=bank.backend,
    )
    if statistical:
        operator = pd_operator(
            psf_operator(
                bank, setup.pixel_grid, source_grid, model_bank, setup.sequence, cfg.beamforming
            )
        )
        floor = 0.0
        if cfg.noise.electronic_noise:
            beamformer = make_beamformer(
                setup.pixel_grid, model_bank, setup.sequence, cfg.beamforming
            )
            variance = electronic_noise_variance(bank, setup.sequence, cfg.noise)
            floor = variance * beamformer.noise_gain
        counts = speckle_frames(cfg, operator, cbv, velocity, source_grid)
        image = statistical_power_doppler(
            operator,
            cbv * source_grid.spacing**2,
            floor,
            counts.blood,
            counts.noise,
            tree.rng("speckle_statistical"),
        ).reshape(setup.pixel_grid.shape)
    else:
        blood = blood_scatterers(
            cbv,
            velocity,
            source_grid,
            cfg.scatterers.n_blood,
            tree.rng("blood_scatterers"),
            cfg.scatterers.blood_transit_length,
        )
        tissue = tissue_scatterers(
            setup.roi,
            cfg.scatterers.n_tissue,
            tissue_amplitude(setup.roi, cfg.scatterers.n_tissue, cfg.noise),
            tree.rng("tissue_scatterers"),
        )
        frames = acquire_frames(
            bank, model_bank, blood, tissue, setup.sequence, setup.pixel_grid, cfg, tree
        )
        n_discard = choose_svd_cutoff(frames, cfg)
        image = power_doppler(svd_clutter_filter(frames, n_discard))
        arrays["frames"] = frames.astype(np.complex64)
        attrs["n_discard"] = n_discard
    arrays["power_doppler"] = image.astype(np.float32)

    save_bundle(args.out, arrays, attrs)
    print(f"saved {args.out}: power Doppler image of shape {image.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
