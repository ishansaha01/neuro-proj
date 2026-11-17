"""Compute a Green's bank from a saved plane medium and a job config, and save it with provenance.

Usage: ``python scripts/compute_bank.py --medium plane.h5 --config bank.yaml --out bank.h5``.
Prints the cost estimate before running; ``--estimate-only`` stops there.

The plane-medium bundle holds one dataset per array field of
`neurosim.head.types.PlaneMedium` and the attributes ``grid_shape``,
``grid_spacing``, ``grid_origin``, ``alpha_power``, ``plane_origin``,
``plane_u_lateral`` and ``plane_u_depth``. The job config is the YAML form of
`BankJob` below.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np

from neurosim.common.config import field, from_dict, load_yaml
from neurosim.common.grid import Grid2D, PlaneSpec
from neurosim.common.io import load_bundle, provenance, save_bundle
from neurosim.fus.backends import BackendConfig, compute_bank, estimate_cost
from neurosim.fus.types import Transducer
from neurosim.head.types import PlaneMedium

_BYTES_PER_GIB = 2**30

# Fields of `PlaneMedium` stored as datasets (everything except the grid, the plane and scalars).
_ARRAY_FIELDS: tuple[str, ...] = tuple(
    f.name
    for f in dataclasses.fields(PlaneMedium)
    if f.name not in ("grid", "plane", "alpha_power")
)


@dataclass(frozen=True)
class RoiConfig:
    """Coarse grid the bank is stored on."""

    x_range: tuple[float, float] = field(doc="Lateral range of the ROI point centres [m].")
    z_range: tuple[float, float] = field(gt=0.0, doc="Depth range of the ROI point centres [m].")
    spacing: float = field(gt=0.0, doc="ROI grid spacing [m].")


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
    roi: RoiConfig = field(doc="ROI grid.")
    frequencies: FrequencyConfig = field(doc="Frequency list.")
    backend: BackendConfig = field(default_factory=BackendConfig, doc="Backend and its settings.")


def load_plane_medium(path: str | os.PathLike[str]) -> tuple[PlaneMedium, int]:
    """Read a plane-medium bundle (module docstring).

    Returns the medium and the root seed stored with it (0 if the bundle has none).
    """
    arrays, attrs = load_bundle(path)
    medium = PlaneMedium(
        grid=Grid2D(
            shape=tuple(attrs["grid_shape"]),
            spacing=attrs["grid_spacing"],
            origin=tuple(attrs["grid_origin"]),
        ),
        plane=PlaneSpec(
            origin=tuple(attrs["plane_origin"]),
            u_lateral=tuple(attrs["plane_u_lateral"]),
            u_depth=tuple(attrs["plane_u_depth"]),
        ),
        alpha_power=attrs["alpha_power"],
        **{name: arrays[name] for name in _ARRAY_FIELDS},
    )
    return medium, int(attrs.get("root_seed", 0))


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
    medium, root_seed = load_plane_medium(args.medium)
    roi = Grid2D.from_extent(job.roi.x_range, job.roi.z_range, job.roi.spacing)
    freqs = np.linspace(job.frequencies.first, job.frequencies.last, job.frequencies.count)

    cost = estimate_cost(medium, job.transducer, roi, freqs, job.backend)
    print(
        f"Green's bank ({job.backend.backend}): {cost['n_simulations']} simulations of "
        f"{cost['grid_points']} grid points x {cost['time_steps']} time steps, "
        f"about {cost['memory_bytes'] / _BYTES_PER_GIB:.2f} GiB and {cost['run_time_s']:.0f} s"
    )
    if args.estimate_only:
        return 0

    bank = compute_bank(medium, job.transducer, roi, freqs, job.backend)
    arrays = {"B": bank.B, "freqs": bank.freqs}
    if bank.static_fmc is not None:
        arrays["static_fmc"] = bank.static_fmc
    # The bank computation draws no random numbers; the seed is that of the medium.
    attrs = provenance(job, root_seed)
    attrs.update(
        roi_shape=list(roi.shape),
        roi_spacing=roi.spacing,
        roi_origin=list(roi.origin),
        c_ref=bank.c_ref,
        backend=bank.backend,
        meta=bank.meta,
    )
    save_bundle(args.out, arrays, attrs)
    print(f"saved {args.out}: B of shape {bank.B.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
