"""Shared fixture of the data evals: the true and the CT-derived plane medium of a seed.

Default settings throughout: ``configs/head_default.yaml`` (0.5 mm box head),
``configs/ct_default.yaml`` (120 kVp, ramp kernel, photon noise on) and
``configs/ct_prior_default.yaml`` (nominal HU mapping), on the default imaging
plane. One deviation, which does not change the plane's medium: the slab is one
slice (``slab_half_slices = 0``) instead of three, because CT slices are
independent and each costs 15-30 s; only the central slice, the imaging plane
itself, enters the CT-derived medium. One seed takes about 20 s in total.

One head takes about 0.5 GB, so a head is released before the next seed is
generated; only plane-sized arrays are kept.
"""

from __future__ import annotations

import dataclasses
import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.grid import Grid2D, plane_to_world, sample_volume
from neurosim.common.seeding import SeedTree
from neurosim.ct.config import CtConfig
from neurosim.data.ct_prior import CtPriorConfig, ct_plane_hu, ct_prior_medium
from neurosim.head.config import HeadConfig
from neurosim.head.generate import generate_head
from neurosim.head.plane import default_plane, sample_plane
from neurosim.head.types import SURFACES, PlaneMedium

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Fine acoustic grid of the evals: 0.25 mm pixels over the plane region the default
# head box holds under the default probe (50 mm wide, 55 mm deep; see
# configs/head_default.yaml), starting at the array face.
PLANE_X_RANGE = (-0.025, 0.025)
PLANE_Z_RANGE = (0.0, 0.055)
PLANE_SPACING = 0.25e-3

_SKULL_SURFACES = [SURFACES.index("skull_outer"), SURFACES.index("skull_inner")]


@dataclass(frozen=True)
class PlanePrior:
    """Truth and CT-derived prior of one seed in the default imaging plane.

    Attributes:
        truth: the true plane medium (`head.plane.sample_plane`).
        ct_medium: the CT-derived plane medium (`data.ct_prior.ct_prior_medium`).
        hu_plane: the CT image the prior was mapped from [HU], shape (nz, nx).
        skull_sdf_center_line: signed distance to the true outer and inner skull
            surfaces along the central depth line x = 0 [m], shape (2, nz).
    """

    truth: PlaneMedium
    ct_medium: PlaneMedium
    hu_plane: np.ndarray
    skull_sdf_center_line: np.ndarray


def _plane_prior(seed: int) -> PlanePrior:
    head_cfg = from_dict(HeadConfig, load_yaml(CONFIG_DIR / "head_default.yaml"))
    ct_cfg = from_dict(CtConfig, load_yaml(CONFIG_DIR / "ct_default.yaml"))
    prior_cfg = dataclasses.replace(
        from_dict(CtPriorConfig, load_yaml(CONFIG_DIR / "ct_prior_default.yaml")),
        slab_half_slices=0,
    )
    grid = Grid2D.from_extent(PLANE_X_RANGE, PLANE_Z_RANGE, PLANE_SPACING)
    tree = SeedTree(seed)
    head = generate_head(head_cfg, tree)
    plane = default_plane(head, head_cfg)
    truth = sample_plane(head, plane, grid)
    ct, ct_medium = ct_prior_medium(head, plane, grid, ct_cfg, prior_cfg, tree)

    depth = grid.axes()[0]
    center_line = plane_to_world(plane, np.stack((np.zeros_like(depth), depth), axis=-1))
    skull_sdf = sample_volume(head.sdf[_SKULL_SURFACES], head.grid, center_line)
    del head
    gc.collect()
    return PlanePrior(
        truth=truth,
        ct_medium=ct_medium,
        hu_plane=ct_plane_hu(ct, grid, plane),
        skull_sdf_center_line=skull_sdf,
    )


@pytest.fixture(scope="session")
def plane_prior() -> Callable[[int], PlanePrior]:
    """Function returning the `PlanePrior` of a seed; each seed is computed once per session."""
    cache: dict[int, PlanePrior] = {}

    def get(seed: int) -> PlanePrior:
        if seed not in cache:
            cache[seed] = _plane_prior(seed)
        return cache[seed]

    return get
