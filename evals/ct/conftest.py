"""Shared fixtures of the CT evals: the small scan config and a small grid."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.grid import Grid3D
from neurosim.common.seeding import SeedTree
from neurosim.ct.config import CtConfig
from neurosim.ct.phantoms import disc_coverage, layered_fractions, radial_distance
from neurosim.ct.simulate import simulate_ct_fractions

# Bone-rod phantom shared by E-CT-005 and E-CT-006: a 20 mm rod of pure ICRU-44
# cortical bone, 40 mm off centre in a 140 mm water cylinder (the layout of a
# clinical CT-number QA phantom).
BONE_ROD_BODY_RADIUS = 0.07
BONE_ROD_CENTER = (0.04, 0.0)
BONE_ROD_RADIUS = 0.01
BONE_ROD_ROI_RADIUS = 0.005

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="session")
def small_cfg() -> CtConfig:
    """The scan configuration every CT test and eval starts from."""
    return from_dict(CtConfig, load_yaml(CONFIG_DIR / "ct_small.yaml"))


@pytest.fixture(scope="session")
def small_grid() -> Grid3D:
    """A 0.2 m x 0.2 m x 16 mm grid of 2 mm voxels, centred on the isocentre."""
    return Grid3D.centered((0.2, 0.2, 0.016), 0.002)


@pytest.fixture(scope="session")
def bone_rod_hu(small_grid) -> Callable[[CtConfig], float]:
    """Function scanning the bone-rod phantom noise-free; returns the rod's mean HU."""
    fractions = layered_fractions(
        small_grid,
        [
            ("water", disc_coverage(small_grid, BONE_ROD_BODY_RADIUS)),
            ("cortical_bone", disc_coverage(small_grid, BONE_ROD_RADIUS, BONE_ROD_CENTER)),
        ],
    )
    roi = radial_distance(small_grid, BONE_ROD_CENTER) < BONE_ROD_ROI_RADIUS

    def measure(cfg: CtConfig) -> float:
        one_slice = dataclasses.replace(cfg, photon_noise=False, slab_z_range=(0.0, 0.002))
        hu = simulate_ct_fractions(fractions, small_grid, one_slice, SeedTree(0)).hu[..., 0]
        return float(hu[roi].mean())

    return measure
