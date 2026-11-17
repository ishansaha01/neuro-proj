"""Shared fixtures: a small medium grid, materials and fast full-wave settings."""

import pytest

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.backends.config import KWaveConfig
from neurosim.fus.backends.phantoms import Material
from neurosim.fus.types import Transducer

SPACING = 0.3e-3  # [m]


@pytest.fixture
def spacing() -> float:
    return SPACING


@pytest.fixture
def grid() -> Grid2D:
    """41 x 41 nodes, 12 mm x 12 mm, x = 0 on a node and z = 0 on the first row."""
    return Grid2D.from_extent((-6.0e-3, 6.0e-3), (0.0, 12.0e-3), SPACING)


@pytest.fixture
def soft() -> Material:
    """The reference medium: lossless at C_REF."""
    return Material(sound_speed=C_REF, density=1000.0)


@pytest.fixture
def bone() -> Material:
    return Material(sound_speed=2800.0, density=1900.0, attenuation=6.0)


@pytest.fixture
def transducer() -> Transducer:
    """Three elements, one node wide, 4 cells apart, centred on nodes."""
    return Transducer(
        n_elements=3,
        pitch=4 * SPACING,
        element_width=SPACING,
        center_frequency=0.5e6,
        fractional_bandwidth=0.8,
    )


@pytest.fixture
def roi() -> Grid2D:
    """3 x 3 ROI points on medium nodes."""
    return Grid2D.from_extent((-3.0e-3, 3.0e-3), (6.0e-3, 9.0e-3), 3.0e-3 / 2)


@pytest.fixture
def fast_kwave() -> KWaveConfig:
    """Full-wave settings for grids of a few thousand points and a few hundred steps."""
    return KWaveConfig(
        cfl=0.3, pml_size=8, pml_search=4, reverberation_tail=2.0e-6, reference_margin=6
    )
