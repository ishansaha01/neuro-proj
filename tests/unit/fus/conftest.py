"""Shared tiny imaging setup for the fUS unit tests (16 elements, 3 angles, 28 frequencies)."""

from __future__ import annotations

import pytest

from neurosim.common.config import from_dict
from neurosim.common.constants import C_REF
from neurosim.fus.config import ImagingConfig, ImagingSetup, imaging_setup
from neurosim.fus.types import GreensBank

TINY_CONFIG = {
    "transducer": {"n_elements": 16},
    "sequence": {"angles_deg": [-5.0, 0.0, 5.0], "frame_rate": 500.0, "n_frames": 6},
    "frequencies": {"n_frequencies": 28},
    "roi": {"x_range": [-0.003, 0.003], "z_range": [0.020, 0.026], "spacing": 1.0e-3},
    "pixels": {"shape": [8, 8]},
    "doppler": {"n_discard": 1},
    "scatterers": {"n_blood": 60, "n_tissue": 60},
}


@pytest.fixture(scope="session")
def tiny_cfg() -> ImagingConfig:
    return from_dict(ImagingConfig, TINY_CONFIG)


@pytest.fixture(scope="session")
def tiny(tiny_cfg: ImagingConfig) -> ImagingSetup:
    return imaging_setup(tiny_cfg)


@pytest.fixture(scope="session")
def water(tiny: ImagingSetup) -> GreensBank:
    return GreensBank.homogeneous(tiny.transducer, tiny.roi, tiny.freqs, C_REF)
