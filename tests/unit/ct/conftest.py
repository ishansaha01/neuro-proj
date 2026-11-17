"""Shared fixtures of the CT unit tests: the small scan config and a small grid."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.grid import Grid3D
from neurosim.ct.config import CtConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@pytest.fixture(scope="session")
def config_dir() -> Path:
    """The repository's configs/ directory."""
    return CONFIG_DIR


@pytest.fixture(scope="session")
def small_cfg() -> CtConfig:
    """The scan configuration every CT test and eval starts from."""
    return from_dict(CtConfig, load_yaml(CONFIG_DIR / "ct_small.yaml"))


@pytest.fixture(scope="session")
def small_grid() -> Grid3D:
    """A 0.2 m x 0.2 m x 16 mm grid of 2 mm voxels, centred on the isocentre."""
    return Grid3D.centered((0.2, 0.2, 0.016), 0.002)
