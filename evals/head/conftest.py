"""Shared fixture of the head evals: the small head configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.head.config import HeadConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="session")
def small_cfg() -> HeadConfig:
    """The head configuration every head eval starts from (configs/head_small.yaml)."""
    return from_dict(HeadConfig, load_yaml(CONFIG_DIR / "head_small.yaml"))
