from pathlib import Path

import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.seeding import SeedTree
from neurosim.head.config import HeadConfig
from neurosim.head.generate import GeneratedHead, generate_head

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
SMALL_SEED = 3


@pytest.fixture(scope="session")
def small_cfg() -> HeadConfig:
    return from_dict(HeadConfig, load_yaml(CONFIG_DIR / "head_small.yaml"))


@pytest.fixture(scope="session")
def small_head(small_cfg: HeadConfig) -> GeneratedHead:
    return generate_head(small_cfg, SeedTree(SMALL_SEED))
