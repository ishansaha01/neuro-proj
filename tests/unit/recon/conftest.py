import dataclasses
from pathlib import Path

import numpy as np
import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.seeding import SeedTree
from neurosim.recon.cbv_images import sample_cbv_images
from neurosim.recon.train import PriorConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
DATA_SEED = 5


@pytest.fixture(scope="session")
def small_cfg() -> PriorConfig:
    """configs/prior_small.yaml with fewer heads, so the data set takes about a second."""
    cfg = from_dict(PriorConfig, load_yaml(CONFIG_DIR / "prior_small.yaml"))
    return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, n_heads=6, heldout_every=3))


@pytest.fixture(scope="session")
def data_seed() -> int:
    """Root seed of the `small_data` fixture."""
    return DATA_SEED


@pytest.fixture(scope="session")
def small_data(small_cfg: PriorConfig) -> dict[str, np.ndarray]:
    return sample_cbv_images(small_cfg.data, SeedTree(DATA_SEED))
