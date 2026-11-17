"""Shared fixtures of the data unit tests: the small run and a still cheaper one for sweeps."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.scene import SceneExtras, build_scene_with_extras
from neurosim.data.trials import simulate_trials
from neurosim.data.types import Scene, TrialSet

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
SMALL_SEED = 3


@pytest.fixture(scope="session")
def small_cfg() -> RunConfig:
    return from_dict(RunConfig, load_yaml(CONFIG_DIR / "run_small.yaml"))


@pytest.fixture(scope="session")
def small_run(small_cfg: RunConfig) -> tuple[Scene, SceneExtras, TrialSet]:
    """The small run of seed `SMALL_SEED`, computed once per session (about 3 s)."""
    tree = SeedTree(SMALL_SEED)
    scene, extras = build_scene_with_extras(small_cfg, tree)
    return scene, extras, simulate_trials(scene, small_cfg, tree, extras)


@pytest.fixture(scope="session")
def quick_cfg(small_cfg: RunConfig) -> RunConfig:
    """The small run with one sample per pixel and four images: about 1.5 s per run."""
    return replace(
        small_cfg, source_refinement=1, trials=replace(small_cfg.trials, n_images=4)
    )


@pytest.fixture(scope="session")
def quick_scene(quick_cfg: RunConfig) -> tuple[Scene, SceneExtras]:
    """The scene of the quick run of seed `SMALL_SEED`, built once per session."""
    return build_scene_with_extras(quick_cfg, SeedTree(SMALL_SEED))
