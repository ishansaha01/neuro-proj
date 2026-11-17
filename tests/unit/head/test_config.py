import dataclasses
from pathlib import Path

import pytest

from neurosim.common.config import ConfigError, from_dict, load_yaml, to_dict
from neurosim.head.config import HeadConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@pytest.mark.parametrize("name", ["head_default.yaml", "head_small.yaml"])
def test_shipped_configs_load_and_round_trip(name):
    cfg = from_dict(HeadConfig, load_yaml(CONFIG_DIR / name))
    assert cfg.grid.region == "box"
    assert from_dict(HeadConfig, to_dict(cfg)) == cfg


def test_default_config_matches_the_roadmap_grid():
    cfg = from_dict(HeadConfig, load_yaml(CONFIG_DIR / "head_default.yaml"))
    assert cfg.grid.spacing == pytest.approx(0.5e-3)
    assert cfg.grid.box_size == pytest.approx((0.08, 0.08, 0.08))
    assert cfg == HeadConfig()


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        HeadConfig().grid.spacing = 1.0


@pytest.mark.parametrize(
    "data",
    [
        {"grid": {"spacing": 0.0}},
        {"grid": {"region": "slab"}},
        {"grid": {"voxel": 1.0e-3}},
        {"materials": {"alpha_power": 1.0}},
        {"materials": {"skull": {"c_bone": 1400.0}}},
        {"microstructure": {"porosity_r2": 0.0}},
        {"microstructure": {"porosity_diploe": 1.5}},
        {"anatomy": {"outer_table_max_fraction": 0.6, "inner_table_max_fraction": 0.5}},
        {"anatomy": {"temporal_window_direction": [0.0, 0.0, 0.0]}},
        {"cbv": {"target_direction": [0.0, 0.0, 0.0]}},
        {"cbv": {"speed_tissue": 0.05}},
        {"task": {"onsets": [0.0, 5.0], "block_duration": 10.0}},
    ],
)
def test_invalid_values_are_rejected(data):
    with pytest.raises(ConfigError):
        from_dict(HeadConfig, data)
