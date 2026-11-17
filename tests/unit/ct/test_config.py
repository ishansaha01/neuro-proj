"""CtConfig: schema, defaults and validation."""

from __future__ import annotations

import dataclasses
import math

import pytest

from neurosim.common.config import ConfigError, from_dict, load_yaml, to_dict
from neurosim.ct.config import CtConfig, FanBeamGeometry


def test_default_yaml_equals_the_dataclass_defaults(config_dir):
    assert from_dict(CtConfig, load_yaml(config_dir / "ct_default.yaml")) == CtConfig()


def test_default_yaml_lists_every_scan_setting(config_dir):
    listed = set(load_yaml(config_dir / "ct_default.yaml"))
    assert listed == set(to_dict(CtConfig())) - {"hu_mapping"}


def test_small_config_loads_and_is_small(small_cfg):
    assert small_cfg.geometry.n_views * small_cfg.geometry.n_detectors < 100_000
    assert small_cfg.kvp == CtConfig().kvp


def test_config_is_frozen_and_hashable(small_cfg):
    with pytest.raises(dataclasses.FrozenInstanceError):
        small_cfg.kvp = 80.0
    assert hash(small_cfg) == hash(dataclasses.replace(small_cfg))


def test_config_round_trips_through_plain_data(small_cfg):
    assert from_dict(CtConfig, to_dict(small_cfg)) == small_cfg


@pytest.mark.parametrize(
    "data",
    [
        {"kernel": "sharp"},
        {"kvp": 10.0},
        {"mas": 0.0},
        {"unknown_key": 1},
        {"geometry": {"n_views": 2}},
        {"geometry": {"source_detector_distance": 0.5}},
        {"monoenergetic_kev": 150.0},
        {"slab_z_from_center": [0.01, -0.01]},
        {"calibration_phantom_diameter": 0.6},
        {"hu_mapping": {"c_bone_range": [3100.0, 2700.0]}},
    ],
)
def test_invalid_values_are_rejected(data):
    with pytest.raises(ConfigError):
        from_dict(CtConfig, data)


def test_geometry_derived_quantities():
    geom = FanBeamGeometry()
    assert geom.angular_pitch == pytest.approx(geom.detector_pitch / geom.source_detector_distance)
    assert math.degrees(2.0 * geom.half_fan_angle) == pytest.approx(49.9, abs=0.1)
    assert geom.fov_radius == pytest.approx(0.251, abs=0.001)
    assert geom.detector_area == pytest.approx(geom.detector_pitch * geom.detector_height)
    fan = geom.fan_angles()
    assert fan.shape == (geom.n_detectors,)
    assert fan[0] == pytest.approx(-fan[-1])
    views = geom.view_angles()
    assert views.shape == (geom.n_views,)
    assert views[0] == 0.0 and views[-1] < 2.0 * math.pi
