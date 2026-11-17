import pytest

from neurosim.common.config import ConfigError, from_dict, to_dict
from neurosim.fus.backends import BackendConfig, BentRayConfig


def test_bent_ray_settings_are_nested_with_defaults():
    cfg = BackendConfig()
    assert cfg.backend == "ray"  # the default backend is unchanged
    assert cfg.bent_ray == BentRayConfig()
    assert cfg.bent_ray.step_fraction == 1.0
    assert cfg.bent_ray.layer_normal_scale == 1.0e-3
    assert cfg.bent_ray.interface_contrast == 0.1
    assert cfg.bent_ray.grazing_taper_deg == 10.0


def test_bent_ray_settings_round_trip():
    data = {"backend": "bent_ray", "bent_ray": {"step_fraction": 0.5, "grazing_taper_deg": 5.0}}
    cfg = from_dict(BackendConfig, data)
    assert cfg.backend == "bent_ray"
    assert cfg.bent_ray.step_fraction == 0.5 and cfg.bent_ray.grazing_taper_deg == 5.0
    assert from_dict(BackendConfig, to_dict(cfg)) == cfg


@pytest.mark.parametrize(
    "data",
    [
        {"bent_ray": {"step_fraction": 0.0}},
        {"bent_ray": {"step_fraction": 1.5}},
        {"bent_ray": {"sweep_tolerance": 0.0}},
        {"bent_ray": {"max_sweeps": 3}},
        {"bent_ray": {"grazing_taper_deg": 0.0}},
        {"bent_ray": {"layer_normal_scale": -1.0e-3}},
        {"bent_ray": {"interface_contrast": 0.0}},
        {"bent_ray": {"max_batch_bytes": 0}},
        {"bent_ray": {"unknown": 1}},
    ],
)
def test_invalid_bent_ray_values_are_refused(data):
    with pytest.raises(ConfigError):
        from_dict(BackendConfig, data)
