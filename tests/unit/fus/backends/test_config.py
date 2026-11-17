import pytest

from neurosim.common.config import ConfigError, from_dict, to_dict
from neurosim.fus.backends import BackendConfig, KWaveConfig, RayConfig


def test_defaults_and_nested_settings():
    cfg = BackendConfig()
    assert cfg.backend == "ray"
    assert cfg.ray == RayConfig()
    assert cfg.kwave == KWaveConfig()
    assert cfg.kwave.n_workers == 1
    assert cfg.kwave.min_points_per_wavelength == 3.0
    assert cfg.kwave.model_dispersion is False


def test_round_trip_through_plain_data():
    cfg = from_dict(BackendConfig, {"backend": "kwave", "kwave": {"cfl": 0.2, "dtype": "float32"}})
    assert cfg.kwave.cfl == 0.2 and cfg.kwave.dtype == "float32"
    assert from_dict(BackendConfig, to_dict(cfg)) == cfg


@pytest.mark.parametrize(
    "data",
    [
        {"backend": "fdtd"},
        {"kwave": {"cfl": 0.5}},
        {"kwave": {"cfl": 0.0}},
        {"kwave": {"min_points_per_wavelength": 1.5}},
        {"kwave": {"n_workers": 0}},
        {"kwave": {"dtype": "float16"}},
        {"ray": {"step_fraction": 1.5}},
        {"ray": {"unknown": 1}},
    ],
)
def test_invalid_values_are_refused(data):
    with pytest.raises(ConfigError):
        from_dict(BackendConfig, data)


def test_config_is_frozen():
    with pytest.raises(AttributeError):
        BackendConfig().backend = "kwave"
