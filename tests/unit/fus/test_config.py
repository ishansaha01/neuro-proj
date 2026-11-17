"""ImagingConfig: defaults, YAML files, derived objects and cross-field validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurosim.common.config import ConfigError, from_dict, load_yaml, to_dict
from neurosim.common.constants import C_REF
from neurosim.fus.config import (
    ImagingConfig,
    imaging_setup,
    pixel_grid_from_config,
    transducer_from_config,
)

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def test_default_yaml_equals_the_dataclass_defaults():
    assert from_dict(ImagingConfig, load_yaml(CONFIGS / "fus_default.yaml")) == ImagingConfig()


def test_roadmap_defaults():
    cfg = ImagingConfig()
    assert cfg.transducer.n_elements == 128
    assert cfg.transducer.center_frequency == 2.0e6
    assert len(cfg.sequence.angles_deg) == 9
    assert (min(cfg.sequence.angles_deg), max(cfg.sequence.angles_deg)) == (-10.0, 10.0)
    assert cfg.pixels.shape == (64, 64)


def test_small_yaml_loads_and_round_trips():
    cfg = from_dict(ImagingConfig, load_yaml(CONFIGS / "fus_small.yaml"))
    assert cfg.transducer.n_elements == 48
    assert from_dict(ImagingConfig, to_dict(cfg)) == cfg


def test_transducer_geometry_follows_the_wavelength():
    transducer = transducer_from_config(ImagingConfig())
    wavelength = C_REF / 2.0e6
    assert transducer.pitch == pytest.approx(0.5 * wavelength)
    assert transducer.element_width == pytest.approx(0.45 * wavelength)
    assert transducer.fractional_bandwidth == 0.6


def test_setup_grids(tiny_cfg):
    setup = imaging_setup(tiny_cfg)
    assert setup.freqs.shape == (28,)
    assert setup.roi.shape == (7, 7)
    assert setup.sequence.angles_deg == (-5.0, 0.0, 5.0)
    z, x = setup.pixel_grid.axes()
    # 8 x 8 pixels centred on the 6 mm x 6 mm ROI, as large as fits.
    assert setup.pixel_grid.spacing == pytest.approx(0.006 / 7)
    np.testing.assert_allclose(
        [x[0], x[-1], z[0], z[-1]], [-0.003, 0.003, 0.020, 0.026], atol=1e-12
    )


def _with(tiny_cfg, **sections):
    data = to_dict(tiny_cfg)
    for name, values in sections.items():
        data[name] = {**data[name], **values}
    return data


def test_explicit_pixel_spacing_must_fit(tiny_cfg):
    cfg = from_dict(ImagingConfig, _with(tiny_cfg, pixels={"spacing": 0.5e-3}))
    assert pixel_grid_from_config(cfg).spacing == 0.5e-3
    with pytest.raises(ConfigError, match="does not fit"):
        from_dict(ImagingConfig, _with(tiny_cfg, pixels={"spacing": 1.0e-3}))


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ({"frequencies": {"n_frequencies": 8}}, "range aliasing"),
        ({"doppler": {"n_discard": 6}}, "n_discard"),
        ({"roi": {"x_range": [0.003, -0.003]}}, "increasing"),
        ({"noise": {"tissue_motion_rms": -1.0}}, "tissue_motion_rms"),
        ({"beamforming": {"window": "hamming"}}, "window"),
    ],
)
def test_invalid_configs_raise(tiny_cfg, sections, message):
    with pytest.raises(ConfigError, match=message):
        from_dict(ImagingConfig, _with(tiny_cfg, **sections))


def test_unknown_key_raises(tiny_cfg):
    with pytest.raises(ConfigError, match="unknown key"):
        from_dict(ImagingConfig, {**to_dict(tiny_cfg), "prf": 5000.0})
