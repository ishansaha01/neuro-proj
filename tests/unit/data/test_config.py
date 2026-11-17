"""data.config: RunConfig validation and the placement of the fine medium grid."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import yaml

from neurosim.common.config import ConfigError, dump_yaml, from_dict, load_yaml
from neurosim.data.config import RunConfig, medium_grid, trial_fidelity
from neurosim.fus.backends.fullwave_plan import element_footprints
from neurosim.fus.config import imaging_setup

from .conftest import CONFIG_DIR


def test_config_round_trips_through_yaml(small_cfg):
    assert from_dict(RunConfig, load_yaml(CONFIG_DIR / "run_small.yaml")) == small_cfg
    assert from_dict(RunConfig, yaml.safe_load(dump_yaml(small_cfg))) == small_cfg


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="medium_spacin"):
        from_dict(RunConfig, {"medium_spacin": 1.0e-4})


def test_absorption_exponents_of_head_and_ct_must_agree(small_cfg):
    hu_mapping = replace(small_cfg.ct.hu_mapping, alpha_power=1.3)
    with pytest.raises(ValueError, match="alpha_power"):
        replace(small_cfg, ct=replace(small_cfg.ct, hu_mapping=hu_mapping))


def test_image_interval_must_hold_one_ensemble(small_cfg):
    # 40 frames at 500 Hz take 0.08 s.
    replace(small_cfg, trials=replace(small_cfg.trials, image_interval=0.08))
    with pytest.raises(ValueError, match="Doppler ensemble"):
        replace(small_cfg, trials=replace(small_cfg.trials, image_interval=0.07))


def test_medium_grid_must_not_be_coarser_than_the_roi(small_cfg):
    with pytest.raises(ValueError, match="medium_spacing"):
        replace(small_cfg, medium_spacing=2.0e-3)


def test_trial_fidelity_overrides_the_imaging_fidelity(small_cfg):
    assert trial_fidelity(small_cfg) == "statistical"
    override = replace(small_cfg, trials=replace(small_cfg.trials, fidelity="scatterer"))
    assert override.imaging.fidelity == "statistical" and trial_fidelity(override) == "scatterer"


def test_medium_grid_covers_array_face_aperture_and_roi(small_cfg):
    grid = medium_grid(small_cfg)
    setup = imaging_setup(small_cfg.imaging)
    z, x = grid.axes()
    roi_z, roi_x = setup.roi.axes()
    margin = small_cfg.medium_margin
    assert z[0] == 0.0  # the array face is the first row
    assert x[grid.shape[1] // 2] == pytest.approx(0.0, abs=1e-12) and grid.shape[1] % 2 == 1
    np.testing.assert_allclose(x, -x[::-1], atol=1e-12)
    rounding = 1e-12  # [m]
    half_width = max(0.5 * setup.transducer.aperture, roi_x[-1]) + margin
    assert half_width - rounding <= x[-1] < half_width + grid.spacing
    assert roi_z[-1] + margin - rounding <= z[-1] < roi_z[-1] + margin + grid.spacing


def test_medium_grid_follows_the_wider_of_aperture_and_roi(small_cfg):
    # 16 elements span 6.2 mm, less than the 12 mm ROI: the ROI edge plus the margin decides.
    narrow = replace(small_cfg.imaging.transducer, n_elements=16)
    grid = medium_grid(replace(small_cfg, imaging=replace(small_cfg.imaging, transducer=narrow)))
    assert grid.axes()[1][-1] == pytest.approx(0.007)


@pytest.mark.parametrize("n_elements", [47, 48])
def test_medium_grid_meets_the_full_wave_array_placement(small_cfg, n_elements):
    """z = 0 on a row and element centres on nodes or half-way, once the pitch is whole cells."""
    transducer = replace(small_cfg.imaging.transducer, n_elements=n_elements)
    imaging = replace(small_cfg.imaging, transducer=transducer)
    pitch = imaging_setup(imaging).transducer.pitch
    cfg = replace(small_cfg, imaging=imaging, medium_spacing=pitch / 3)
    array_row, columns, pitch_cells = element_footprints(
        medium_grid(cfg), imaging_setup(imaging).transducer
    )
    assert array_row == 0 and pitch_cells == 3 and columns.shape[0] == n_elements
