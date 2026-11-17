"""HU -> acoustics map: nominal constants, sampling and the map itself."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.common.seeding import SeedTree
from neurosim.common.units import db_cm_mhz_to_np_m, np_m_to_db_cm
from neurosim.ct.acoustics import (
    BONE_THRESHOLD_HU,
    SOFT_TISSUE_ALPHA0,
    hu_to_acoustic,
    nominal_hu_mapping,
    sample_hu_mapping,
)
from neurosim.ct.calibration import effective_hu
from neurosim.ct.types import HuMapping


def test_nominal_mapping_uses_aubry_2003_constants(small_cfg):
    mapping = nominal_hu_mapping(small_cfg)
    assert isinstance(mapping, HuMapping)
    assert (mapping.c_water, mapping.c_bone) == (1500.0, 2900.0)
    assert (mapping.rho_water, mapping.rho_bone) == (1000.0, 2100.0)
    assert mapping.alpha_exponent == 0.5
    bone = BASIS_MATERIALS.index("cortical_bone")
    assert mapping.hu_bone == pytest.approx(effective_hu(small_cfg)[bone])


def test_nominal_absorption_reproduces_the_published_values_at_their_frequency(small_cfg):
    mapping = nominal_hu_mapping(small_cfg)
    hu_cfg = small_cfg.hu_mapping
    at_frequency = np_m_to_db_cm(
        db_cm_mhz_to_np_m(
            np.array([mapping.alpha_min, mapping.alpha_max]),
            hu_cfg.absorption_frequency,
            hu_cfg.alpha_power,
        )
    )
    # Aubry 2003: 0.2 dB/mm and 8 dB/mm.
    np.testing.assert_allclose(at_frequency, [2.0, 80.0])


def test_configured_hu_bone_overrides_the_scanner_model(small_cfg):
    hu_cfg = dataclasses.replace(small_cfg.hu_mapping, hu_bone=2000.0)
    assert nominal_hu_mapping(dataclasses.replace(small_cfg, hu_mapping=hu_cfg)).hu_bone == 2000.0


def test_map_end_points_and_units(small_cfg):
    mapping = nominal_hu_mapping(small_cfg)
    hu = np.array([-1000.0, 0.0, 0.5 * mapping.hu_bone, mapping.hu_bone, 2.0 * mapping.hu_bone])
    sound_speed, density, attenuation = hu_to_acoustic(hu, mapping)
    for values in (sound_speed, density, attenuation):
        assert values.shape == hu.shape and values.dtype == np.float32
    np.testing.assert_allclose(sound_speed, [1500.0, 1500.0, 2200.0, 2900.0, 2900.0])
    np.testing.assert_allclose(density, [1000.0, 1000.0, 1550.0, 2100.0, 2100.0])
    assert attenuation[0] == attenuation[1] == SOFT_TISSUE_ALPHA0
    assert attenuation[3] == attenuation[4] == pytest.approx(mapping.alpha_min)
    half = mapping.alpha_min + (mapping.alpha_max - mapping.alpha_min) * 0.5**0.5
    assert attenuation[2] == pytest.approx(half, rel=1e-6)


def test_absorption_law_applies_to_bone_voxels_only(small_cfg):
    mapping = nominal_hu_mapping(small_cfg)
    below, above = hu_to_acoustic(
        np.array([BONE_THRESHOLD_HU - 1.0, BONE_THRESHOLD_HU + 1.0]), mapping
    )[2]
    assert below == SOFT_TISSUE_ALPHA0 and above > mapping.alpha_min


def test_sampled_mapping_is_deterministic_and_stream_addressed(small_cfg):
    first = sample_hu_mapping(SeedTree(3), small_cfg, 0)
    assert sample_hu_mapping(SeedTree(3), small_cfg, 0) == first
    assert sample_hu_mapping(SeedTree(3), small_cfg, 1) != first
    assert sample_hu_mapping(SeedTree(4), small_cfg, 0) != first
    assert sample_hu_mapping(SeedTree(3), small_cfg) != first


def test_sampled_constants_stay_in_their_ranges(small_cfg):
    hu_cfg = small_cfg.hu_mapping
    nominal = nominal_hu_mapping(small_cfg)
    samples = [sample_hu_mapping(SeedTree(11), small_cfg, k) for k in range(200)]

    def spread(name: str) -> tuple[float, float]:
        values = [getattr(sample, name) for sample in samples]
        return min(values), max(values)

    low, high = spread("hu_bone")
    assert nominal.hu_bone * hu_cfg.hu_bone_relative_range[0] <= low < high
    assert high <= nominal.hu_bone * hu_cfg.hu_bone_relative_range[1]
    for name, bounds in (
        ("c_water", hu_cfg.c_water_range),
        ("c_bone", hu_cfg.c_bone_range),
        ("rho_bone", hu_cfg.rho_bone_range),
        ("alpha_exponent", hu_cfg.absorption_exponent_range),
    ):
        low, high = spread(name)
        assert bounds[0] <= low < high <= bounds[1]
        # 200 uniform draws cover most of the interval.
        assert high - low > 0.8 * (bounds[1] - bounds[0])
    assert spread("rho_water") == (hu_cfg.rho_water, hu_cfg.rho_water)
    low, high = spread("alpha_max")
    assert low >= 13.0 / 1.5**hu_cfg.alpha_power - 1e-9
    assert high <= nominal.alpha_max + 1e-9
    assert all(sample.alpha_min < sample.alpha_max for sample in samples)
