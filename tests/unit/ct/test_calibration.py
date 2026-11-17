"""Water correction and HU calibration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.ct.attenuation import basis_attenuation
from neurosim.ct.calibration import (
    apply_water_correction,
    effective_hu,
    hu_calibration,
    process_signal,
    water_correction_table,
)
from neurosim.ct.detector import detector_signal, flat_field_counts

WATER = BASIS_MATERIALS.index("water")


def _water_signal(cfg, thickness: np.ndarray) -> np.ndarray:
    energies, counts = flat_field_counts(cfg)
    attenuation = basis_attenuation(energies)[WATER : WATER + 1]
    return detector_signal(thickness[np.newaxis], attenuation, counts, energies)


def test_table_is_monotonic_and_starts_at_zero(small_cfg):
    table = water_correction_table(small_cfg)
    assert table.thickness[0] == 0.0 and table.projection[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(table.projection) > 0.0)
    assert table.mu_reference == pytest.approx(20.0, rel=0.1)  # water near 65 keV [1/m]


def test_correction_makes_water_projections_linear(small_cfg):
    thickness = np.array([0.0133, 0.0871, 0.1602, 0.2468])
    corrected = process_signal(_water_signal(small_cfg, thickness), small_cfg)
    mu_reference = water_correction_table(small_cfg).mu_reference
    np.testing.assert_allclose(corrected, mu_reference * thickness, rtol=1e-4)


def test_correction_leaves_the_phantom_diameter_unchanged(small_cfg):
    uncorrected_cfg = dataclasses.replace(small_cfg, water_correction=False)
    diameter = np.array([small_cfg.calibration_phantom_diameter])
    np.testing.assert_allclose(
        process_signal(_water_signal(small_cfg, diameter), small_cfg),
        process_signal(_water_signal(small_cfg, diameter), uncorrected_cfg),
        rtol=1e-4,
    )


def test_correction_extrapolates_beyond_the_table(small_cfg):
    table = water_correction_table(small_cfg)
    top = table.projection[-1]
    values = apply_water_correction(np.array([top, top + 1.0, top + 2.0]), table)
    assert np.all(np.isfinite(values))
    assert values[2] - values[1] == pytest.approx(values[1] - values[0])
    assert values[1] > values[0]


def test_process_signal_floors_starved_rays(small_cfg):
    energies, counts = flat_field_counts(small_cfg)
    projection = process_signal(np.array([0.0, -5.0, counts @ energies]), small_cfg)
    assert np.all(np.isfinite(projection))
    assert projection[0] == projection[1] > 10.0
    assert projection[2] == pytest.approx(0.0, abs=1e-9)


def test_hu_calibration_is_cached_and_physical(small_cfg):
    calibration = hu_calibration(small_cfg)
    assert hu_calibration(dataclasses.replace(small_cfg)) is calibration
    assert calibration.mu_water == pytest.approx(
        water_correction_table(small_cfg).mu_reference, rel=0.01
    )
    assert abs(calibration.mu_air) < 0.01 * calibration.mu_water


def test_effective_hu_fixed_points_and_order(small_cfg):
    air, water, soft_tissue, bone = effective_hu(small_cfg)
    assert air == pytest.approx(-1000.0) and water == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < soft_tissue < 100.0 < bone


def test_effective_bone_hu_falls_with_tube_voltage(small_cfg):
    bone = BASIS_MATERIALS.index("cortical_bone")
    low = effective_hu(dataclasses.replace(small_cfg, kvp=80.0))[bone]
    high = effective_hu(dataclasses.replace(small_cfg, kvp=140.0))[bone]
    assert low > effective_hu(small_cfg)[bone] > high
