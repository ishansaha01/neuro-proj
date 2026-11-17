import dataclasses

import numpy as np
import pytest

from neurosim.common.grid import Grid2D
from neurosim.head.ablations import (
    apply_skull_effects,
    homogenize_skull,
    match_skull_impedance,
    remove_skull,
    zero_skull_absorption,
)
from neurosim.head.materials import skull_mask
from neurosim.head.plane import default_plane, sample_plane
from neurosim.head.types import PlaneMedium, SkullEffects, Tissue

GRID = Grid2D.from_extent((-6.0e-3, 6.0e-3), (0.0, 26.0e-3), 0.25e-3)
MAPS = ("sound_speed", "density", "attenuation")
TRANSFORMS = (homogenize_skull, zero_skull_absorption, match_skull_impedance, remove_skull)


@pytest.fixture(scope="module")
def medium(small_head, small_cfg) -> PlaneMedium:
    return sample_plane(small_head, default_plane(small_head, small_cfg), GRID)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_transform_is_pure_and_touches_only_skull_acoustics(transform, medium):
    before = {f.name: np.copy(getattr(medium, f.name)) for f in dataclasses.fields(medium)}
    result = transform(medium)
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(medium, name), values)
    outside = ~skull_mask(medium.labels)
    for name in MAPS:
        np.testing.assert_array_equal(getattr(result, name)[outside], before[name][outside])
    for name in ("labels", "porosity", "hidden_c", "cbv_baseline", "activation", "blood_velocity"):
        np.testing.assert_array_equal(getattr(result, name), before[name])
    assert result.grid == medium.grid and result.plane == medium.plane
    assert all(getattr(result, name).dtype == np.float32 for name in MAPS)


def test_homogenize_keeps_the_means_and_the_absorption(medium):
    result = homogenize_skull(medium)
    in_skull = skull_mask(medium.labels)
    assert np.ptp(result.sound_speed[in_skull]) == 0.0
    assert np.ptp(result.density[in_skull]) == 0.0
    assert result.sound_speed[in_skull][0] == pytest.approx(
        medium.sound_speed[in_skull].mean(), rel=1e-6
    )
    assert result.density[in_skull][0] == pytest.approx(medium.density[in_skull].mean(), rel=1e-6)
    np.testing.assert_array_equal(result.attenuation, medium.attenuation)


def test_zero_absorption_changes_only_the_attenuation(medium):
    result = zero_skull_absorption(medium)
    assert not result.attenuation[skull_mask(medium.labels)].any()
    np.testing.assert_array_equal(result.sound_speed, medium.sound_speed)
    np.testing.assert_array_equal(result.density, medium.density)


def test_matched_impedance_equals_the_scalp_impedance(medium):
    result = match_skull_impedance(medium)
    in_skull = skull_mask(medium.labels)
    scalp = medium.labels == Tissue.SCALP
    scalp_impedance = medium.density[scalp][0] * medium.sound_speed[scalp][0]
    impedance = result.density[in_skull] * result.sound_speed[in_skull]
    np.testing.assert_allclose(impedance, scalp_impedance, rtol=1e-6)
    np.testing.assert_array_equal(result.sound_speed, medium.sound_speed)


def test_removed_skull_has_grey_matter_properties(medium):
    result = remove_skull(medium)
    in_skull = skull_mask(medium.labels)
    grey = medium.labels == Tissue.GREY
    for name in MAPS:
        assert np.all(getattr(result, name)[in_skull] == getattr(medium, name)[grey][0]), name


def test_apply_skull_effects_composes_the_switches(medium):
    assert apply_skull_effects(medium, SkullEffects()) is medium
    in_skull = skull_mask(medium.labels)
    everything_off = apply_skull_effects(
        medium, SkullEffects(aberration=False, attenuation=False, reverberation=False)
    )
    assert np.ptp(everything_off.sound_speed[in_skull]) == 0.0
    assert not everything_off.attenuation[in_skull].any()
    scalp = medium.labels == Tissue.SCALP
    impedance = everything_off.density[in_skull] * everything_off.sound_speed[in_skull]
    np.testing.assert_allclose(
        impedance, medium.density[scalp][0] * medium.sound_speed[scalp][0], rtol=1e-6
    )
    only_attenuation_off = apply_skull_effects(medium, SkullEffects(attenuation=False))
    np.testing.assert_array_equal(only_attenuation_off.sound_speed, medium.sound_speed)
    np.testing.assert_array_equal(
        only_attenuation_off.attenuation, zero_skull_absorption(medium).attenuation
    )


def test_medium_without_the_needed_tissue_is_an_error(medium):
    no_skull = dataclasses.replace(medium, labels=np.full_like(medium.labels, Tissue.WHITE))
    for transform in TRANSFORMS:
        with pytest.raises(ValueError, match="no skull pixel"):
            transform(no_skull)
    skull_only = dataclasses.replace(medium, labels=np.full_like(medium.labels, Tissue.DIPLOE))
    with pytest.raises(ValueError, match="no SCALP pixel"):
        match_skull_impedance(skull_only)
    with pytest.raises(ValueError, match="no GREY pixel"):
        remove_skull(skull_only)
