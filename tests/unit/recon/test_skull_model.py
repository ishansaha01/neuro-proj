"""The CT skull-model family (`recon.skull_model`)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from neurosim.common.grid import Grid2D
from neurosim.ct.acoustics import hu_to_acoustic
from neurosim.recon.config import SkullModelConfig
from neurosim.recon.skull_model import (
    SkullParameters,
    build_family,
    cosine_modes,
    delay_only,
    fermat_bank,
    fermat_travel_time,
    hu_knots,
    knot_slowness,
    lateral_slowness_map,
    model_bank,
    model_medium,
    path_bins,
)

from .scene_fixtures import tiny_ct_family, tiny_ct_run  # noqa: F401


def test_hu_knots_cover_the_range_and_repeat_the_threshold():
    knots = hu_knots(1234.0, 300.0, 50.0)
    assert knots[0] == 0.0 and knots[-1] == 1234.0
    assert np.sum(knots == 300.0) == 2
    assert np.all(np.diff(knots) <= 50.0 + 1e-9)


def test_path_bins_interpolate_any_function_of_hu():
    grid = Grid2D(shape=(3, 4), spacing=1e-3, origin=(0.0, 0.0))
    hu = np.array(
        [[-200.0, 0.0, 120.0, 299.0], [300.0, 310.0, 990.0, 1500.0], [25.0, 75.0, 650.0, 20.0]]
    )
    knots = hu_knots(1500.0, 300.0, 50.0)
    lateral_knots = np.linspace(0.0, 3e-3, 3)
    bins = path_bins(hu, grid, knots, 300.0, lateral_knots)
    n_hu = knots.shape[0]
    # The HU entries weigh 1 in total; a linear function of the clipped HU is reproduced.
    np.testing.assert_allclose(bins.weight[:, :2].sum(axis=1), 1.0)
    values = 2.0 + 0.01 * knots
    interpolated = (bins.weight[:, :2] * values[bins.index[:, :2]]).sum(axis=1)
    np.testing.assert_allclose(interpolated, 2.0 + 0.01 * np.clip(hu.reshape(-1), 0.0, None))
    # Soft pixels use the first knot group, bone pixels the second: a step at the threshold.
    step = (knots >= 300.0) & (np.arange(n_hu) >= np.argmax(knots == 300.0) + 1)
    in_bone = (bins.weight[:, :2] * step[bins.index[:, :2]]).sum(axis=1)
    np.testing.assert_allclose(in_bone, (hu.reshape(-1) >= 300.0).astype(float))
    # Only bone pixels weigh on the lateral knots, and there they weigh 1.
    np.testing.assert_allclose(bins.weight[:, 2:].sum(axis=1), (hu.reshape(-1) >= 300.0) * 1.0)
    assert bins.index[:, 2:].min() >= n_hu and bins.n_bins == n_hu + 3


def test_cosine_modes_are_orthogonal_with_unit_rms():
    modes = cosine_modes(12, 5)
    np.testing.assert_allclose(modes.T @ modes / 12, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(modes.mean(axis=0), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="lateral knots"):
        cosine_modes(4, 4)


def test_knot_slowness_is_the_hu_map(tiny_ct_family):
    family = tiny_ct_family
    nominal = family.nominal
    speed, _, _ = hu_to_acoustic(family.hu_knots, nominal, family.bone_threshold_hu)
    slowness = knot_slowness(
        torch.from_numpy(family.hu_knots),
        torch.tensor(nominal.c_bone),
        torch.tensor(nominal.c_water),
        torch.tensor(nominal.hu_bone),
    )
    np.testing.assert_allclose(slowness.numpy(), 1.0 / speed, rtol=1e-6)


def test_family_shapes_and_nominal_member(tiny_ct_run, tiny_ct_family):
    scene, family = tiny_ct_run[0], tiny_ct_family
    n_elements, n_nodes = 16, scene.bank_ct.roi.shape[0] * scene.bank_ct.roi.shape[1]
    assert family.hu_path.shape == (n_elements, n_nodes, family.hu_knots.shape[0])
    assert family.lateral_path.shape == (n_elements, n_nodes, family.lateral_knots.shape[0])
    assert family.n_lateral_modes == 3
    # The histogram holds the whole traced path, which is at least the chord.
    assert np.all(family.hu_path.sum(axis=2) >= family.path_length * (1 - 1e-5))
    assert np.all(family.lateral_path.sum(axis=2) <= family.hu_path.sum(axis=2) * (1 + 1e-5))
    nominal = family.nominal_parameters()
    np.testing.assert_array_equal(fermat_travel_time(family, nominal), family.travel_time)
    # The nominal member's fixed-path bank is the scene's CT bank.
    np.testing.assert_allclose(fermat_bank(family, nominal).B, scene.bank_ct.B, atol=1e-6)


def test_travel_time_follows_the_parameters(tiny_ct_family):
    family = tiny_ct_family
    nominal = family.nominal_parameters()
    slower = dataclasses.replace(nominal, c_bone=nominal.c_bone - 150.0)
    in_bone = family.lateral_path.sum(axis=2) > 1e-4
    change = fermat_travel_time(family, slower) - family.travel_time
    assert np.all(change[in_bone] > 0.0)
    # A lateral mode changes the travel time linearly in its coefficient.
    lateral = np.array([1.0e-5, 0.0, 0.0])
    once = fermat_travel_time(family, dataclasses.replace(nominal, lateral=lateral))
    twice = fermat_travel_time(family, dataclasses.replace(nominal, lateral=2 * lateral))
    np.testing.assert_allclose(twice - family.travel_time, 2 * (once - family.travel_time))


def test_model_medium_and_bank_of_a_member(tiny_ct_family):
    family = tiny_ct_family
    nominal = family.nominal_parameters()
    member = SkullParameters(
        c_bone=nominal.c_bone + 80.0,
        c_water=nominal.c_water + 10.0,
        hu_bone=nominal.hu_bone * 1.05,
        lateral=np.array([5.0e-6, -3.0e-6, 2.0e-6]),
        common_amplitude=0.5,
    )
    medium = model_medium(family, member)
    is_bone = family.ct_hu >= family.bone_threshold_hu
    without = model_medium(family, dataclasses.replace(member, lateral=np.zeros(3)))
    extra = 1.0 / medium.sound_speed.astype(np.float64) - 1.0 / without.sound_speed
    # The medium stores its sound speed in float32: slowness is kept to about 1e-7 relative.
    np.testing.assert_allclose(extra, lateral_slowness_map(family, member.lateral), atol=1e-10)
    np.testing.assert_allclose(extra[~is_bone], 0.0, atol=1e-10)
    assert np.abs(extra[is_bone]).max() > 1e-6
    bank = model_bank(family, member)
    full = model_bank(family, dataclasses.replace(member, common_amplitude=1.0))
    np.testing.assert_allclose(bank.B, 0.5 * full.B, rtol=1e-6)
    # First-order agreement of the fixed-path bank with the re-traced one where both reach.
    fixed = fermat_bank(family, member)
    both = (np.abs(full.B) > 0) & (np.abs(fixed.B) > 0)
    assert both.mean() > 0.5


def test_delay_only_keeps_phases_and_support(tiny_ct_run):
    bank = tiny_ct_run[0].bank_ct
    unit = delay_only(bank)
    reached = np.abs(bank.B) > 0
    np.testing.assert_allclose(np.abs(unit.B[reached]), 1.0, rtol=1e-6)
    assert np.all(unit.B[~reached] == 0)
    np.testing.assert_allclose(
        np.angle(unit.B[reached] * np.conj(bank.B[reached])), 0.0, atol=1e-5
    )


def test_family_needs_a_ct_and_the_bent_ray_backend(tiny_ct_run):
    scene = tiny_ct_run[0]
    with pytest.raises(ValueError, match="no CT image"):
        build_family(dataclasses.replace(scene, ct_hu=None), SkullModelConfig(n_lateral_modes=3))
    meta = {**scene.meta, "model_backend": {**scene.meta["model_backend"], "backend": "ray"}}
    with pytest.raises(ValueError, match="bent_ray"):
        build_family(dataclasses.replace(scene, meta=meta), SkullModelConfig(n_lateral_modes=3))
