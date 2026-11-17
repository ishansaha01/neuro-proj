import dataclasses

import numpy as np
import pytest

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.common.units import db_to_np
from neurosim.fus.backends import (
    BackendConfig,
    BentRayConfig,
    bent_ray_travel_times,
    compute_bank,
    estimate_cost,
    ray_travel_times,
)
from neurosim.fus.backends.bent_ray import SOLVER, grazing_window, trace_bent_rays
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium
from neurosim.fus.types import Transducer

FREQS = np.linspace(0.4e6, 0.6e6, 5)
BENT = BackendConfig(backend="bent_ray")
RAY = BackendConfig(backend="ray")


@pytest.fixture
def single() -> Transducer:
    """One element at x = 0."""
    return Transducer(1, 1.2e-3, 0.3e-3, 0.5e6, 0.8)


def test_reference_medium_gives_a_unit_bank(grid, soft, transducer, roi):
    bank = compute_bank(homogeneous_medium(grid, soft, 1.5), transducer, roi, FREQS, BENT)
    assert bank.backend == "bent_ray" and bank.static_fmc is None and bank.c_ref == C_REF
    assert bank.B.shape == (5, 3, 3, 5) and bank.B.dtype == np.complex64
    np.testing.assert_allclose(bank.B, 1.0, atol=1e-5)


def test_bank_of_a_homogeneous_medium_is_the_far_field_greens_ratio(grid, transducer, roi):
    slow = Material(1400.0, 1300.0)
    bank = compute_bank(homogeneous_medium(grid, slow, 1.5), transducer, roi, FREQS, BENT)
    points = roi.points()
    distance = np.hypot(
        points[None, ..., 0] - transducer.element_x()[:, None, None], points[None, ..., 1]
    )
    phase = 2.0 * np.pi * FREQS[:, None, None, None] * distance * (1.0 / 1400.0 - 1.0 / C_REF)
    np.testing.assert_allclose(bank.B, np.sqrt(1400.0 / C_REF) * np.exp(1j * phase), rtol=1e-4)


def test_agrees_with_the_ray_backend_at_normal_incidence(grid, soft, bone, single, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.3)
    roi = Grid2D.from_extent((0.0, 0.0), (6.0e-3, 9.0e-3), 3.0e-3)
    bent = compute_bank(medium, single, roi, FREQS, BENT)
    ray = compute_bank(medium, single, roi, FREQS, RAY)
    assert np.abs(ray.B).max() < 0.9  # interfaces and absorption both act
    # Both use the midpoint rule with a step of one cell, sampled from opposite ends of
    # the ray: each is within spacing / 8 of path per interface, in delay and absorption.
    np.testing.assert_allclose(np.abs(bent.B), np.abs(ray.B), rtol=5e-3)
    delay_bound = 2 * spacing / 8 * (1.0 / soft.sound_speed - 1.0 / bone.sound_speed)
    phase_bound = 2.0 * np.pi * FREQS[:, None, None, None] * delay_bound
    assert np.all(np.abs(np.angle(bent.B / ray.B)) < phase_bound)


def test_refracted_arrival_is_earlier_and_weaker_than_the_straight_ray(soft, spacing):
    grid = Grid2D.from_extent((-3.0e-3, 12.0e-3), (0.0, 15.0e-3), spacing)
    medium = slab_medium(
        grid, soft, Material(2800.0, 1900.0), 9.5 * spacing, 12 * spacing, alpha_power=1.5
    )
    transducer = Transducer(1, 1.2e-3, spacing, 0.5e6, 0.8)
    point = np.array([[9.0e-3, 15.0e-3]])
    bent_time = bent_ray_travel_times(medium, transducer, point)
    straight_time = ray_travel_times(medium, transducer, point)
    assert bent_time.shape == (1, 1) and bent_time.dtype == np.float64
    assert bent_time[0, 0] < straight_time[0, 0] - 100.0e-9
    roi = Grid2D.from_extent((9.0e-3, 9.0e-3), (15.0e-3, 15.0e-3), spacing)
    bent = compute_bank(medium, transducer, roi, FREQS, BENT)
    ray = compute_bank(medium, transducer, roi, FREQS, RAY)
    assert np.all(np.abs(bent.B) < 0.8 * np.abs(ray.B))


def test_absorption_follows_the_power_law_along_the_path(grid, soft, single, spacing):
    lossy = Material(C_REF, soft.density, attenuation=5.0)
    thickness = 10 * spacing
    medium = slab_medium(grid, soft, lossy, 9.5 * spacing, thickness, alpha_power=1.3)
    roi = Grid2D.from_extent((0.0, 0.0), (9.0e-3, 9.0e-3), 3.0e-3)
    bank = compute_bank(medium, single, roi, FREQS, BENT)
    loss_db = 5.0 * (FREQS / 1.0e6) ** 1.3 * (thickness / 1.0e-2)
    np.testing.assert_allclose(np.abs(bank.B[:, 0, 0, 0]), np.exp(-db_to_np(loss_db)), rtol=2e-3)
    np.testing.assert_allclose(np.angle(bank.B), 0.0, atol=1e-5)


def test_smooth_impedance_ramp_conserves_energy(soft, single):
    fine = Grid2D.from_extent((-1.5e-3, 1.5e-3), (0.0, 12.0e-3), 0.05e-3)
    medium = homogeneous_medium(fine, soft, alpha_power=1.5)
    z = fine.axes()[0]
    ramp = np.interp(z, [3.0e-3, 9.0e-3], [1000.0, 3000.0]).astype(np.float32)  # density x 3
    medium = dataclasses.replace(medium, density=np.repeat(ramp[:, None], fine.shape[1], axis=1))
    integrals = trace_bent_rays(medium, single, np.array([[0.0, 11.0e-3]]), BentRayConfig())
    assert 0.995 < integrals.transmission[0, 0] <= 1.0
    assert integrals.spreading[0, 0] == pytest.approx(np.sqrt(3.0), rel=1e-3)


def test_reciprocity_with_the_density_ratio(spacing):
    """rho(r_e) G(r | r_e) = rho(r) G(r_e | r): the ends are swapped by flipping the medium."""
    grid = Grid2D.from_extent((-6.0e-3, 6.0e-3), (0.0, 12.0e-3), spacing)
    top, bottom = Material(1500.0, 1000.0), Material(1650.0, 1200.0)
    bone = Material(2800.0, 1900.0, attenuation=4.0)
    medium = slab_medium(grid, top, bone, 9.5 * spacing, 8 * spacing, alpha_power=1.5)
    rows = np.arange(grid.shape[0])[:, None] >= 18
    below = {
        name: np.where(rows & (medium.sound_speed == top.sound_speed), value, getattr(medium, name))
        for name, value in (("sound_speed", bottom.sound_speed), ("density", bottom.density))
    }
    medium = dataclasses.replace(medium, **{k: v.astype(np.float32) for k, v in below.items()})
    flipped = dataclasses.replace(
        medium,
        sound_speed=medium.sound_speed[::-1].copy(),
        density=medium.density[::-1].copy(),
        attenuation=medium.attenuation[::-1].copy(),
    )
    transducer = Transducer(2, 20 * spacing, spacing, 0.5e6, 0.8)  # elements at -3 mm and +3 mm
    roi = Grid2D.from_extent((-3.0e-3, 3.0e-3), (12.0e-3, 12.0e-3), 6.0e-3)
    forward = compute_bank(medium, transducer, roi, FREQS, BENT).B[:, 1, 0, 0]  # +3 mm -> -3 mm
    backward = compute_bank(flipped, transducer, roi, FREQS, BENT).B[:, 0, 0, 1]  # -3 mm -> +3 mm
    np.testing.assert_allclose(backward / forward, top.density / bottom.density, rtol=0.01)


def test_grazing_window():
    cosines = np.sin(np.deg2rad([0.0, 2.5, 5.0, 10.0, 30.0, 90.0]))
    window = grazing_window(cosines, 10.0)
    np.testing.assert_allclose(window, [0.0, 0.15625, 0.5, 1.0, 1.0, 1.0], atol=1e-12)


def test_head_wave_first_arrival_gets_no_amplitude(spacing):
    grid = Grid2D.from_extent((-3.0e-3, 15.0e-3), (0.0, 6.0e-3), spacing)
    slow, fast = Material(1500.0, 1000.0), Material(3000.0, 1900.0)
    medium = slab_medium(grid, slow, fast, 9.5 * spacing, 11 * spacing, alpha_power=1.5)
    transducer = Transducer(1, 1.2e-3, spacing, 0.5e6, 0.8)
    # Far along the interface and above it: the head wave arrives before the direct wave.
    roi = Grid2D.from_extent((12.0e-3, 12.0e-3), (1.5e-3, 1.5e-3), spacing)
    bank = compute_bank(medium, transducer, roi, FREQS, BENT)
    assert np.all(np.abs(bank.B) < 0.02)
    assert bank.meta["fraction_post_critical"] == 1.0
    near = Grid2D.from_extent((0.9e-3, 0.9e-3), (1.5e-3, 1.5e-3), spacing)
    direct = compute_bank(medium, transducer, near, FREQS, BENT)
    assert np.all(np.abs(direct.B) > 0.9) and direct.meta["fraction_post_critical"] == 0.0


def test_meta_is_deterministic_and_complete(grid, soft, bone, transducer, roi, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    first = compute_bank(medium, transducer, roi, FREQS, BENT)
    second = compute_bank(medium, transducer, roi, FREQS, BENT)
    np.testing.assert_array_equal(first.B, second.B)
    assert first.meta == second.meta
    assert first.meta["solver"] == SOLVER
    assert first.meta["grid_spacing_m"] == spacing and first.meta["ray_step_m"] == spacing
    assert first.meta["eikonal_sweeps"] >= 4
    assert 0.0 <= first.meta["fraction_post_critical"] <= 1.0
    assert first.meta["fraction_forced_closure"] == 0.0
    accuracy = first.meta["travel_time_accuracy"]
    assert 0.0 <= accuracy["rms_s"] <= accuracy["max_abs_s"] < 0.25 * spacing / C_REF
    assert not any("time_s" in key or "wall" in key for key in first.meta)
    assert np.all(np.isfinite(first.B))


def test_result_does_not_depend_on_the_batch_size(grid, soft, bone, transducer, roi, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    one_by_one = dataclasses.replace(BENT, bent_ray=BentRayConfig(max_batch_bytes=1))
    together = compute_bank(medium, transducer, roi, FREQS, BENT)
    separate = compute_bank(medium, transducer, roi, FREQS, one_by_one)
    np.testing.assert_array_equal(together.B, separate.B)
    assert estimate_cost(medium, transducer, roi, FREQS, one_by_one)["batch_elements"] == 1


def test_cost_estimate(grid, soft, transducer, roi):
    medium = homogeneous_medium(grid, soft, 1.5)
    cost = estimate_cost(medium, transducer, roi, FREQS, BENT)
    assert cost["grid_points"] == 41 * 41 and cost["n_simulations"] == 3
    assert cost["time_steps"] == 0 and cost["batch_elements"] == 3
    assert cost["memory_bytes"] > 8 * 5 * 3 * 9 and cost["run_time_s"] > 0.0
    assert cost["ray_samples"] > 9 * 3 * 20


def test_points_outside_the_medium_are_refused(grid, soft, transducer):
    medium = homogeneous_medium(grid, soft, 1.5)
    with pytest.raises(ValueError, match="inside the medium grid"):
        bent_ray_travel_times(medium, transducer, np.array([[0.0, 13.0e-3]]))
    wide = Transducer(3, 7.0e-3, 0.3e-3, 0.5e6, 0.8)
    with pytest.raises(ValueError, match="inside the medium grid"):
        bent_ray_travel_times(medium, wide, np.array([[0.0, 6.0e-3]]))


def test_travel_times_keep_the_shape_of_the_points(grid, soft, transducer):
    medium = homogeneous_medium(grid, soft, 1.5)
    points = Grid2D.from_extent((-3.0e-3, 3.0e-3), (3.0e-3, 9.0e-3), 3.0e-3).points()
    times = bent_ray_travel_times(medium, transducer, points)
    assert times.shape == (3, 3, 3)
    distance = np.hypot(
        points[None, ..., 0] - transducer.element_x()[:, None, None], points[None, ..., 1]
    )
    np.testing.assert_allclose(times, distance / C_REF, rtol=1e-9)
