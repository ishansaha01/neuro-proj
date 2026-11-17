import dataclasses

import numpy as np
import pytest

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.common.units import db_to_np
from neurosim.fus.backends import (
    BackendConfig,
    RayConfig,
    compute_bank,
    estimate_cost,
    ray_travel_times,
)
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium
from neurosim.fus.backends.ray import trace_rays
from neurosim.fus.types import Transducer

FREQS = np.linspace(0.4e6, 0.6e6, 5)
RAY = BackendConfig(backend="ray")


def impedance(material: Material) -> float:
    return material.sound_speed * material.density


@pytest.fixture
def single() -> Transducer:
    """One element at x = 0."""
    return Transducer(1, 1.2e-3, 0.3e-3, 0.5e6, 0.8)


@pytest.fixture
def below() -> np.ndarray:
    """One point on the array normal, below every test slab [m], shape (1, 2)."""
    return np.array([[0.0, 9.0e-3]])


def test_travel_times_in_a_homogeneous_medium(grid, transducer):
    medium = homogeneous_medium(grid, Material(1480.0, 1000.0), alpha_power=1.5)
    points = Grid2D.from_extent((-3.0e-3, 3.0e-3), (3.0e-3, 9.0e-3), 3.0e-3).points()
    times = ray_travel_times(medium, transducer, points)
    assert times.shape == (3, 3, 3) and times.dtype == np.float64
    distance = np.hypot(
        points[None, ..., 0] - transducer.element_x()[:, None, None], points[None, ..., 1]
    )
    np.testing.assert_allclose(times, distance / 1480.0, rtol=1e-6)


def test_slab_adds_the_analytic_delay_on_its_normal(grid, soft, single, below, spacing):
    fast = Material(2800.0, 1000.0)
    thickness = 10 * spacing
    medium = slab_medium(grid, soft, fast, 9.5 * spacing, thickness, alpha_power=1.5)
    delay = ray_travel_times(medium, single, below)[0, 0] - below[0, 1] / C_REF
    expected = thickness * (1.0 / fast.sound_speed - 1.0 / C_REF)
    assert delay == pytest.approx(expected, rel=0.01)


def test_one_way_path_through_a_slab_loses_both_interfaces(
    grid, soft, bone, single, below, spacing
):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    integrals = trace_rays(medium, single, below, RayConfig())
    z1, z2 = impedance(soft), impedance(bone)
    assert integrals.transmission[0, 0] == pytest.approx(4 * z1 * z2 / (z1 + z2) ** 2, rel=1e-6)
    assert integrals.transmission[0, 0] < 1.0


def test_single_interface_is_flux_normalised_and_symmetric(grid, soft, bone, single, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 40 * spacing, alpha_power=1.5)
    inside = np.array([[0.0, 9.0e-3]])
    integrals = trace_rays(medium, single, inside, RayConfig())
    z1, z2 = impedance(soft), impedance(bone)
    expected = 2 * np.sqrt(z1 * z2) / (z1 + z2)
    assert integrals.transmission[0, 0] == pytest.approx(expected, rel=1e-6)
    # Pressure transmission 2 Z2 / (Z1 + Z2) = flux-normalised loss * sqrt(Z2 / Z1).
    assert expected * np.sqrt(z2 / z1) == pytest.approx(2 * z2 / (z1 + z2))


def test_impedance_matched_slab_loses_nothing(grid, soft, single, below, spacing):
    matched = soft.impedance_matched(2800.0)
    medium = slab_medium(grid, soft, matched, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    integrals = trace_rays(medium, single, below, RayConfig())
    assert integrals.transmission[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_smooth_impedance_ramp_conserves_energy(soft, single):
    fine = Grid2D.from_extent((-1.5e-3, 1.5e-3), (0.0, 12.0e-3), 0.05e-3)
    medium = homogeneous_medium(fine, soft, alpha_power=1.5)
    z = fine.axes()[0]
    ramp = np.interp(z, [3.0e-3, 9.0e-3], [1000.0, 3000.0]).astype(np.float32)  # density x 3
    medium = dataclasses.replace(medium, density=np.repeat(ramp[:, None], fine.shape[1], axis=1))
    integrals = trace_rays(medium, single, np.array([[0.0, 11.0e-3]]), RayConfig())
    # 120 small steps instead of one: the loss falls as 1 / steps to about 2e-3
    # (a single sharp step would give 0.866).
    assert 0.995 < integrals.transmission[0, 0] <= 1.0
    # Pressure rises as sqrt(rho_r / rho_e): intensity p^2 / (rho c) is conserved.
    assert integrals.spreading[0, 0] == pytest.approx(np.sqrt(3.0), rel=1e-3)


@pytest.mark.parametrize("step_fraction", [1.0, 0.5, 0.23])
def test_interface_loss_does_not_depend_on_the_ray_step(
    grid, soft, bone, transducer, spacing, step_fraction
):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    points = np.array([[2.4e-3, 9.0e-3], [-3.0e-3, 10.5e-3]])  # oblique rays
    integrals = trace_rays(medium, transducer, points, RayConfig(step_fraction=step_fraction))
    z1, z2 = impedance(soft), impedance(bone)
    np.testing.assert_allclose(integrals.transmission, 4 * z1 * z2 / (z1 + z2) ** 2, rtol=1e-6)


def test_absorption_follows_the_power_law(grid, soft, single, below, spacing):
    lossy = Material(C_REF, soft.density, attenuation=5.0)
    thickness = 10 * spacing
    medium = slab_medium(grid, soft, lossy, 9.5 * spacing, thickness, alpha_power=1.3)
    roi = Grid2D.from_extent((0.0, 0.0), (9.0e-3, 9.0e-3), 3.0e-3)
    bank = compute_bank(medium, single, roi, FREQS, RAY)
    loss_db = 5.0 * (FREQS / 1.0e6) ** 1.3 * (thickness / 1.0e-2)
    np.testing.assert_allclose(np.abs(bank.B[:, 0, 0, 0]), np.exp(-db_to_np(loss_db)), rtol=2e-3)
    np.testing.assert_allclose(np.angle(bank.B), 0.0, atol=1e-5)


def test_bank_of_a_homogeneous_medium_is_the_far_field_greens_ratio(grid, transducer, roi):
    slow = Material(1400.0, 1300.0)
    bank = compute_bank(homogeneous_medium(grid, slow, 1.5), transducer, roi, FREQS, RAY)
    points = roi.points()
    distance = np.hypot(
        points[None, ..., 0] - transducer.element_x()[:, None, None], points[None, ..., 1]
    )
    # exp(-i w t) convention: a slower medium delays, so the phase grows with frequency.
    excess_delay = distance * (1.0 / 1400.0 - 1.0 / C_REF)
    phase = 2.0 * np.pi * FREQS[:, None, None, None] * excess_delay
    expected = np.sqrt(1400.0 / C_REF) * np.exp(1j * phase)
    np.testing.assert_allclose(bank.B, expected, rtol=1e-4)
    assert np.all(np.diff(np.unwrap(np.angle(bank.B), axis=0), axis=0) > 0.0)


def test_bank_contract_fields(grid, soft, bone, transducer, roi, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    bank = compute_bank(medium, transducer, roi, FREQS, RAY)
    assert bank.backend == "ray" and bank.c_ref == C_REF and bank.static_fmc is None
    assert bank.B.shape == (5, 3, 3, 5) and bank.B.dtype == np.complex64
    assert bank.freqs.dtype == np.float64 and bank.roi == roi
    assert np.all(np.isfinite(bank.B))


def test_reference_medium_gives_unit_bank(grid, soft, transducer, roi):
    bank = compute_bank(homogeneous_medium(grid, soft, 1.5), transducer, roi, FREQS, RAY)
    np.testing.assert_allclose(bank.B, 1.0, atol=1e-5)


def test_chunking_does_not_change_the_result(grid, soft, bone, transducer, roi, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    whole = trace_rays(medium, transducer, roi.points().reshape(-1, 2), RayConfig())
    chunked = trace_rays(
        medium, transducer, roi.points().reshape(-1, 2), RayConfig(max_chunk_bytes=10_000)
    )
    # A chunk of short rays gets fewer, longer segments; the integrals agree to sampling error.
    np.testing.assert_allclose(chunked.travel_time, whole.travel_time, rtol=5e-3)
    np.testing.assert_allclose(chunked.transmission, whole.transmission, rtol=1e-6)
    np.testing.assert_array_equal(chunked.path_length, whole.path_length)


def test_points_outside_the_medium_are_refused(grid, soft, transducer):
    medium = homogeneous_medium(grid, soft, 1.5)
    with pytest.raises(ValueError, match="inside the medium grid"):
        ray_travel_times(medium, transducer, np.array([[0.0, 20.0e-3]]))


def test_cost_estimate_has_the_common_keys(grid, soft, transducer, roi):
    cost = estimate_cost(homogeneous_medium(grid, soft, 1.5), transducer, roi, FREQS, RAY)
    assert cost["n_simulations"] == 0 and cost["time_steps"] == 0
    assert cost["grid_points"] == 41 * 41
    assert cost["memory_bytes"] > 8 * FREQS.size * 3 * 15
    assert cost["run_time_s"] > 0.0
