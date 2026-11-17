"""Torch assembly of the phase-screen operator and its inputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurosim.fus import operators
from neurosim.fus.config import imaging_setup
from neurosim.fus.operators import operator_factors, psf_operator
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.types import GreensBank
from neurosim.recon import forward
from neurosim.recon.forward import (
    element_factors,
    expected_power_doppler,
    power_doppler_operator,
    psf_matrix,
    reference_bank,
    refined_source_points,
    scene_screen_model,
    screen_model_from_factors,
    screen_tensors,
)

from .scene_fixtures import tiny_run, tiny_scene, tiny_scene_refined  # noqa: F401


def test_refined_source_points_are_cell_midpoints(tiny_scene):
    grid = tiny_scene.pixel_grid
    centres = grid.points().reshape(-1, 2)
    np.testing.assert_allclose(refined_source_points(grid, 1), centres)
    refined = refined_source_points(grid, 2).reshape(4, -1, 2)
    assert refined.shape == (4, centres.shape[0], 2)
    np.testing.assert_allclose(refined.mean(axis=0), centres, atol=1e-12)
    np.testing.assert_allclose(np.abs(refined - centres).max(), 0.25 * grid.spacing)
    with pytest.raises(ValueError):
        refined_source_points(grid, 0)


def test_factors_have_documented_shapes(tiny_scene):
    setup = imaging_setup(tiny_scene.imaging)
    bank = reference_bank(tiny_scene)
    sources = refined_source_points(tiny_scene.pixel_grid, 1)
    factors = operator_factors(bank, tiny_scene.pixel_grid, sources, bank, setup.sequence)
    nf, n_el, n_pix = setup.freqs.shape[0], setup.transducer.n_elements, sources.shape[0]
    n_angles = len(setup.sequence.angles_deg)
    assert factors.pulse.shape == (nf,)
    assert factors.tx_weights.shape == (nf, n_angles, n_pix)
    assert factors.tx_phasors.shape == (nf, n_angles, n_el)
    assert factors.rx_weights.shape == (nf, n_el, n_pix)
    assert factors.greens.shape == (nf, n_el, n_pix)
    assert factors.greens.dtype == np.complex64
    with pytest.raises(ValueError):
        screen_model_from_factors(factors, 3)


def test_factor_export_is_independent_of_chunking(tiny_scene, monkeypatch):
    setup = imaging_setup(tiny_scene.imaging)
    bank = reference_bank(tiny_scene)
    sources = refined_source_points(tiny_scene.pixel_grid, 1)
    whole = operator_factors(bank, tiny_scene.pixel_grid, sources, bank, setup.sequence)
    monkeypatch.setattr(operators, "_CHUNK_VALUES", 1)
    chunked = operator_factors(bank, tiny_scene.pixel_grid, sources, bank, setup.sequence)
    np.testing.assert_allclose(chunked.greens, whole.greens, rtol=1e-6)
    np.testing.assert_array_equal(chunked.tx_phasors, whole.tx_phasors)


def test_element_factors_are_amplitude_and_delay_phase():
    freqs = torch.tensor([1.0e6, 2.0e6], dtype=torch.float64)
    delays = torch.tensor([0.0, 125.0e-9], dtype=torch.float64)
    log_amplitudes = torch.log(torch.tensor([1.0, 0.5], dtype=torch.float64))
    factors = element_factors(freqs, delays, log_amplitudes)
    # 125 ns is an eighth of a period at 1 MHz and a quarter at 2 MHz.
    expected = torch.tensor(
        [[1.0, 0.5 * np.exp(0.25j * np.pi)], [1.0, 0.5j]], dtype=torch.complex128
    )
    torch.testing.assert_close(factors, expected, atol=1e-12, rtol=0.0)


def test_assembly_matches_psf_operator_for_a_screen(tiny_scene):
    setup = imaging_setup(tiny_scene.imaging)
    homogeneous = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, 1540.0)
    screen = tiny_scene.screen_true
    reference = psf_operator(
        apply_phase_screen(homogeneous, screen.delays, screen.amplitudes),
        tiny_scene.pixel_grid,
        tiny_scene.pixel_grid,
        homogeneous,
        setup.sequence,
    )
    model = scene_screen_model(tiny_scene, 1)
    assembled = psf_matrix(model, *screen_tensors(screen)).numpy()
    assert np.abs(assembled - reference).max() < 1e-5 * np.abs(reference).max()


def test_operator_matches_the_scene_operator_with_refinement(tiny_scene_refined):
    model = scene_screen_model(tiny_scene_refined, 2)
    operator = power_doppler_operator(model, *screen_tensors(tiny_scene_refined.screen_true))
    reference = tiny_scene_refined.operators["skull_das"]
    assert operator.shape == reference.shape
    assert np.abs(operator.numpy() - reference).max() < 1e-5 * reference.max()


def test_pixel_subset_and_chunked_gradient_agree_with_the_whole(tiny_scene, monkeypatch):
    model = scene_screen_model(tiny_scene, 1, torch.complex128)
    delays, log_amplitudes = screen_tensors(tiny_scene.screen_ct)
    pixel_index = torch.tensor([3, 17, 100, 255])
    whole = power_doppler_operator(model, delays, log_amplitudes)
    subset = power_doppler_operator(model, delays, log_amplitudes, pixel_index)
    torch.testing.assert_close(subset, whole[pixel_index], rtol=1e-10, atol=0.0)

    def gradient() -> torch.Tensor:
        variable = delays.clone().requires_grad_(True)
        total = power_doppler_operator(model, variable, log_amplitudes, pixel_index).sum()
        return torch.autograd.grad(total, variable)[0]

    direct = gradient()
    monkeypatch.setattr(forward, "_CHUNK_VALUES", 1)  # one frequency per recomputed chunk
    torch.testing.assert_close(gradient(), direct, rtol=1e-9, atol=0.0)


def test_expected_image_is_linear_in_cbv_plus_floor():
    operator = torch.tensor([[1.0, 2.0], [0.0, 4.0], [3.0, 0.0]])
    floor = torch.tensor([0.5, 0.5, 1.0])
    cbv = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    expected = expected_power_doppler(operator, cbv, 0.5, floor)
    torch.testing.assert_close(expected, torch.tensor([[2.0, 2.5, 2.5], [1.5, 0.5, 4.0]]))


def test_scene_operators_by_offset_equal_the_single_model(tiny_scene_refined):
    scene = tiny_scene_refined
    screens = (scene.screen_true, scene.screen_ct)
    by_offset = forward.scene_screen_operators(scene, screens, 2)
    model = scene_screen_model(scene, 2)
    for operator, screen in zip(by_offset, screens):
        whole = power_doppler_operator(model, *screen_tensors(screen)).to(torch.float32)
        assert operator.dtype == torch.float32
        torch.testing.assert_close(operator, whole, rtol=1e-4, atol=1e-6 * float(whole.max()))
