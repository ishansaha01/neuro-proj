"""fus.speckle: independent frames of moving blood in one power Doppler image."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import erf

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.fus.resolution import axial_correlation_length, lateral_correlation_length
from neurosim.fus.speckle import (
    SpeckleFrames,
    cell_flow_correlation,
    flow_correlation,
    independent_frames,
    pixel_independent_frames,
    speckle_frames,
)

LATERAL, AXIAL = 1.5e-3, 0.6e-3  # correlation lengths of an anisotropic resolution cell [m]
WAVEVECTOR = np.array([0.0, 2.0 * 2.0 * np.pi * 1.8e6 / C_REF])  # 2 k along depth [rad/m]
N_FRAMES, FRAME_RATE = 100, 500.0


def _frames(speed: float, angle_deg: float, n_frames: int = N_FRAMES) -> float:
    """Independent frames of uniform flow at an angle from the array axis."""
    angle = np.deg2rad(angle_deg)
    velocity = speed * np.array([np.cos(angle), np.sin(angle)])
    correlation = flow_correlation(velocity, LATERAL, AXIAL, WAVEVECTOR, n_frames, FRAME_RATE)
    return float(independent_frames(correlation))


def test_flow_correlation_shape_decay_and_doppler_phase():
    velocity = np.array([[0.0, 0.0], [3.0e-3, 0.0], [0.0, 3.0e-3]])
    correlation = flow_correlation(velocity, LATERAL, AXIAL, WAVEVECTOR, 8, FRAME_RATE)
    assert correlation.shape == (3, 8) and correlation.dtype == np.complex128
    np.testing.assert_allclose(correlation[:, 0], 1.0)
    np.testing.assert_allclose(correlation[0], 1.0)  # static blood stays correlated
    lags = np.arange(8) / FRAME_RATE
    # |g|^2 = exp(-pi (v tau / l)^2): the intensity correlation of a Gaussian cell.
    np.testing.assert_allclose(
        np.abs(correlation[1]) ** 2, np.exp(-np.pi * (3.0e-3 * lags / LATERAL) ** 2)
    )
    np.testing.assert_allclose(
        np.abs(correlation[2]) ** 2, np.exp(-np.pi * (3.0e-3 * lags / AXIAL) ** 2)
    )
    # Lateral flow under the array centre has no Doppler shift; axial flow has 2 v / lambda.
    np.testing.assert_allclose(correlation[1].imag, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.angle(correlation[2, 1]), WAVEVECTOR[1] * 3.0e-3 / FRAME_RATE, rtol=1e-12
    )


def test_static_blood_is_one_frame_and_fast_blood_saturates_at_the_ensemble():
    assert _frames(0.0, 45.0) == pytest.approx(1.0, abs=1e-12)
    # Crossing many correlation lengths per frame: every frame is new.
    assert _frames(5.0, 45.0) == pytest.approx(N_FRAMES, rel=1e-6)
    for speed in (1e-4, 1e-3, 2e-2, 0.3):
        assert 1.0 <= _frames(speed, 30.0) <= N_FRAMES
    np.testing.assert_array_equal(independent_frames(np.zeros((2, 5))), 1.0)  # no signal
    np.testing.assert_allclose(independent_frames(np.eye(5)[0]), 5.0)  # white: n frames
    assert independent_frames(3.0 * np.ones(7)) == pytest.approx(1.0)  # any scale


def test_frames_grow_with_speed_and_with_the_ensemble_duration():
    speeds = np.geomspace(1e-4, 1.0, 25)
    counts = np.array([_frames(speed, 45.0) for speed in speeds])
    assert np.all(np.diff(counts) > 0.0)
    durations = np.array([_frames(5e-3, 45.0, n_frames) for n_frames in (10, 25, 50, 100, 200)])
    assert np.all(np.diff(durations) > 0.0)


def test_frames_follow_the_anisotropy_of_the_resolution_cell():
    lateral, oblique, axial = (_frames(2e-2, angle) for angle in (0.0, 45.0, 90.0))
    assert lateral < oblique < axial  # the cell is shortest along depth
    # The direction enters through 1 / l^2 = cos^2 / l_x^2 + sin^2 / l_z^2 alone.
    length = 1.0 / np.hypot(np.cos(np.pi / 4.0) / LATERAL, np.sin(np.pi / 4.0) / AXIAL)
    isotropic = flow_correlation(
        np.array([2e-2, 0.0]), length, length, WAVEVECTOR, N_FRAMES, FRAME_RATE
    )
    assert float(independent_frames(isotropic)) == pytest.approx(oblique, rel=1e-12)
    assert _frames(2e-2, 135.0) == pytest.approx(oblique, rel=1e-12)


def test_long_ensemble_matches_the_degrees_of_freedom_of_integrated_speckle():
    """Goodman's M for a Gaussian correlation, with x = ensemble duration / decorrelation time:

    1 / [erf(sqrt(pi) x) / x - (1 - exp(-pi x^2)) / (pi x^2)].
    """
    for speed, angle in ((1e-3, 45.0), (5e-3, 0.0), (2e-2, 45.0), (2e-2, 90.0)):
        radians = np.deg2rad(angle)
        length = 1.0 / np.hypot(np.cos(radians) / LATERAL, np.sin(radians) / AXIAL)
        ratio = speed * N_FRAMES / FRAME_RATE / length  # ensemble duration / decorrelation time
        degrees = 1.0 / (
            erf(np.sqrt(np.pi) * ratio) / ratio
            - (1.0 - np.exp(-np.pi * ratio**2)) / (np.pi * ratio**2)
        )
        assert _frames(speed, angle) == pytest.approx(degrees, rel=2e-3)


def test_independent_frames_is_the_variance_of_the_mean_power_of_a_gaussian_process():
    """Monte Carlo: var / mean^2 of the mean of |E_i|^2 for a process with a given correlation."""
    n_frames, n_draws = 12, 60_000
    correlation = flow_correlation(
        np.array([4e-3, 6e-3]), LATERAL, AXIAL, WAVEVECTOR, n_frames, FRAME_RATE
    )
    lag = np.arange(n_frames)
    difference = lag[:, None] - lag[None, :]
    covariance = np.where(
        difference >= 0, correlation[np.abs(difference)], np.conj(correlation[np.abs(difference)])
    )
    factor = np.linalg.cholesky(covariance + 1e-12 * np.eye(n_frames))
    rng = SeedTree(0).rng("blood_scatterers")
    white = (
        rng.standard_normal((n_draws, n_frames)) + 1j * rng.standard_normal((n_draws, n_frames))
    ) / np.sqrt(2.0)
    power = np.mean(np.abs(white @ factor.T) ** 2, axis=1)
    measured = power.mean() ** 2 / power.var()
    assert measured == pytest.approx(float(independent_frames(correlation)), rel=0.03)


def test_pixel_frames_mix_the_correlations_of_the_cells_by_power():
    correlation = flow_correlation(
        np.array([[0.0, 0.0], [2e-2, 0.0], [0.0, 2e-2]]), LATERAL, AXIAL, WAVEVECTOR, 50, FRAME_RATE
    ).astype(np.complex64)
    alone = independent_frames(correlation)
    operator = np.array(
        [
            [1.0, 0.0, 0.0],  # sees the static cell only
            [0.0, 1.0, 0.0],  # the laterally moving cell only
            [0.0, 0.5, 0.0],  # the same cell at half the power: the same count
            [1.0, 1.0, 0.0],  # a mixture
            [0.0, 0.0, 0.0],  # no blood
        ],
        dtype=np.float32,
    )
    cbv = np.array([0.02, 0.02, 0.02])
    counts = pixel_independent_frames(operator, cbv, correlation)
    assert counts.shape == (5,) and counts.dtype == np.float32
    np.testing.assert_allclose(counts[:3], alone[[0, 1, 1]], rtol=1e-5)
    assert alone[0] < counts[3] < alone[1]
    assert counts[4] == 1.0
    # Only the power H * cbv of each cell counts: no blood in the moving cell, static pixel.
    static = pixel_independent_frames(operator, np.array([0.02, 0.0, 0.02]), correlation)
    assert static[3] == pytest.approx(1.0)
    # Two cells of equal power and equal velocity are one speckle, not two.
    twin = np.stack((correlation[1], correlation[1], correlation[2]))
    np.testing.assert_allclose(
        pixel_independent_frames(np.array([[1.0, 1.0, 0.0]]), cbv, twin), alone[1], rtol=1e-5
    )
    # Equal powers at different Doppler shifts beat against each other: more frames than
    # the same two cells without their phases.
    pair = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    axial_speeds = np.array([[0.0, 1e-3], [0.0, 1.5e-3], [0.0, 3e-3]])
    slow = flow_correlation(axial_speeds, LATERAL, AXIAL, WAVEVECTOR, 50, FRAME_RATE)
    with_beat = pixel_independent_frames(pair, cbv, slow.astype(np.complex64))
    without = pixel_independent_frames(pair, cbv, np.abs(slow).astype(np.complex64))
    assert with_beat[0] > 1.2 * without[0]


def test_cell_correlation_uses_the_resolution_of_each_cell(tiny, tiny_cfg):
    grid = tiny.pixel_grid
    velocity = np.zeros((2, *grid.shape))
    velocity[0] = 1e-2  # lateral flow everywhere
    arguments = (grid, tiny.transducer, tiny.sequence, tiny_cfg.beamforming, tiny.freqs, C_REF)
    correlation = cell_flow_correlation(velocity, *arguments)
    n_frames = tiny.sequence.n_frames
    assert correlation.shape == (grid.shape[0] * grid.shape[1], n_frames)
    assert correlation.dtype == np.complex64
    points = grid.points().reshape(-1, 2)
    lateral = lateral_correlation_length(
        tiny.transducer, tiny.sequence, tiny_cfg.beamforming, points, tiny.freqs, C_REF
    )
    lag = 1.0 / tiny.sequence.frame_rate
    np.testing.assert_allclose(
        np.abs(correlation[:, 1]) ** 2, np.exp(-np.pi * (1e-2 * lag / lateral) ** 2), rtol=1e-5
    )
    # Axial flow decorrelates with the axial length, the same for every cell.
    axial_flow = cell_flow_correlation(velocity[::-1], *arguments)
    axial = axial_correlation_length(tiny.transducer, tiny.freqs, C_REF)
    np.testing.assert_allclose(
        np.abs(axial_flow[:, 1]) ** 2, np.exp(-np.pi * (1e-2 * lag / axial) ** 2), rtol=1e-5
    )
    with pytest.raises(ValueError, match="velocity has shape"):
        cell_flow_correlation(velocity[:, :-1], *arguments)


def test_speckle_frames_derives_the_counts_or_takes_the_override(tiny, tiny_cfg):
    grid = tiny.pixel_grid
    n_pix = grid.shape[0] * grid.shape[1]
    operator = np.eye(n_pix, dtype=np.float32)
    cbv = np.full(grid.shape, 0.03)
    speeds = np.linspace(0.0, 2.0, n_pix).reshape(grid.shape)  # up to 4 mm per frame
    velocity = np.stack((np.zeros(grid.shape), speeds))  # axial flow, faster cell by cell
    derived = speckle_frames(tiny_cfg, operator, cbv, velocity, grid)
    assert isinstance(derived, SpeckleFrames)
    assert derived.blood.shape == (n_pix,) and derived.blood.dtype == np.float32
    assert derived.noise == tiny_cfg.sequence.n_frames == 6
    assert derived.blood[0] == 1.0 and derived.blood[-1] == pytest.approx(6.0, rel=1e-3)
    assert np.all(np.diff(derived.blood) >= 0.0)

    forced = replace(tiny_cfg, doppler=replace(tiny_cfg.doppler, effective_frames=4))
    override = speckle_frames(forced, operator, cbv, velocity, grid)
    np.testing.assert_array_equal(override.blood, np.full(n_pix, 4.0, dtype=np.float32))
    assert override.blood.dtype == np.float32 and override.noise == 4.0


def test_uniform_flow_gives_every_pixel_the_count_of_its_cells(tiny, tiny_cfg):
    """A blurring operator does not change the count when all cells move alike."""
    grid = Grid2D(shape=(1, 5), spacing=1e-4, origin=(0.023, -2e-4))  # cells close together
    velocity = np.broadcast_to(np.array([3e-3, 4e-3])[:, None, None], (2, 1, 5))
    cbv = np.linspace(0.01, 0.05, 5).reshape(1, 5)
    sharp = speckle_frames(tiny_cfg, np.eye(5, dtype=np.float32), cbv, velocity, grid)
    rng = SeedTree(1).rng("speckle_statistical")
    blur = rng.uniform(0.1, 1.0, size=(3, 5)).astype(np.float32)
    blurred = speckle_frames(tiny_cfg, blur, cbv, velocity, grid)
    np.testing.assert_allclose(blurred.blood, sharp.blood.mean(), rtol=2e-3)
