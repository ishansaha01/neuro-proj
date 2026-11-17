"""Blood and tissue scatterers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.common.grid import Grid2D
from neurosim.fus.config import NoiseConfig
from neurosim.fus.scatterers import (
    advance,
    blood_scatterers,
    tissue_amplitude,
    tissue_scatterers,
)

GRID = Grid2D(shape=(4, 6), spacing=1.0e-3, origin=(0.020, -0.0025))
TRANSIT = 1.0e-3


def _inside_cells(xz: np.ndarray) -> bool:
    return bool(
        np.all(xz[:, 0] >= -0.003)
        and np.all(xz[:, 0] <= 0.003)
        and np.all(xz[:, 1] >= 0.0195)
        and np.all(xz[:, 1] <= 0.0235)
    )


def _state(n: int = 4000, velocity=(0.01, -0.02), seed: int = 0):
    cbv = np.full(GRID.shape, 0.01)
    cbv[1, 2] = 0.05
    velocity_map = np.broadcast_to(np.array(velocity)[:, None, None], (2, *GRID.shape))
    return cbv, blood_scatterers(cbv, velocity_map, GRID, n, np.random.default_rng(seed), TRANSIT)


def test_density_follows_cbv_and_backscatter_is_normalised():
    cbv, state = _state(n=20000)
    scatterers = state.scatterers()
    assert scatterers.xz.shape == (20000, 2) and _inside_cells(state.home_xz)
    np.testing.assert_allclose(
        np.sum(scatterers.amplitude**2), cbv.sum() * GRID.spacing**2, rtol=1e-12
    )
    assert set(np.unique(np.sign(scatterers.amplitude))) == {-1.0, 1.0}
    z, x = GRID.axes()
    in_cell = (np.abs(state.home_xz[:, 0] - x[2]) < 0.0005) & (
        np.abs(state.home_xz[:, 1] - z[1]) < 0.0005
    )
    expected = 20000 * 0.05 / cbv.sum()
    assert abs(in_cell.sum() - expected) < 5 * np.sqrt(expected)


def test_same_generator_state_gives_the_same_scatterers():
    _, a = _state(n=50, seed=3)
    _, b = _state(n=50, seed=3)
    np.testing.assert_array_equal(a.scatterers().xz, b.scatterers().xz)
    np.testing.assert_array_equal(a.amplitude, b.amplitude)


def test_scatterers_move_along_the_velocity_and_stay_in_the_grid():
    _, state = _state(n=500)
    dt = 1e-3  # 0.022 mm of travel: no scatterer near a path end crosses more than once
    before, after = state.scatterers().xz, advance(state, dt).scatterers().xz
    step = after - before
    moved_freely = np.linalg.norm(step, axis=1) < 0.5 * TRANSIT * 0.5
    assert moved_freely.mean() > 0.9
    expected = np.broadcast_to(np.array([0.01, -0.02]) * dt, step[moved_freely].shape)
    np.testing.assert_allclose(step[moved_freely], expected, atol=1e-12)
    for time in (0.0, 0.013, 0.4, 7.0):
        assert _inside_cells(advance(state, time).scatterers().xz)


def test_advance_composes_and_is_periodic_per_scatterer():
    _, state = _state(n=200)
    np.testing.assert_allclose(
        advance(advance(state, 0.01), 0.02).path_position,
        advance(state, 0.03).path_position,
        atol=1e-15,
    )
    period = state.path_length / np.linalg.norm(state.velocity, axis=1)
    assert period.min() < period.max()  # path lengths differ: no common return time
    one = replace(
        state,
        home_xz=state.home_xz[:1],
        velocity=state.velocity[:1],
        path_position=state.path_position[:1],
        path_length=state.path_length[:1],
        amplitude=state.amplitude[:1],
    )
    np.testing.assert_allclose(
        advance(one, float(period[0])).scatterers().xz, one.scatterers().xz, atol=1e-12
    )


def test_zero_velocity_scatterers_stay_at_home():
    _, state = _state(n=100, velocity=(0.0, 0.0))
    np.testing.assert_allclose(advance(state, 1.0).scatterers().xz, state.home_xz, atol=1e-15)


@pytest.mark.parametrize(
    ("cbv", "velocity"),
    [
        (np.zeros(GRID.shape), np.zeros((2, *GRID.shape))),
        (np.full(GRID.shape, -0.1), np.zeros((2, *GRID.shape))),
        (np.ones((3, 3)), np.zeros((2, *GRID.shape))),
        (np.ones(GRID.shape), np.zeros(GRID.shape)),
    ],
)
def test_invalid_maps_raise(cbv, velocity):
    with pytest.raises(ValueError):
        blood_scatterers(cbv, velocity, GRID, 10, np.random.default_rng(0), TRANSIT)


def test_tissue_scatterers_fill_the_grid_with_the_requested_power():
    tissue = tissue_scatterers(GRID, 20000, 2.5, np.random.default_rng(1))
    assert tissue.xz.shape == (20000, 2) and _inside_cells(tissue.xz)
    assert np.sqrt(np.mean(tissue.amplitude**2)) == pytest.approx(2.5, rel=0.03)
    assert abs(np.mean(tissue.amplitude)) < 0.06
    assert tissue.xz[:, 0].mean() == pytest.approx(0.0, abs=1e-4)


def test_tissue_amplitude_realises_the_backscatter_ratio():
    noise = NoiseConfig(tissue_to_blood_db=30.0, cbv_reference=0.04)
    n = 500
    amplitude = tissue_amplitude(GRID, n, noise)
    area = 24 * GRID.spacing**2
    tissue_backscatter = n * amplitude**2 / area  # per unit area
    assert tissue_backscatter / noise.cbv_reference == pytest.approx(1000.0, rel=1e-12)
