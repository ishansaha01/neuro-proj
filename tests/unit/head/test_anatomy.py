import dataclasses

import numpy as np
import pytest

from neurosim.common.grid import Grid3D
from neurosim.common.seeding import SeedTree
from neurosim.head.anatomy import (
    draw_geometry,
    max_scalp_radius,
    radial_sdf,
    surface_radii,
    unit_vector,
)
from neurosim.head.config import AnatomyConfig
from neurosim.head.types import SURFACES, Tissue, labels_from_sdf

CFG = AnatomyConfig()


def random_directions(n: int) -> np.ndarray:
    vectors = SeedTree(5).rng("anatomy", 7).standard_normal((n, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_surfaces_are_strictly_nested_and_inside_the_bound(seed):
    geometry = draw_geometry(CFG, SeedTree(seed).rng("anatomy"))
    radii = surface_radii(geometry, random_directions(4000))
    assert radii.shape == (len(SURFACES), 4000)
    assert radii.dtype == np.float32
    assert np.all(np.diff(radii, axis=0) < 0.0)
    assert np.all(radii[-1] > 0.0)
    assert radii[0].max() <= max(max_scalp_radius(CFG))


def test_same_stream_gives_the_same_head_and_seeds_differ():
    directions = random_directions(100)
    first = surface_radii(draw_geometry(CFG, SeedTree(0).rng("anatomy")), directions)
    again = surface_radii(draw_geometry(CFG, SeedTree(0).rng("anatomy")), directions)
    other = surface_radii(draw_geometry(CFG, SeedTree(1).rng("anatomy")), directions)
    np.testing.assert_array_equal(first, again)
    assert np.max(np.abs(first - other)) > 1.0e-3


def test_skull_is_thinner_at_both_temporal_windows_than_at_the_vertex():
    no_variation = dataclasses.replace(CFG, skull_variation=0.0)
    geometry = draw_geometry(no_variation, SeedTree(0).rng("anatomy"))
    window = unit_vector(CFG.temporal_window_direction)
    directions = np.array([window, window * [-1.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    radii = surface_radii(geometry, directions)
    skull = radii[SURFACES.index("skull_outer")] - radii[SURFACES.index("skull_inner")]
    assert skull[:2] == pytest.approx(CFG.skull_thickness_temporal, rel=0.01)
    assert skull[2] == pytest.approx(CFG.skull_thickness_vault, rel=0.01)


def test_radial_sdf_matches_radii_and_labels():
    geometry = draw_geometry(CFG, SeedTree(0).rng("anatomy"))
    grid = Grid3D.centered((0.2, 0.24, 0.22), 8.0e-3)
    sdf = radial_sdf(geometry, grid)
    assert sdf.shape == (len(SURFACES), *grid.shape)
    assert sdf.dtype == np.float32
    assert np.all(np.isfinite(sdf))
    points = grid.coords().reshape(-1, 3)
    radius = np.linalg.norm(points, axis=1)
    expected = radius - surface_radii(geometry, points / radius[:, None])
    np.testing.assert_allclose(sdf.reshape(len(SURFACES), -1), expected, atol=2e-6)
    labels = labels_from_sdf(sdf)
    assert set(np.unique(labels)) == set(Tissue)
    corner = labels[0, 0, 0]
    centre = labels[tuple(n // 2 for n in grid.shape)]
    assert (corner, centre) == (Tissue.BACKGROUND, Tissue.WHITE)


def test_sdf_is_finite_at_the_head_centre():
    geometry = draw_geometry(CFG, SeedTree(0).rng("anatomy"))
    grid = Grid3D(shape=(3, 3, 3), spacing=1.0e-3, origin=(-1.0e-3, -1.0e-3, -1.0e-3))
    sdf = radial_sdf(geometry, grid)
    assert np.all(np.isfinite(sdf))
    assert np.all(sdf[:, 1, 1, 1] < 0.0)


def test_layers_that_do_not_fit_are_rejected():
    tiny = dataclasses.replace(CFG, semi_axes=(0.02, 0.02, 0.02))
    with pytest.raises(ValueError, match="do not fit"):
        draw_geometry(tiny, SeedTree(0).rng("anatomy"))
