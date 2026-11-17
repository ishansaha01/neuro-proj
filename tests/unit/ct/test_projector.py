"""fan_beam_project: shapes, linearity and geometry conventions."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.ct.projector import enclosing_radius, fan_beam_project, samples_per_ray

SPACING = 0.002


@pytest.fixture(scope="module")
def square() -> np.ndarray:
    image = np.zeros((40, 40))
    image[10:30, 10:30] = 1.0
    return image


def test_shape_for_slice_and_stack(small_cfg, square):
    geom = small_cfg.geometry
    single = fan_beam_project(square, SPACING, geom)
    stack = fan_beam_project(np.stack((square, 2.0 * square)), SPACING, geom)
    assert single.shape == (geom.n_views, geom.n_detectors)
    assert stack.shape == (2, geom.n_views, geom.n_detectors)
    np.testing.assert_allclose(stack[0], single)
    np.testing.assert_allclose(stack[1], 2.0 * single)


def test_central_rays_measure_the_width_of_a_square(small_cfg, square):
    geom = small_cfg.geometry
    sinogram = fan_beam_project(square, SPACING, geom)
    centre = geom.n_detectors // 2
    # View 0 looks along x: the two central rays cross 20 pixels (interpolated edges).
    width = sinogram[0, centre - 1 : centre + 1].mean()
    assert width == pytest.approx(20 * SPACING, rel=0.01)


def test_empty_image_projects_to_zero(small_cfg):
    assert not fan_beam_project(np.zeros((16, 16)), SPACING, small_cfg.geometry).any()


def test_chunking_does_not_change_the_result(small_cfg, square):
    geom = small_cfg.geometry
    whole = fan_beam_project(square, SPACING, geom, max_chunk_samples=10**9)
    chunked = fan_beam_project(square, SPACING, geom, max_chunk_samples=1000)
    np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-12)


def test_center_offset_moves_the_object(small_cfg):
    geom = small_cfg.geometry
    big = np.zeros((60, 60))
    big[35:45, 20:30] = 1.0
    crop = big[30:50, 15:35]
    # The crop's centre sits at pixel (39.5, 24.5) of the big grid, whose centre is (29.5, 29.5).
    offset = ((39.5 - 29.5) * SPACING, (24.5 - 29.5) * SPACING)
    full = fan_beam_project(big, SPACING, geom, ray_step_fraction=0.25)
    moved = fan_beam_project(crop, SPACING, geom, center_offset=offset, ray_step_fraction=0.25)
    np.testing.assert_allclose(moved, full, atol=0.03 * SPACING * 10)


def test_view_and_fan_angle_conventions(small_cfg):
    geom = small_cfg.geometry
    image = np.zeros((41, 41))
    image[30, 20] = 1.0  # a point on the +x axis, 20 mm from the isocentre
    sinogram = fan_beam_project(image, SPACING, geom, ray_step_fraction=0.25)
    centre = 0.5 * (geom.n_detectors - 1)
    # Source on +x (view 0): the point is on the central ray.
    assert abs(np.argmax(sinogram[0]) - centre) <= 0.5
    # Source on +y (quarter turn): the ray to the point is rotated counter-clockwise
    # from the central ray, which is a positive fan angle.
    quarter = geom.n_views // 4
    expected = np.arctan2(0.020, geom.source_isocenter_distance) / geom.angular_pitch + centre
    assert abs(np.argmax(sinogram[quarter]) - expected) <= 1.0


def test_bad_image_rank_raises(small_cfg):
    with pytest.raises(ValueError, match="shape"):
        fan_beam_project(np.zeros(5), SPACING, small_cfg.geometry)


def test_sampling_helpers():
    assert enclosing_radius((30, 40), 0.001, (0.0, 0.0)) == pytest.approx(0.025)
    assert enclosing_radius((30, 40), 0.001, (0.003, 0.004)) == pytest.approx(0.030)
    assert samples_per_ray((30, 40), 0.001, (0.0, 0.0), 0.5) == 100
