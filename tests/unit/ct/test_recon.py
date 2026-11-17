"""fbp_fan_beam: filter properties, shapes and a round trip through the projector."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.common.seeding import SeedTree
from neurosim.ct.projector import fan_beam_project
from neurosim.ct.recon import (
    backproject,
    fbp_fan_beam,
    filter_sinogram,
    filter_taps,
    pixel_coordinates,
)

SPACING = 0.002


def test_filter_taps_follow_kak_slaney(small_cfg):
    geom = small_cfg.geometry
    taps = filter_taps(geom, "ramp")
    centre = geom.n_detectors - 1
    alpha = geom.angular_pitch
    assert taps.shape == (2 * geom.n_detectors - 1,)
    assert taps[centre] == pytest.approx(1.0 / (8.0 * alpha**2))
    assert taps[centre + 2] == 0.0
    expected = -0.5 / (np.pi * np.sin(alpha)) ** 2
    assert taps[centre + 1] == pytest.approx(expected)
    np.testing.assert_allclose(taps, taps[::-1])


@pytest.mark.parametrize("kernel", ["shepp-logan", "hann"])
def test_smoother_kernels_have_a_smaller_central_tap(small_cfg, kernel):
    geom = small_cfg.geometry
    centre = geom.n_detectors - 1
    assert 0.0 < filter_taps(geom, kernel)[centre] < filter_taps(geom, "ramp")[centre]


def test_unknown_kernel_raises(small_cfg):
    with pytest.raises(ValueError, match="kernel"):
        filter_taps(small_cfg.geometry, "sharp")


def test_sinogram_shape_is_checked(small_cfg):
    with pytest.raises(ValueError, match="sinogram shape"):
        filter_sinogram(np.zeros((3, 5)), small_cfg.geometry, "ramp")


def test_pixel_coordinates_are_centred_and_indexed_ix_iy():
    x, y = pixel_coordinates((4, 6), 0.001, (0.01, -0.02))
    assert x.shape == y.shape == (4, 6)
    assert x.mean() == pytest.approx(0.01) and y.mean() == pytest.approx(-0.02)
    assert x[1, 0] - x[0, 0] == pytest.approx(0.001) and x[0, 1] == x[0, 0]


def test_round_trip_recovers_a_uniform_square(small_cfg):
    geom = small_cfg.geometry
    image = np.zeros((40, 40))
    image[10:30, 12:32] = 25.0  # [1/m]
    sinogram = fan_beam_project(image, SPACING, geom)
    recon = fbp_fan_beam(sinogram, geom, "ramp", image.shape, SPACING)
    assert recon.shape == image.shape and recon.dtype == np.float64
    assert recon[14:26, 16:28].mean() == pytest.approx(25.0, rel=0.01)
    assert abs(recon[:5, :5].mean()) < 0.5


def test_backproject_accepts_arbitrary_points_and_chunks(small_cfg):
    geom = small_cfg.geometry
    rng = SeedTree(3).rng("ct_photon_noise")
    filtered = rng.normal(size=(geom.n_views, geom.n_detectors))
    x = np.array([[0.0, 0.01], [0.02, -0.03]])
    y = np.array([[0.0, 0.02], [-0.01, 0.04]])
    whole = backproject(filtered, geom, x, y)
    chunked = backproject(filtered, geom, x, y, max_chunk_samples=17)
    assert whole.shape == x.shape
    np.testing.assert_allclose(chunked, whole, rtol=1e-12)
