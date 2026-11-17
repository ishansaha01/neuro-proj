import numpy as np
import pytest

from neurosim.common.grid import Grid2D
from neurosim.fus.backends.geometry import (
    check_geometry,
    check_medium,
    checked_frequencies,
    element_centres,
    inside_grid,
)
from neurosim.fus.backends.phantoms import homogeneous_medium
from neurosim.fus.types import Transducer


def test_element_centres_are_on_the_array_face(transducer, spacing):
    centres = element_centres(transducer)
    assert centres.shape == (3, 2)
    np.testing.assert_allclose(centres[:, 0], np.array([-4, 0, 4]) * spacing)
    assert np.all(centres[:, 1] == 0.0)


def test_checked_frequencies_returns_float64():
    freqs = checked_frequencies(np.array([1, 2, 3], dtype=np.int64) * 1_000_000)
    assert freqs.dtype == np.float64 and freqs.shape == (3,)
    assert checked_frequencies([2.0e6]).shape == (1,)


@pytest.mark.parametrize(
    "freqs",
    [[], [[1.0e6, 2.0e6]], [0.0, 1.0e6], [2.0e6, 1.0e6], [1.0e6, 1.0e6], [1.0e6, 2.0e6, 4.0e6]],
)
def test_checked_frequencies_refuses_bad_lists(freqs):
    with pytest.raises(ValueError):
        checked_frequencies(np.array(freqs))


def test_inside_grid_uses_the_outer_pixel_faces(grid, spacing):
    points = np.array(
        [[-6.0e-3 - 0.4 * spacing, 0.0], [-6.0e-3 - 0.6 * spacing, 0.0], [0.0, 12.2e-3]]
    )
    np.testing.assert_array_equal(inside_grid(grid, points), [True, False, False])


def test_check_medium_refuses_bad_maps(grid, soft):
    medium = homogeneous_medium(grid, soft, alpha_power=1.5)
    check_medium(medium)
    medium.sound_speed[3, 4] = 0.0
    with pytest.raises(ValueError, match="positive"):
        check_medium(medium)
    medium.sound_speed[3, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        check_medium(medium)
    medium = homogeneous_medium(grid, soft, alpha_power=1.5)
    medium.attenuation[0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        check_medium(medium)


def test_check_geometry_refuses_roi_and_array_outside_the_medium(grid, soft, transducer, roi):
    medium = homogeneous_medium(grid, soft, alpha_power=1.5)
    check_geometry(medium, transducer, roi)
    with pytest.raises(ValueError, match="inside the medium grid"):
        too_wide = Grid2D.from_extent((0.0, 9.0e-3), (6.0e-3, 9.0e-3), 3.0e-3)
        check_geometry(medium, transducer, too_wide)
    with pytest.raises(ValueError, match="in front of the array"):
        check_geometry(medium, transducer, Grid2D.from_extent((0.0, 3.0e-3), (0.0, 3.0e-3), 3.0e-3))
    wide = Transducer(64, 0.3e-3, 0.3e-3, 0.5e6, 0.8)
    with pytest.raises(ValueError, match="array element"):
        check_geometry(medium, wide, roi)
