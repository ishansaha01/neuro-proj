import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium


def test_slab_occupies_whole_rows_between_pixel_faces(grid, soft, bone, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 5 * spacing, alpha_power=1.2)
    in_slab = medium.sound_speed == np.float32(bone.sound_speed)
    assert np.all(in_slab[10:15]) and not np.any(in_slab[:10]) and not np.any(in_slab[15:])
    assert np.all(medium.density[in_slab] == np.float32(bone.density))
    assert np.all(medium.attenuation[~in_slab] == 0.0)
    assert medium.alpha_power == 1.2
    assert medium.sound_speed.dtype == np.float32 and medium.sound_speed.shape == grid.shape
    assert medium.blood_velocity.shape == (2, *grid.shape)


def test_thickness_profile_is_per_column(soft, bone, spacing):
    grid = Grid2D.from_extent((0.0, 2 * spacing), (0.0, 9 * spacing), spacing)
    thickness = np.array([0.0, 2.0, 4.0]) * spacing
    medium = slab_medium(grid, soft, bone, 1.5 * spacing, thickness, alpha_power=1.5)
    rows_in_slab = np.count_nonzero(medium.density == np.float32(bone.density), axis=0)
    np.testing.assert_array_equal(rows_in_slab, [0, 2, 4])


def test_homogeneous_medium_is_uniform(grid, bone):
    medium = homogeneous_medium(grid, bone, alpha_power=1.5)
    assert np.ptp(medium.sound_speed) == 0.0 and medium.sound_speed[0, 0] == bone.sound_speed
    assert np.all(medium.attenuation == np.float32(bone.attenuation))


def test_impedance_matched_material_keeps_rho_c():
    soft = Material(1540.0, 1000.0, attenuation=0.5)
    matched = soft.impedance_matched(2800.0)
    assert matched.sound_speed == 2800.0
    np.testing.assert_allclose(matched.sound_speed * matched.density, 1540.0 * 1000.0)
    assert matched.attenuation == 0.0
