import numpy as np
import pytest

from neurosim.common.constants import BASIS_MATERIALS, C_WATER, RHO_WATER
from neurosim.common.units import db_cm_mhz_to_np_m, np_m_to_db_cm
from neurosim.head.materials import (
    MaterialParams,
    acoustic_properties,
    brain_mask,
    material_fractions,
    power_law_prefactor,
    skull_mask,
    skull_sound_speed_from_porosity,
)
from neurosim.head.types import Tissue

PARAMS = MaterialParams()


def all_labels() -> np.ndarray:
    return np.arange(len(Tissue), dtype=np.uint8).reshape(2, 4)


def test_masks_select_the_contract_labels():
    labels = all_labels()
    assert labels[skull_mask(labels)].tolist() == [2, 3, 4]
    assert labels[brain_mask(labels)].tolist() == [6, 7]


def test_power_law_prefactor_reproduces_the_quoted_absorption():
    alpha0 = power_law_prefactor(8.0, 0.5e6, 1.1)
    assert np_m_to_db_cm(db_cm_mhz_to_np_m(alpha0, 0.5e6, 1.1)) == pytest.approx(8.0)
    assert power_law_prefactor(3.0, 1.0e6, 1.7) == pytest.approx(3.0)


def test_acoustic_properties_shapes_dtypes_and_soft_tissue_rows():
    labels = all_labels()
    zeros = np.zeros(labels.shape, dtype=np.float32)
    maps = acoustic_properties(labels, zeros, zeros, PARAMS)
    for values in maps:
        assert values.shape == labels.shape
        assert values.dtype == np.float32
    sound_speed, density, attenuation = maps
    assert sound_speed[labels == Tissue.BACKGROUND] == pytest.approx(C_WATER)
    assert density[labels == Tissue.CSF] == pytest.approx(RHO_WATER)
    assert sound_speed[labels == Tissue.SCALP] == pytest.approx(PARAMS.scalp.sound_speed)
    assert density[labels == Tissue.GREY] == pytest.approx(PARAMS.grey.density)
    expected = power_law_prefactor(
        PARAMS.white.absorption, PARAMS.white.absorption_frequency, PARAMS.alpha_power
    )
    assert attenuation[labels == Tissue.WHITE] == pytest.approx(expected)


def test_skull_follows_the_mixing_law_at_its_end_points():
    labels = np.full(2, Tissue.DIPLOE, dtype=np.uint8)
    porosity = np.array([0.0, 1.0], dtype=np.float32)
    sound_speed, density, attenuation = acoustic_properties(
        labels, porosity, np.zeros(2, dtype=np.float32), PARAMS
    )
    skull = PARAMS.skull
    assert sound_speed == pytest.approx([skull.c_bone, skull.c_fluid])
    assert density == pytest.approx([skull.rho_bone, skull.rho_fluid])
    at_reference = np_m_to_db_cm(
        db_cm_mhz_to_np_m(attenuation, skull.absorption_frequency, PARAMS.alpha_power)
    )
    assert at_reference == pytest.approx([skull.absorption_min, skull.absorption_max], rel=1e-5)
    assert skull_sound_speed_from_porosity(np.float64(0.5), skull) == pytest.approx(
        0.5 * (skull.c_bone + skull.c_fluid)
    )


def test_hidden_c_changes_only_the_skull_sound_speed():
    labels = all_labels()
    porosity = np.full(labels.shape, 0.3, dtype=np.float32)
    zeros = np.zeros(labels.shape, dtype=np.float32)
    hidden = np.full(labels.shape, 120.0, dtype=np.float32)
    base = acoustic_properties(labels, porosity, zeros, PARAMS)
    shifted = acoustic_properties(labels, porosity, hidden, PARAMS)
    in_skull = skull_mask(labels)
    assert shifted[0][in_skull] - base[0][in_skull] == pytest.approx(120.0)
    np.testing.assert_array_equal(shifted[0][~in_skull], base[0][~in_skull])
    np.testing.assert_array_equal(shifted[1], base[1])
    np.testing.assert_array_equal(shifted[2], base[2])


def test_material_fractions_per_tissue():
    labels = all_labels()
    porosity = np.full(labels.shape, 0.25, dtype=np.float32)
    fractions = material_fractions(labels, porosity)
    assert fractions.shape == (len(BASIS_MATERIALS), *labels.shape)
    assert fractions.dtype == np.float32
    np.testing.assert_allclose(fractions.sum(axis=0), 1.0, atol=1e-6)
    by_name = dict(zip(BASIS_MATERIALS, fractions))
    assert np.all(by_name["air"] == 0.0)
    assert by_name["water"][labels == Tissue.CSF] == 1.0
    assert by_name["water"][labels == Tissue.BACKGROUND] == 1.0
    assert by_name["soft_tissue"][labels == Tissue.SCALP] == 1.0
    assert by_name["soft_tissue"][labels == Tissue.WHITE] == 1.0
    assert by_name["cortical_bone"][skull_mask(labels)] == pytest.approx(0.75)
    assert by_name["soft_tissue"][skull_mask(labels)] == pytest.approx(0.25)


def test_shape_mismatch_is_an_error():
    labels = all_labels()
    wrong = np.zeros((3, 3), dtype=np.float32)
    right = np.zeros(labels.shape, dtype=np.float32)
    with pytest.raises(ValueError):
        acoustic_properties(labels, wrong, right, PARAMS)
    with pytest.raises(ValueError):
        material_fractions(labels, wrong)
