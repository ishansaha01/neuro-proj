import math

import numpy as np
import pytest

from neurosim.common import units


def test_db_to_np_hand_value():
    # 20 dB is an amplitude ratio of 10, and ln(10) Np.
    assert units.db_to_np(20.0) == pytest.approx(math.log(10.0), rel=1e-12)


def test_np_to_db_hand_value():
    # 1 Np = 20 / ln(10) dB = 8.685889638 dB.
    assert units.np_to_db(1.0) == pytest.approx(8.685889638, rel=1e-9)


def test_db_np_round_trip():
    levels_db = np.array([-60.0, -3.0, 0.0, 0.5, 40.0])
    np.testing.assert_allclose(units.np_to_db(units.db_to_np(levels_db)), levels_db, rtol=1e-12)


def test_db_cm_mhz_to_np_m_hand_values():
    # 0.5 dB/(cm MHz) at 2 MHz, y = 1: 1 dB/cm = 100 dB/m = 100 * ln(10)/20 Np/m.
    assert units.db_cm_mhz_to_np_m(0.5, 2.0e6, 1.0) == pytest.approx(11.51292546, rel=1e-9)
    # Same prefactor with y = 2: 0.5 * 2**2 = 2 dB/cm.
    assert units.db_cm_mhz_to_np_m(0.5, 2.0e6, 2.0) == pytest.approx(23.02585093, rel=1e-9)
    # At 1 MHz the exponent does not matter.
    assert units.db_cm_mhz_to_np_m(8.0, 1.0e6, 1.3) == pytest.approx(92.10340372, rel=1e-9)


def test_np_m_to_db_cm_hand_value_and_round_trip():
    assert units.np_m_to_db_cm(11.51292546) == pytest.approx(1.0, rel=1e-9)
    alpha0 = np.array([0.2, 0.6, 8.0])
    np.testing.assert_allclose(
        units.np_m_to_db_cm(units.db_cm_mhz_to_np_m(alpha0, 1.0e6, 1.1)), alpha0, rtol=1e-12
    )


def test_hu_from_mu_hand_values():
    mu_water, mu_air = 20.0, 0.02
    assert units.hu_from_mu(mu_water, mu_water, mu_air) == 0.0
    assert units.hu_from_mu(mu_air, mu_water, mu_air) == pytest.approx(-1000.0, rel=1e-12)
    # Twice the attenuation of water with mu_air = 0: +1000 HU.
    assert units.hu_from_mu(40.0, 20.0, 0.0) == pytest.approx(1000.0, rel=1e-12)


def test_mu_from_hu_hand_value_and_round_trip():
    assert units.mu_from_hu(500.0, 20.0, 0.0) == pytest.approx(30.0, rel=1e-12)
    hu = np.array([-1000.0, -100.0, 0.0, 40.0, 1800.0])
    np.testing.assert_allclose(
        units.hu_from_mu(units.mu_from_hu(hu, 19.3, 0.021), 19.3, 0.021), hu, rtol=1e-12, atol=1e-9
    )


def test_wavelength_hand_value():
    # 1540 m/s at 2 MHz: 0.77 mm.
    assert units.wavelength(1540.0, 2.0e6) == pytest.approx(7.7e-4, rel=1e-12)


def test_conversions_are_vectorised_and_keep_float32():
    alpha0 = np.full((2, 3), 0.5, dtype=np.float32)
    frequencies = np.array([1.0e6, 2.0e6, 3.0e6], dtype=np.float32)
    alpha = units.db_cm_mhz_to_np_m(alpha0, frequencies, 1.0)
    assert alpha.shape == (2, 3)
    assert alpha.dtype == np.float32
    np.testing.assert_allclose(alpha[0], 0.5 * 11.51292546 * np.array([1, 2, 3]), rtol=1e-6)
    assert units.hu_from_mu(np.zeros((4, 5), dtype=np.float32), 20.0, 0.0).dtype == np.float32
