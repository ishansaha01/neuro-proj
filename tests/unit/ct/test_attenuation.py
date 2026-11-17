"""basis_attenuation: shapes, units and ordering."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.common.constants import BASIS_MATERIALS
from neurosim.ct.attenuation import DENSITIES, MASS_FRACTIONS, basis_attenuation, mass_attenuation


def test_tables_cover_the_basis_materials():
    assert set(MASS_FRACTIONS) == set(BASIS_MATERIALS) == set(DENSITIES)


@pytest.mark.parametrize("material", BASIS_MATERIALS)
def test_mass_fractions_sum_to_one(material):
    assert sum(MASS_FRACTIONS[material].values()) == pytest.approx(1.0, abs=1e-6)


def test_basis_attenuation_shape_and_order():
    energies = np.array([40.0, 60.0, 80.0, 100.0])
    mu = basis_attenuation(energies)
    assert mu.shape == (len(BASIS_MATERIALS), energies.size)
    air, water, soft_tissue, bone = mu
    assert np.all(air < 1e-2 * water)
    assert np.all(bone > soft_tissue) and np.all(soft_tissue > water)
    assert np.all(np.diff(mu, axis=1) < 0.0)


def test_water_at_60_kev_is_in_si_units():
    # NIST: mu/rho = 0.2059 cm^2/g at 60 keV, so mu is about 20.6 1/m.
    mu_water = basis_attenuation([60.0])[BASIS_MATERIALS.index("water"), 0]
    assert mu_water == pytest.approx(20.6, rel=0.01)
    assert mass_attenuation("water", [60.0])[0] == pytest.approx(0.02059, rel=0.01)


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        mass_attenuation("titanium", [60.0])
