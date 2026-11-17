import numpy as np
import pytest

from neurosim.common.constants import C_REF
from neurosim.fus.backends import BackendConfig, compute_bank
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium

FREQS = np.linspace(0.4e6, 0.6e6, 3)
ANALYTIC = BackendConfig(backend="analytic")


@pytest.mark.parametrize("density", [1000.0, 1050.0])
def test_reference_medium_gives_unit_bank_whatever_the_density(grid, transducer, roi, density):
    medium = homogeneous_medium(grid, Material(C_REF, density), alpha_power=1.5)
    bank = compute_bank(medium, transducer, roi, FREQS, ANALYTIC)
    assert bank.backend == "analytic" and bank.c_ref == C_REF and bank.static_fmc is None
    assert bank.B.shape == (3, 3, 3, 5) and bank.B.dtype == np.complex64
    assert np.all(bank.B == 1.0)


@pytest.mark.parametrize(
    "material, message",
    [
        (Material(1500.0, 1000.0), "sound_speed == C_REF"),
        (Material(C_REF, 1000.0, attenuation=0.5), "lossless"),
    ],
)
def test_other_homogeneous_media_are_refused(grid, transducer, roi, material, message):
    medium = homogeneous_medium(grid, material, alpha_power=1.5)
    with pytest.raises(ValueError, match=message):
        compute_bank(medium, transducer, roi, FREQS, ANALYTIC)


def test_heterogeneous_media_are_refused(grid, soft, transducer, roi, spacing):
    faster = slab_medium(grid, soft, Material(2000.0, 1000.0), 9.5 * spacing, 3 * spacing, 1.5)
    with pytest.raises(ValueError, match="sound_speed == C_REF"):
        compute_bank(faster, transducer, roi, FREQS, ANALYTIC)
    denser = slab_medium(grid, soft, Material(C_REF, 1500.0), 9.5 * spacing, 3 * spacing, 1.5)
    with pytest.raises(ValueError, match="uniform density"):
        compute_bank(denser, transducer, roi, FREQS, ANALYTIC)
