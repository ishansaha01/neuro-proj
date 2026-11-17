"""reference_greens: the fast asymptotic route equals the Hankel-function route."""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.common.constants import C_REF
from neurosim.fus import reference
from neurosim.fus.types import Transducer

# Stated equivalence tolerance of the two routes (relative).
TOLERANCE = 1e-9
WAVELENGTH = C_REF / 2.0e6


def _hankel(
    transducer: Transducer, xz: np.ndarray, freqs: np.ndarray, n_sub: int = 5
) -> np.ndarray:
    k = 2.0 * np.pi * freqs / C_REF
    return reference._greens_hankel(reference._source_x(transducer, n_sub), xz[:, 0], xz[:, 1], k)


def _points(rng: np.random.Generator, n: int, z_min: float) -> np.ndarray:
    return np.stack((rng.uniform(-0.03, 0.03, n), rng.uniform(z_min, z_min + 0.03, n)), axis=1)


@pytest.mark.parametrize(
    ("pitch_wavelengths", "band", "nf"),
    [
        (0.5, (0.9e6, 3.1e6), 40),  # default array, band truncated at -20 dB
        (0.5, (0.45e6, 3.55e6), 64),  # wide band (-40 dB)
        (1.0, (0.9e6, 3.1e6), 33),  # wide elements
    ],
)
def test_asymptotic_route_matches_hankel(pitch_wavelengths, band, nf):
    pitch = pitch_wavelengths * WAVELENGTH
    transducer = Transducer(24, pitch, 0.9 * pitch, 2.0e6, 0.6)
    freqs = np.linspace(*band, nf)
    # Closest allowed range: k r = 20 at the lowest frequency, on axis and off the array end.
    z_min = 1.000001 * reference._ASYMPTOTIC_MIN_ARGUMENT * C_REF / (2.0 * np.pi * freqs[0])
    xz = _points(np.random.default_rng(1), 150, z_min)
    xz[0] = (0.0, z_min)
    xz[1] = (0.03, z_min)
    fast = reference.reference_greens(transducer, xz, freqs, C_REF)
    exact = _hankel(transducer, xz, freqs)
    assert np.max(np.abs(fast - exact) / np.abs(exact)) < TOLERANCE


def test_asymptotic_route_is_actually_used(monkeypatch):
    transducer = Transducer(8, 0.5 * WAVELENGTH, 0.45 * WAVELENGTH, 2.0e6, 0.6)
    xz = _points(np.random.default_rng(2), 5, 0.02)

    def fail(*args, **kwargs):
        raise AssertionError("Hankel route used")

    monkeypatch.setattr(reference, "_greens_hankel", fail)
    reference.reference_greens(transducer, xz, np.linspace(1e6, 3e6, 8), C_REF)


@pytest.mark.parametrize(
    ("freqs", "z"),
    [
        (np.linspace(1e6, 3e6, 8), 0.002),  # too close to the array: k r < 20
        (np.array([1.0e6, 1.5e6, 2.7e6]), 0.03),  # frequencies not uniformly spaced
        (np.array([2.0e6]), 0.03),  # a single frequency
    ],
)
def test_falls_back_to_hankel(monkeypatch, freqs, z):
    transducer = Transducer(8, 0.5 * WAVELENGTH, 0.45 * WAVELENGTH, 2.0e6, 0.6)
    xz = np.array([[0.001, z], [-0.002, z + 0.001]])
    expected = _hankel(transducer, xz, freqs)

    def fail(*args, **kwargs):
        raise AssertionError("asymptotic route used")

    monkeypatch.setattr(reference, "_greens_asymptotic", fail)
    np.testing.assert_allclose(
        reference.reference_greens(transducer, xz, freqs, C_REF), expected, rtol=1e-13
    )


def test_result_does_not_depend_on_block_size(monkeypatch):
    transducer = Transducer(8, 0.5 * WAVELENGTH, 0.45 * WAVELENGTH, 2.0e6, 0.6)
    freqs = np.linspace(1e6, 3e6, 12)
    xz = _points(np.random.default_rng(3), 37, 0.02)
    whole = reference.reference_greens(transducer, xz, freqs, C_REF)
    monkeypatch.setattr(reference, "_BLOCK_SIZE", 12 * 8 * 5 * 4)  # four points per block
    np.testing.assert_allclose(
        reference.reference_greens(transducer, xz, freqs, C_REF), whole, rtol=1e-12
    )


def test_shapes_and_other_n_sub():
    transducer = Transducer(8, 0.5 * WAVELENGTH, 0.45 * WAVELENGTH, 2.0e6, 0.6)
    freqs = np.linspace(1e6, 3e6, 6)
    xz = _points(np.random.default_rng(4), 12, 0.02).reshape(3, 4, 2)
    out = reference.reference_greens(transducer, xz, freqs, C_REF, n_sub=3)
    assert out.shape == (6, 8, 3, 4)
    assert out.dtype == np.complex128
    np.testing.assert_allclose(
        out.reshape(6, 8, 12),
        _hankel(transducer, xz.reshape(-1, 2), freqs, n_sub=3),
        rtol=TOLERANCE,
    )
    assert reference.reference_greens(transducer, xz[:0, 0], freqs, C_REF).shape == (6, 8, 0)


def test_point_source_value():
    """A single point source at the element centre gives (i/4) H0(k r) of the textbook."""
    from scipy.special import hankel1

    transducer = Transducer(1, 0.5 * WAVELENGTH, 0.45 * WAVELENGTH, 2.0e6, 0.6)
    freqs = np.linspace(1e6, 3e6, 5)
    xz = np.array([[0.003, 0.04]])
    r = np.hypot(0.003, 0.04)
    expected = 0.25j * hankel1(0, 2.0 * np.pi * freqs / C_REF * r)
    out = reference.reference_greens(transducer, xz, freqs, C_REF, n_sub=1)
    np.testing.assert_allclose(out[:, 0, 0], expected, rtol=TOLERANCE)
