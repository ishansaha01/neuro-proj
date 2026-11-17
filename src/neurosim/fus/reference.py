"""Green's function of the homogeneous reference medium (docs/ARCHITECTURE.md, section 5).

This is the `G_ref` that every `GreensBank` is defined relative to, so it has
exactly one implementation, `reference_greens`. It evaluates the same quantity
by the cheaper of two routes, which agree to better than 1e-9 relative
(`tests/unit/fus/test_reference.py`):

1. `_greens_hankel`: the Hankel function itself (`scipy.special.hankel1`).
   Valid everywhere, about 180 ns per (frequency, source point, field point).
2. `_greens_asymptotic`: for uniformly spaced frequencies and `k r >= 20`
   everywhere (about three wavelengths from the array), the large-argument
   expansion of the Hankel function, with the phase advanced from one frequency
   to the next by a multiplication instead of a new exponential. About 8 ns per
   (frequency, source point, field point).
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1

from neurosim.fus.types import Transducer

# Smallest k*r for which the asymptotic expansion is used [rad].
_ASYMPTOTIC_MIN_ARGUMENT = 20.0
# Relative truncation error allowed for the expansion. The error is bounded by the
# first omitted term (DLMF 10.17(iii)); the number of terms is chosen per call from
# the smallest k*r so that this term is below the tolerance.
_ASYMPTOTIC_TOLERANCE = 2e-11
# Largest number of terms ever needed: |a_10| / 20**10 = 1.1e-11 at the smallest argument.
_ASYMPTOTIC_MAX_TERMS = 10
# Largest relative deviation from uniform spacing for a frequency grid to count as uniform.
_UNIFORM_TOLERANCE = 1e-9
# Work-array size of the asymptotic route [complex numbers]: nf * source points *
# field points per block. Run time is flat between 0.25e6 and 4e6 on the development machine.
_BLOCK_SIZE = 1_000_000


def _hankel0_coefficients(n_terms: int) -> np.ndarray:
    """Coefficients a_n(0) of the large-argument expansion of H0^(1).

    `H0^(1)(x) ~ sqrt(2 / (pi x)) exp(i (x - pi/4)) sum_n i^n a_n / x^n` with
    `a_0 = 1`, `a_n = a_{n-1} * (-(2n - 1)^2) / (8 n)` (DLMF 10.17.1 and 10.17.5,
    https://dlmf.nist.gov/10.17; Abramowitz & Stegun 9.2.7).

    Returns:
        float64 array, shape (n_terms,) [dimensionless].
    """
    coefficients = np.ones(n_terms)
    for n in range(1, n_terms):
        coefficients[n] = coefficients[n - 1] * (-((2 * n - 1) ** 2)) / (8.0 * n)
    return coefficients


def _n_asymptotic_terms(min_argument: float) -> int:
    """Smallest even number of terms whose first omitted term is below the tolerance."""
    coefficients = _hankel0_coefficients(_ASYMPTOTIC_MAX_TERMS + 1)
    for n_terms in range(2, _ASYMPTOTIC_MAX_TERMS, 2):
        if abs(coefficients[n_terms]) / min_argument**n_terms <= _ASYMPTOTIC_TOLERANCE:
            return n_terms
    return _ASYMPTOTIC_MAX_TERMS


def _is_uniform(k: np.ndarray) -> bool:
    """True if the wavenumbers are ascending and uniformly spaced (at least two)."""
    if k.shape[0] < 2:
        return False
    steps = np.diff(k)
    deviation = np.abs(steps - steps[0])
    return bool(steps[0] > 0.0 and np.all(deviation <= _UNIFORM_TOLERANCE * steps[0]))


def _source_x(transducer: Transducer, n_sub: int) -> np.ndarray:
    """Lateral positions of the point sources of every element [m], shape (n_sub, n_elements)."""
    offsets = (np.arange(n_sub) - (n_sub - 1) / 2.0) / n_sub * transducer.element_width
    return transducer.element_x()[None, :] + offsets[:, None]


def _greens_hankel(source_x: np.ndarray, x: np.ndarray, z: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Element-averaged `(i/4) H0^(1)(k r)` from the Hankel function.

    Args:
        source_x: point-source positions [m], shape (n_sub, n_elements).
        x, z: field-point coordinates [m], shape (n_points,).
        k: wavenumbers [rad/m], shape (nf,).

    Returns:
        complex128 array, shape (nf, n_elements, n_points).
    """
    n_sub, n_elements = source_x.shape
    out = np.zeros((k.shape[0], n_elements, x.shape[0]), dtype=np.complex128)
    for sub_x in source_x:
        r = np.hypot(x[None, :] - sub_x[:, None], z[None, :])  # (n_elements, n_points)
        out += 0.25j * hankel1(0, k[:, None, None] * r[None, :, :])
    out /= n_sub
    return out


def _greens_asymptotic(
    source_x: np.ndarray, x: np.ndarray, z: np.ndarray, k: np.ndarray, n_terms: int
) -> np.ndarray:
    """Element-averaged `(i/4) H0^(1)(k r)` from the large-argument expansion.

    For uniformly spaced `k` and an even `n_terms`. Per source-point distance `r`:

    - phase `exp(i k_f r)`: computed once at the first frequency and advanced by
      `exp(i dk r)` (doubling the filled rows each step, so rounding errors grow
      with log2(nf) rather than nf);
    - amplitude `sqrt(2 / (pi k r)) sum_n i^n a_n / (k r)^n`: separable in `k` and
      `r`, so all frequencies come from one real matrix product. Even `n` are the
      real part and odd `n` the imaginary part; both are written straight into
      the interleaved (re, im) memory of a complex array.

    Arguments and return value as for `_greens_hankel`.
    """
    n_sub, n_elements = source_x.shape
    nf, n_points = k.shape[0], x.shape[0]
    n_sources = n_sub * n_elements
    block = max(1, _BLOCK_SIZE // (nf * n_sources))

    # frequency_factor[f, n] = i^n a_n sqrt(2/pi) k_f^-(n + 1/2), as a real number that
    # multiplies 1 (even n) or i (odd n).
    orders = np.arange(n_terms)
    signs = np.where(orders // 2 % 2 == 0, 1.0, -1.0)
    frequency_factor = (
        np.sqrt(2.0 / np.pi)
        * signs
        * _hankel0_coefficients(n_terms)
        * k[:, None] ** (-(orders + 0.5))
    )
    step_wavenumber = (k[-1] - k[0]) / (nf - 1)
    # (i/4) exp(-i pi/4) / n_sub: Green's-function prefactor, expansion phase and element mean.
    prefactor = 0.25j * np.exp(-0.25j * np.pi) / n_sub

    out = np.empty((nf, n_elements, n_points), dtype=np.complex128)
    for start in range(0, n_points, block):
        stop = min(start + block, n_points)
        n_block = stop - start
        m = n_sources * n_block
        r = np.hypot(
            x[None, None, start:stop] - source_x[:, :, None], z[None, None, start:stop]
        ).reshape(m)

        phase = np.empty((nf, m), dtype=np.complex128)
        np.exp(1j * (k[0] * r), out=phase[0])
        step = np.exp(1j * (step_wavenumber * r))
        filled = 1
        while filled < nf:
            count = min(filled, nf - filled)
            np.multiply(phase[:count], step, out=phase[filled : filled + count])
            filled += count
            step = step * step

        # distance_factor[n, j, n odd] = r_j^-(n + 1/2); the other half stays zero.
        inverse_r = 1.0 / r
        inverse_r2 = inverse_r * inverse_r
        distance_factor = np.zeros((n_terms, m, 2))
        distance_factor[0, :, 0] = np.sqrt(inverse_r)
        distance_factor[1, :, 1] = distance_factor[0, :, 0] * inverse_r
        for n in range(2, n_terms):
            previous = distance_factor[n - 2, :, n % 2]
            np.multiply(previous, inverse_r2, out=distance_factor[n, :, n % 2])

        amplitude = np.empty((nf, m), dtype=np.complex128)
        np.matmul(
            frequency_factor,
            distance_factor.reshape(n_terms, 2 * m),
            out=amplitude.view(np.float64),
        )
        element_sum = np.einsum(
            "fsj,fsj->fj",
            amplitude.reshape(nf, n_sub, n_elements * n_block),
            phase.reshape(nf, n_sub, n_elements * n_block),
        )
        out[:, :, start:stop] = element_sum.reshape(nf, n_elements, n_block)
    out *= prefactor
    return out


def reference_greens(
    transducer: Transducer,
    xz: np.ndarray,
    freqs: np.ndarray,
    c_ref: float,
    n_sub: int = 5,
) -> np.ndarray:
    """One-way 2D free-space transfer function from each array element to each point.

    A point (line) source in 2D radiates `(i/4) H0^(1)(k r)` under the
    exp(-i w t) convention (Morse & Ingard, Theoretical Acoustics, 1968, eq. 7.3.14).
    A finite element is the mean of `n_sub` point sources spread uniformly over
    its width, which gives the element directivity.

    Args:
        transducer: the array (elements at z = 0).
        xz: field points [m], shape (..., 2), columns (x, z); z > 0.
        freqs: frequencies [Hz], shape (nf,).
        c_ref: sound speed of the reference medium [m/s].
        n_sub: point sources per element (odd, so one sits at the element centre).

    Returns:
        complex128 array, shape (nf, n_elements, ...) [dimensionless, 2D Green's function].
    """
    xz = np.asarray(xz, dtype=np.float64)
    field_shape = xz.shape[:-1]
    x = xz[..., 0].reshape(-1)
    z = xz[..., 1].reshape(-1)
    k = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64) / c_ref  # [rad/m], (nf,)
    source_x = _source_x(transducer, n_sub)

    min_argument = 0.0
    if x.shape[0] > 0 and _is_uniform(k):
        # Smallest distance from any point to the array line segment that holds the sources.
        lateral_gap = np.maximum(0.0, np.maximum(source_x.min() - x, x - source_x.max()))
        min_argument = k[0] * np.hypot(lateral_gap, z).min()
    if min_argument < _ASYMPTOTIC_MIN_ARGUMENT:
        out = _greens_hankel(source_x, x, z, k)
    else:
        out = _greens_asymptotic(source_x, x, z, k, _n_asymptotic_terms(min_argument))
    return out.reshape(k.shape[0], source_x.shape[1], *field_shape)
