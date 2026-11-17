"""Smooth Gaussian random fields: on a 3D grid (FFT filtering) and on the unit sphere.

Both kinds are stationary, zero-mean, unit-variance fields with the squared-
exponential covariance ``exp(-d**2 / (2 * correlation_length**2))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft

from neurosim.common.grid import Grid3D


def grid_random_field(
    rng: np.random.Generator, grid: Grid3D, correlation_length: float
) -> np.ndarray:
    """Gaussian random field on a grid, by filtering white noise in the Fourier domain.

    The field is periodic over the grid box and its realisation is tied to the
    grid (another shape or spacing gives another realisation).

    rng: generator of the stream that owns this field.
    grid: grid the field lives on.
    correlation_length: correlation length of the squared-exponential covariance [m].

    Returns a float32 array of shape ``grid.shape`` with zero mean and unit
    variance over the grid.
    """
    noise = rng.standard_normal(grid.shape, dtype=np.float32)
    spectrum = fft.rfftn(noise, workers=1)
    del noise
    # Amplitude filter exp(-k^2 l^2 / 4), i.e. power spectrum exp(-k^2 l^2 / 2),
    # the transform of the squared-exponential covariance. Applied axis by axis.
    for axis, n in enumerate(grid.shape):
        frequencies = fft.rfftfreq(n) if axis == 2 else fft.fftfreq(n)
        wavenumber = 2.0 * np.pi * frequencies / grid.spacing
        gain = np.exp(-0.25 * (wavenumber * correlation_length) ** 2).astype(np.float32)
        shape = [1, 1, 1]
        shape[axis] = gain.size
        spectrum *= gain.reshape(shape)
    values = fft.irfftn(spectrum, s=grid.shape, workers=1)
    del spectrum
    values -= values.mean(dtype=np.float64)
    values /= values.std(dtype=np.float64)
    return values


@dataclass(frozen=True)
class SphereFields:
    """A bank of independent smooth random fields on the unit sphere.

    Each field is a sum of random plane waves evaluated on unit vectors (random
    Fourier features, Rahimi & Recht 2007): ``g(n) = sqrt(2 / K) * sum_k
    cos(w_k . n + p_k)`` with ``w_k ~ N(0, I / l**2)`` and ``p_k ~ U(0, 2 pi)``.
    Its covariance is ``exp(-|n - n'|**2 / (2 l**2))``: `l` is a correlation
    length in chord length on the unit sphere, close to the angle in radians.
    The fields do not depend on any grid.

    frequencies: `w`, float32, shape (n_fields, n_modes, 3) [rad per unit chord].
    phases: `p`, float32, shape (n_fields, n_modes) [rad].
    """

    frequencies: np.ndarray
    phases: np.ndarray

    @classmethod
    def draw(
        cls, rng: np.random.Generator, correlation_lengths: tuple[float, ...], n_modes: int
    ) -> SphereFields:
        """Draw one field per entry of `correlation_lengths` (chord length, about radians)."""
        lengths = np.asarray(correlation_lengths, dtype=np.float64)
        frequencies = rng.standard_normal((lengths.size, n_modes, 3)) / lengths[:, None, None]
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(lengths.size, n_modes))
        return cls(frequencies.astype(np.float32), phases.astype(np.float32))

    def evaluate(self, directions: np.ndarray) -> np.ndarray:
        """Evaluate every field at unit vectors.

        directions: unit vectors, shape (n, 3).
        Returns float32 values of shape (n_fields, n); unit variance over realisations.
        """
        n_fields, n_modes, _ = self.frequencies.shape
        directions = np.asarray(directions, dtype=np.float32)
        arguments = directions @ self.frequencies.reshape(n_fields * n_modes, 3).T
        arguments += self.phases.reshape(1, n_fields * n_modes)
        np.cos(arguments, out=arguments)
        waves = arguments.reshape(directions.shape[0], n_fields, n_modes)
        return (np.sqrt(2.0 / n_modes) * waves.sum(axis=2)).T.astype(np.float32)
