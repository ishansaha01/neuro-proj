"""Gauge directions of a phase screen's delays (see `recon.calibrate` for why they exist).

A common delay of all elements and a linear tilt of the delays across the
aperture cannot be separated by the power Doppler image from a shift or a
rotation of the unknown CBV image. This module names those directions, builds
aperture modes free of them, and removes them from a delay profile.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct


def dct_modes(n_elements: int) -> np.ndarray:
    """Orthonormal DCT-II modes across the aperture as columns, smoothest (constant) first.

    Returns float64, shape (n_elements, n_elements).
    """
    return dct(np.eye(n_elements), type=2, norm="ortho", axis=0).T


def gauge_directions(element_x: np.ndarray, gauge: str) -> np.ndarray:
    """Gauge directions of the delays as columns, shape (n_elements, 1 or 2)."""
    constant = np.ones_like(element_x)
    if gauge == "mean":
        return constant[:, None]
    if gauge == "mean_tilt":
        return np.stack((constant, element_x - element_x.mean()), axis=1)
    raise ValueError(f"unknown gauge {gauge!r}")


def gauge_free_basis(element_x: np.ndarray, gauge: str) -> np.ndarray:
    """Orthonormal aperture modes orthogonal to the gauge directions, smoothest first.

    The DCT-II modes across the elements, made orthogonal to the gauge directions
    and to each other in order of increasing spatial frequency (Gram-Schmidt). Mode
    `k` has `k + n_gauge` sign changes across the aperture, give or take one
    (removing the tilt adds two sign changes to the odd modes).

    element_x: lateral element positions [m], shape (n_elements,).
    gauge: "mean" or "mean_tilt" (`CalibrationConfig.gauge`).

    Returns float64, shape (n_elements, n_elements - n_gauge), orthonormal columns.
    """
    n = element_x.shape[0]
    modes = dct_modes(n)
    directions = gauge_directions(element_x, gauge)
    n_gauge = directions.shape[1]
    stacked = np.concatenate((directions, modes[:, 1:]), axis=1)
    q, _ = np.linalg.qr(stacked)
    return q[:, n_gauge:n]


def remove_gauge(delays: np.ndarray, element_x: np.ndarray, gauge: str) -> np.ndarray:
    """Project the gauge directions out of a delay profile (or a difference of two).

    delays: [s], shape (n_elements,).
    Returns the least-squares residual after fitting the gauge directions, same shape.
    """
    directions = gauge_directions(element_x, gauge)
    coefficients, *_ = np.linalg.lstsq(directions, delays, rcond=None)
    return delays - directions @ coefficients
