"""Likelihood of a power Doppler image: the one definition every method uses.

Speckle model (`fus.acquire.statistical_power_doppler`; Goodman, Statistical
Optics, 1985, on the sum of independent speckle intensities): a pixel of one
frame is a circular complex Gaussian (blood with random phases plus electronic
noise), so its power is exponential about its mean, and the average of `L`
independent frames is gamma distributed with shape `L` and that mean,

    y_p ~ Gamma(shape = L, mean = mu_p),   mu = H (cbv * cell_area) + noise_floor
    log p(y | mu) = sum_p [ L log L - log Gamma(L) + (L - 1) log y_p - L log mu_p - L y_p / mu_p ]

The floor is part of the mean, not an additive noise term: blood and noise add in
the complex frame, and the power of the sum is what is averaged.

This density is exact for the statistical data tier. The approximations are in
what it leaves out, and they are the tier's own (same docstring): pixels are
taken as independent (real speckle is correlated over a resolution cell, so for
scatterer-fidelity data this is a composite likelihood that overstates the
information by about the number of pixels per resolution cell); blood and noise
share one shape `L`; the SVD filter's loss is ignored; `H` is treated as exact. A
Gaussian approximation was the alternative; it is not used because the gamma
density costs the same and stays valid for small `L` and for pixels at the floor.

An average of `n` images with different means is not exactly gamma. For the
time-averaged image used in calibration it is approximated by a gamma with shape
`n L` and the average mean, which matches the first two moments when the means
differ by much less than themselves (a task response of tens of percent in a
small region).
"""

from __future__ import annotations

import math

import torch


def gamma_log_likelihood(
    measured: torch.Tensor, expected: torch.Tensor, shape: float
) -> torch.Tensor:
    """Log density of measured power Doppler images given their expected images.

    measured: power Doppler values (> 0), shape (batch, n_pix).
    expected: expected values `mu` (> 0), same shape.
    shape: gamma shape `L`, the number of independent speckle realisations per image.

    Returns the log density summed over pixels [nat], shape (batch,).
    """
    constant = shape * math.log(shape) - math.lgamma(shape)
    per_pixel = (
        constant
        + (shape - 1.0) * torch.log(measured)
        - shape * torch.log(expected)
        - shape * measured / expected
    )
    return per_pixel.sum(dim=-1)


def gamma_deviance(measured: torch.Tensor, expected: torch.Tensor, shape: float) -> torch.Tensor:
    """Deviance of the gamma model: twice the log-likelihood gap to a perfect fit.

    ``D = 2 L sum_p [ y_p / mu_p - 1 - log(y_p / mu_p) ]``, zero when `expected`
    equals `measured`. When the model is right, `D` is approximately chi-square
    with one degree of freedom per pixel (mean `n_pix` up to a factor
    ``1 + 1 / (6 L)``), so ``D / n_pix`` near 1 is a fit at the speckle level and a
    larger value measures model mismatch.

    Arguments as for `gamma_log_likelihood`. Returns shape (batch,).
    """
    ratio = measured / expected
    return 2.0 * shape * (ratio - 1.0 - torch.log(ratio)).sum(dim=-1)
