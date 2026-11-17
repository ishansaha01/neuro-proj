"""Likelihood of a power Doppler image: the one definition every method uses.

Generative model (`fus.acquire.statistical_power_doppler`, `fus.speckle`). A
pixel of one frame is a circular complex Gaussian: blood with random phases plus
electronic noise. The image is the mean power of the ensemble, with mean

    mu_p = b_p + f_p,   b = H (cbv * cell_area),   f = noise_floor

The floor is part of the mean, not an additive noise term: blood and noise add in
the complex frame, and the power of the sum is what is averaged. Successive blood
frames are correlated, so the blood of pixel `p` counts as `N_b[p]` independent
frames (`Scene.effective_frames`, between 1 and the ensemble length); the noise
is new in every frame and counts as `N_n` frames (`Scene.noise_frames`). The
variance is

    var_p = b_p^2 / N_b[p] + (2 b_p f_p + f_p^2) / N_n
          = b_p^2 / N_b[p] + (mu_p^2 - b_p^2) / N_n

The exact density (a gamma mixture of non-central chi-square densities) has no
closed form. It is approximated by the gamma density with the same mean and
variance (`speckle_shape`),

    y_p ~ Gamma(shape = L_p, mean = mu_p),   L_p = mu_p^2 / var_p
    log p(y | mu) = sum_p [ L_p log L_p - log Gamma(L_p) + (L_p - 1) log y_p
                            - L_p log mu_p - L_p y_p / mu_p ]

`L_p` depends on the CBV through `b_p`, so it is recomputed with the expected
image and differentiated with it. The approximation is exact when
`N_b = N_n = L` (then `L_p = L` whatever the floor: the single-shape model used
before the blood count became per pixel) and for a pixel without floor
(`L_p = N_b[p]`). It is poorest for slow blood well below the floor, where the
true density is close to a gamma density of shape `N_b` shifted by the floor and
is more skewed than the matched gamma.

What the model leaves out is the statistical data tier's own list: pixels are
taken as independent (real speckle is correlated over a resolution cell, so for
scatterer-fidelity data this is a composite likelihood that overstates the
information by about the number of pixels per resolution cell); the SVD filter's
loss is ignored; `H` is treated as exact; and `N_b` is taken as known, although
it follows from the blood velocity and the baseline CBV, which a scanner does not
know (it could estimate `N_b` from the slow-time autocorrelation of the frames).

An average of `n` images with different means is not exactly of this form. For
the time-averaged image used in calibration both counts are multiplied by `n`
and the average mean is used, which matches the first two moments when the means
differ by much less than themselves (a task response of tens of percent in a
small region).
"""

from __future__ import annotations

import torch


def speckle_shape(
    expected: torch.Tensor,
    noise_floor: torch.Tensor,
    blood_frames: torch.Tensor | float,
    noise_frames: float,
) -> torch.Tensor:
    """Gamma shape `L = mean^2 / variance` of each pixel under the generative model.

    expected: expected power Doppler values `mu = b + f` (> 0), shape (..., n_pix).
    noise_floor: electronic-noise floor `f` (0 <= f <= mu), shape (n_pix,).
    blood_frames: independent frames `N_b` of the blood signal, scalar or shape (n_pix,).
    noise_frames: independent frames `N_n` of the electronic noise.

    Returns the shape of `expected`, values between `min(N_b, N_n)` and `max(N_b, N_n)`.
    """
    blood_squared = (expected - noise_floor).square()
    expected_squared = expected.square()
    variance = blood_squared / blood_frames + (expected_squared - blood_squared) / noise_frames
    return expected_squared / variance


def gamma_log_likelihood(
    measured: torch.Tensor, expected: torch.Tensor, shape: torch.Tensor | float
) -> torch.Tensor:
    """Log density of measured power Doppler images given their expected images.

    measured: power Doppler values (> 0), shape (batch, n_pix).
    expected: expected values `mu` (> 0), same shape.
    shape: gamma shape `L` of each pixel (`speckle_shape`), a scalar or a tensor
        broadcastable to `expected`.

    Returns the log density summed over pixels [nat], shape (batch,).
    """
    shape = torch.as_tensor(shape, dtype=expected.dtype)
    per_pixel = (
        shape * torch.log(shape)
        - torch.lgamma(shape)
        + (shape - 1.0) * torch.log(measured)
        - shape * torch.log(expected)
        - shape * measured / expected
    )
    return per_pixel.sum(dim=-1)


def gamma_deviance(
    measured: torch.Tensor, expected: torch.Tensor, shape: torch.Tensor | float
) -> torch.Tensor:
    """Deviance of the gamma model about its mean, at the model's own shape.

    ``D = 2 sum_p L_p [ y_p / mu_p - 1 - log(y_p / mu_p) ]``, zero when `expected`
    equals `measured`; for a shape that does not depend on `expected` it is twice
    the log-likelihood gap to a perfect fit. When the model is right, `D` is
    approximately chi-square with one degree of freedom per pixel (each term has
    mean ``1 + 1 / (6 L_p)``), so ``D / n_pix`` near 1 is a fit at the speckle
    level and a larger value measures model mismatch.

    Arguments as for `gamma_log_likelihood`. Returns shape (batch,).
    """
    ratio = measured / expected
    return 2.0 * (shape * (ratio - 1.0 - torch.log(ratio))).sum(dim=-1)


def power_doppler_log_likelihood(
    measured: torch.Tensor,
    expected: torch.Tensor,
    noise_floor: torch.Tensor,
    blood_frames: torch.Tensor | float,
    noise_frames: float,
) -> torch.Tensor:
    """Log-likelihood of measured images under the speckle model of the module docstring.

    `gamma_log_likelihood` with the moment-matched shape of `speckle_shape`.

    measured: power Doppler values (> 0), shape (batch, n_pix).
    expected: expected values `mu = b + f`, same shape.
    noise_floor: floor `f` of the pixels, shape (n_pix,).
    blood_frames: independent blood frames of the pixels, scalar or shape (n_pix,).
    noise_frames: independent noise frames.

    Returns shape (batch,) [nat].
    """
    shape = speckle_shape(expected, noise_floor, blood_frames, noise_frames)
    return gamma_log_likelihood(measured, expected, shape)


def power_doppler_deviance(
    measured: torch.Tensor,
    expected: torch.Tensor,
    noise_floor: torch.Tensor,
    blood_frames: torch.Tensor | float,
    noise_frames: float,
) -> torch.Tensor:
    """`gamma_deviance` with the moment-matched shape of `speckle_shape`.

    Arguments as for `power_doppler_log_likelihood`. Returns shape (batch,).
    """
    shape = speckle_shape(expected, noise_floor, blood_frames, noise_frames)
    return gamma_deviance(measured, expected, shape)
