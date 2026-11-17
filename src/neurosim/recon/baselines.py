"""Baselines without a learned prior, on the scene's pixel grid.

- The four raw power Doppler images of a `TrialSet` (`raw_power_doppler`):
  `skull_das` (uncorrected), `skull_ct`, `skull_true` and `clean_das` (no skull:
  the upper bound of what beamforming alone can give). They are in power Doppler
  units, not CBV; every metric that compares them with CBV is scale invariant.
- A classical regularised deconvolution (`gamma_deconvolution`): non-negative
  maximum likelihood under the speckle model of `recon.likelihood`, by
  multiplicative updates stopped early, optionally with a Tikhonov penalty.

The two diffusion baselines (DPS with the skull fixed at the CT screen, and
prior-only samples) are `recon.dps.reconstruct_trials` with
`H(screen_ct)` and `recon.dps.prior_samples`; `recon.experiment` wires them.
"""

from __future__ import annotations

import numpy as np
import torch

from neurosim.data.types import BEAMFORMERS, TrialSet
from neurosim.recon.config import DeconvolutionConfig
from neurosim.recon.forward import expected_power_doppler


def raw_power_doppler(trials: TrialSet) -> dict[str, np.ndarray]:
    """The measured power Doppler series of every beamformer key, float32, (n, nz, nx) each."""
    return {key: np.asarray(trials.power_doppler[key], dtype=np.float32) for key in BEAMFORMERS}


def gamma_deconvolution(
    operator: torch.Tensor,
    noise_floor: torch.Tensor,
    cell_area: float,
    measured: np.ndarray,
    cfg: DeconvolutionConfig,
) -> np.ndarray:
    """Non-negative deconvolution of power Doppler images under gamma speckle.

    Maximises the likelihood of `recon.likelihood.gamma_log_likelihood` over
    ``cbv >= 0`` with ``mu = H (cbv * cell_area) + noise_floor``, i.e. minimises
    ``sum_p [log mu_p + y_p / mu_p]`` (the gamma shape only scales it). Its gradient
    with respect to ``cbv`` is ``cell_area * H^T (1 / mu - y / mu^2)``; the
    multiplicative update divides the negative part by the positive part,

        cbv <- cbv * [H^T (y / mu^2)] / [H^T (1 / mu)]

    This is the multiplicative update for the Itakura-Saito divergence, which is
    the gamma deviance (Fevotte, Bertin & Durrieu 2009, "Nonnegative matrix
    factorization with the Itakura-Saito divergence", Neural Computation
    21(3):793-830, with the dictionary fixed); it is the counterpart for
    multiplicative noise of the Richardson-Lucy update for Poisson noise. It keeps
    the iterate non-negative and its fixed points are stationary points of the
    likelihood. There is no explicit penalty: the iteration starts from a flat
    image whose expected power matches the mean measured power above the floor and
    is stopped after `cfg.n_iterations` updates, which is the regulariser.

    operator: power Doppler operator `H`, float32, shape (n_pix, n_cells).
    noise_floor: float32, shape (n_pix,).
    cell_area: area of one CBV cell [m^2].
    measured: power Doppler images, shape (n, nz, nx).
    cfg: number of updates.

    Returns CBV fraction, float32, shape (n, nz, nx).
    """
    shape = measured.shape
    y = torch.from_numpy(np.ascontiguousarray(measured, dtype=np.float32)).flatten(1)
    column_gain = operator.sum(dim=0) * cell_area  # image power per unit CBV of each cell
    excess = (y - noise_floor).mean(dim=1, keepdim=True).clamp_min(1e-3 * y.mean())
    cbv = (excess / column_gain.mean() * operator.shape[1] / operator.shape[0]).expand(
        -1, operator.shape[1]
    )
    for _ in range(cfg.n_iterations):
        expected = expected_power_doppler(operator, cbv, cell_area, noise_floor)
        numerator = (y / expected.square()) @ operator
        denominator = (1.0 / expected) @ operator
        cbv = cbv * numerator / denominator
    return cbv.reshape(shape).numpy().astype(np.float32)
