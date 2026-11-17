"""CBV signal in time: hemodynamic response, task time course, CBV at a time.

``cbv(x, t) = cbv_baseline(x) * (1 + a(t) * activation(x))`` (docs/ARCHITECTURE.md
section 4, rule 3), with ``a(t)`` the block design convolved with the
hemodynamic response function (HRF).

The HRF is the canonical double-gamma function of SPM (``spm_hrf`` with its
default parameters: response peaking near 5 s, undershoot near 15 s, ratio
1/6, unit dispersion), measured for BOLD fMRI. Using it for the CBV response
seen by fUS is a modelling assumption.
# assumed, to verify: SPM defaults quoted from memory; BOLD HRF used for CBV.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, special

from neurosim.head.config import TaskConfig

# Shape parameters of the two gamma densities (unit scale, time in seconds) and
# the weight of the undershoot: SPM's delays of 6 s and 16 s and ratio 1/6.
_RESPONSE_SHAPE = 6.0
_UNDERSHOOT_SHAPE = 16.0
_UNDERSHOOT_RATIO = 1.0 / 6.0


def _gamma_density(t: np.ndarray, shape: float) -> np.ndarray:
    """Gamma probability density with unit scale; zero for t < 0."""
    positive = np.maximum(t, 0.0)
    return np.exp(special.xlogy(shape - 1.0, positive) - positive - special.gammaln(shape))


def _unnormalised_hrf(t: np.ndarray) -> np.ndarray:
    return _gamma_density(t, _RESPONSE_SHAPE) - _UNDERSHOOT_RATIO * _gamma_density(
        t, _UNDERSHOOT_SHAPE
    )


def _unnormalised_step_response(t: np.ndarray) -> np.ndarray:
    """Integral of the unnormalised HRF from 0 to t (regularised incomplete gamma functions)."""
    positive = np.maximum(t, 0.0)
    return special.gammainc(_RESPONSE_SHAPE, positive) - _UNDERSHOOT_RATIO * special.gammainc(
        _UNDERSHOOT_SHAPE, positive
    )


# Peak of the unnormalised HRF, found once numerically between the mode of the
# response term (shape - 1 = 5 s) and its start.
_HRF_PEAK = -optimize.minimize_scalar(
    lambda t: -_unnormalised_hrf(np.float64(t)),
    bounds=(0.0, _RESPONSE_SHAPE),
    method="bounded",
    options={"xatol": 1.0e-10},
).fun
# The step response is largest where the HRF changes sign, i.e. where the two
# gamma terms are equal: t^(16 - 6) = 6 * Gamma(16) / Gamma(6).
_HRF_ZERO_CROSSING = math.exp(
    (math.lgamma(_UNDERSHOOT_SHAPE) - math.lgamma(_RESPONSE_SHAPE) - math.log(_UNDERSHOOT_RATIO))
    / (_UNDERSHOOT_SHAPE - _RESPONSE_SHAPE)
)
_STEP_RESPONSE_PEAK = float(_unnormalised_step_response(np.float64(_HRF_ZERO_CROSSING)))


def hrf(t: ArrayLike) -> np.ndarray:
    """Hemodynamic response function, scaled to a peak of 1.

    t: time since the stimulus [s], any shape. Zero for t < 0.
    Returns the dimensionless response, float64, same shape.
    """
    return _unnormalised_hrf(np.asarray(t, dtype=np.float64)) / _HRF_PEAK


def task_labels(task_cfg: TaskConfig, t: ArrayLike) -> np.ndarray:
    """Task state at given times: 1 inside a task block, 0 otherwise.

    t: times [s], any shape. A block covers ``onset <= t < onset + block_duration``.
    Returns int8, same shape.
    """
    t = np.asarray(t, dtype=np.float64)
    onsets = np.asarray(task_cfg.onsets, dtype=np.float64)
    since_onset = t[..., None] - onsets
    in_block = (since_onset >= 0.0) & (since_onset < task_cfg.block_duration)
    return in_block.any(axis=-1).astype(np.int8)


def task_timecourse(task_cfg: TaskConfig, t: ArrayLike) -> np.ndarray:
    """Fractional CBV change a(t) at the centre of the activated patch.

    The block design (1 during a block, 0 otherwise) is convolved with the HRF.
    The result is scaled so that a block long enough to reach the largest
    possible response peaks at `task_cfg.peak_fractional_change`; shorter
    blocks peak lower. The convolution of a block with a gamma density is a
    difference of incomplete gamma functions, so it is evaluated exactly at any
    time, without a time grid.

    t: times [s], any shape.
    Returns a(t), dimensionless, float64, same shape.
    """
    t = np.asarray(t, dtype=np.float64)
    since_onset = t[..., None] - np.asarray(task_cfg.onsets, dtype=np.float64)
    response = _unnormalised_step_response(since_onset) - _unnormalised_step_response(
        since_onset - task_cfg.block_duration
    )
    return task_cfg.peak_fractional_change * response.sum(axis=-1) / _STEP_RESPONSE_PEAK


def cbv_at(cbv_baseline: np.ndarray, activation: np.ndarray, a: ArrayLike) -> np.ndarray:
    """Blood volume fraction at given values of the task time course.

    cbv_baseline: resting blood volume fraction, shape (...).
    activation: spatial pattern of the task response in [0, 1], shape (...).
    a: fractional CBV change a(t), a scalar or an array of shape (n_times,).

    Returns ``cbv_baseline * (1 + a * activation)``, float32, shape (...) for a
    scalar `a` and (n_times, ...) otherwise.
    """
    a = np.asarray(a, dtype=np.float32)
    a = a.reshape(a.shape + (1,) * np.ndim(cbv_baseline))
    return (cbv_baseline * (1.0 + a * activation)).astype(np.float32)
