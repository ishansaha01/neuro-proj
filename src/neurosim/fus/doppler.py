"""SVD clutter filter and power Doppler (docs/ARCHITECTURE.md, section 5, step 3).

Demene et al. (2015), "Spatiotemporal clutter filtering of ultrafast ultrasound
data highly increases Doppler and fUltrasound sensitivity", IEEE TMI
34(11):2271-2285: the frames of one ensemble are stacked into a space-time
(Casorati) matrix; tissue, being spatially coherent and slow, concentrates in
the largest singular components, which are removed.
"""

from __future__ import annotations

import numpy as np

from neurosim.fus.config import ImagingConfig


def _casorati_svd(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of the (n_frames, n_pix) matrix of an ensemble, singular values descending.

    Returns:
        (temporal, singular_values, spatial): shapes (n_frames, r), (r,), (r, n_pix)
        with r = min(n_frames, n_pix); `frames = (temporal * singular_values) @ spatial`.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"frames must have shape (n_frames, nz, nx), got {frames.shape}")
    return np.linalg.svd(frames.reshape(frames.shape[0], -1), full_matrices=False)


def svd_clutter_filter(frames: np.ndarray, n_discard: int) -> np.ndarray:
    """Remove the `n_discard` largest singular components of an ensemble of frames.

    Args:
        frames: complex IQ frames, shape (n_frames, nz, nx).
        n_discard: number of largest singular components to remove, 0..min(n_frames, nz*nx).

    Returns:
        filtered frames, same shape, complex128.
    """
    temporal, singular_values, spatial = _casorati_svd(frames)
    if not 0 <= n_discard <= singular_values.shape[0]:
        raise ValueError(
            f"n_discard must be between 0 and {singular_values.shape[0]}, got {n_discard}"
        )
    kept = singular_values.copy()
    kept[:n_discard] = 0.0
    return ((temporal * kept) @ spatial).reshape(frames.shape).astype(np.complex128)


def choose_svd_cutoff(frames: np.ndarray, cfg: ImagingConfig) -> int:
    """Number of leading singular components to discard, by the rule in `cfg.doppler`.

    Rules (`cfg.doppler.cutoff_rule`):

    - "fixed": `cfg.doppler.n_discard`, as in Demene et al. (2015), who set the
      threshold by hand.
    - "energy": the fewest leading components that hold at least
      `cfg.doppler.energy_fraction` of the total energy (sum of squared singular
      values). Suited to clutter that dominates the energy.
    - "frequency": the leading components whose temporal singular vector has a
      mean Doppler frequency below `cfg.doppler.cutoff_frequency`, counted up to
      the first component at or above it. The mean frequency is the lag-one
      autocorrelation estimate `|arg sum_t v[t+1] conj(v[t])| * frame_rate / (2 pi)`
      (Kasai et al. 1985, IEEE Trans. Sonics Ultrason. 32(3):458-464). A simple
      data-adaptive rule in the spirit of Song et al. (2017, IEEE TMI
      36(1):251-262); assumed, to verify against their thresholds.

    The result is capped at one less than the number of components, so the
    filter never removes everything.

    Args:
        frames: complex IQ frames, shape (n_frames, nz, nx).
        cfg: imaging configuration (`doppler` and `sequence.frame_rate` are used).

    Returns:
        the number of components to pass to `svd_clutter_filter`.
    """
    temporal, singular_values, _ = _casorati_svd(frames)
    n_components = singular_values.shape[0]
    rule = cfg.doppler.cutoff_rule
    if rule == "fixed":
        n_discard = cfg.doppler.n_discard
    elif rule == "energy":
        cumulative = np.cumsum(singular_values**2)
        reached = cumulative >= cfg.doppler.energy_fraction * cumulative[-1]
        n_discard = int(np.argmax(reached)) + 1
    else:
        lag_one = np.sum(temporal[1:] * np.conj(temporal[:-1]), axis=0)
        frequency = np.abs(np.angle(lag_one)) * cfg.sequence.frame_rate / (2.0 * np.pi)
        fast = frequency >= cfg.doppler.cutoff_frequency
        n_discard = int(np.argmax(fast)) if fast.any() else n_components
    return min(n_discard, n_components - 1)


def power_doppler(filtered_frames: np.ndarray) -> np.ndarray:
    """Power Doppler image: mean of |IQ|^2 over the ensemble.

    Args:
        filtered_frames: clutter-filtered IQ frames, shape (n_frames, nz, nx).

    Returns:
        float64 array, shape (nz, nx) [IQ units squared].
    """
    frames = np.asarray(filtered_frames)
    return (frames.real**2 + frames.imag**2).mean(axis=0)
