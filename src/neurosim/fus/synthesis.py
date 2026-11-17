"""Channel-data synthesis in the frequency domain (docs/ARCHITECTURE.md, section 5, step 1).

Blood and tissue are weak point scatterers (Born approximation), so for
scatterers `k` with amplitude `s_k` at `r_k`, transmit angle `a` and receive
element `j`:

    RF(f, a, j) = P(f) * sum_k s_k * G_tx(r_k, a, f) * G(r_k, j, f)
    G_tx(r, a, f) = sum_e exp(+i w d_e(a)) * G(r, e, f)

`G = B * G_ref` is the one-way transfer function of the medium (`fus.greens`),
`P` the two-way pulse spectrum (`fus.pulse`) and `d_e(a) = x_e sin(a) / c_ref`
the firing delay of element `e` for a plane wave steered by `a`, shifted so the
earliest element fires at t = 0. All elements transmit with equal weight.

The time convention is exp(-i w t): the spectra are those of signals whose time
origin is the firing of the earliest element. They are sampled on the bank's
frequency grid, which makes them periodic in time (`fus.window`).
"""

from __future__ import annotations

import numpy as np

from neurosim.fus.greens import medium_greens
from neurosim.fus.pulse import pulse_spectrum
from neurosim.fus.types import GreensBank, PlaneWaveSequence, Scatterers, Transducer

# Scatterers handled per pass of `channel_spectra`, as a count of complex Green's-function
# values (nf * n_elements * scatterers); 2e6 values are 32 MB.
_CHUNK_VALUES = 2_000_000


def transmit_delays(
    transducer: Transducer, sequence: PlaneWaveSequence, c_ref: float
) -> np.ndarray:
    """Element firing delays of every plane-wave transmit.

    Returns:
        float64 array, shape (n_angles, n_elements) [s]; the smallest delay of each angle is 0.
    """
    angles = np.deg2rad(np.asarray(sequence.angles_deg, dtype=np.float64))
    delays = transducer.element_x()[None, :] * np.sin(angles)[:, None] / c_ref
    return delays - delays.min(axis=1, keepdims=True)


def transmit_phasors(bank: GreensBank, sequence: PlaneWaveSequence) -> np.ndarray:
    """Per-element transmit factors `exp(+i w d_e(a))` on the bank's frequency grid.

    Returns:
        complex128 array, shape (nf, n_angles, n_elements).
    """
    delays = transmit_delays(bank.transducer, sequence, bank.c_ref)
    return np.exp(2j * np.pi * bank.freqs[:, None, None] * delays[None, :, :])


def unit_scatterer_factors(
    bank: GreensBank,
    sequence: PlaneWaveSequence,
    xz: np.ndarray,
    freq_slice: slice = slice(None),
) -> tuple[np.ndarray, np.ndarray]:
    """The two factors of the channel spectrum of a unit point scatterer at each point.

    A unit scatterer at `r_k` gives `RF(f, a, j) = P(f) * transmit[f, a, k] * receive[f, j, k]`.

    Args:
        bank: the medium.
        sequence: gives the steering angles.
        xz: scatterer positions [m], shape (n, 2), columns (x, z).
        freq_slice: which of `bank.freqs` to evaluate.

    Returns:
        (transmit, receive): `G_tx`, complex128, shape (nf_selected, n_angles, n), and
        `G`, complex128, shape (nf_selected, n_elements, n).
    """
    receive = medium_greens(bank, xz, freq_slice)
    transmit = np.matmul(transmit_phasors(bank, sequence)[freq_slice], receive)
    return transmit, receive


def transmit_field(bank: GreensBank, sequence: PlaneWaveSequence, xz: np.ndarray) -> np.ndarray:
    """Plane-wave transmit field `G_tx` of every steering angle at arbitrary points.

    Args:
        bank: the medium.
        sequence: gives the steering angles.
        xz: points [m], shape (..., 2), columns (x, z); inside the bank ROI.

    Returns:
        complex128 array, shape (nf, n_angles, ...) [dimensionless, per unit source spectrum].
    """
    xz = np.asarray(xz, dtype=np.float64)
    transmit, _ = unit_scatterer_factors(bank, sequence, xz.reshape(-1, 2))
    return transmit.reshape(*transmit.shape[:2], *xz.shape[:-1])


def channel_spectra(
    bank: GreensBank, scatterers: Scatterers, sequence: PlaneWaveSequence
) -> np.ndarray:
    """Frequency-domain channel data of a set of point scatterers (single scattering).

    Args:
        bank: the medium; its ROI must cover every scatterer.
        scatterers: positions [m] and amplitudes.
        sequence: gives the steering angles.

    Returns:
        complex128 array, shape (nf, n_angles, n_elements) [scatterer amplitude units].
    """
    xz = np.asarray(scatterers.xz, dtype=np.float64).reshape(-1, 2)
    amplitude = np.asarray(scatterers.amplitude).reshape(-1)
    if amplitude.shape[0] != xz.shape[0]:
        raise ValueError(f"{xz.shape[0]} scatterer positions but {amplitude.shape[0]} amplitudes")
    nf, n_elements = bank.B.shape[:2]
    # Accumulated as (nf, n_elements, n_angles): the product then needs the transpose of
    # the small transmit factor rather than of the large receive factor.
    spectra = np.zeros((nf, n_elements, len(sequence.angles_deg)), dtype=np.complex128)
    chunk = max(1, _CHUNK_VALUES // (nf * n_elements))
    for start in range(0, xz.shape[0], chunk):
        stop = start + chunk
        transmit, receive = unit_scatterer_factors(bank, sequence, xz[start:stop])
        transmit *= amplitude[start:stop]
        spectra += np.matmul(receive, np.ascontiguousarray(transmit.transpose(0, 2, 1)))
    spectra *= pulse_spectrum(bank.transducer, bank.freqs)[:, None, None]
    return np.ascontiguousarray(spectra.transpose(0, 2, 1))


def static_clutter_spectra(bank: GreensBank, sequence: PlaneWaveSequence) -> np.ndarray:
    """Plane-wave channel data of the static medium (skull echoes), from `bank.static_fmc`.

    `static_fmc[f, e_tx, e_rx]` is the echo at element `e_rx` when element `e_tx`
    fires; a plane-wave transmit is the delayed sum over `e_tx`, and the pulse
    spectrum multiplies the result. The static echoes do not depend on the
    scatterers, so they are the same in every frame.

    Range aliasing: `static_fmc` is known only on the bank's frequency grid, so
    its echoes are periodic in time with period `1/df` (`fus.window`). Echoes
    that arrive before or after the ROI's time window (the skull surface, late
    reverberation) are therefore folded into it and reach the image at wrapped
    ranges, where a scanner would have gated them out in time. No gating is
    possible here because the arrival time is not recoverable from `nf` samples.
    What is modelled is thus the *whole* static echo train folded onto the ROI,
    an upper bound on the static clutter a gated scanner would see. A backend
    that wants gated clutter must gate in time before sampling `static_fmc`.
    Being identical in every frame, the folded echoes are removed by the SVD
    clutter filter like any other static signal.

    Args:
        bank: the medium; `static_fmc` must be present.
        sequence: gives the steering angles.

    Returns:
        complex128 array, shape (nf, n_angles, n_elements) [per unit source spectrum].
    """
    if bank.static_fmc is None:
        raise ValueError("the bank has no static_fmc")
    spectra = np.matmul(transmit_phasors(bank, sequence), bank.static_fmc.astype(np.complex128))
    spectra *= pulse_spectrum(bank.transducer, bank.freqs)[:, None, None]
    return spectra
