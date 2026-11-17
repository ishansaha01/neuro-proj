"""Delay-and-sum beamforming with coherent plane-wave compounding, as phase conjugation.

Every (frequency, angle, element) sample of the channel spectra is multiplied
by the conjugate *phase* of what a model medium predicts for a scatterer at the
pixel, `M(p; f, a, j) = P(f) G_tx(p, a, f) G(p, j, f)`, weighted by the receive
apodisation `w_j(p)`, and summed:

    IQ(p) = sum_{f,a,j} w_j(p) RF(f, a, j) conj(M / |M|) / sum_{f,a,j} w_j(p) |M|

Removing a phase `exp(+i w tau)` is a delay by `tau`, so:

- with `GreensBank.homogeneous` as the model, `tau` is the travel time at
  `c_ref` and this is conventional delay-and-sum (what a scanner does);
- with a bank computed from the true or a CT-derived skull it is
  aberration-corrected beamforming.

Normalisation. The denominator is the value the numerator takes for a unit
scatterer at the pixel in the model medium, so the image is in units of
scatterer amplitude: a lone scatterer of amplitude `s` at a pixel gives
`IQ = s` when the model is the true medium. For the homogeneous model this is a
depth gain that undoes geometric spreading (the scanner's time-gain
compensation). For a skull model it also undoes what the model knows of the
skull's amplitude effects (attenuation, focusing and defocusing of the
transmitted wave), which conjugating phases alone cannot: phases restore the
coherence of the sum, not the strength of the transmitted field at the pixel.
The price is noise: `Beamformer.noise_gain` grows where the model field is weak.

The weights depend only on the model bank, the sequence and the pixel grid, so
they are computed once (`make_beamformer`) and reused for every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.fus.config import BeamformingConfig
from neurosim.fus.pulse import pulse_spectrum
from neurosim.fus.synthesis import unit_scatterer_factors
from neurosim.fus.types import GreensBank, PlaneWaveSequence, Transducer

# Pixels handled per pass of `make_beamformer`, as a count of complex Green's-function
# values (nf * n_elements * pixels); 2e6 values are 32 MB.
_CHUNK_VALUES = 2_000_000


@dataclass(frozen=True)
class Beamformer:
    """Precomputed weights of a delay-and-sum beamformer.

    Attributes:
        pixel_grid: the image grid.
        tx_weights: conjugate unit phasors of the model transmit field, complex64,
            shape (nf, n_angles, n_pix).
        rx_weights: receive apodisation times conjugate unit phasors of the model
            receive Green's function, divided by the pixel's normalisation (module
            docstring), complex64, shape (nf, n_elements, n_pix).
        noise_gain: image noise power per unit channel noise variance, float64,
            shape (nz, nx): the sum over (f, angle, element) of the squared weights.
            White channel noise of variance `v` per sample gives image power `v * noise_gain`.
    """

    pixel_grid: Grid2D
    tx_weights: np.ndarray
    rx_weights: np.ndarray
    noise_gain: np.ndarray


def receive_apodization(
    transducer: Transducer, pixel_grid: Grid2D, beamforming: BeamformingConfig
) -> np.ndarray:
    """Receive weight of every element for every pixel.

    With `f_number > 0` the active aperture of a pixel at `(x, z)` is
    `|x_element - x| <= z / (2 f_number)`; with 0 it is the whole array, centred
    on the array centre. The window ("boxcar" or "hann") spans the active aperture.

    Returns:
        float64 array, shape (n_elements, n_pix), values in [0, 1].
    """
    points = pixel_grid.points().reshape(-1, 2)
    element_x = transducer.element_x()[:, None]
    if beamforming.f_number > 0.0:
        centre = points[None, :, 0]
        half_width = points[None, :, 1] / (2.0 * beamforming.f_number)
    else:
        centre = np.zeros((1, points.shape[0]))
        half_width = np.full((1, points.shape[0]), 0.5 * transducer.aperture)
    position = (element_x - centre) / half_width  # -1 .. 1 across the active aperture
    active = np.abs(position) <= 1.0
    if beamforming.window == "hann":
        return np.where(active, 0.5 * (1.0 + np.cos(np.pi * position)), 0.0)
    return active.astype(np.float64)


def _conjugate_phase(values: np.ndarray) -> np.ndarray:
    """`conj(v) / |v|` as complex64, and 0 where `v` is 0."""
    magnitude = np.abs(values)
    phase = np.zeros(values.shape, dtype=np.complex128)
    np.divide(np.conj(values), magnitude, out=phase, where=magnitude > 0.0)
    return phase.astype(np.complex64)


def make_beamformer(
    pixel_grid: Grid2D,
    model_bank: GreensBank,
    sequence: PlaneWaveSequence,
    beamforming: BeamformingConfig = BeamformingConfig(),
) -> Beamformer:
    """Precompute the beamforming weights for a model medium.

    Args:
        pixel_grid: image grid; must lie inside the model bank's ROI.
        model_bank: the medium the beamformer assumes (homogeneous: conventional DAS).
        sequence: gives the steering angles.
        beamforming: receive aperture options.
    """
    points = pixel_grid.points().reshape(-1, 2)
    n_pix = points.shape[0]
    nf, n_elements = model_bank.B.shape[:2]
    n_angles = len(sequence.angles_deg)
    apodization = receive_apodization(model_bank.transducer, pixel_grid, beamforming)

    tx_weights = np.empty((nf, n_angles, n_pix), dtype=np.complex64)
    rx_weights = np.empty((nf, n_elements, n_pix), dtype=np.complex64)
    # Response of the un-normalised sum to a unit scatterer at the pixel in the model
    # medium, and the sum of its squared weights.
    unit_response = np.empty(n_pix)
    squared_weights = np.empty(n_pix)
    pulse = pulse_spectrum(model_bank.transducer, model_bank.freqs)
    chunk = max(1, _CHUNK_VALUES // (nf * n_elements))
    for start in range(0, n_pix, chunk):
        stop = start + chunk
        transmit, receive = unit_scatterer_factors(model_bank, sequence, points[start:stop])
        tx_phase = _conjugate_phase(transmit)
        rx_phase = _conjugate_phase(receive) * apodization[None, :, start:stop]
        tx_weights[:, :, start:stop] = tx_phase
        rx_weights[:, :, start:stop] = rx_phase
        unit_response[start:stop] = np.einsum(
            "f,fp,fp->p",
            pulse,
            np.abs(transmit).sum(axis=1),
            (np.abs(receive) * apodization[None, :, start:stop]).sum(axis=1),
        )
        squared_weights[start:stop] = np.einsum(
            "fp,fp->p", (np.abs(tx_phase) ** 2).sum(axis=1), (np.abs(rx_phase) ** 2).sum(axis=1)
        )

    inverse_response = np.zeros(n_pix)
    np.divide(1.0, unit_response, out=inverse_response, where=unit_response > 0.0)
    rx_weights *= inverse_response.astype(np.float32)
    noise_gain = squared_weights * inverse_response**2
    return Beamformer(
        pixel_grid=pixel_grid,
        tx_weights=tx_weights,
        rx_weights=rx_weights,
        noise_gain=noise_gain.reshape(pixel_grid.shape),
    )


def beamform(beamformer: Beamformer, spectra: np.ndarray) -> np.ndarray:
    """Apply precomputed beamforming weights to channel spectra.

    Args:
        beamformer: from `make_beamformer`.
        spectra: channel spectra, complex, shape (nf, n_angles, n_elements).

    Returns:
        complex128 IQ image, shape (nz, nx) [scatterer amplitude units].
    """
    expected = (*beamformer.tx_weights.shape[:2], beamformer.rx_weights.shape[1])
    if spectra.shape != expected:
        raise ValueError(f"spectra have shape {spectra.shape}, expected {expected}")
    # (nf, n_angles, n_elements) @ (nf, n_elements, n_pix) -> (nf, n_angles, n_pix)
    partial = np.matmul(spectra.astype(np.complex64), beamformer.rx_weights)
    partial *= beamformer.tx_weights
    return partial.sum(axis=(0, 1), dtype=np.complex128).reshape(beamformer.pixel_grid.shape)


def das(
    spectra: np.ndarray,
    pixel_grid: Grid2D,
    model_bank: GreensBank,
    sequence: PlaneWaveSequence,
    beamforming: BeamformingConfig = BeamformingConfig(),
) -> np.ndarray:
    """Delay-and-sum image of one set of channel spectra (see the module docstring).

    Builds the weights and applies them; to beamform many frames with the same
    model, call `make_beamformer` once and `beamform` per frame.

    Args:
        spectra: channel spectra, complex, shape (nf, n_angles, n_elements).
        pixel_grid: image grid.
        model_bank: the medium the beamformer assumes.
        sequence: gives the steering angles.
        beamforming: receive aperture options.

    Returns:
        complex128 IQ image, shape (nz, nx).
    """
    return beamform(make_beamformer(pixel_grid, model_bank, sequence, beamforming), spectra)
