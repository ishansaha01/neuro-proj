"""Linear image-domain operators derived from synthesis and beamforming.

docs/ARCHITECTURE.md, section 5, step 4.

`A[p, s]` is the IQ value at pixel `p` produced by a unit scatterer at source
node `s`. The channel spectrum of one scatterer is a product of a transmit and a
receive factor (`synthesis.unit_scatterer_factors`), and the beamformer weights
are a product of a transmit and a receive weight (`beamform.Beamformer`), so the
double sum over angles and elements separates:

    A[p, s] = sum_f P(f) * (sum_a tx_w[f, a, p] T[f, a, s]) * (sum_j rx_w[f, j, p] G[f, j, s])

These are the same factors and weights that `channel_spectra` and `beamform`
use, combined in a different order; eval E-FUS-013 checks the two paths agree.

Factorised form. With `T[f, a, s] = sum_e D[f, a, e] G[f, e, s]` (`D` the firing
phasors of `synthesis.transmit_phasors`), every term is a chain of matrix
products over the element index:

    A = sum_f P(f) * (tx_w[f]^T D[f] G[f]) o (rx_w[f]^T G[f])        (o: elementwise)

A per-element complex factor `u[f, e]` on the medium (a phase screen,
`fus.screen`: `G -> u G`, on transmit and on receive by reciprocity) therefore
enters as a diagonal scaling between two of these products,

    A(u) = sum_f P(f) * (tx_w[f]^T (D[f] diag(u[f])) G[f]) o ((rx_w[f]^T diag(u[f])) G[f])

`operator_factors` exports `P`, `tx_w`, `D`, `rx_w` and `G`; `psf_operator`
contracts the same arrays with `u = 1`. `recon.forward` contracts them in torch
for any `u` (eval E-RECON-005).

Weights of a varying model. `tx_w` and `rx_w` are functions of the model bank
(`beamform.make_beamformer`). `weight_factors` exports what they are made of
besides the bank's own values: the reference Green's function at the pixels,
the nodes and weights of the bank's bilinear interpolation, the firing phasors,
the receive apodisation and the pulse spectrum. `recon.rebeamform` builds the
weights from these in torch for any model bank on that grid, so that they can be
differentiated with respect to the model (eval E-RECON-012).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.fus.beamform import Beamformer, make_beamformer, receive_apodization
from neurosim.fus.config import BeamformingConfig
from neurosim.fus.greens import bilinear_weights, check_same_sampling, medium_greens
from neurosim.fus.pulse import pulse_spectrum
from neurosim.fus.reference import reference_greens
from neurosim.fus.synthesis import transmit_phasors
from neurosim.fus.types import GreensBank, PlaneWaveSequence

# Frequencies handled per pass, as a count of complex Green's-function values
# (frequencies * n_elements * sources); 4e6 values are 64 MB.
_CHUNK_VALUES = 4_000_000
# The same count for `operator_factors`, which holds its complex64 result besides the
# working arrays of a pass (complex128, and several times larger inside `reference_greens`).
_EXPORT_CHUNK_VALUES = 1_000_000
# Fewest frequencies per pass: with several, the last pass still has at least two,
# which `reference_greens` needs for its fast route.
_MIN_CHUNK = 4


@dataclass(frozen=True)
class OperatorFactors:
    """The per-frequency arrays whose products give the point-spread operator.

    With a per-element factor `u[f, e]` on the medium (`u = 1`: the bank itself),

        A[p, s] = sum_f pulse[f] * transmit[f, p, s] * receive[f, p, s]
        transmit[f] = tx_weights[f].T @ ((tx_phasors[f] * u[f]) @ greens[f])
        receive[f]  = (rx_weights[f].T * u[f]) @ greens[f]

    Attributes:
        freqs: frequencies [Hz], float64, shape (nf,).
        pulse: two-way pulse spectrum `P(f)`, float64, shape (nf,).
        tx_weights: transmit beamforming weights, complex64, shape (nf, n_angles, n_pix).
        tx_phasors: firing phasors `exp(+i w d_e(a))`, complex64, shape (nf, n_angles, n_elements).
        rx_weights: receive beamforming weights (apodised, normalised), complex64,
            shape (nf, n_elements, n_pix).
        greens: one-way transfer function of the bank from every element to every
            source point, complex64, shape (nf, n_elements, n_src).
    """

    freqs: np.ndarray
    pulse: np.ndarray
    tx_weights: np.ndarray
    tx_phasors: np.ndarray
    rx_weights: np.ndarray
    greens: np.ndarray


@dataclass(frozen=True)
class WeightFactors:
    """What the beamformer weights of a model bank are made of, at a set of pixels.

    For a model bank with values `B[f, e, node]` on the nodes of its ROI grid, the
    model's transfer function to pixel `p` is

        M[f, e, p] = reference[f, e, p] * sum_c node_weights[c, p] B[f, e, nodes[c, p]]

    and `beamform.make_beamformer` forms from it (module docstring of `fus.beamform`)

        receive = M,   transmit[f, a, p] = sum_e tx_phasors[f, a, e] M[f, e, p]
        tx_weights = conj(transmit) / |transmit|
        rx_weights = apodization * conj(receive) / |receive| / unit_response
        unit_response[p] = sum_f pulse[f] (sum_a |transmit|) (sum_e apodization |receive|)
        noise_gain[p] = sum_f (sum_a |tx phase|^2) (sum_e |rx phase|^2) / unit_response^2

    Attributes:
        freqs: frequencies [Hz], float64, shape (nf,).
        pulse: two-way pulse spectrum, float64, shape (nf,).
        tx_phasors: firing phasors, complex64, shape (nf, n_angles, n_elements).
        reference: reference Green's function from every element to every selected
            pixel, complex64, shape (nf, n_elements, n_selected).
        apodization: receive apodisation, float64, shape (n_elements, n_selected).
        nodes: flat indices of the four ROI nodes around each pixel, int64, (4, n_selected).
        node_weights: their bilinear weights, float32, shape (4, n_selected).
    """

    freqs: np.ndarray
    pulse: np.ndarray
    tx_phasors: np.ndarray
    reference: np.ndarray
    apodization: np.ndarray
    nodes: np.ndarray
    node_weights: np.ndarray


def weight_factors(
    model_bank: GreensBank,
    pixel_grid: Grid2D,
    sequence: PlaneWaveSequence,
    beamforming: BeamformingConfig = BeamformingConfig(),
    pixel_index: np.ndarray | None = None,
) -> WeightFactors:
    """The arrays `make_beamformer` combines with a model bank's values (`WeightFactors`).

    Args:
        model_bank: any bank on the array, frequency grid and ROI grid of the model
            family; its values `B` are not used.
        pixel_grid: image grid.
        sequence: gives the steering angles.
        beamforming: receive aperture options.
        pixel_index: flat indices of the pixels to export; None = all pixels.
    """
    points = pixel_grid.points().reshape(-1, 2)
    apodization = receive_apodization(model_bank.transducer, pixel_grid, beamforming)
    if pixel_index is not None:
        points, apodization = points[pixel_index], apodization[:, pixel_index]
    nodes, node_weights = bilinear_weights(model_bank.roi, points)
    reference = reference_greens(
        model_bank.transducer, points, model_bank.freqs, model_bank.c_ref
    )
    return WeightFactors(
        freqs=np.asarray(model_bank.freqs, dtype=np.float64),
        pulse=pulse_spectrum(model_bank.transducer, model_bank.freqs),
        tx_phasors=transmit_phasors(model_bank, sequence).astype(np.complex64),
        reference=reference.astype(np.complex64),
        apodization=apodization,
        nodes=nodes,
        node_weights=node_weights,
    )


def _medium_factor_chunks(
    bank: GreensBank,
    sequence: PlaneWaveSequence,
    sources: np.ndarray,
    chunk_values: int | None = None,
) -> Iterator[tuple[slice, np.ndarray, np.ndarray]]:
    """The medium-side factors of the operator, a few frequencies at a time.

    Yields `(freq_slice, tx_phasors, greens)`: the firing phasors, complex128,
    shape (n_selected, n_angles, n_elements), and the bank's transfer function to
    the source points, complex128, shape (n_selected, n_elements, n_src). The
    slices are contiguous and cover the frequency grid; a slice holds about
    `chunk_values` Green's-function values (default `_CHUNK_VALUES`).
    """
    nf, n_elements = bank.B.shape[:2]
    budget = _CHUNK_VALUES if chunk_values is None else chunk_values
    n_chunks = int(np.ceil(nf / max(_MIN_CHUNK, budget // (n_elements * sources.shape[0]))))
    bounds = np.linspace(0, nf, n_chunks + 1).astype(int)
    phasors = transmit_phasors(bank, sequence)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        freq_slice = slice(int(start), int(stop))
        yield freq_slice, phasors[freq_slice], medium_greens(bank, sources, freq_slice)


def _check_inputs(bank: GreensBank, model_bank: GreensBank) -> np.ndarray:
    """Check the two banks match and return the pulse spectrum on their frequency grid."""
    check_same_sampling(bank, model_bank)
    return pulse_spectrum(bank.transducer, bank.freqs)


def operator_factors(
    bank: GreensBank,
    pixel_grid: Grid2D,
    sources: np.ndarray,
    model_bank: GreensBank,
    sequence: PlaneWaveSequence,
    beamforming: BeamformingConfig = BeamformingConfig(),
    beamformer: Beamformer | None = None,
) -> OperatorFactors:
    """The arrays `psf_operator` multiplies together, for assembly outside this package.

    Memory: `greens` holds nf * n_elements * n_src complex64 values and the two
    weight arrays nf * (n_angles + n_elements) * n_pix. To keep `greens` small the
    sources can be exported in several calls that share the weights (`beamformer`).

    Args:
        bank: the medium behind the per-element factor (the homogeneous bank for a
            phase screen in the reference medium).
        pixel_grid: image grid.
        sources: scatterer positions [m], shape (n_src, 2), columns (x, z).
        model_bank: the medium the beamformer assumes.
        sequence: gives the steering angles.
        beamforming: receive aperture options.
        beamformer: weights of an earlier call with the same `pixel_grid`, `model_bank`,
            `sequence` and `beamforming` (`make_beamformer`), to avoid recomputing them;
            None = compute them.
    """
    pulse = _check_inputs(bank, model_bank)
    if beamformer is None:
        beamformer = make_beamformer(pixel_grid, model_bank, sequence, beamforming)
    sources = np.asarray(sources, dtype=np.float64).reshape(-1, 2)
    nf, n_elements = bank.B.shape[:2]
    tx_phasors = np.empty((nf, len(sequence.angles_deg), n_elements), dtype=np.complex64)
    greens = np.empty((nf, n_elements, sources.shape[0]), dtype=np.complex64)
    for freq_slice, chunk_phasors, chunk_greens in _medium_factor_chunks(
        bank, sequence, sources, _EXPORT_CHUNK_VALUES
    ):
        tx_phasors[freq_slice] = chunk_phasors
        greens[freq_slice] = chunk_greens
    return OperatorFactors(
        freqs=np.asarray(bank.freqs, dtype=np.float64),
        pulse=pulse,
        tx_weights=beamformer.tx_weights,
        tx_phasors=tx_phasors,
        rx_weights=beamformer.rx_weights,
        greens=greens,
    )


def _accumulate_operator(
    operator: np.ndarray,
    beamformer: Beamformer,
    pulse: np.ndarray,
    freq_slice: slice,
    tx_phasors: np.ndarray,
    greens: np.ndarray,
) -> None:
    """Add the terms of one frequency chunk (`u = 1` in `OperatorFactors`) to `operator`."""
    transmit = np.matmul(tx_phasors, greens)
    transmit_part = np.empty(operator.shape, dtype=np.complex64)
    receive_part = np.empty(operator.shape, dtype=np.complex64)
    for offset, f in enumerate(range(freq_slice.start, freq_slice.stop)):
        np.matmul(
            beamformer.tx_weights[f].T, transmit[offset].astype(np.complex64), out=transmit_part
        )
        np.matmul(beamformer.rx_weights[f].T, greens[offset].astype(np.complex64), out=receive_part)
        transmit_part *= receive_part
        transmit_part *= np.float32(pulse[f])
        operator += transmit_part


def psf_operator(
    bank: GreensBank,
    pixel_grid: Grid2D,
    source_grid: Grid2D,
    model_bank: GreensBank,
    sequence: PlaneWaveSequence,
    beamforming: BeamformingConfig = BeamformingConfig(),
) -> np.ndarray:
    """Point-spread operator: IQ image of a unit scatterer at every source node.

    Accumulated frequency by frequency; peak memory is about four arrays of
    (n_pix, n_src) complex64 plus the beamformer weights.

    Args:
        bank: the true medium.
        pixel_grid: image grid.
        source_grid: grid of scatterer positions (for the statistical tier, the CBV grid).
        model_bank: the medium the beamformer assumes.
        sequence: gives the steering angles.
        beamforming: receive aperture options.

    Returns:
        complex64 array, shape (n_pix, n_src), pixels and sources in C order of their grids.
    """
    pulse = _check_inputs(bank, model_bank)
    beamformer = make_beamformer(pixel_grid, model_bank, sequence, beamforming)
    sources = source_grid.points().reshape(-1, 2)
    n_pix, n_src = beamformer.rx_weights.shape[2], sources.shape[0]
    operator = np.zeros((n_pix, n_src), dtype=np.complex64)
    for freq_slice, tx_phasors, greens in _medium_factor_chunks(bank, sequence, sources):
        _accumulate_operator(operator, beamformer, pulse, freq_slice, tx_phasors, greens)
    return operator


def pd_operator(A: np.ndarray) -> np.ndarray:
    """Power Doppler operator `H = |A|^2`.

    For scatterers with independent random phases the expected blood power
    Doppler image is linear in their backscatter: `E[PD] = H @ q`, with `q` the
    backscatter of each source cell (for blood, `cbv * cell_area`; `fus.config`).

    Args:
        A: point-spread operator, complex, shape (n_pix, n_src).

    Returns:
        float32 array, shape (n_pix, n_src).
    """
    return (A.real.astype(np.float32) ** 2 + A.imag.astype(np.float32) ** 2).astype(np.float32)
