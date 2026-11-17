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
"""

from __future__ import annotations

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.fus.beamform import make_beamformer
from neurosim.fus.config import BeamformingConfig
from neurosim.fus.greens import check_same_sampling
from neurosim.fus.pulse import pulse_spectrum
from neurosim.fus.synthesis import unit_scatterer_factors
from neurosim.fus.types import GreensBank, PlaneWaveSequence

# Frequencies handled per pass, as a count of complex Green's-function values
# (frequencies * n_elements * sources); 4e6 values are 64 MB.
_CHUNK_VALUES = 4_000_000
# Fewest frequencies per pass: with several, the last pass still has at least two,
# which `reference_greens` needs for its fast route.
_MIN_CHUNK = 4


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
    check_same_sampling(bank, model_bank)
    beamformer = make_beamformer(pixel_grid, model_bank, sequence, beamforming)
    sources = source_grid.points().reshape(-1, 2)
    pulse = pulse_spectrum(bank.transducer, bank.freqs)
    nf, n_elements = bank.B.shape[:2]
    n_pix, n_src = beamformer.rx_weights.shape[2], sources.shape[0]

    operator = np.zeros((n_pix, n_src), dtype=np.complex64)
    transmit_part = np.empty((n_pix, n_src), dtype=np.complex64)
    receive_part = np.empty((n_pix, n_src), dtype=np.complex64)
    n_chunks = int(np.ceil(nf / max(_MIN_CHUNK, _CHUNK_VALUES // (n_elements * n_src))))
    bounds = np.linspace(0, nf, n_chunks + 1).astype(int)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        transmit, receive = unit_scatterer_factors(bank, sequence, sources, slice(start, stop))
        for offset, f in enumerate(range(start, stop)):
            np.matmul(
                beamformer.tx_weights[f].T, transmit[offset].astype(np.complex64), out=transmit_part
            )
            np.matmul(
                beamformer.rx_weights[f].T, receive[offset].astype(np.complex64), out=receive_part
            )
            transmit_part *= receive_part
            transmit_part *= np.float32(pulse[f])
            operator += transmit_part
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
