"""Electronic noise and tissue micro-motion: the seeded, switchable disturbances of a frame.

Electronic noise level. The noise is white, circular complex Gaussian, added to
every (frequency, angle, element) sample of the channel spectra with variance
`v = E|n|^2`. `NoiseConfig.electronic_snr_db` sets `v` against the *nominal blood
echo power*

    P_nominal = cbv_reference * ROI area * mean_{f,a,j} |RF_unit(f, a, j)|^2

where `RF_unit` is the channel spectrum of one unit-amplitude scatterer at the
ROI centre in the homogeneous reference medium. In the units of `fus.config`
(unit backscatter per unit blood area) this is the incoherent channel power of
blood at volume fraction `cbv_reference` filling the ROI, with the Green's
functions frozen at the ROI centre: `v = P_nominal / 10^(snr_db / 10)`.
The reference medium is used whatever the bank, so a skull and its paired
no-skull run get the same noise, as the electronics would give.

Through the beamformer, noise of variance `v` gives an image noise floor of
`v * Beamformer.noise_gain` (sum of squared weights); that is the floor of the
statistical tier.
"""

from __future__ import annotations

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.common.units import db_to_np
from neurosim.fus.config import NoiseConfig
from neurosim.fus.synthesis import channel_spectra
from neurosim.fus.types import GreensBank, PlaneWaveSequence, Scatterers


def nominal_blood_echo_power(
    bank: GreensBank, sequence: PlaneWaveSequence, cbv_reference: float
) -> float:
    """Nominal incoherent channel power of blood filling the bank's ROI (module docstring).

    Args:
        bank: gives the array, frequency grid, reference sound speed and ROI; its
            `B` is not used (the reference medium is).
        sequence: gives the steering angles.
        cbv_reference: reference blood volume fraction.

    Returns:
        mean power per channel sample [scatterer amplitude units squared].
    """
    (z0, x0), (nz, nx), spacing = bank.roi.origin, bank.roi.shape, bank.roi.spacing
    centre = np.array([[x0 + 0.5 * (nx - 1) * spacing, z0 + 0.5 * (nz - 1) * spacing]])
    around_centre = Grid2D(
        shape=(2, 2),
        spacing=spacing,
        origin=(centre[0, 1] - 0.5 * spacing, centre[0, 0] - 0.5 * spacing),
    )
    reference = GreensBank.homogeneous(bank.transducer, around_centre, bank.freqs, bank.c_ref)
    unit = channel_spectra(reference, Scatterers(xz=centre, amplitude=np.ones(1)), sequence)
    area = nz * nx * spacing**2
    return float(cbv_reference * area * np.mean(np.abs(unit) ** 2))


def electronic_noise_variance(
    bank: GreensBank, sequence: PlaneWaveSequence, noise: NoiseConfig
) -> float:
    """Variance `E|n|^2` of the electronic noise per channel-spectrum sample."""
    snr = float(np.exp(2.0 * db_to_np(noise.electronic_snr_db)))  # power ratio
    return nominal_blood_echo_power(bank, sequence, noise.cbv_reference) / snr


def electronic_noise(
    tree: SeedTree, frame: int, shape: tuple[int, int, int], variance: float
) -> np.ndarray:
    """Electronic noise of one frame, from stream `us_electronic_noise`, sub-index `frame`.

    Args:
        tree: seed tree of the run.
        frame: absolute frame index (frame `n` gets the same noise whichever
            other frames are simulated).
        shape: (nf, n_angles, n_elements).
        variance: `E|n|^2` per sample.

    Returns:
        complex128 array of the given shape.
    """
    rng = tree.rng("us_electronic_noise", frame)
    parts = rng.standard_normal((2, *shape))
    return np.sqrt(0.5 * variance) * (parts[0] + 1j * parts[1])


def tissue_displacement(tree: SeedTree, noise: NoiseConfig, times: np.ndarray) -> np.ndarray:
    """Rigid in-plane displacement of the tissue at the given times.

    A sum of sinusoids, one per entry of `noise.tissue_motion_frequencies`, each
    with a random direction and phase drawn once from stream `tissue_motion` and
    the same amplitude, scaled so that the rms magnitude over long times is
    `noise.tissue_motion_rms`. It is a function of absolute time only.

    Args:
        tree: seed tree of the run.
        noise: gives the rms displacement and the component frequencies.
        times: [s], shape (n,).

    Returns:
        float64 array, shape (n, 2), columns (dx, dz) [m].
    """
    frequencies = np.asarray(noise.tissue_motion_frequencies, dtype=np.float64)
    rng = tree.rng("tissue_motion")
    direction = rng.uniform(0.0, 2.0 * np.pi, size=frequencies.shape[0])
    phase = rng.uniform(0.0, 2.0 * np.pi, size=frequencies.shape[0])
    # Each component has mean square amplitude^2 / 2; the components are orthogonal in time.
    amplitude = noise.tissue_motion_rms * np.sqrt(2.0 / frequencies.shape[0])
    oscillation = np.sin(2.0 * np.pi * np.asarray(times)[:, None] * frequencies[None, :] + phase)
    unit_vectors = np.stack((np.cos(direction), np.sin(direction)), axis=1)
    return amplitude * oscillation @ unit_vectors
