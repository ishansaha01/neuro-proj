"""E-FUS-010: the point-spread function in water has the widths diffraction theory predicts.

Theory. For a scatterer on the array axis at depth z and a pixel displaced by x
laterally (same depth), every receive element's phase differs by
`k x x_j / z` (Fresnel), so the receive sum over a uniform aperture D is
`sinc(D x f / (c z))` (Goodman, Introduction to Fourier Optics, 3rd ed., 2005,
section 4.4.1: Fraunhofer pattern of a rectangular aperture; at one frequency
its full width at half maximum is 1.207 lambda z / D). A plane wave steered by
`a` adds the phase `k x sin(a)`, so compounding multiplies by the mean of
`exp(i k x sin(a))` over the angles (Montaldo et al. 2009, IEEE TUFFC
56(3):489-506). For a pixel displaced by dz in depth the two-way phase is
`2 k dz`, so the axial profile is the Fourier transform of the pulse-echo
spectrum (for a Gaussian of -6 dB fractional bandwidth b: 0.441 lambda / b).

Each frequency contributes with weight `P(f) f^(-3/2)`: `P` is the two-way
pulse spectrum, a 2D Green's function has magnitude `(8 pi k r)^(-1/2)`, and the
plane wave radiated by a line of such sources has magnitude proportional to
`1/k`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.common.units import wavelength
from neurosim.fus.beamform import das
from neurosim.fus.config import ImagingConfig, imaging_setup
from neurosim.fus.pulse import pulse_spectrum
from neurosim.fus.synthesis import channel_spectra
from neurosim.fus.types import GreensBank, Scatterers

EVAL_ID = "E-FUS-010"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
SAMPLES_PER_WAVELENGTH = 16  # profile sampling
HALF_LENGTH_SAMPLES = 48  # profiles extend 3 wavelengths either side of the scatterer
TOLERANCE = 0.10


def _fwhm(coordinate: np.ndarray, profile: np.ndarray) -> float:
    """Full width at half maximum of the main lobe of |profile| (linear interpolation) [m]."""
    magnitude = np.abs(profile)
    peak = int(np.argmax(magnitude))
    half = 0.5 * magnitude[peak]

    def crossing(step: int) -> float:
        index = peak
        while magnitude[index + step] >= half:
            index += step
        inside, outside = magnitude[index], magnitude[index + step]
        fraction = (inside - half) / (inside - outside)
        return coordinate[index] + fraction * (coordinate[index + step] - coordinate[index])

    return crossing(+1) - crossing(-1)


@eval_case(
    id=EVAL_ID,
    claim=(
        "the image of a single scatterer in water has the lateral and axial full widths at "
        "half maximum that diffraction theory gives for the aperture, depth and pulse used"
    ),
    reference=(
        "Lateral: FWHM of |sum_f P(f) f^-1.5 sinc(D x f/(c z)) mean_a exp(2 pi i f x sin(a)/c)| "
        "(Fraunhofer pattern of a rectangular aperture, Goodman 2005 sec. 4.4.1; plane-wave "
        "compounding, Montaldo et al. 2009); axial: FWHM of |sum_f P(f) f^-1.5 exp(4 pi i f dz/c)| "
        "(Fourier transform of the pulse-echo spectrum)"
    ),
    metric=(
        "simulated / theoretical FWHM minus 1, for the lateral profile with one plane wave at "
        "0 degrees, the lateral profile with the compounded sequence, and the axial profile"
    ),
    threshold=(
        "each |ratio - 1| < 0.10 (fixed in advance: the theory is paraxial, and at f-number "
        "z/D = 1.6 the neglected terms, the 1/sqrt(r) amplitude taper across the aperture and "
        "the element directivity are each a few percent)"
    ),
    seeds=(0,),
)
def test_e_fus_010_psf_water() -> None:
    setup = imaging_setup(from_dict(ImagingConfig, load_yaml(CONFIG)))
    transducer, freqs = setup.transducer, setup.freqs
    bank = GreensBank.homogeneous(transducer, setup.roi, freqs, C_REF)
    depth = float(np.mean(setup.roi.axes()[0]))  # ROI centre, on the array axis
    scatterer = Scatterers(xz=np.array([[0.0, depth]]), amplitude=np.ones(1))

    step = float(wavelength(C_REF, transducer.center_frequency)) / SAMPLES_PER_WAVELENGTH
    n_samples = 2 * HALF_LENGTH_SAMPLES + 1
    offsets = step * (np.arange(n_samples) - HALF_LENGTH_SAMPLES)
    lateral_line = Grid2D(shape=(1, n_samples), spacing=step, origin=(depth, offsets[0]))
    axial_line = Grid2D(shape=(n_samples, 1), spacing=step, origin=(depth + offsets[0], 0.0))

    weights = pulse_spectrum(transducer, freqs) * freqs**-1.5
    fine = np.linspace(offsets[0], offsets[-1], 4001)
    delay = freqs[:, None] * fine[None, :] / C_REF  # f * offset / c, (nf, n_fine)
    aperture_pattern = np.sinc(transducer.aperture * delay / depth)
    axial_theory = _fwhm(fine, weights @ np.exp(4j * np.pi * delay))

    single = replace(setup.sequence, angles_deg=(0.0,))
    errors = {}
    for name, sequence in (("lateral_single", single), ("lateral_compounded", setup.sequence)):
        spectra = channel_spectra(bank, scatterer, sequence)
        sines = np.sin(np.deg2rad(np.asarray(sequence.angles_deg)))
        steering = np.exp(
            2j * np.pi * freqs[:, None, None] * fine[None, None, :] * sines[None, :, None] / C_REF
        ).mean(axis=1)
        theory = _fwhm(fine, weights @ (aperture_pattern * steering))
        simulated = _fwhm(offsets, das(spectra, lateral_line, bank, sequence)[0])
        errors[name] = simulated / theory - 1.0
    axial = das(channel_spectra(bank, scatterer, setup.sequence), axial_line, bank, setup.sequence)
    errors["axial"] = _fwhm(offsets, axial[:, 0]) / axial_theory - 1.0

    report(EVAL_ID, errors, passed=all(abs(error) < TOLERANCE for error in errors.values()))
