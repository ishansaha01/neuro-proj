"""fus.resolution: speckle correlation lengths of the resolution cell from diffraction theory."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.config import BeamformingConfig
from neurosim.fus.operators import psf_operator
from neurosim.fus.resolution import (
    GAUSSIAN_LENGTH_PER_FWHM,
    axial_correlation_length,
    gaussian_fwhm,
    image_spectrum,
    lateral_correlation_length,
    mean_wavenumber,
    receive_sine_aperture,
    transmit_sine_aperture,
)
from neurosim.fus.types import GreensBank, Transducer

POINT = np.array([[0.0, 0.023]])  # centre of the tiny ROI, columns (x, z) [m]


def _narrow_band(transducer: Transducer) -> np.ndarray:
    """Two frequencies 1 Hz apart at the pulse centre: a single-frequency image."""
    return transducer.center_frequency + np.array([0.0, 1.0])


def test_image_spectrum_is_the_pulse_tilted_towards_low_frequencies(tiny):
    weights = image_spectrum(tiny.transducer, tiny.freqs)
    assert weights.shape == tiny.freqs.shape and np.all(weights > 0.0)
    assert tiny.freqs[np.argmax(weights)] < tiny.transducer.center_frequency
    wavenumber = mean_wavenumber(tiny.transducer, tiny.freqs, C_REF)
    mean_frequency = wavenumber * C_REF / (2.0 * np.pi)
    # A 2 MHz pulse of 60 % bandwidth: the tilt f^-3 of the power moves the mean to 1.78 MHz.
    assert mean_frequency == pytest.approx(1.78e6, rel=0.01)


def test_axial_length_of_a_gaussian_spectrum(tiny):
    """`0.441 lambda / b` (FWHM) times 1.064 for a Gaussian spectrum; the tilt barely changes it."""
    transducer = tiny.transducer
    wavelength = C_REF / transducer.center_frequency
    gaussian = 0.441 * wavelength / transducer.fractional_bandwidth * GAUSSIAN_LENGTH_PER_FWHM
    length = axial_correlation_length(transducer, tiny.freqs, C_REF)
    assert length == pytest.approx(gaussian, rel=0.06)
    assert gaussian_fwhm(GAUSSIAN_LENGTH_PER_FWHM * 2.0e-3) == pytest.approx(2.0e-3)
    # Twice the bandwidth, half the length.
    wide = replace(transducer, fractional_bandwidth=0.5 * transducer.fractional_bandwidth)
    assert axial_correlation_length(wide, tiny.freqs, C_REF) > 1.5 * length


def test_sine_apertures(tiny):
    transducer = tiny.transducer
    half = 0.5 * transducer.aperture
    centre, width = receive_sine_aperture(transducer, POINT, BeamformingConfig())
    assert centre[0] == pytest.approx(0.0, abs=1e-12)
    assert width[0] == pytest.approx(2.0 * half / np.hypot(half, POINT[0, 1]))
    # Off axis the aperture is seen obliquely: narrower in sine space, centred off zero.
    off_centre, off_width = receive_sine_aperture(
        transducer, np.array([[0.004, 0.023]]), BeamformingConfig()
    )
    assert off_centre[0] < 0.0 and off_width[0] < width[0]
    # An f-number aperture inside the array: |x_e - x| <= z / (2 F), whatever the depth.
    _, limited = receive_sine_aperture(transducer, POINT, BeamformingConfig(f_number=4.0))
    assert limited[0] == pytest.approx(2.0 * np.sin(np.arctan(1.0 / 8.0)))
    assert transmit_sine_aperture(replace(tiny.sequence, angles_deg=(3.0,))) == (
        pytest.approx(np.sin(np.deg2rad(3.0))),
        0.0,
    )
    centre_tx, width_tx = transmit_sine_aperture(tiny.sequence)  # -5, 0, 5 degrees
    assert centre_tx == pytest.approx(0.0, abs=1e-12)
    assert width_tx == pytest.approx(3.0 * np.sin(np.deg2rad(5.0)))


def test_lateral_length_matches_the_closed_forms_at_one_frequency(tiny):
    transducer = tiny.transducer
    freqs = _narrow_band(transducer)
    wavelength = C_REF / transducer.center_frequency
    points = np.array([[0.0, 0.023], [0.003, 0.02], [-0.002, 0.04]])
    boxcar = BeamformingConfig()
    _, receive = receive_sine_aperture(transducer, points, boxcar)
    single = replace(tiny.sequence, angles_deg=(0.0,))
    # One plane wave, boxcar aperture: the equivalent width of sinc^2, lambda / A.
    np.testing.assert_allclose(
        lateral_correlation_length(transducer, single, boxcar, points, freqs, C_REF),
        wavelength / receive,
        rtol=1e-3,
    )
    # Hann aperture: int w^4 / (int w^2)^2 = (35 / 128) / (3 / 8)^2 per unit width.
    np.testing.assert_allclose(
        lateral_correlation_length(
            transducer, single, BeamformingConfig(window="hann"), points, freqs, C_REF
        ),
        35.0 / 18.0 * wavelength / receive,
        rtol=1e-3,
    )
    # Compounding: trapezoidal transfer function, for either order of the two widths.
    for angles in ((-5.0, 0.0, 5.0), (-20.0, -10.0, 0.0, 10.0, 20.0)):
        sequence = replace(tiny.sequence, angles_deg=angles)
        _, transmit = transmit_sine_aperture(sequence)
        wide, narrow = np.maximum(receive, transmit), np.minimum(receive, transmit)
        np.testing.assert_allclose(
            lateral_correlation_length(transducer, sequence, boxcar, points, freqs, C_REF),
            wavelength * (wide - 0.6 * narrow) / (wide - narrow / 3.0) ** 2,
            rtol=1e-3,
        )


def test_lateral_length_grows_with_depth_and_with_the_f_number(tiny):
    transducer = tiny.transducer
    single = replace(tiny.sequence, angles_deg=(0.0,))
    depths = np.linspace(0.02, 0.035, 7)  # f-number 6 stays inside the 6 mm array
    points = np.stack((np.zeros_like(depths), depths), axis=1)
    lengths = lateral_correlation_length(
        transducer, single, BeamformingConfig(), points, tiny.freqs, C_REF
    )
    assert lengths.shape == depths.shape and np.all(np.diff(lengths) > 0.0)
    limited = lateral_correlation_length(
        transducer, single, BeamformingConfig(f_number=6.0), points, tiny.freqs, C_REF
    )
    assert np.all(limited > lengths)
    # A fixed f-number gives one sine-space aperture, hence one length, at every depth.
    np.testing.assert_allclose(limited, limited[0], rtol=1e-12)
    # The bandwidth shortens the length a little against the mean wavelength.
    wavenumber = mean_wavenumber(transducer, tiny.freqs, C_REF)
    centre = wavenumber * C_REF / (2.0 * np.pi) + np.array([0.0, 1.0])
    narrow = lateral_correlation_length(
        transducer, single, BeamformingConfig(), points, centre, C_REF
    )
    assert np.all(lengths < 1.05 * narrow) and np.all(lengths > 0.85 * narrow)


def test_point_without_receive_aperture_gets_the_transmit_aperture_alone(tiny):
    transducer = tiny.transducer
    limited = BeamformingConfig(f_number=5.0)
    points = np.array([[0.05, 0.02], [0.0, 0.02]])  # far beside the array, and under it
    _, width = receive_sine_aperture(transducer, points, limited)
    assert width[0] == 0.0 and width[1] > 0.0
    freqs = _narrow_band(transducer)
    wavelength = C_REF / transducer.center_frequency
    _, transmit = transmit_sine_aperture(tiny.sequence)
    lengths = lateral_correlation_length(
        transducer, tiny.sequence, limited, points, freqs, C_REF
    )
    assert lengths[0] == pytest.approx(wavelength / transmit, rel=1e-3)
    assert lengths[1] < lengths[0]
    # One plane wave and no aperture: no lateral resolution at all.
    single = replace(tiny.sequence, angles_deg=(0.0,))
    unresolved = lateral_correlation_length(transducer, single, limited, points, freqs, C_REF)
    assert unresolved[0] > 1.0e3 * unresolved[1]


def _simulated_correlation_lengths(
    transducer: Transducer, sequence, freqs: np.ndarray
) -> tuple[float, float]:
    """Lateral and axial integrals of |C|^2 of the simulated point-spread function [m]."""
    spacing, half_width = 0.1e-3, 3.0e-3
    x0, z0 = POINT[0]
    sources = Grid2D.from_extent(
        (x0 - half_width, x0 + half_width), (z0 - half_width, z0 + half_width), spacing
    )
    water = GreensBank.homogeneous(transducer, sources, freqs, C_REF)
    pixel = Grid2D(shape=(1, 1), spacing=spacing, origin=(z0, x0))
    psf = psf_operator(water, pixel, sources, water, sequence)[0].reshape(sources.shape)
    padded = [2 * n for n in sources.shape]
    correlation = np.fft.ifft2(np.abs(np.fft.fft2(psf.astype(np.complex128), padded)) ** 2)
    intensity = np.fft.fftshift(np.abs(correlation / correlation[0, 0]) ** 2)
    centre_z, centre_x = (n // 2 for n in intensity.shape)
    return (
        float(intensity[centre_z].sum() * spacing),
        float(intensity[:, centre_x].sum() * spacing),
    )


def test_lengths_match_the_simulated_point_spread_function(tiny):
    """The diffraction model against the autocorrelation of `psf_operator` in water.

    40 elements at 23 mm depth (f-number 1.5), so that the point-spread function
    fits well inside the 6 mm window the autocorrelation is integrated over.
    """
    transducer = replace(tiny.transducer, n_elements=40)
    lateral, axial = _simulated_correlation_lengths(transducer, tiny.sequence, tiny.freqs)
    model_lateral = lateral_correlation_length(
        transducer, tiny.sequence, BeamformingConfig(), POINT, tiny.freqs, C_REF
    )[0]
    model_axial = axial_correlation_length(transducer, tiny.freqs, C_REF)
    assert model_lateral == pytest.approx(lateral, rel=0.08)
    assert model_axial == pytest.approx(axial, rel=0.08)
    assert lateral > 1.5 * axial  # an anisotropic cell
