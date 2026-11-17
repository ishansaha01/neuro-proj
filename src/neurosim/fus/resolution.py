"""Size of the resolution cell from diffraction theory, as speckle correlation lengths.

The quantity. Let `h(r)` be the point-spread function (PSF): the complex value of
a fixed pixel for a unit scatterer at `r`. If every scatterer in front of that
pixel is displaced by `d`, the pixel's field before and after has the normalised
correlation

    C(d) = int h(r + d) conj(h(r)) dr / int |h(r)|^2 dr

(the scatterers have independent random phases, so only each scatterer's own two
contributions correlate). The *correlation length* along a unit vector `u` is the
equivalent width of the intensity correlation,

    l_u = int |C(s u)|^2 ds                                                [m]

For a Gaussian PSF of amplitude full width at half maximum (FWHM) `w` along `u`,
`|C(s u)|^2 = exp(-4 ln2 s^2 / w^2)` and `l_u = sqrt(pi / (4 ln 2)) w = 1.064 w`
(`gaussian_fwhm` converts back). `fus.speckle` needs `l_u`, not `w`: it is what
sets how far blood has to move before the speckle is renewed.

How it is computed. `C` is the Fourier transform of the PSF's spatial power
spectrum `S(K) = |h^(K)|^2` (Wiener-Khinchin), so by Parseval

    l_u = 2 pi int S_u(kappa)^2 dkappa / (int S_u(kappa) dkappa)^2

with `S_u` the spectrum projected on `u`. A pixel value is a sum over frequency
`f`, steering angle `a` and receive element `j`; near the pixel each term is a
plane wave in the scatterer position with wavevector
`K = (2 pi f / c) (sin a + sin t_j, cos a + cos t_j)`, `t_j` the angle from the
depth axis at which element `j` is seen from the pixel (Fraunhofer / local
plane-wave picture: Goodman, Introduction to Fourier Optics, 3rd ed., 2005,
sec. 4.4.1, for the aperture; Montaldo et al. 2009, IEEE TUFFC 56(3):489-506,
for plane-wave compounding as a synthetic transmit aperture).

- Axial (`u` = depth). `K_z ~ 4 pi f / c`, so `S_z` is the squared amplitude
  spectrum `W(f)^2` of the pixel value and

      l_z = (c / 2) int W^4 df / (int W^2 df)^2

  For a Gaussian `W` of -6 dB fractional bandwidth `b` this is `0.470 lambda / b`,
  i.e. 1.064 times the familiar axial FWHM `0.441 lambda / b`.
- Lateral (`u` along the array). `K_x = 2 pi s / lambda`, `s = sin a + sin t_j`:
  at one frequency the lateral transfer function `T(s)` is the receive window
  over the sine-space aperture of the pixel (width `A`), convolved with the
  transmit set (width `B`), and the lateral power spectrum in cycles per metre
  is `lambda T^2(sigma lambda) / int T^2 ds`. Frequencies add in power with
  weights `p_f = W_f^2 / sum W^2`, so

      l_x = int [sum_f p_f lambda_f T^2(sigma lambda_f)]^2 dsigma / (int T^2 ds)^2

  For one frequency this is `lambda int T^4 ds / (int T^2 ds)^2`: with a boxcar
  receive window `T` is a trapezoid and `l_x = lambda (A - 3 B / 5) / (A - B / 3)^2`
  for `A >= B`; for a single plane wave (`B = 0`) it is `lambda / A`, the
  equivalent width of `sinc^2`, which for a paraxial pixel on the axis is
  `lambda z / D` = 0.829 times the FWHM `1.207 lambda z / D` of the sinc. The
  bandwidth shortens `l_x` against the value at the mean wavelength, most for a
  sinc-like PSF (by 4 to 8 % in the cases below).
- `W(f) = P(f) f^(-3/2)` (`image_spectrum`): the two-way pulse spectrum times
  the frequency dependence of the 2D propagator. A 2D Green's function has
  magnitude `(8 pi k r)^(-1/2)` and the plane wave radiated by a line of such
  sources has magnitude proportional to `1/k` (docstring of eval E-FUS-010). The
  tilt moves the mean frequency of `W^2` below the pulse centre (1.78 MHz for a
  2 MHz pulse of 60 % bandwidth).

Assumptions. Their size was measured by integrating `|C|^2` of the simulated PSF
(`fus.operators.psf_operator`, water, 2 MHz, 60 % bandwidth) and comparing:
48 elements at 30 mm depth, lateral 1.45 mm here against 1.48 mm simulated,
axial 0.575 against 0.576; 128 elements at 35 mm (f-number 0.7), lateral 0.74
against 0.73 on the axis and 0.81 against 0.80 at 12 mm off axis; at 15 mm
(f-number 0.3) lateral 0.49 against 0.47 on the axis and 0.56 against 0.68 under
the edge of the array (-18 %, the worst case found); axial 0.576 against 0.59 to
0.61. The elliptical combination of `fus.speckle` gave 0.76 mm at 45 degrees for
the first case, as simulated.

- Paraxial in depth (`cos a + cos t_j ~ 2`), lateral and axial spectra separable.
- Uniform weight over the sine-space aperture. Elements are evenly spaced in
  `x`, not in `sin t`, and their amplitude falls with distance and angle; both
  are ignored.
- The transmit set is taken as a filled aperture of sine-space width
  `n (max - min) / (n - 1)` (angles evenly spaced in `sin a`).
- A homogeneous medium at the reference sound speed. A skull tilts the spectrum
  further (ROADMAP: effective centre near 1.3 MHz through the default skull
  against 1.7 MHz without) and shades the aperture, which lengthens the lateral
  correlation length by a few tens of percent; this is not modelled.
"""

from __future__ import annotations

import numpy as np

from neurosim.common.units import wavelength
from neurosim.fus.config import BeamformingConfig
from neurosim.fus.pulse import pulse_spectrum
from neurosim.fus.types import PlaneWaveSequence, Transducer

# Exponent of the frequency weighting of a pixel value in the 2D model (module docstring):
# -1/2 from the receive Green's function and -1 from the transmitted plane wave.
IMAGE_SPECTRUM_EXPONENT = -1.5
# Correlation length of a Gaussian PSF in units of its amplitude FWHM: the integral of
# exp(-4 ln2 s^2) over s.
GAUSSIAN_LENGTH_PER_FWHM = float(np.sqrt(np.pi / (4.0 * np.log(2.0))))
# Midpoint samples across the support of the lateral transfer function (numerical
# integration of a piecewise-smooth function; 512 give the boxcar closed form to 1e-5).
_N_SPECTRUM_SAMPLES = 512
# Receive-aperture widths at which the lateral length is evaluated before it is
# interpolated to the points (it is a smooth function of the width alone).
_N_WIDTH_SAMPLES = 65
# Sine-space width that stands for "no receive aperture": with it the transfer function
# is that of the transmit set alone, and for a single plane wave the length is
# wavelength / 1e-6, about a kilometre, i.e. no lateral resolution.
_MIN_SINE_WIDTH = 1.0e-6


def gaussian_fwhm(correlation_length: np.ndarray | float) -> np.ndarray | float:
    """Amplitude FWHM of the Gaussian PSF that has the given correlation length [m]."""
    return correlation_length / GAUSSIAN_LENGTH_PER_FWHM


def image_spectrum(transducer: Transducer, freqs: np.ndarray) -> np.ndarray:
    """Amplitude weight `W(f)` of each frequency in a pixel value (module docstring).

    Args:
        transducer: the array (pulse centre frequency and bandwidth).
        freqs: frequency grid of the synthesis [Hz], shape (nf,), uniformly spaced.

    Returns:
        float64 array, shape (nf,), arbitrary scale.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    return pulse_spectrum(transducer, freqs) * freqs**IMAGE_SPECTRUM_EXPONENT


def _power_weights(transducer: Transducer, freqs: np.ndarray) -> np.ndarray:
    """Share `p_f = W_f^2 / sum W^2` of each frequency in the power of a pixel, shape (nf,)."""
    power = image_spectrum(transducer, freqs) ** 2
    return power / power.sum()


def mean_wavenumber(transducer: Transducer, freqs: np.ndarray, sound_speed: float) -> float:
    """Power-weighted mean wavenumber `2 pi f / c` of the image [rad/m].

    Args:
        transducer: the array.
        freqs: frequency grid of the synthesis [Hz], shape (nf,).
        sound_speed: sound speed of the medium the image is formed in [m/s].
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    mean_frequency = float(np.sum(_power_weights(transducer, freqs) * freqs))
    return 2.0 * np.pi / float(wavelength(sound_speed, mean_frequency))


def axial_correlation_length(
    transducer: Transducer, freqs: np.ndarray, sound_speed: float
) -> float:
    """Correlation length of the PSF along depth, `(c / 2) int W^4 df / (int W^2 df)^2` [m].

    Args:
        transducer: the array.
        freqs: frequency grid of the synthesis [Hz], shape (nf,), uniformly spaced.
        sound_speed: sound speed of the medium the image is formed in [m/s].
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    share = _power_weights(transducer, freqs)
    spacing = float(freqs[1] - freqs[0])
    return float(0.5 * sound_speed * np.sum(share**2) / spacing)


def receive_sine_aperture(
    transducer: Transducer, points: np.ndarray, beamforming: BeamformingConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Active receive aperture of each point in sine space.

    The active aperture is that of `fus.beamform.receive_apodization` (the whole
    array, or the elements within `z / (2 f_number)` of the point's lateral
    position), limited to the array. Its two ends are seen from the point at
    angles `t` from the depth axis; the aperture in sine space runs between
    their sines.

    Args:
        transducer: the array.
        points: plane points [m], shape (..., 2), columns (x, z), z > 0.
        beamforming: receive aperture options.

    Returns:
        `(centre, width)` of the interval of `sin t`, float64 arrays of shape (...).
    """
    points = np.asarray(points, dtype=np.float64)
    x, z = points[..., 0], points[..., 1]
    half_array = 0.5 * transducer.aperture
    if beamforming.f_number > 0.0:
        half_width = z / (2.0 * beamforming.f_number)
        first = np.clip(x - half_width, -half_array, half_array)
        last = np.clip(x + half_width, -half_array, half_array)
    else:
        first = np.full_like(x, -half_array)
        last = np.full_like(x, half_array)
    sine_first = (first - x) / np.hypot(first - x, z)
    sine_last = (last - x) / np.hypot(last - x, z)
    return 0.5 * (sine_first + sine_last), sine_last - sine_first


def transmit_sine_aperture(sequence: PlaneWaveSequence) -> tuple[float, float]:
    """Synthetic transmit aperture of a plane-wave sequence in sine space.

    `n` angles evenly spaced in `sin a` act as a filled aperture of width
    `n (max - min) / (n - 1)` (each angle stands for one spacing); one angle has
    width 0.

    Returns:
        `(centre, width)` of the interval of `sin a`.
    """
    sines = np.sin(np.deg2rad(np.asarray(sequence.angles_deg, dtype=np.float64)))
    n_angles = sines.shape[0]
    width = n_angles * (sines.max() - sines.min()) / (n_angles - 1) if n_angles > 1 else 0.0
    return float(sines.mean()), float(width)


def _window_integral(window: str, position: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Integral of the receive window from the lower end of its aperture to `position`.

    The window spans `[-width / 2, width / 2]` with peak value 1.
    """
    inside = np.clip(position, -0.5 * width, 0.5 * width)
    ramp = inside + 0.5 * width
    if window == "hann":
        return 0.5 * ramp + width / (4.0 * np.pi) * np.sin(2.0 * np.pi * inside / width)
    return ramp


def _transfer_power(
    window: str, sine: np.ndarray, receive_width: np.ndarray, transmit_width: float
) -> np.ndarray:
    """Squared lateral transfer function `T(s)^2` of one frequency (arbitrary scale).

    `T` is the receive window on `[-A/2, A/2]` convolved with a boxcar of width
    `B`: `F(s + B/2) - F(s - B/2)` with `F` the integral of the window; for
    `B = 0` it is the window itself.

    Args:
        window: "boxcar" or "hann".
        sine: positions `s` about the centre of the support, any shape.
        receive_width: `A` > 0, broadcastable to `sine`.
        transmit_width: `B` >= 0.
    """
    if transmit_width > 0.0:
        transfer = _window_integral(
            window, sine + 0.5 * transmit_width, receive_width
        ) - _window_integral(window, sine - 0.5 * transmit_width, receive_width)
    elif window == "hann":
        inside = np.abs(sine) <= 0.5 * receive_width
        transfer = np.where(inside, 0.5 * (1.0 + np.cos(2.0 * np.pi * sine / receive_width)), 0.0)
    else:
        transfer = (np.abs(sine) <= 0.5 * receive_width).astype(np.float64)
    return transfer**2


def _lateral_length_of_width(
    receive_width: np.ndarray,
    transmit_width: float,
    window: str,
    wavelengths: np.ndarray,
    share: np.ndarray,
) -> np.ndarray:
    """Lateral correlation length for receive apertures of given sine-space widths [m].

    The integral for `l_x` of the module docstring, by the midpoint rule over the
    spatial frequencies `sigma` [cycles/m] that the shortest wavelength reaches.

    Args:
        receive_width: `A` > 0, shape (m,).
        transmit_width: `B` >= 0.
        window: receive window.
        wavelengths: wavelength of every frequency [m], shape (nf,).
        share: power share `p_f` of every frequency, shape (nf,), sum 1.

    Returns:
        float64 array, shape (m,).
    """
    receive_width = receive_width[:, None]
    support = receive_width + transmit_width
    fraction = ((np.arange(_N_SPECTRUM_SAMPLES) + 0.5) / _N_SPECTRUM_SAMPLES - 0.5)[None, :]
    norm = _transfer_power(window, fraction * support, receive_width, transmit_width).sum(
        axis=1
    ) * (support[:, 0] / _N_SPECTRUM_SAMPLES)
    frequency = fraction * support / wavelengths.min()  # sigma, shape (m, n_samples)
    spectrum = np.zeros(frequency.shape)
    for wavelength_f, share_f in zip(wavelengths, share):
        spectrum += (share_f * wavelength_f) * _transfer_power(
            window, frequency * wavelength_f, receive_width, transmit_width
        )
    step = support[:, 0] / wavelengths.min() / _N_SPECTRUM_SAMPLES
    return (spectrum**2).sum(axis=1) * step / norm**2


def lateral_correlation_length(
    transducer: Transducer,
    sequence: PlaneWaveSequence,
    beamforming: BeamformingConfig,
    points: np.ndarray,
    freqs: np.ndarray,
    sound_speed: float,
) -> np.ndarray:
    """Correlation length of the PSF along the array at each point (module docstring) [m].

    The length depends on the point only through the sine-space width of its
    receive aperture; it is evaluated on a ladder of widths spanning those of the
    points, and its reciprocal (nearly linear in the width) is interpolated.

    A point whose f-number aperture misses the array has no receive aperture and
    no image; it gets the length of the transmit set alone (`_MIN_SINE_WIDTH`),
    which matters only as the resolution attributed to blood at that point when
    another pixel sees it.

    Args:
        transducer: the array.
        sequence: gives the steering angles.
        beamforming: receive aperture and window.
        points: plane points [m], shape (..., 2), columns (x, z), z > 0.
        freqs: frequency grid of the synthesis [Hz], shape (nf,), uniformly spaced.
        sound_speed: sound speed of the medium the image is formed in [m/s].

    Returns:
        float64 array, shape (...).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    _, receive_width = receive_sine_aperture(transducer, points, beamforming)
    _, transmit_width = transmit_sine_aperture(sequence)
    receive_width = np.maximum(receive_width, _MIN_SINE_WIDTH)
    ladder = np.linspace(receive_width.min(), receive_width.max(), _N_WIDTH_SAMPLES)
    lengths = _lateral_length_of_width(
        ladder,
        transmit_width,
        beamforming.window,
        np.asarray(wavelength(sound_speed, freqs), dtype=np.float64),
        _power_weights(transducer, freqs),
    )
    if ladder[-1] == ladder[0]:
        return np.full(receive_width.shape, lengths[0])
    return 1.0 / np.interp(receive_width, ladder, 1.0 / lengths)
