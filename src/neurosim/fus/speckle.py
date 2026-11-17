"""How many independent speckle realisations one power Doppler image of moving blood holds.

A power Doppler pixel is the mean power of `n` compounded frames,
`I = (1/n) sum_i |E_i|^2`, taken `dt = 1 / frame_rate` apart. Blood moves little
between frames, so the frames are correlated and `I` fluctuates far more than
the mean of `n` independent speckle intensities would (ROADMAP D-018).

Derivation.

1. *Variance of the mean power.* The blood part of a pixel is a sum over many
   scatterers with independent phases: a circular complex Gaussian process in
   slow time with correlation `Gamma(tau) = E[E(t + tau) conj(E(t))]`. For such a
   process `cov(|E_i|^2, |E_j|^2) = |Gamma(t_i - t_j)|^2` (moment theorem for
   complex Gaussian processes, Reed 1962, IRE Trans. Inf. Theory 8(3):194-195),
   hence, with `g = Gamma / Gamma(0)`,

       var(I) / mean(I)^2 = (1/n^2) sum_ij |g(t_i - t_j)|^2
                          = (1/n) [1 + 2 sum_{k=1}^{n-1} (1 - k/n) |g(k dt)|^2]  =:  1 / N

   `N` is the number of independent frames (`independent_frames`): 1 for static
   speckle (`g = 1`), `n` when successive frames are uncorrelated. It is the
   discrete form of the "number of degrees of freedom" of integrated speckle,
   `M = [(2/T) int_0^T (1 - tau/T) |g(tau)|^2 dtau]^-1` (Goodman, Speckle
   Phenomena in Optics, 2007, sec. 4.7, "Statistics of integrated and blurred
   speckle"; also Goodman, Statistical Optics, 1985, on integrated intensity),
   which for long ensembles tends to `T / tau_c`, `tau_c = int |g|^2 dtau`. The
   first two moments of `I` are exact; its density is *approximated* by a gamma
   density of shape `N` (Goodman 2007, sec. 4.7.2), which is exact only when `N`
   equal and independent cells are averaged.

2. *Correlation of blood in uniform flow.* Blood that moves with velocity `v`
   in front of a pixel displaces every scatterer by `v tau`; the pixel's field
   correlation is then the autocorrelation `C(v tau)` of the point-spread function
   (`fus.resolution`). For a Gaussian resolution cell with correlation lengths
   `l_x` (lateral) and `l_z` (axial)

       |g(tau)|^2 = exp(-pi tau^2 [(v_x / l_x)^2 + (v_z / l_z)^2])

   so the decorrelation time is `tau_c = l_u / |v|`, with
   `1 / l_u^2 = cos^2(phi) / l_x^2 + sin^2(phi) / l_z^2` for flow at angle `phi`
   from the array axis: blood has to cross one correlation length of the
   resolution cell along its direction of flow. For `n dt >> tau_c`,
   `N -> |v| n dt / l_u + 1/pi`; the fit of ROADMAP D-018,
   `1 + |v| n dt / (1.06 FWHM)`, has the same slope (`l_u` = 1.064 FWHM for a
   Gaussian) and an offset that is too large by up to `1 - 1/pi` = 0.7. The
   phase of `g` is the Doppler shift, `g = |g| exp(i K . v tau)`, `K` the mean
   wavevector of the echo (`2 k` along depth for a pixel under the array centre).

3. *Blood of several velocities in one pixel.* A pixel sees the source cells `s`
   with powers `H[p, s] cbv_s`. Cells scatter independently, so their Doppler
   spectra add in power and the pixel's field correlation is the power-weighted
   mean of theirs,

       g_p(tau) = sum_s H[p, s] cbv_s g_s(tau) / sum_s H[p, s] cbv_s

   and `N_p` follows from step 1 (`pixel_independent_frames`). The fields of the
   cells interfere in the pixel, so their *intensities* do not add as independent
   variables: `sum_s mean_s^2 / N_s` would give two cells of equal power and equal
   velocity twice the frames of one. The cross terms carry the beat of the cells'
   Doppler shifts, which is why the phase of `g_s` is kept.

Assumptions: the Gaussian shape of `|C|^2` (its integral, which sets `N` for long
ensembles, is that of the diffraction model of `fus.resolution`); each cell's blood
treated as a homogeneous patch filling the resolution cell, with the cell's own
resolution (the point-spread function varies slowly over the cells one pixel
sees); one velocity per cell (no velocity spread inside a cell, no flow out of
the imaging plane); pure translation, without the re-entrant paths of
`fus.scatterers`; the baseline CBV as weights; no clutter filter.

Electronic noise is independent from frame to frame: its count is `n`. How blood
and noise combine in a pixel is in `fus.acquire.statistical_power_doppler`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.config import BeamformingConfig, ImagingConfig, imaging_setup
from neurosim.fus.resolution import (
    axial_correlation_length,
    lateral_correlation_length,
    mean_wavenumber,
    receive_sine_aperture,
    transmit_sine_aperture,
)
from neurosim.fus.types import PlaneWaveSequence, Transducer


@dataclass(frozen=True)
class SpeckleFrames:
    """Independent-frame counts of one power Doppler image (module docstring).

    Attributes:
        blood: number of independent frames of the blood signal of each pixel, in
            [1, n_frames], float32, shape (n_pix,).
        noise: number of independent frames of the electronic noise (the same for
            every pixel).
    """

    blood: np.ndarray
    noise: float


def flow_correlation(
    velocity: np.ndarray,
    lateral_length: np.ndarray | float,
    axial_length: np.ndarray | float,
    wavevector: np.ndarray,
    n_frames: int,
    frame_rate: float,
) -> np.ndarray:
    """Field correlation `g(k dt)`, k = 0 .. n_frames - 1, of blood in uniform flow (step 2).

    Args:
        velocity: in-plane blood velocity [m/s], shape (..., 2), components (vx, vz).
        lateral_length: correlation length of the resolution cell along the array [m],
            scalar or shape (...).
        axial_length: correlation length along depth [m], scalar or shape (...).
        wavevector: mean wavevector of the echo [rad/m], shape (..., 2) or (2,),
            components (Kx, Kz); it sets the Doppler phase only.
        n_frames: frames per power Doppler image.
        frame_rate: compounded frame rate [Hz].

    Returns:
        complex128 array, shape (..., n_frames); `g(0) = 1`.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    lags = np.arange(n_frames) / frame_rate
    rate_squared = (velocity[..., 0] / lateral_length) ** 2 + (velocity[..., 1] / axial_length) ** 2
    doppler = np.sum(np.asarray(wavevector, dtype=np.float64) * velocity, axis=-1)  # [rad/s]
    return np.exp(
        -0.5 * np.pi * rate_squared[..., None] * lags**2 + 1j * doppler[..., None] * lags
    )


def independent_frames(correlation: np.ndarray) -> np.ndarray:
    """Number of independent frames `N` of a process with the given field correlation (step 1).

    Args:
        correlation: field correlation `Gamma(k dt)` at the lags k = 0 .. n - 1 of an
            n-frame ensemble, complex or real, shape (..., n); any scale.

    Returns:
        float64 array, shape (...), values in [1, n]; 1 where `Gamma(0)` is 0 (no signal).
    """
    n_frames = correlation.shape[-1]
    lag = np.arange(n_frames)
    weights = np.where(lag == 0, 1.0, 2.0 * (1.0 - lag / n_frames))
    power = np.abs(correlation) ** 2
    weighted = power @ weights
    has_signal = power[..., 0] > 0.0
    return np.where(has_signal, n_frames * power[..., 0] / np.where(has_signal, weighted, 1.0), 1.0)


def cell_flow_correlation(
    velocity: np.ndarray,
    source_grid: Grid2D,
    transducer: Transducer,
    sequence: PlaneWaveSequence,
    beamforming: BeamformingConfig,
    freqs: np.ndarray,
    sound_speed: float,
) -> np.ndarray:
    """Field correlation of the blood of every source cell, with the cell's own resolution.

    `flow_correlation` with the correlation lengths of `fus.resolution` at the cell
    centre and the mean wavevector `k (sin a + sin t, cos a + cos t)` of the
    transmit set and of the cell's receive aperture.

    Args:
        velocity: in-plane blood velocity per cell [m/s], shape (2, nz, nx), (vx, vz).
        source_grid: grid of the cells.
        transducer: the array.
        sequence: steering angles, frame rate and frames per image.
        beamforming: receive aperture options.
        freqs: frequency grid of the synthesis [Hz], shape (nf,).
        sound_speed: sound speed the image is formed with [m/s].

    Returns:
        complex64 array, shape (n_src, n_frames), n_src = nz * nx in C order.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.shape != (2, *source_grid.shape):
        raise ValueError(
            f"velocity has shape {velocity.shape}, expected {(2, *source_grid.shape)}"
        )
    points = source_grid.points().reshape(-1, 2)
    lateral = lateral_correlation_length(
        transducer, sequence, beamforming, points, freqs, sound_speed
    )
    axial = axial_correlation_length(transducer, freqs, sound_speed)
    receive_sine, _ = receive_sine_aperture(transducer, points, beamforming)
    transmit_sine, _ = transmit_sine_aperture(sequence)
    wavenumber = mean_wavenumber(transducer, freqs, sound_speed)
    wavevector = wavenumber * np.stack(
        (
            transmit_sine + receive_sine,
            np.sqrt(1.0 - transmit_sine**2) + np.sqrt(1.0 - receive_sine**2),
        ),
        axis=-1,
    )
    correlation = flow_correlation(
        velocity.reshape(2, -1).T,
        lateral,
        axial,
        wavevector,
        sequence.n_frames,
        sequence.frame_rate,
    )
    return correlation.astype(np.complex64)


def pixel_independent_frames(
    operator: np.ndarray, cbv_cells: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    """Independent frames of the blood signal of every pixel (step 3).

    Args:
        operator: power Doppler operator `H`, shape (n_pix, n_src).
        cbv_cells: blood volume (or backscatter) of each source cell, shape (n_src,)
            or the shape of the source grid; only its distribution matters.
        correlation: field correlation of the blood of each cell
            (`cell_flow_correlation`), shape (n_src, n_frames).

    Returns:
        float32 array, shape (n_pix,), values in [1, n_frames]; 1 for a pixel that sees
        no blood.
    """
    weights = np.asarray(cbv_cells, dtype=np.float32).reshape(-1, 1)
    operator = np.asarray(operator, dtype=np.float32)
    # Two real single-precision products on contiguous arrays (the fast BLAS route),
    # instead of one complex product with a complex copy of the operator.
    real = operator @ np.ascontiguousarray(weights * correlation.real, dtype=np.float32)
    imaginary = operator @ np.ascontiguousarray(weights * correlation.imag, dtype=np.float32)
    return independent_frames(real + 1j * imaginary).astype(np.float32)


def speckle_frames(
    cfg: ImagingConfig,
    operator: np.ndarray,
    cbv_cells: np.ndarray,
    velocity: np.ndarray,
    source_grid: Grid2D,
) -> SpeckleFrames:
    """Independent-frame counts of the statistical tier for one operator.

    The one place that interprets `cfg.doppler.effective_frames`: a number forces
    that count for blood and noise in every pixel; None derives the blood count
    from the blood velocity (module docstring) and gives the noise `n_frames`.

    Args:
        cfg: imaging configuration (array, sequence, frequency grid, beamforming, override).
        operator: power Doppler operator `H` of the image, shape (n_pix, n_src).
        cbv_cells: blood volume of each source cell, shape (n_src,) or the source grid's.
        velocity: in-plane blood velocity per source cell [m/s], shape (2, nz, nx).
        source_grid: grid of the source cells.
    """
    forced = cfg.doppler.effective_frames
    if forced is not None:
        return SpeckleFrames(
            blood=np.full(operator.shape[0], forced, dtype=np.float32), noise=float(forced)
        )
    setup = imaging_setup(cfg)
    correlation = cell_flow_correlation(
        velocity,
        source_grid,
        setup.transducer,
        setup.sequence,
        cfg.beamforming,
        setup.freqs,
        C_REF,
    )
    return SpeckleFrames(
        blood=pixel_independent_frames(operator, cbv_cells, correlation),
        noise=float(setup.sequence.n_frames),
    )
