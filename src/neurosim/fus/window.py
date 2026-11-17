"""Unambiguous time window of a sampled-frequency simulation.

Channel data are synthesised at `nf` uniformly spaced frequencies with spacing
`df`. A signal known only at those frequencies is periodic in time with period
`1 / df`: an echo arriving at time `t` cannot be told from one at `t + 1/df`.
In the beamformer this means that a scatterer whose round-trip time on some
(angle, element) channel differs from a pixel's by a multiple of `1/df` adds
coherently at that pixel (range aliasing, `c / (2 df)` in range).

The imaging chain never leaves the frequency domain, so no explicit time
origin is needed: each channel's window is implicitly the `1/df`-long interval
that holds the echoes of the ROI. The requirement is therefore that, on every
channel, the spread of round-trip times over the ROI plus the pulse length
fits in `1/df`. `check_time_window` enforces it; `ImagingConfig` and
`acquire_frames` both call it.

What is *not* covered: echoes from outside the ROI. Blood and tissue
scatterers only exist inside the ROI, so none arise from them. The static
echoes in `GreensBank.static_fmc` (skull surface, reverberation) do arrive
outside the window and are folded into it; see `synthesis.static_clutter_spectra`.
"""

from __future__ import annotations

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.fus.pulse import pulse_duration
from neurosim.fus.types import PlaneWaveSequence, Transducer

# Level at which the pulse envelope is taken to have ended, below its peak [dB].
# assumed, to verify: -40 dB keeps aliased pulse tails 40 dB below the main echo.
_PULSE_END_DB = 40.0
# Points per side of the lattice on which the ROI's round-trip times are sampled.
_SPAN_SAMPLES = 9


def unambiguous_window(freqs: np.ndarray) -> float:
    """Period in time of a signal sampled at uniformly spaced frequencies, `1 / df` [s]."""
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.shape[0] < 2:
        raise ValueError("at least two frequencies are needed to define a time window")
    return float((freqs.shape[0] - 1) / (freqs[-1] - freqs[0]))


def round_trip_span(
    transducer: Transducer, sequence: PlaneWaveSequence, grid: Grid2D, c_ref: float
) -> float:
    """Largest spread of round-trip times over a grid on any (angle, element) channel [s].

    The round-trip time of point `(x, z)` for steering angle `a` and receive
    element at `x_j` is `(x sin a + z cos a) / c + hypot(x - x_j, z) / c` (plane
    wave out, direct path back, in the reference medium). The spread is sampled
    on a lattice over the grid's full extent, outer cell faces included.

    Args:
        transducer: the array.
        sequence: gives the steering angles.
        grid: region holding every scatterer and pixel.
        c_ref: reference sound speed [m/s].
    """
    (z0, x0), (nz, nx) = grid.origin, grid.shape
    z = np.linspace(z0 - 0.5 * grid.spacing, z0 + (nz - 0.5) * grid.spacing, _SPAN_SAMPLES)
    x = np.linspace(x0 - 0.5 * grid.spacing, x0 + (nx - 0.5) * grid.spacing, _SPAN_SAMPLES)
    z, x = (axis.reshape(-1) for axis in np.meshgrid(z, x, indexing="ij"))
    angles = np.deg2rad(np.asarray(sequence.angles_deg, dtype=np.float64))
    transmit = x[None, :] * np.sin(angles)[:, None] + z[None, :] * np.cos(angles)[:, None]
    receive = np.hypot(x[None, :] - transducer.element_x()[:, None], z[None, :])
    round_trip = (transmit[:, None, :] + receive[None, :, :]) / c_ref  # (n_angles, n_elements, n)
    return float(np.ptp(round_trip, axis=2).max())


def check_time_window(
    freqs: np.ndarray,
    transducer: Transducer,
    sequence: PlaneWaveSequence,
    grid: Grid2D,
    c_ref: float,
) -> None:
    """Raise if echoes from different parts of the ROI would alias onto each other.

    Args:
        freqs: frequency grid [Hz], uniformly spaced.
        transducer: the array.
        sequence: gives the steering angles.
        grid: region holding every scatterer and pixel.
        c_ref: reference sound speed [m/s].

    Raises:
        ValueError: if round-trip span + pulse length exceeds `1 / df`.
    """
    window = unambiguous_window(freqs)
    needed = round_trip_span(transducer, sequence, grid, c_ref) + pulse_duration(
        transducer, _PULSE_END_DB
    )
    if needed > window:
        n_needed = int(np.ceil(needed * (freqs[-1] - freqs[0]))) + 1
        raise ValueError(
            f"range aliasing: the ROI's round-trip span plus pulse length is {needed * 1e6:.1f} us "
            f"but the frequency spacing only gives an unambiguous window of {window * 1e6:.1f} us; "
            f"use at least {n_needed} frequencies or a smaller ROI"
        )
