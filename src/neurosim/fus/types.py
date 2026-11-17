"""Contract types for the fUS model (see docs/ARCHITECTURE.md, section 5).

`GreensBank` is the only interface between the solver backends and the imaging
chain. Change these types only together with docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from neurosim.common.grid import Grid2D, sample_plane_map


@dataclass(frozen=True)
class Transducer:
    """Linear array at depth z = 0, centred on x = 0, elements along +x.

    Attributes:
        n_elements: number of elements.
        pitch: centre-to-centre element spacing [m].
        element_width: active width of one element [m].
        center_frequency: pulse centre frequency [Hz].
        fractional_bandwidth: -6 dB two-way bandwidth as a fraction of the centre frequency.
    """

    n_elements: int
    pitch: float
    element_width: float
    center_frequency: float
    fractional_bandwidth: float

    def element_x(self) -> np.ndarray:
        """Lateral positions of the element centres [m], shape (n_elements,)."""
        return (np.arange(self.n_elements) - (self.n_elements - 1) / 2.0) * self.pitch

    @property
    def aperture(self) -> float:
        """Total aperture width [m]."""
        return self.n_elements * self.pitch


@dataclass(frozen=True)
class PlaneWaveSequence:
    """Plane-wave compounding sequence for one Doppler ensemble.

    Attributes:
        angles_deg: steering angles [degrees], one transmit per angle per frame.
        frame_rate: compounded frame rate [Hz].
        n_frames: compounded frames per power Doppler image (ensemble length).
    """

    angles_deg: tuple[float, ...]
    frame_rate: float
    n_frames: int


@dataclass(frozen=True)
class Scatterers:
    """Point scatterers in the imaging plane.

    Attributes:
        xz: positions [m], float64, shape (n, 2), columns (x, z).
        amplitude: scattering amplitude (arbitrary linear units), shape (n,), real or complex.
    """

    xz: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class PhaseScreen:
    """Low-dimensional skull model: one delay and one amplitude per array element.

    Applied to a bank as `B * amplitude_e * exp(+i 2 pi f delay_e)`
    (`neurosim.fus.screen.apply_phase_screen`).

    Attributes:
        delays: extra one-way delay per element [s], float64, shape (n_elements,).
        amplitudes: one-way amplitude factor per element (dimensionless, > 0), shape (n_elements,).
    """

    delays: np.ndarray
    amplitudes: np.ndarray


@dataclass(frozen=True)
class GreensBank:
    """Everything one medium does to the sound field, relative to a homogeneous reference.

    `B[f, e, iz, ix] = G_medium(r, e, f) / G_ref(r, e, f)`: the one-way transfer
    function from array element `e` to ROI point `r = roi.points()[iz, ix]`,
    divided by that of the homogeneous lossless reference medium with sound speed
    `c_ref`. With the exp(-i w t) convention, an extra one-way delay `tau` and an
    amplitude factor `a` give `B = a * exp(+i w tau)`. By reciprocity the same `B`
    serves transmit and receive.

    Attributes:
        freqs: frequencies [Hz], float64, shape (nf,), ascending and uniformly spaced.
        roi: coarse grid the bank is stored on; it must cover every scatterer and pixel.
        B: complex64, shape (nf, n_elements, nz_roi, nx_roi).
        c_ref: sound speed of the reference medium [m/s].
        transducer: the array the bank was computed for.
        backend: name of the backend that produced it ("analytic", "ray", "bent_ray", "kwave").
        static_fmc: optional full-matrix response of the static medium (skull echoes) at
            the array, complex64, shape (nf, n_elements_tx, n_elements_rx), in units of
            pressure spectrum per unit source spectrum (same normalisation as G_medium).
        meta: free-form provenance (backend settings, grid size, run time).
    """

    freqs: np.ndarray
    roi: Grid2D
    B: np.ndarray
    c_ref: float
    transducer: Transducer
    backend: str
    static_fmc: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (self.freqs.shape[0], self.transducer.n_elements, *self.roi.shape)
        if self.B.shape != expected:
            raise ValueError(f"B has shape {self.B.shape}, expected {expected}")

    @classmethod
    def homogeneous(
        cls, transducer: Transducer, roi: Grid2D, freqs: np.ndarray, c_ref: float
    ) -> "GreensBank":
        """Bank of the reference medium itself: B = 1 everywhere (the analytic backend)."""
        shape = (freqs.shape[0], transducer.n_elements, *roi.shape)
        return cls(
            freqs=np.asarray(freqs, dtype=np.float64),
            roi=roi,
            B=np.ones(shape, dtype=np.complex64),
            c_ref=c_ref,
            transducer=transducer,
            backend="analytic",
        )

    def interp(self, xz: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of B at arbitrary plane points.

        Args:
            xz: points [m], shape (..., 2), columns (x, z); must lie inside `roi`.

        Returns:
            complex64 array of shape (nf, n_elements, ...).
        """
        nf, ne = self.B.shape[:2]
        flat = self.B.reshape(nf * ne, *self.roi.shape)
        out = sample_plane_map(flat, self.roi, xz, order=1, fill=0.0)
        return out.reshape(nf, ne, *np.shape(xz)[:-1]).astype(np.complex64)
