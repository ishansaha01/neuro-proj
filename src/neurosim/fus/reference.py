"""Green's function of the homogeneous reference medium (docs/ARCHITECTURE.md, section 5).

This is the `G_ref` that every `GreensBank` is defined relative to, so it has
exactly one implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1

from neurosim.fus.types import Transducer


def reference_greens(
    transducer: Transducer,
    xz: np.ndarray,
    freqs: np.ndarray,
    c_ref: float,
    n_sub: int = 5,
) -> np.ndarray:
    """One-way 2D free-space transfer function from each array element to each point.

    A point (line) source in 2D radiates `(i/4) H0^(1)(k r)` under the
    exp(-i w t) convention (Morse & Ingard, Theoretical Acoustics, 1968, eq. 7.3.14).
    A finite element is the mean of `n_sub` point sources spread uniformly over
    its width, which gives the element directivity.

    Args:
        transducer: the array (elements at z = 0).
        xz: field points [m], shape (..., 2), columns (x, z); z > 0.
        freqs: frequencies [Hz], shape (nf,).
        c_ref: sound speed of the reference medium [m/s].
        n_sub: point sources per element (odd, so one sits at the element centre).

    Returns:
        complex128 array, shape (nf, n_elements, ...) [dimensionless, 2D Green's function].
    """
    xz = np.asarray(xz, dtype=np.float64)
    field_shape = xz.shape[:-1]
    x = xz[..., 0].reshape(-1)
    z = xz[..., 1].reshape(-1)
    k = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64) / c_ref  # [rad/m], (nf,)

    offsets = (np.arange(n_sub) - (n_sub - 1) / 2.0) / n_sub * transducer.element_width
    element_x = transducer.element_x()

    out = np.zeros((k.shape[0], element_x.shape[0], x.shape[0]), dtype=np.complex128)
    for offset in offsets:
        dx = x[None, :] - (element_x[:, None] + offset)  # (n_elements, n_points)
        r = np.hypot(dx, z[None, :])
        out += 0.25j * hankel1(0, k[:, None, None] * r[None, :, :])
    out /= n_sub
    return out.reshape(k.shape[0], element_x.shape[0], *field_shape)
