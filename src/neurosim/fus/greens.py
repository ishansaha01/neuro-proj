"""One-way transfer function of the medium, `G = B * G_ref`, at arbitrary plane points.

`G_ref` (fast-varying, analytic) comes from `fus.reference`; `B` (slowly varying)
is interpolated bilinearly from the bank's coarse ROI grid. This module is the
only place where the two are multiplied, so synthesis, beamforming and the
operators all see the same medium.
"""

from __future__ import annotations

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.fus.reference import reference_greens
from neurosim.fus.types import GreensBank

# How far beyond the outermost ROI nodes a point may lie [ROI spacings]. Scatterers
# are generated up to the outer faces of the outermost ROI cells (half a spacing
# beyond the nodes) and tissue micro-motion may push them a little further.
_EXTENSION_SPACINGS = 1.0


def check_same_sampling(bank: GreensBank, model_bank: GreensBank) -> None:
    """Raise if two banks do not share the array and the frequency grid.

    Channel data synthesised with one bank can only be beamformed with the
    weights of another if both describe the same elements at the same frequencies.
    """
    if bank.transducer != model_bank.transducer:
        raise ValueError(
            f"the banks are for different arrays: {bank.transducer} and {model_bank.transducer}"
        )
    if not np.array_equal(bank.freqs, model_bank.freqs):
        raise ValueError("the banks are sampled on different frequency grids")


def bilinear_weights(grid: Grid2D, xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corner nodes and weights for bilinear interpolation on a plane grid.

    Inside the hull of the node centres this is the interpolation of
    `GreensBank.interp` (`common.grid.sample_plane_map`, order 1). Outside it the
    value of the nearest edge node is used, up to one spacing beyond the
    outermost nodes; `sample_plane_map` does the same up to half a spacing and
    returns its fill value beyond.

    Args:
        grid: the grid the map lives on.
        xz: points [m], shape (n, 2), columns (x, z).

    Returns:
        (nodes, weights): flat node indices into a (nz * nx) map, int64, shape (4, n);
        weights, float32, shape (4, n), summing to 1 for every point.

    Raises:
        ValueError: if a point lies more than one spacing outside the outermost nodes.
    """
    xz = np.asarray(xz, dtype=np.float64)
    nz, nx = grid.shape
    z0, x0 = grid.origin
    iz = (xz[:, 1] - z0) / grid.spacing
    ix = (xz[:, 0] - x0) / grid.spacing
    lowest = min(iz.min(initial=0.0), ix.min(initial=0.0))
    excess = max((iz - (nz - 1)).max(initial=0.0), (ix - (nx - 1)).max(initial=0.0), -lowest)
    if excess > _EXTENSION_SPACINGS:
        raise ValueError(
            f"points lie up to {excess:.2f} grid spacings outside the bank ROI "
            f"(allowed: {_EXTENSION_SPACINGS}); the ROI must cover every scatterer and pixel"
        )
    iz = np.clip(iz, 0.0, nz - 1)
    ix = np.clip(ix, 0.0, nx - 1)
    iz0 = np.minimum(np.floor(iz).astype(np.int64), max(nz - 2, 0))
    ix0 = np.minimum(np.floor(ix).astype(np.int64), max(nx - 2, 0))
    iz1 = np.minimum(iz0 + 1, nz - 1)
    ix1 = np.minimum(ix0 + 1, nx - 1)
    fz = iz - iz0
    fx = ix - ix0
    nodes = np.stack((iz0 * nx + ix0, iz0 * nx + ix1, iz1 * nx + ix0, iz1 * nx + ix1))
    weights = np.stack(((1 - fz) * (1 - fx), (1 - fz) * fx, fz * (1 - fx), fz * fx))
    return nodes, weights.astype(np.float32)


def medium_greens(bank: GreensBank, xz: np.ndarray, freq_slice: slice = slice(None)) -> np.ndarray:
    """One-way transfer function from every element to every point in the bank's medium.

    `G(r, e, f) = B(r, e, f) * G_ref(r, e, f)` with `B` interpolated bilinearly.
    By reciprocity it serves both transmit and receive.

    Args:
        bank: the medium.
        xz: points [m], shape (n, 2), columns (x, z); inside the bank ROI.
        freq_slice: which of `bank.freqs` to evaluate (a contiguous slice keeps
            the grid uniform, which `reference_greens` needs for its fast route).

    Returns:
        complex128 array, shape (nf_selected, n_elements, n) [dimensionless, 2D Green's function].
    """
    xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
    greens = reference_greens(bank.transducer, xz, bank.freqs[freq_slice], bank.c_ref)
    nodes, weights = bilinear_weights(bank.roi, xz)
    ratio_map = bank.B[freq_slice].reshape(greens.shape[0], greens.shape[1], -1)
    ratio = np.take(ratio_map, nodes[0], axis=2)
    ratio *= weights[0]
    for corner_nodes, corner_weights in zip(nodes[1:], weights[1:]):
        corner = np.take(ratio_map, corner_nodes, axis=2)
        corner *= corner_weights
        ratio += corner
    greens *= ratio
    return greens
