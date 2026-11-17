"""Blood and tissue point scatterers.

Blood: every scatterer has the same strength, so the *density* of scatterers
carries the blood volume. The strength is set so that the total backscatter
equals the blood area, `sum_k |s_k|^2 = sum(cbv * cell_area)`, whatever the
number of scatterers (`fus.config` explains the normalisation). Each scatterer
has a random sign: a point scatterer stands for a local fluctuation of the
medium about its mean, which is as often negative as positive. With all signs
equal the mean scatterer density would itself act as a reflector, and every
sharp change of it (the ROI boundary, the steps between CBV cells) would give a
coherent echo that real blood does not produce. With random signs the expected
power of the blood signal is exactly the sum of the scatterers' powers.

Blood motion: a scatterer drawn at a home position travels a straight path
centred on its home, along the local in-plane blood velocity (the velocity of
its home cell). On reaching the end of the path it re-enters at the start
(`advance`); the same happens at the faces of the grid, which is periodic. This
is deterministic and keeps the time-averaged scatterer density equal to the CBV
map blurred by the path length: plain advection along an arbitrary velocity map
would carry the CBV pattern away with the flow. The re-entry is the in-plane
picture of a blood cell leaving the resolution cell while another enters. Path
lengths are spread uniformly over 0.5 to 1.5 times the nominal transit length,
so that scatterers of equal speed do not all return to their starting points at
the same instant (which would repeat the whole speckle pattern periodically).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.common.units import db_to_np
from neurosim.fus.config import NoiseConfig
from neurosim.fus.types import Scatterers

# Range of a scatterer's path length, as fractions of the nominal transit length.
_PATH_LENGTH_RANGE = (0.5, 1.5)


@dataclass(frozen=True)
class BloodState:
    """Blood scatterers at one instant.

    Attributes:
        home_xz: centre of each scatterer's path [m], float64, shape (n, 2), columns (x, z).
        velocity: in-plane velocity of each scatterer [m/s], float64, shape (n, 2), (vx, vz).
        path_position: distance travelled along the path from its start [m], float64,
            shape (n,), in [0, path_length).
        path_length: length of each scatterer's path [m], float64, shape (n,).
        amplitude: scattering amplitude of each scatterer, float64, shape (n,): one
            magnitude with random signs.
        grid: the grid of the CBV map; scatterers stay inside its cells.
    """

    home_xz: np.ndarray
    velocity: np.ndarray
    path_position: np.ndarray
    path_length: np.ndarray
    amplitude: np.ndarray
    grid: Grid2D

    def scatterers(self) -> Scatterers:
        """Current positions and amplitudes."""
        speed = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        direction = np.divide(
            self.velocity, speed, out=np.zeros_like(self.velocity), where=speed > 0.0
        )
        xz = self.home_xz + direction * (self.path_position - 0.5 * self.path_length)[:, None]
        return Scatterers(xz=_wrap_into_grid(xz, self.grid), amplitude=self.amplitude)


def _cell_box(grid: Grid2D) -> tuple[np.ndarray, np.ndarray]:
    """Lower corner and size of the box covered by the grid's cells, as (x, z) pairs [m]."""
    (z0, x0), (nz, nx) = grid.origin, grid.shape
    lower = np.array([x0 - 0.5 * grid.spacing, z0 - 0.5 * grid.spacing])
    return lower, np.array([nx * grid.spacing, nz * grid.spacing])


def _wrap_into_grid(xz: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Map points periodically into the box covered by the grid's cells."""
    lower, size = _cell_box(grid)
    return lower + np.mod(xz - lower, size)


def blood_scatterers(
    cbv_map: np.ndarray,
    velocity_map: np.ndarray,
    grid: Grid2D,
    n_scatterers: int,
    rng: np.random.Generator,
    transit_length: float,
) -> BloodState:
    """Draw blood scatterers with number density proportional to CBV.

    A cell is chosen with probability proportional to its CBV, then a position
    uniformly inside the cell, a path length, a uniformly distributed starting
    point along the path, and a random sign.

    Args:
        cbv_map: blood volume fraction, shape (nz, nx), >= 0, not all zero.
        velocity_map: in-plane blood velocity [m/s], shape (2, nz, nx), components (vx, vz).
        grid: grid of both maps.
        n_scatterers: number of scatterers (a numerical choice; enough for several
            per resolution cell where there is blood).
        rng: generator of the `blood_scatterers` stream.
        transit_length: nominal (mean) path length before re-entry [m].
    """
    cbv_map = np.asarray(cbv_map, dtype=np.float64)
    velocity_map = np.asarray(velocity_map, dtype=np.float64)
    if cbv_map.shape != grid.shape or velocity_map.shape != (2, *grid.shape):
        raise ValueError(
            f"cbv_map {cbv_map.shape} and velocity_map {velocity_map.shape} must have shapes "
            f"{grid.shape} and {(2, *grid.shape)}"
        )
    total = cbv_map.sum()
    if cbv_map.min() < 0.0 or not total > 0.0:
        raise ValueError("cbv_map must be non-negative and not all zero")

    cell = rng.choice(cbv_map.size, size=n_scatterers, p=cbv_map.reshape(-1) / total)
    iz, ix = np.unravel_index(cell, grid.shape)
    jitter = rng.uniform(-0.5, 0.5, size=(n_scatterers, 2))
    z, x = grid.axes()
    home_xz = np.stack((x[ix], z[iz]), axis=1) + jitter * grid.spacing
    path_length = transit_length * rng.uniform(*_PATH_LENGTH_RANGE, size=n_scatterers)
    path_position = path_length * rng.uniform(0.0, 1.0, size=n_scatterers)
    magnitude = np.sqrt(total * grid.spacing**2 / n_scatterers)
    return BloodState(
        home_xz=home_xz,
        velocity=velocity_map[:, iz, ix].T.copy(),
        path_position=path_position,
        path_length=path_length,
        amplitude=magnitude * rng.choice((-1.0, 1.0), size=n_scatterers),
        grid=grid,
    )


def advance(state: BloodState, dt: float) -> BloodState:
    """Move every scatterer along its path for a time `dt` [s], with re-entry at the path end."""
    speed = np.linalg.norm(state.velocity, axis=1)
    path_position = np.mod(state.path_position + speed * dt, state.path_length)
    return replace(state, path_position=path_position)


def tissue_amplitude(grid: Grid2D, n_scatterers: int, noise: NoiseConfig) -> float:
    """Rms amplitude of tissue scatterers that realises `noise.tissue_to_blood_db`.

    Tissue backscatter per unit area, `n_scatterers * amplitude^2 / area`, is set
    to `tissue_to_blood_db` above that of blood at `cbv_reference` (which is
    `cbv_reference` in the units of `fus.config`).

    Args:
        grid: region the tissue scatterers fill (its cells).
        n_scatterers: number of tissue scatterers.
        noise: gives `tissue_to_blood_db` and `cbv_reference`.
    """
    area = grid.shape[0] * grid.shape[1] * grid.spacing**2
    power_ratio = float(np.exp(2.0 * db_to_np(noise.tissue_to_blood_db)))
    return float(np.sqrt(power_ratio * noise.cbv_reference * area / n_scatterers))


def tissue_scatterers(
    grid: Grid2D, n_scatterers: int, amplitude: float, rng: np.random.Generator
) -> Scatterers:
    """Draw static tissue scatterers uniformly over the cells of a grid.

    Amplitudes are real and normally distributed with standard deviation
    `amplitude`, the usual model of fully developed speckle.

    Args:
        grid: region to fill.
        n_scatterers: number of scatterers.
        amplitude: rms scattering amplitude.
        rng: generator of the `tissue_scatterers` stream.
    """
    lower, size = _cell_box(grid)
    xz = lower + rng.uniform(0.0, 1.0, size=(n_scatterers, 2)) * size
    return Scatterers(xz=xz, amplitude=amplitude * rng.standard_normal(n_scatterers))
