"""The only module that talks to k-wave-python (0.6.2, pure-Python k-space solver).

Facts read from the installed solver source (`kwave/kspaceFirstOrder.py`,
`kwave/solvers/kspace_solver.py`) that this wrapper relies on:

- Grid arrays are C-ordered with shape `kgrid.N`. We pass `N = (nz, nx)`, so our
  plane maps `[iz, ix]` go in unchanged (the spacing is isotropic).
- With `pml_inside=False` the grid is padded by the layer thickness on every
  side; medium maps are padded with their edge values, source and sensor masks
  with zeros.
- A binary sensor mask returns `out["p"]` of shape (n_sensor_points, n_steps)
  with the points in C order of the mask, i.e. sorted by (iz, ix).
- An additive pressure source (`p_mode` unset) adds
  `source.p * 2 dt / (c0 dx)` to the density at every source point and time
  step, with `c0` the sound speed at that point: a mass source. One row of
  `source.p` per source point, in C order of the mask.
- `alpha_coeff` is the power-law prefactor in dB/(MHz^y cm), the unit of
  `PlaneMedium.attenuation`; `alpha_power` is one scalar. The dispersion term
  contains tan(pi y / 2) and is singular at y = 1; `alpha_mode="no_dispersion"`
  drops it (`KWaveConfig.model_dispersion` explains why that is the default).
  y = 2 switches to the Stokes form, which has no dispersion term.
- The k-space correction uses the largest sound speed of the grid as its
  reference; `medium.sound_speed_ref` is ignored by the Python solver.
"""

from __future__ import annotations

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.pml import get_optimal_pml_size

from neurosim.fus.backends.config import KWaveConfig

# Half-width of the band of alpha_power around 1 in which the dispersion term is refused.
# k-Wave's own guard for this singularity (kwave/utils/checks.py) warns in the same spirit.
_DISPERSION_SINGULAR_HALF_WIDTH = 0.05


def _pml_size(n_points: int, cfg: KWaveConfig) -> int:
    # k-Wave refuses a grid that is not larger than twice the layer.
    thickest = min(cfg.pml_size + cfg.pml_search, (n_points - 1) // 2)
    if thickest < cfg.pml_size:
        raise ValueError(
            f"a grid axis of {n_points} points is too short for an absorbing layer of "
            f"kwave.pml_size = {cfg.pml_size} points; k-Wave needs more than twice the layer"
        )
    if thickest == cfg.pml_size:
        return cfg.pml_size
    return int(get_optimal_pml_size([n_points], [cfg.pml_size, thickest])[0])


def pml_sizes(shape: tuple[int, int], cfg: KWaveConfig) -> tuple[int, int]:
    """Thickness of the absorbing layer along (z, x) [grid points].

    At least `cfg.pml_size`; up to `cfg.pml_search` thicker where that gives the
    padded grid smaller prime factors (faster FFTs), using k-Wave's own search.

    Raises:
        ValueError: if an axis has no more than `2 * cfg.pml_size` points.
    """
    return _pml_size(shape[0], cfg), _pml_size(shape[1], cfg)


def check_absorption_model(attenuation: np.ndarray, alpha_power: float, cfg: KWaveConfig) -> None:
    """Refuse absorption settings the solver cannot model.

    Args:
        attenuation: alpha0 map [dB/(cm MHz^y)], shape (nz, nx).
        alpha_power: power-law exponent y.
        cfg: backend settings.
    """
    if not np.any(attenuation > 0.0):
        return
    if not 0.0 <= alpha_power < 3.0:
        raise ValueError(f"k-Wave needs 0 <= alpha_power < 3, got {alpha_power}")
    if cfg.model_dispersion and abs(alpha_power - 1.0) < _DISPERSION_SINGULAR_HALF_WIDTH:
        raise ValueError(
            f"alpha_power = {alpha_power} is within {_DISPERSION_SINGULAR_HALF_WIDTH} of 1, where "
            "the dispersion term of k-Wave's power-law absorption is singular; set "
            "kwave.model_dispersion = False to model the absorption without dispersion"
        )


def simulate(
    *,
    sound_speed: np.ndarray,
    density: np.ndarray,
    attenuation: np.ndarray,
    alpha_power: float,
    spacing: float,
    dt: float,
    n_steps: int,
    source_mask: np.ndarray,
    source_signals: np.ndarray,
    sensor_mask: np.ndarray,
    pml: tuple[int, int],
    cfg: KWaveConfig,
) -> np.ndarray:
    """Run one 2D k-Wave simulation and return the recorded pressure.

    Args:
        sound_speed: [m/s], shape (nz, nx).
        density: [kg/m^3], shape (nz, nx).
        attenuation: alpha0 [dB/(cm MHz^y)], shape (nz, nx); all zero for a lossless run.
        alpha_power: power-law exponent y.
        spacing: grid spacing [m].
        dt: time step [s].
        n_steps: number of time steps.
        source_mask: bool, shape (nz, nx); the driven grid points.
        source_signals: additive pressure-source amplitude of every driven point
            (rows in C order of `source_mask`), shape (n_source_points, n_samples),
            n_samples <= n_steps; the source is silent afterwards.
        sensor_mask: bool, shape (nz, nx); the recorded grid points.
        pml: absorbing-layer thickness along (z, x) [grid points], added outside the grid.
        cfg: backend settings (layer absorption, precision, dispersion switch).

    Returns:
        pressure time series, shape (n_sensor_points, n_steps), rows in C order of
        `sensor_mask`, in units of the source amplitude, dtype `cfg.dtype`.
    """
    kgrid = kWaveGrid(list(sound_speed.shape), [spacing, spacing])
    kgrid.setTime(int(n_steps), float(dt))

    absorbing = bool(np.any(attenuation > 0.0))
    medium = kWaveMedium(
        sound_speed=np.asarray(sound_speed, dtype=np.float64),
        density=np.asarray(density, dtype=np.float64),
        alpha_coeff=np.asarray(attenuation, dtype=np.float64) if absorbing else None,
        alpha_power=float(alpha_power) if absorbing else None,
        alpha_mode=None if cfg.model_dispersion or not absorbing else "no_dispersion",
    )
    source = kSource()
    source.p_mask = np.asarray(source_mask, dtype=bool)
    source.p = np.asarray(source_signals, dtype=np.float64)
    sensor = kSensor(mask=np.asarray(sensor_mask, dtype=bool), record=["p"])

    result = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend="python",
        device="cpu",
        quiet=True,
        pml_inside=False,
        pml_size=pml,
        pml_alpha=cfg.pml_alpha,
        dtype=cfg.dtype,
    )
    return result["p"]
