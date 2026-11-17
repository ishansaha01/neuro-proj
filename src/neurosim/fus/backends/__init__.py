"""Green's-bank backends. Nothing outside this package imports an acoustic solver.

`compute_bank` turns a plane medium into a `GreensBank`,

    B[f, e, iz, ix] = G_medium(r, e, f) / G_ref(r, e, f),

with the exp(-i w t) convention: an extra one-way delay tau and an amplitude
factor a give `B = a exp(+i w tau)` (docs/ARCHITECTURE.md, section 5).

Normalisation of G_medium. `G_ref = (i/4) H0(k_ref r)` solves
`laplace(G) + k_ref^2 G = -delta`. In a medium with variable density the
backends use the same source term: `G_medium` solves

    rho div( grad(G) / rho ) + (w / c)^2 G = -delta(r - r_e)

(a mass source of the same strength in both media). Consequences: a homogeneous
medium (c1, rho1) has `B = H0(k1 r) / H0(k_ref r)` whatever its density, and
`B = 1` for the reference sound speed; reciprocity reads
`rho(r_e) G(r | r_e) = rho(r) G(r_e | r)`, so the same B serves transmit and
receive up to the constant density ratio between the array face and the ROI.

Backends: `analytic` (reference medium, B = 1), `ray` (`ray.py`: straight rays),
`bent_ray` (`bent_ray.py`: refracted first arrivals from the eikonal equation),
`kwave` (`fullwave.py`; the solver is imported only when this backend is used).
"""

from __future__ import annotations

import logging

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.fus.backends.analytic import analytic_bank
from neurosim.fus.backends.bent_ray import (
    bent_ray_bank,
    bent_ray_travel_times,
    estimate_bent_ray_cost,
)
from neurosim.fus.backends.config import BackendConfig, BentRayConfig, KWaveConfig, RayConfig
from neurosim.fus.backends.ray import estimate_ray_cost, ray_bank, ray_travel_times
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.types import PlaneMedium

__all__ = [
    "BackendConfig",
    "BentRayConfig",
    "KWaveConfig",
    "RayConfig",
    "bent_ray_travel_times",
    "compute_bank",
    "estimate_cost",
    "ray_travel_times",
]

logger = logging.getLogger(__name__)

# Bytes of one stored bank value (complex64).
_BANK_ITEMSIZE = 8


def estimate_cost(
    medium: PlaneMedium,
    transducer: Transducer,
    roi: Grid2D,
    freqs: np.ndarray,
    cfg: BackendConfig,
) -> dict[str, float | int]:
    """Estimate the cost of `compute_bank` without running it.

    Args: as for `compute_bank`.

    Returns:
        dict with at least `grid_points` (grid points of one simulation, absorbing
        layer included; the medium grid for `ray` and `bent_ray`), `time_steps`, `n_simulations`,
        `memory_bytes` (peak estimate) and `run_time_s` (wall-clock estimate). The
        `kwave` backend adds `reference_grid_points` and `phase_error_bound_rad`.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    bank_bytes = _BANK_ITEMSIZE * freqs.size * transducer.n_elements * roi.shape[0] * roi.shape[1]
    if cfg.backend == "analytic":
        return {
            "grid_points": 0,
            "time_steps": 0,
            "n_simulations": 0,
            "memory_bytes": bank_bytes,
            "run_time_s": 0.0,
        }
    if cfg.backend == "ray":
        return estimate_ray_cost(medium, transducer, roi, freqs, cfg.ray)
    if cfg.backend == "bent_ray":
        return estimate_bent_ray_cost(medium, transducer, roi, freqs, cfg.bent_ray)
    from neurosim.fus.backends import fullwave_plan

    plan = fullwave_plan.build_plan(medium, transducer, roi, freqs, cfg.kwave)
    absorbing = bool(np.any(medium.attenuation > 0.0))
    return fullwave_plan.estimate_cost(
        plan, freqs.size, transducer.n_elements, absorbing, cfg.kwave
    )


def compute_bank(
    medium: PlaneMedium,
    transducer: Transducer,
    roi: Grid2D,
    freqs: np.ndarray,
    cfg: BackendConfig,
) -> GreensBank:
    """Compute the Green's bank of a plane medium with the configured backend.

    Args:
        medium: plane medium; sound_speed [m/s], density [kg/m^3] and attenuation
            (alpha0 [dB/(cm MHz^y)], exponent `alpha_power`) on a fine grid whose
            z = 0 row is the array face, arrays of shape (nz, nx).
        transducer: the array (elements at z = 0).
        roi: coarse grid the bank is stored on; inside the medium grid, z > 0.
        freqs: frequencies [Hz], shape (nf,), ascending and uniformly spaced.
        cfg: backend choice and settings.

    Returns:
        `GreensBank` with B complex64 of shape (nf, n_elements, nz_roi, nx_roi).
        `static_fmc` (complex64, shape (nf, n_elements, n_elements)) is set by the
        `kwave` backend only.
    """
    logger.info(
        "computing Green's bank with backend %r; estimated cost: %s",
        cfg.backend,
        estimate_cost(medium, transducer, roi, freqs, cfg),
    )
    if cfg.backend == "analytic":
        return analytic_bank(medium, transducer, roi, freqs)
    if cfg.backend == "ray":
        return ray_bank(medium, transducer, roi, freqs, cfg.ray)
    if cfg.backend == "bent_ray":
        return bent_ray_bank(medium, transducer, roi, freqs, cfg.bent_ray)
    from neurosim.fus.backends import fullwave

    return fullwave.fullwave_bank(medium, transducer, roi, freqs, cfg.kwave)
