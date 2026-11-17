"""Full-wave backend: one k-Wave simulation per element, referenced to the reference medium.

Method. Each element (the medium-grid nodes of the z = 0 row under its width)
is driven with one short broadband pulse. The pressure is recorded at the ROI
nodes and at every element, and converted to spectra at exactly the requested
frequencies. The bank is the ratio of that spectrum to the spectrum of the same
simulation (same solver, source, grid spacing and time step) in the homogeneous
lossless reference medium:

    B[f, e, r] = P_medium(f, r; e) / P_ref(f, r; e)

The ratio cancels the pulse, the solver's source scaling and time staggering,
and the directivity of the discretised element. The reference medium is
translation invariant, so one reference simulation (a few if its recording
would not fit in memory) is shifted and mirrored to serve every element
(`fullwave_plan.ReferenceLayout`).

Source normalisation. Both simulations inject the same mass source: k-Wave
divides an additive pressure source by the local sound speed, so the drive of
every source point is multiplied by `c(point) / c_ref`. `G_medium` is therefore
the Green's function of `rho div(grad p / rho) + (w / c)^2 p = -delta`, which
in a homogeneous medium is `(i/4) H0(k r)` whatever the density.

Static echoes. `static_fmc[f, e_tx, e_rx]` is the medium recording minus the
reference recording, averaged over the nodes of the receiving element, divided
by the source factor `P_ref(f, r) / G_ref(f, r)` (the pressure spectrum the
solver produces per unit Green's function). The factor is taken as the median
over all reference points; its spread is stored in `meta`. Where the medium at
the array face is not the reference medium, the difference of the direct
element-to-element wave is part of `static_fmc`.

Accuracy limits are those of the solver: see `fullwave_plan.phase_error_bound`
for the phase error of the k-space scheme (it sets the choice of `cfl`),
`KWaveConfig.min_points_per_wavelength` for the grid and
`KWaveConfig.model_dispersion` for what absorption does to the phase speed.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from neurosim.common.config import to_dict
from neurosim.common.constants import C_REF, RHO_WATER
from neurosim.common.grid import Grid2D
from neurosim.fus.backends.config import KWaveConfig
from neurosim.fus.backends.fullwave_plan import (
    FullWavePlan,
    build_plan,
    estimate_cost,
    reference_shifts,
)
from neurosim.fus.backends.fullwave_solver import simulate
from neurosim.fus.backends.spectra import spectra_at
from neurosim.fus.reference import reference_greens
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.types import PlaneMedium

logger = logging.getLogger(__name__)

# Final part of a recording inspected for pressure that has not died away [fraction of its length].
_TAIL_FRACTION = 0.05


@dataclass(frozen=True)
class _Problem:
    """Inputs shared by every simulation of one bank; maps are float64 of shape (nz, nx)."""

    sound_speed: np.ndarray
    density: np.ndarray
    attenuation: np.ndarray
    alpha_power: float
    transducer: Transducer
    freqs: np.ndarray
    plan: FullWavePlan
    cfg: KWaveConfig


def _recording_positions(
    sensor_mask: np.ndarray, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    """Index in a recording of the grid points (rows, columns); sensors are ordered by (iz, ix)."""
    recorded = np.flatnonzero(sensor_mask)
    return np.searchsorted(recorded, np.ravel_multi_index((rows, columns), sensor_mask.shape))


def _tail_level_db(recording: np.ndarray) -> float:
    """Largest pressure in the final part of any time series, relative to that series' peak [dB]."""
    n_tail = max(1, int(_TAIL_FRACTION * recording.shape[1]))
    peak = np.abs(recording).max(axis=1)
    tail = np.abs(recording[:, -n_tail:]).max(axis=1)
    audible = peak > 0.0
    if not np.any(audible):
        return -np.inf
    with np.errstate(divide="ignore"):
        return float(20.0 * np.log10(np.max(tail[audible] / peak[audible])))


def _element_spectra(problem: _Problem, element: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate one transmitting element in the medium.

    Returns:
        pressure spectra at the ROI nodes, complex128, shape (nf, nz_roi, nx_roi);
        element-averaged pressure spectra at every element, shape (nf, n_elements);
        tail level of the recording [dB].
    """
    plan = problem.plan
    footprint = plan.element_columns[element]
    source_mask = np.zeros(plan.medium_shape, dtype=bool)
    source_mask[plan.array_row, footprint] = True
    # The same mass source as in the reference run (module docstring).
    drive = plan.pulse[None, :] * problem.sound_speed[plan.array_row, footprint][:, None] / C_REF

    sensor_mask = np.zeros(plan.medium_shape, dtype=bool)
    sensor_mask[np.ix_(plan.roi_rows, plan.roi_columns)] = True
    sensor_mask[plan.array_row, plan.element_columns.ravel()] = True

    recording = simulate(
        sound_speed=problem.sound_speed,
        density=problem.density,
        attenuation=problem.attenuation,
        alpha_power=problem.alpha_power,
        spacing=plan.spacing,
        dt=plan.dt,
        n_steps=plan.n_steps,
        source_mask=source_mask,
        source_signals=drive,
        sensor_mask=sensor_mask,
        pml=plan.pml,
        cfg=problem.cfg,
    )
    spectra = spectra_at(recording, plan.dt, problem.freqs)
    at_roi = _recording_positions(sensor_mask, plan.roi_rows[:, None], plan.roi_columns[None, :])
    at_elements = _recording_positions(sensor_mask, plan.array_row, plan.element_columns)
    return spectra[:, at_roi], spectra[:, at_elements].mean(axis=-1), _tail_level_db(recording)


def _all_element_spectra(problem: _Problem) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
    """Yield `_element_spectra` for every element in order, on `cfg.n_workers` processes."""
    n_elements = problem.transducer.n_elements
    run = functools.partial(_element_spectra, problem)
    workers = min(problem.cfg.n_workers, n_elements)
    if workers == 1:
        yield from map(run, range(n_elements))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(run, range(n_elements))


def _reference_spectra(problem: _Problem) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the reference element in the homogeneous lossless reference medium.

    Returns:
        pressure spectra on the ROI rows at the columns `first_column + roi_shifts`,
        complex128, shape (nf, nz_roi, n_roi_shifts); and on the array row at
        `first_column + array_shifts`, shape (nf, n_array_shifts).
    """
    plan = problem.plan
    layout = plan.reference
    n_footprint = plan.element_columns.shape[1]
    source_mask = np.zeros(layout.shape, dtype=bool)
    source_mask[plan.array_row, layout.first_column + np.arange(n_footprint)] = True
    # The pressure field of a mass source in a homogeneous medium does not depend on the
    # density, so any value serves; water is used.
    homogeneous = {
        "sound_speed": np.full(layout.shape, C_REF),
        "density": np.full(layout.shape, RHO_WATER),
        "attenuation": np.zeros(layout.shape),
    }

    roi_spectra = np.empty(
        (problem.freqs.size, plan.roi_rows.size, layout.roi_shifts.size), dtype=np.complex128
    )
    array_spectra = np.empty((problem.freqs.size, layout.array_shifts.size), dtype=np.complex128)
    roi_row_index = {int(row): index for index, row in enumerate(plan.roi_rows)}
    for group in layout.row_groups:
        sensor_mask = np.zeros(layout.shape, dtype=bool)
        columns_of_row = {
            row: layout.first_column
            + (layout.array_shifts if row == plan.array_row else layout.roi_shifts)
            for row in group
        }
        for row, columns in columns_of_row.items():
            sensor_mask[row, columns] = True
        recording = simulate(
            **homogeneous,
            alpha_power=problem.alpha_power,
            spacing=plan.spacing,
            dt=plan.dt,
            n_steps=plan.n_steps,
            source_mask=source_mask,
            source_signals=plan.pulse[None, :],
            sensor_mask=sensor_mask,
            pml=layout.pml,
            cfg=problem.cfg,
        )
        spectra = spectra_at(recording, plan.dt, problem.freqs)
        for row, columns in columns_of_row.items():
            at_row = spectra[:, _recording_positions(sensor_mask, row, columns)]
            if row == plan.array_row:
                array_spectra[:] = at_row
            else:
                roi_spectra[:, roi_row_index[row]] = at_row
    return roi_spectra, array_spectra


def _check_reference_level(reference: np.ndarray, freqs: np.ndarray, cfg: KWaveConfig) -> None:
    """Refuse to divide by a reference spectrum that is too weak anywhere.

    Args:
        reference: reference pressure spectra, shape (nf, nz_roi, n_roi_shifts); every
            entry is the denominator of at least one bank value.
        freqs: frequencies [Hz], shape (nf,).
        cfg: backend settings (`min_reference_level_db`).
    """
    magnitude = np.abs(reference).reshape(freqs.size, -1)
    level_db = 20.0 * np.log10(magnitude.min(axis=1) / magnitude.max(axis=1))
    worst = int(np.argmin(level_db))
    if level_db[worst] < cfg.min_reference_level_db:
        raise ValueError(
            f"the reference field at {freqs[worst]:.4g} Hz is {level_db[worst]:.1f} dB below its "
            f"maximum at some ROI point (limit kwave.min_reference_level_db = "
            f"{cfg.min_reference_level_db} dB): G_medium / G_ref is ill-conditioned there, "
            "typically in a directivity null of a wide element. Restrict the ROI or the "
            "frequency list, or lower the limit"
        )


def _source_factor(problem: _Problem, reference: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Pressure spectrum the solver produces per unit reference Green's function.

    Args:
        problem: the bank problem.
        reference: reference pressure spectra, shape (nf, nz_roi, n_roi_shifts).

    Returns:
        the factor `P_ref / G_ref` at each frequency (median over the reference points),
        complex128, shape (nf,); and the largest and the median relative deviation of the
        point-wise ratio from it [dimensionless].
    """
    plan = problem.plan
    n_footprint = plan.element_columns.shape[1]
    # One element at x = 0 made of point sources on the driven grid nodes.
    grid_element = dataclasses.replace(
        problem.transducer, n_elements=1, element_width=n_footprint * plan.spacing
    )
    x = (plan.reference.roi_shifts - 0.5 * (n_footprint - 1)) * plan.spacing
    z = (plan.roi_rows - plan.array_row) * plan.spacing
    z_map, x_map = np.meshgrid(z, x, indexing="ij")
    greens = reference_greens(
        grid_element, np.stack((x_map, z_map), axis=-1), problem.freqs, C_REF, n_sub=n_footprint
    )[:, 0]
    ratio = (reference / greens).reshape(problem.freqs.size, -1)
    factor = np.median(ratio.real, axis=1) + 1j * np.median(ratio.imag, axis=1)
    deviation = np.abs(ratio / factor[:, None] - 1.0)
    return factor, float(deviation.max()), float(np.median(deviation))


def fullwave_bank(
    medium: PlaneMedium,
    transducer: Transducer,
    roi: Grid2D,
    freqs: np.ndarray,
    cfg: KWaveConfig,
) -> GreensBank:
    """Green's bank from full-wave simulations (module docstring).

    Args:
        medium: plane medium; sound_speed [m/s], density [kg/m^3], attenuation
            [dB/(cm MHz^y)] of shape (nz, nx) on a grid whose z = 0 row is the array
            face, and the power-law exponent alpha_power. The element pitch must be
            a whole number of grid cells.
        transducer: the array.
        roi: grid the bank is stored on. Its points are snapped to the nearest medium
            grid nodes (`meta["roi_snap_distance_m"]`).
        freqs: frequencies [Hz], shape (nf,), ascending and uniformly spaced.
        cfg: backend settings.

    Returns:
        `GreensBank` with B complex64 of shape (nf, n_elements, nz_roi, nx_roi),
        `static_fmc` complex64 of shape (nf, n_elements, n_elements) and `c_ref = C_REF`.
    """
    started = time.perf_counter()
    plan = build_plan(medium, transducer, roi, freqs, cfg)
    freqs = np.asarray(freqs, dtype=np.float64)
    n_elements = transducer.n_elements
    absorbing = bool(np.any(medium.attenuation > 0.0))
    cost = estimate_cost(plan, freqs.size, n_elements, absorbing, cfg)

    problem = _Problem(
        sound_speed=medium.sound_speed.astype(np.float64),
        density=medium.density.astype(np.float64),
        attenuation=medium.attenuation.astype(np.float64),
        alpha_power=float(medium.alpha_power),
        transducer=transducer,
        freqs=freqs,
        plan=plan,
        cfg=cfg,
    )
    layout = plan.reference
    reference_roi, reference_array = _reference_spectra(problem)
    _check_reference_level(reference_roi, freqs, cfg)
    source_factor, factor_max_deviation, factor_median_deviation = _source_factor(
        problem, reference_roi
    )

    bank = np.empty((freqs.size, n_elements, *roi.shape), dtype=np.complex64)
    received = np.empty((freqs.size, n_elements, n_elements), dtype=np.complex128)
    tail_level_db = -np.inf
    for element, (at_roi, at_elements, tail) in enumerate(_all_element_spectra(problem)):
        shifts = reference_shifts(plan.roi_columns, plan.element_columns[element])
        bank[:, element] = at_roi / reference_roi[:, :, np.searchsorted(layout.roi_shifts, shifts)]
        received[:, element] = at_elements
        tail_level_db = max(tail_level_db, tail)
    if tail_level_db > cfg.max_tail_level_db:
        logger.warning(
            "the last %.0f %% of a recording still reaches %.1f dB relative to its peak "
            "(limit %.1f dB); increase kwave.reverberation_tail",
            100.0 * _TAIL_FRACTION,
            tail_level_db,
            cfg.max_tail_level_db,
        )

    # Element-averaged reference field on the array row, by element separation.
    n_footprint = plan.element_columns.shape[1]
    separation_shifts = (
        plan.pitch_cells * np.arange(n_elements)[:, None] + np.arange(n_footprint)[None, :]
    )
    reference_received = reference_array[
        :, np.searchsorted(layout.array_shifts, separation_shifts)
    ].mean(axis=-1)
    separation = np.abs(np.arange(n_elements)[:, None] - np.arange(n_elements)[None, :])
    static_fmc = (received - reference_received[:, separation]) / source_factor[:, None, None]

    return GreensBank(
        freqs=freqs,
        roi=roi,
        B=bank,
        c_ref=C_REF,
        transducer=transducer,
        backend="kwave",
        static_fmc=static_fmc.astype(np.complex64),
        meta={
            "config": to_dict(cfg),
            "cost_estimate": cost,
            "run_time_s": time.perf_counter() - started,
            "medium_grid_shape": plan.medium_shape,
            "reference_grid_shape": layout.shape,
            "pml_points": plan.pml,
            "time_step_s": plan.dt,
            "time_steps": plan.n_steps,
            "points_per_wavelength": plan.points_per_wavelength,
            "roi_snap_distance_m": plan.roi_snap_distance,
            "element_width_on_grid_m": n_footprint * plan.spacing,
            "source_factor_max_relative_deviation": factor_max_deviation,
            "source_factor_median_relative_deviation": factor_median_deviation,
            "tail_level_db": tail_level_db,
        },
    )
