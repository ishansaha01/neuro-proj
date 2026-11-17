"""Contract types for runs and datasets (see docs/ARCHITECTURE.md, section 7).

A `Scene` is everything about one skull and one probe placement that does not
change over time. A `TrialSet` is the time-varying part: the CBV ground truth and
the measured power Doppler images. `recon` consumes exactly these two types.
Change them only together with docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurosim.common.grid import Grid2D, PlaneSpec
from neurosim.fus.config import ImagingConfig
from neurosim.fus.types import GreensBank

# Beamformer keys. Each names one way of turning channel data into an image.
SKULL_DAS = "skull_das"  # real skull, conventional DAS at C_REF (what a scanner gives)
SKULL_CT = "skull_ct"  # real skull, beamformed with the CT-derived skull model (baseline)
SKULL_TRUE = "skull_true"  # real skull, beamformed with the true skull model (baseline)
CLEAN_DAS = "clean_das"  # skull removed, conventional DAS (the paired clean target)
BEAMFORMERS: tuple[str, ...] = (SKULL_DAS, SKULL_CT, SKULL_TRUE, CLEAN_DAS)


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
class Scene:
    """One skull, one probe placement, one imaging setup.

    The CBV image lives on `pixel_grid` (shape (nz, nx), n_pix = nz * nx, C order).
    The expected blood power Doppler image for beamformer `k` is
    `operators[k] @ (cbv.ravel() * cell_area) + noise_floor[k]`.

    Attributes:
        imaging: the imaging configuration (array, sequence, frequencies, grids, noise).
        plane: imaging plane in the head frame.
        pixel_grid: grid of the CBV image and of every power Doppler image.
        cell_area: area of one pixel [m^2].
        cbv_baseline: resting CBV fraction on `pixel_grid`, float32, (nz, nx).
        activation: task response pattern in [0, 1] on `pixel_grid`, float32, (nz, nx).
        brain_mask: True where the pixel is brain tissue, bool, (nz, nx).
        bank_skull: Green's bank of the true medium (after any skull-effect switches).
        bank_clean: Green's bank of the same plane with the skull removed.
        bank_ct: Green's bank of the CT-derived medium (ray backend); the skull prior.
        screen_ct: phase screen fitted to `bank_ct`; initial skull estimate for blind recon.
        screen_true: phase screen fitted to `bank_skull`; for diagnostics only.
        operators: power Doppler operator H = |A|^2 per beamformer key in `BEAMFORMERS`,
            float32, (n_pix, n_pix).
        noise_floor: electronic-noise power Doppler floor per beamformer key, float32, (n_pix,).
        effective_frames: number of independent speckle realisations per image.
        meta: provenance and diagnostics (config hash, seed, medium grid, cost, CT settings,
            R^2 of CT-derived vs true skull sound speed).
    """

    imaging: ImagingConfig
    plane: PlaneSpec
    pixel_grid: Grid2D
    cell_area: float
    cbv_baseline: np.ndarray
    activation: np.ndarray
    brain_mask: np.ndarray
    bank_skull: GreensBank
    bank_clean: GreensBank
    bank_ct: GreensBank
    screen_ct: PhaseScreen
    screen_true: PhaseScreen
    operators: dict[str, np.ndarray]
    noise_floor: dict[str, np.ndarray]
    effective_frames: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialSet:
    """Time-varying data of one scene: one entry per power Doppler image.

    Attributes:
        time: acquisition time of each image [s], float64, (n,).
        amplitude: fractional CBV change a(t) in the activated region, float32, (n,).
        label: task state per image (0 = rest, 1 = task), int8, (n,).
        cbv: ground-truth CBV fraction per image on the scene's pixel grid, float32, (n, nz, nx).
        power_doppler: measured power Doppler per beamformer key in `BEAMFORMERS`,
            float32, (n, nz, nx).
        fidelity: "statistical" or "scatterer".
    """

    time: np.ndarray
    amplitude: np.ndarray
    label: np.ndarray
    cbv: np.ndarray
    power_doppler: dict[str, np.ndarray]
    fidelity: str
