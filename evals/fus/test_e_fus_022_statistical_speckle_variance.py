"""E-FUS-022: speckle variance of the scatterer tier against the statistical tier's gamma model.

The statistical tier draws every power Doppler pixel from a gamma distribution
with shape `effective_frames` (`acquire.statistical_power_doppler`), so its
relative speckle variance is `1 / effective_frames`, and by default
`effective_frames = sequence.n_frames` (`data.scene`: every frame of the
ensemble an independent speckle realisation). In the scatterer tier blood
moves from frame to frame, and frames stay correlated until the blood has
crossed a resolution cell. E-FUS-014 compares the two tiers only for freshly
redrawn scatterers (one frame per draw). This eval compares the variance of a
whole ensemble with moving blood.

Set-up (blood only, the simplest case that isolates the claim):

- No tissue, no micro-motion, no electronic noise, no static clutter and no SVD
  filter: the speckle term of the statistical tier models none of them.
- Array, angles and frame rate of `configs/fus_small.yaml` (48 elements, 500 Hz, the
  default frame rate) in water (homogeneous bank). The ROI is shrunk to a 6 mm box at
  30 mm depth and the frequency grid to the fewest bins the range-aliasing check
  (`fus.window`) accepts for it, which changes the cost and not the physics.
- Uniform CBV in a 5.5 mm box. Uniform blood velocity: in the head model the flow
  direction follows vessel tangents that vary over `CbvConfig.vessel_correlation`
  (4 mm), several resolution cells, so within a cell the flow is uniform. The
  direction is 45 degrees to the array, the in-plane projection of a vessel having no
  preferred direction. Speeds: `CbvConfig.speed_tissue` (1 mm/s) and
  `CbvConfig.speed_vessel` (20 mm/s), the two ends of the head model's range.
  Scatterers re-enter their paths as in production (`blood_transit_length`).
- Ensemble: 100 frames at 500 Hz, 0.2 s. The default ensemble is 200 frames, 0.4 s;
  100 is what the time budget allows with enough realisations (below). The tier's
  assumed value is taken for the sequence actually used: `effective_frames = 100`.
  What sets the number of independent frames is the ensemble duration times the
  blood speed, so the shortfall grows with the duration for slow blood; the
  diagnostic formula below extrapolates to the default.
- 5 independent draws of the blood per speed (`blood_scatterers` sub-streams
  (speed, draw)), about 30 speckle cells in the image each.

Estimator. Relative variance = variance over the draws / mean^2. With few draws
the per-pixel ratio is badly conditioned, so numerator and denominator are
pooled over pixels before dividing: unbiased variance (n - 1) on top, the
unbiased estimate of the squared mean (`mean^2 - variance / n`) below. The mean
is uniform up to the roll-off at the box edge. Standard error: jackknife over
the draws; expected about 25 % for the slow speed (exponential-like pixels,
`sqrt(8 / (5 * 30))`) and less for the fast one.

Scatterer density. With n scatterers of equal strength per effective PSF area,
one frame has relative variance `1 + 1/n` instead of 1, and the excess `1/n` is
a fluctuation of the local scatterer count, which does not average out over an
ensemble: it adds up to `1/n` to the relative variance of a power Doppler pixel
whatever the number of independent frames. Blood itself has millions of cells
per resolution cell, so this term is an artefact of the reference. The bound
`1 / (density * PSF area)` is computed from the PSF (`psf_operator`, PSF area =
`(sum |h|^2)^2 / sum |h|^4`) and checked against the single-frame relative
variance of the simulated frames. At 30 scatterers per mm^2 it is about 0.03,
which is larger than the assumed 1/100. The eval therefore gives the tier the
benefit of it: the upper bar is applied to the ratio after the bound has been
subtracted. It can resolve a true ratio above roughly 2 + 0.03 * 100 = 5, not
one between 2 and 5.

Diagnostic formula (no part of the verdict). For uniform flow the speckle at a
pixel is renewed when the scatterers have crossed the PSF: for a Gaussian PSF
of full width at half maximum w along the flow the intensity correlation is
`exp(-4 ln2 d^2 / w^2)` after a displacement d, whose integral is
`sqrt(pi / (4 ln 2)) w = 1.06 w`. Estimate: `1 + duration * speed / (1.06 w)`
independent frames, at most `n_frames`, with w from the diffraction widths
quoted in E-FUS-010 (lateral `1.207 lambda z / D` for the receive aperture
alone, axial `0.441 lambda / fractional bandwidth`), combined for the flow
direction as for an elliptical Gaussian.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.common.units import wavelength
from neurosim.fus.acquire import acquire_frames, statistical_power_doppler
from neurosim.fus.config import ImagingConfig, SequenceConfig, imaging_setup
from neurosim.fus.doppler import power_doppler
from neurosim.fus.operators import psf_operator
from neurosim.fus.scatterers import blood_scatterers
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.config import CbvConfig

EVAL_ID = "E-FUS-022"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
ROI_X, ROI_Z = (-3.0e-3, 3.0e-3), (27.0e-3, 33.0e-3)  # ROI nodes [m]
BLOOD_X, BLOOD_Z = (-2.5e-3, 2.5e-3), (27.5e-3, 32.5e-3)  # centres of the outer blood cells [m]
BLOOD_CELL = 0.5e-3  # CBV grid spacing [m]; the map is uniform, so it only sets the box
PIXEL_X, PIXEL_Z = (-2.4e-3, 2.4e-3), (27.6e-3, 32.4e-3)  # image, inside the blood box [m]
PIXEL_SPACING = 0.4e-3  # [m]
PSF_SAMPLING = 0.1e-3  # grid on which the PSF area is integrated [m]
N_FREQUENCIES = 30  # fewest bins `fus.window.check_time_window` accepts for this ROI
N_FRAMES = 100
N_REALISATIONS = 5
SCATTERER_DENSITY = 30.0e6  # blood scatterers per m^2 (30 per mm^2, as in E-FUS-014)
FLOW_ANGLE_DEG = 45.0  # flow direction, from the array axis towards depth
N_TIER_DRAWS = 200_000  # pixels drawn from the statistical tier to confirm its variance
RATIO_RANGE = (0.5, 2.0)
# Full widths at half maximum of the PSF, as quoted in E-FUS-010: lateral in units of
# lambda z / D (Goodman 2005, sec. 4.4.1, rectangular aperture), axial in units of
# lambda / fractional bandwidth (Gaussian pulse-echo spectrum).
LATERAL_FWHM_FACTOR = 1.207
AXIAL_FWHM_FACTOR = 0.441
# Integral of the intensity correlation of a Gaussian PSF, in units of its FWHM.
DECORRELATION_FRACTION = float(np.sqrt(np.pi / (4.0 * np.log(2.0))))


def _relative_variance(samples: np.ndarray) -> float:
    """Variance over independent draws / squared mean, pooled over all other axes.

    Args:
        samples: non-negative values, shape (n_draws, ...); axis 0 holds independent draws.

    Returns:
        sum of the unbiased variances / sum of the unbiased squared-mean estimates.
    """
    n_draws = samples.shape[0]
    variance = samples.var(axis=0, ddof=1)
    squared_mean = samples.mean(axis=0) ** 2 - variance / n_draws
    return float(variance.sum() / squared_mean.sum())


def _jackknife_error(samples: np.ndarray) -> float:
    """Delete-one jackknife standard error of `_relative_variance` over the draws (axis 0)."""
    n_draws = samples.shape[0]
    estimates = np.array(
        [_relative_variance(np.delete(samples, draw, axis=0)) for draw in range(n_draws)]
    )
    return float(np.sqrt((n_draws - 1) / n_draws * np.sum((estimates - estimates.mean()) ** 2)))


def _psf_width_along_flow(transducer: Transducer, depth: float, angle: float) -> float:
    """Diffraction estimate of the PSF full width at half maximum along a direction [m].

    Args:
        transducer: the array.
        depth: depth of the pixel [m].
        angle: direction, from the array axis towards depth [rad].
    """
    lam = float(wavelength(C_REF, transducer.center_frequency))
    lateral = LATERAL_FWHM_FACTOR * lam * depth / transducer.aperture
    axial = AXIAL_FWHM_FACTOR * lam / transducer.fractional_bandwidth
    return float(1.0 / np.hypot(np.cos(angle) / lateral, np.sin(angle) / axial))


@eval_case(
    id=EVAL_ID,
    claim=(
        "for blood moving at the head model's default speeds (1 and 20 mm/s) the relative "
        "variance of a scatterer-tier power Doppler pixel over independent blood realisations "
        "equals the 1 / effective_frames of the statistical tier's gamma speckle, with "
        "effective_frames = n_frames as the default configuration assigns"
    ),
    reference=(
        "The scatterer tier (explicit blood scatterers moving frame by frame, blood only): "
        "variance of its power Doppler image over independent draws of the blood. The "
        "statistical tier's value is exact: a gamma variable of shape k has variance / mean^2 "
        "= 1/k (mean of k independent speckle intensities, Goodman, Statistical Optics, 1985)"
    ),
    metric=(
        "max_ratio: largest over the two speeds of (scatterer-tier relative variance) / "
        "(1 / effective_frames); relative variance = variance over draws / mean^2, pooled over "
        "the pixels of the blood region. max_ratio_density_corrected: the same after "
        "subtracting the largest possible finite-scatterer-density term from the measured value"
    ),
    threshold=(
        "for both speeds, ratio >= 0.5 and density-corrected ratio <= 2 (fixed in advance: a "
        "factor 2 in variance is a factor 1.4 in the speckle standard deviation, the accuracy "
        "a noise model needs before a likelihood or an uncertainty built on it means much; the "
        "measurement has a standard error of about 25 % (5 draws x about 30 speckle cells), a "
        "quarter of the way from 1 to the upper bar; the reference's finite scatterer density "
        "adds at most 1 / (scatterers per PSF area), about 0.03, to the measured value, more "
        "than the assumed 1/100, so the upper bar is applied after subtracting that bound)"
    ),
    seeds=(0,),
)
def test_e_fus_022_statistical_speckle_variance() -> None:
    base = from_dict(ImagingConfig, load_yaml(CONFIG))
    default_sequence = SequenceConfig()
    cfg = replace(
        base,
        sequence=replace(
            base.sequence, frame_rate=default_sequence.frame_rate, n_frames=N_FRAMES
        ),
        frequencies=replace(base.frequencies, n_frequencies=N_FREQUENCIES),
        roi=replace(base.roi, x_range=ROI_X, z_range=ROI_Z),
        noise=replace(
            base.noise,
            electronic_noise=False,
            tissue_clutter=False,
            tissue_motion=False,
            static_clutter=False,
        ),
    )
    setup = imaging_setup(cfg)
    sequence, transducer = setup.sequence, setup.transducer
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    bank = GreensBank.homogeneous(transducer, setup.roi, setup.freqs, C_REF)
    blood_grid = Grid2D.from_extent(BLOOD_X, BLOOD_Z, BLOOD_CELL)
    pixel_grid = Grid2D.from_extent(PIXEL_X, PIXEL_Z, PIXEL_SPACING)
    cbv = np.full(blood_grid.shape, cfg.noise.cbv_reference)
    blood_area = blood_grid.shape[0] * blood_grid.shape[1] * blood_grid.spacing**2
    n_scatterers = round(SCATTERER_DENSITY * blood_area)
    flow_angle = np.deg2rad(FLOW_ANGLE_DEG)
    flow_direction = np.array([np.cos(flow_angle), np.sin(flow_angle)])  # (x, z)

    # The statistical tier: what the default configuration assigns to this sequence
    # (`data.scene`), and the variance its draws actually have.
    effective_frames = cfg.doppler.effective_frames or sequence.n_frames
    assumed = 1.0 / effective_frames
    tier_draws = statistical_power_doppler(
        np.ones((N_TIER_DRAWS, 1)),
        np.ones(1),
        0.0,
        effective_frames,
        tree.rng("speckle_statistical"),
    )
    tier_variance = float(tier_draws.var() / tier_draws.mean() ** 2)

    # Finite-density bound from the PSF of the central pixel.
    depth = 0.5 * (PIXEL_Z[0] + PIXEL_Z[1])
    psf_grid = Grid2D.from_extent(ROI_X, ROI_Z, PSF_SAMPLING)
    centre_pixel = Grid2D(shape=(1, 1), spacing=PIXEL_SPACING, origin=(depth, 0.0))
    psf = psf_operator(bank, centre_pixel, psf_grid, bank, sequence)[0]
    psf_power = np.abs(psf.astype(np.complex128)) ** 2
    psf_area = float(psf_power.sum() ** 2 / np.sum(psf_power**2) * psf_grid.spacing**2)
    density_bound = 1.0 / (SCATTERER_DENSITY * psf_area)

    duration = sequence.n_frames / sequence.frame_rate
    lam = float(wavelength(C_REF, transducer.center_frequency))
    decorrelation_length = DECORRELATION_FRACTION * _psf_width_along_flow(
        transducer, depth, flow_angle
    )
    default_duration = default_sequence.n_frames / default_sequence.frame_rate

    head_cbv = CbvConfig()
    speeds = {"slow": head_cbv.speed_tissue, "fast": head_cbv.speed_vessel}
    values: dict[str, float] = {}
    ratios, corrected_ratios, frame_powers, lines = [], [], [], []
    for speed_index, (name, speed) in enumerate(speeds.items()):
        velocity = np.broadcast_to(
            (speed * flow_direction)[:, None, None], (2, *blood_grid.shape)
        )
        intensity = np.empty((N_REALISATIONS, sequence.n_frames, *pixel_grid.shape))
        images = np.empty((N_REALISATIONS, *pixel_grid.shape))
        for draw in range(N_REALISATIONS):
            blood = blood_scatterers(
                cbv,
                velocity,
                blood_grid,
                n_scatterers,
                tree.rng("blood_scatterers", speed_index, draw),
                cfg.scatterers.blood_transit_length,
            )
            frames = acquire_frames(bank, bank, blood, None, sequence, pixel_grid, cfg, tree)
            intensity[draw] = np.abs(frames) ** 2
            images[draw] = power_doppler(frames)
        frame_powers.append(intensity)

        measured = _relative_variance(images)
        corrected = measured - density_bound
        estimate = min(sequence.n_frames, 1.0 + duration * speed / decorrelation_length)
        default_estimate = min(
            default_sequence.n_frames, 1.0 + default_duration * speed / decorrelation_length
        )
        ratios.append(measured / assumed)
        corrected_ratios.append(corrected / assumed)
        values.update(
            {
                f"{name}_speed_mm_s": 1.0e3 * speed,
                f"{name}_relative_variance": measured,
                f"{name}_relative_variance_standard_error": _jackknife_error(images),
                f"{name}_relative_variance_density_corrected": corrected,
                f"{name}_independent_frames": 1.0 / measured,
                f"{name}_ratio": ratios[-1],
                f"{name}_ratio_density_corrected": corrected_ratios[-1],
                f"{name}_travel_wavelengths": duration * speed / lam,
                f"{name}_estimated_independent_frames": estimate,
                f"{name}_estimated_independent_frames_default_ensemble": default_estimate,
            }
        )
        lines.append(
            f"{name} ({1.0e3 * speed:g} mm/s, {duration * speed / lam:.2f} wavelengths "
            f"travelled): relative variance {measured:.3f} "
            f"+- {values[f'{name}_relative_variance_standard_error']:.3f}, "
            f"{1.0 / measured:.2f} independent frames (formula {estimate:.2f}), "
            f"assumed {assumed:.4f}, ratio {ratios[-1]:.1f} "
            f"(density-corrected {corrected_ratios[-1]:.1f})"
        )

    single_frame = np.concatenate(frame_powers)
    values.update(
        {
            "max_ratio": max(ratios),
            "min_ratio": min(ratios),
            "max_ratio_density_corrected": max(corrected_ratios),
            "assumed_relative_variance": assumed,
            "statistical_tier_sampled_relative_variance": tier_variance,
            "n_frames": float(sequence.n_frames),
            "ensemble_duration_s": duration,
            "default_n_frames": float(default_sequence.n_frames),
            "default_ensemble_duration_s": default_duration,
            "single_frame_relative_variance": _relative_variance(single_frame),
            "single_frame_relative_variance_standard_error": _jackknife_error(single_frame),
            "finite_density_bound": density_bound,
            "scatterers_per_psf_area": SCATTERER_DENSITY * psf_area,
            "decorrelation_length_mm": 1.0e3 * decorrelation_length,
        }
    )
    note = (
        "; ".join(lines)
        + f"; ensemble {sequence.n_frames} frames = {duration:g} s (default "
        f"{default_sequence.n_frames} frames = {default_duration:g} s); single-frame relative "
        f"variance {values['single_frame_relative_variance']:.3f} "
        f"+- {values['single_frame_relative_variance_standard_error']:.3f} against "
        f"1 + {density_bound:.3f} expected for {SCATTERER_DENSITY * psf_area:.1f} scatterers "
        "per PSF area; formula: 1 + duration * speed / decorrelation length, decorrelation "
        f"length {1.0e3 * decorrelation_length:.2f} mm = {DECORRELATION_FRACTION:.2f} x the "
        "diffraction FWHM of the PSF along the flow"
    )
    low, high = RATIO_RANGE
    report(
        EVAL_ID,
        values,
        passed=min(ratios) >= low and max(corrected_ratios) <= high,
        note=note,
    )
