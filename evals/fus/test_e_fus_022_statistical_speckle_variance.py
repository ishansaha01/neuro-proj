"""E-FUS-022: speckle variance of the scatterer tier against the statistical tier's speckle model.

The statistical tier draws the blood part of every power Doppler pixel from a
gamma distribution whose shape is the number of independent blood frames
(`acquire.statistical_power_doppler`), so its relative speckle variance is
`1 / effective_frames`. The default configuration (`doppler.effective_frames`
None) derives that number per pixel from the blood velocity, the ensemble and
the resolution cell (`fus.speckle.speckle_frames`, the function the pipeline
calls in `data.scene`); until ROADMAP D-018 was fixed it was `sequence.n_frames`,
every frame of the ensemble an independent speckle realisation. In the scatterer
tier blood moves from frame to frame, and frames stay correlated until the blood
has crossed a resolution cell. E-FUS-014 compares the two tiers only for freshly
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
  (4 mm), several resolution cells, so within a cell the flow is uniform.
- Ensemble: 100 frames at 500 Hz, 0.2 s. The default ensemble is 200 frames, 0.4 s;
  100 is what the time budget allows with enough realisations (below). The tier's
  assumed value is taken for the sequence actually used. What sets the number of
  independent frames is the ensemble duration times the blood speed, so the model's
  value for the default ensemble is reported as well.
- Independent draws of the blood per case (`blood_scatterers` sub-streams
  (case, draw)), about 30 speckle cells in the image each.

Cases (all under the same bar):

- `slow` and `fast`, the two cases this eval has had from the start, unchanged:
  `CbvConfig.speed_tissue` (1 mm/s) and `CbvConfig.speed_vessel` (20 mm/s), the two
  ends of the head model's range, at 45 degrees to the array, the in-plane
  projection of a vessel having no preferred direction; scatterers re-enter their
  paths as in production (`blood_transit_length`); 5 draws.
- `medium_translation` (5 mm/s at 45 degrees), `fast_lateral_translation` and
  `fast_axial_translation` (20 mm/s along the array and along depth), added with
  the speckle model, which claims that the number of independent frames follows
  the anisotropy of the resolution cell (correlation lengths 1.46 mm lateral
  against 0.57 mm axial here). These isolate what the model describes, blood
  translating through the resolution cell: the scatterer paths are made far
  longer than the distance travelled (`TRANSLATION_PATH_LENGTH`), so no scatterer
  re-enters during the ensemble; the blood box is periodic and wider than the
  distance travelled. 3 draws each (the time budget): an expected standard error
  of 20 to 30 %, of which the jackknife estimate from 3 draws is itself a rough
  measure. For lateral flow the reference is biased a little towards more
  independent frames: scatterers vanish at the edge of the 5.5 mm box while still
  inside the 1.4 mm wide point-spread function of the outer pixels (measured once
  with 4 draws: 3.9 independent frames in the central 2.4 mm, 4.7 outside it,
  model 3.1).

What the added cases do not cover, and why. With production paths (1 mm nominal,
0.5 to 1.5 mm) a scatterer returns to the start of its own path, so its
contribution to a pixel repeats. For flow with a depth component the carrier
phase scrambles the repeats and the re-entry only adds decorrelation: from the
autocorrelation of the simulated point-spread function averaged over the paths,
the scatterer tier then has 1.1 to 1.5 times the independent frames of pure
translation (the two original cases). For lateral flow there is no carrier
along the path and a 1 mm loop never leaves a 1.4 mm wide resolution cell: the
scatterer tier's speckle stops renewing, about 1.3 independent frames at any
speed by the same calculation, where translating blood gives 3.1 at 20 mm/s
(measured once outside this eval, 4 draws: relative variance 1.06 +- 0.25
against the assumed 0.32, a ratio of 3.3 where the calculation gives 2.4). That
is a property of the re-entrant paths of `fus.scatterers`, not of the
statistical tier; it is not part of this eval's verdict and needs its own eval.

Estimator. Relative variance = variance over the draws / mean^2. With few draws
the per-pixel ratio is badly conditioned, so numerator and denominator are
pooled over pixels before dividing: unbiased variance (n - 1) on top, the
unbiased estimate of the squared mean (`mean^2 - variance / n`) below. The mean
is uniform up to the roll-off at the box edge. Standard error: jackknife over
the draws; expected about 25 % for the slow speed (exponential-like pixels,
`sqrt(8 / (5 * 30))`) and less for the fast one. The tier's assumed value is
pooled over the same pixels with the same weights, `sum_p mean_p^2 / N_p /
sum_p mean_p^2`, `mean = H @ cbv` and `N_p` from `fus.speckle.speckle_frames`,
and confirmed by drawing images from `statistical_power_doppler`.

Scatterer density. With n scatterers of equal strength per effective PSF area,
one frame has relative variance `1 + 1/n` instead of 1, and the excess `1/n` is
a fluctuation of the local scatterer count, which does not average out over an
ensemble: it adds up to `1/n` to the relative variance of a power Doppler pixel
whatever the number of independent frames. Blood itself has millions of cells
per resolution cell, so this term is an artefact of the reference. The bound
`1 / (density * PSF area)` is computed from the PSF (`psf_operator`, PSF area =
`(sum |h|^2)^2 / sum |h|^4`) and checked against the single-frame relative
variance of the simulated frames. At 30 scatterers per mm^2 it is about 0.03.
The eval gives the tier the benefit of it: the upper bar is applied to the
ratio after the bound has been subtracted. (With translating blood new
scatterers keep entering the resolution cell, so the count fluctuation does
average down and the bound overstates it; it is subtracted all the same.)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from neurosim.fus.operators import pd_operator, psf_operator
from neurosim.fus.resolution import axial_correlation_length, lateral_correlation_length
from neurosim.fus.scatterers import blood_scatterers
from neurosim.fus.speckle import speckle_frames
from neurosim.fus.types import GreensBank
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
N_REALISATIONS_ADDED = 3  # draws of each case added with the speckle model (time budget)
SCATTERER_DENSITY = 30.0e6  # blood scatterers per m^2 (30 per mm^2, as in E-FUS-014)
FLOW_ANGLE_DEG = 45.0  # flow direction, from the array axis towards depth
MEDIUM_SPEED = 5.0e-3  # between the two default speeds of the head model [m/s]
# Nominal scatterer path of the translation cases [m]: 250 times the 4 mm travelled at
# 20 mm/s in the ensemble, so that under 1 % of the scatterers re-enter their paths.
TRANSLATION_PATH_LENGTH = 1.0
N_TIER_DRAWS = 200_000  # pixels drawn from the statistical tier to confirm its variance
RATIO_RANGE = (0.5, 2.0)


@dataclass(frozen=True)
class FlowCase:
    """One uniform flow whose speckle variance is compared between the tiers.

    Attributes:
        name: prefix of the case's reported values.
        speed: blood speed [m/s].
        angle_deg: flow direction, from the array axis towards depth [degrees].
        transit_length: nominal scatterer path before re-entry [m]; None = the
            production value `scatterers.blood_transit_length`.
        n_realisations: independent draws of the blood.
    """

    name: str
    speed: float
    angle_deg: float
    transit_length: float | None
    n_realisations: int


def _cases() -> tuple[FlowCase, ...]:
    """The cases of the module docstring; the first two are the original ones, in their order."""
    head_cbv = CbvConfig()
    return (
        FlowCase("slow", head_cbv.speed_tissue, FLOW_ANGLE_DEG, None, N_REALISATIONS),
        FlowCase("fast", head_cbv.speed_vessel, FLOW_ANGLE_DEG, None, N_REALISATIONS),
        FlowCase(
            "medium_translation",
            MEDIUM_SPEED,
            FLOW_ANGLE_DEG,
            TRANSLATION_PATH_LENGTH,
            N_REALISATIONS_ADDED,
        ),
        FlowCase(
            "fast_lateral_translation",
            head_cbv.speed_vessel,
            0.0,
            TRANSLATION_PATH_LENGTH,
            N_REALISATIONS_ADDED,
        ),
        FlowCase(
            "fast_axial_translation",
            head_cbv.speed_vessel,
            90.0,
            TRANSLATION_PATH_LENGTH,
            N_REALISATIONS_ADDED,
        ),
    )


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


def _pooled_relative_variance(mean: np.ndarray, frames: np.ndarray) -> float:
    """Relative variance the statistical tier assigns, pooled as `_relative_variance` pools.

    Args:
        mean: expected blood power Doppler of every pixel, shape (n_pix,).
        frames: independent blood frames of every pixel, shape (n_pix,).

    Returns:
        sum of the variances `mean^2 / frames` / sum of the squared means.
    """
    return float(np.sum(mean**2 / frames) / np.sum(mean**2))


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

    # The statistical tier for this image: its operator and expected blood image. The
    # frame counts the default configuration assigns come from the pipeline's function,
    # `fus.speckle.speckle_frames`, for each case's flow (below).
    operator = pd_operator(psf_operator(bank, pixel_grid, blood_grid, bank, sequence))
    cbv_cells = cbv * blood_grid.spacing**2
    expected = operator.astype(np.float64) @ cbv_cells.reshape(-1)
    n_tier_images = N_TIER_DRAWS // expected.shape[0]
    default_cfg = replace(
        cfg, sequence=replace(cfg.sequence, n_frames=default_sequence.n_frames)
    )

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
    default_duration = default_sequence.n_frames / default_sequence.frame_rate

    values: dict[str, float] = {}
    ratios, corrected_ratios, frame_powers, lines = [], [], [], []
    for case_index, case in enumerate(_cases()):
        name, speed = case.name, case.speed
        flow_angle = np.deg2rad(case.angle_deg)
        flow_direction = np.array([np.cos(flow_angle), np.sin(flow_angle)])  # (x, z)
        velocity = np.broadcast_to(
            (speed * flow_direction)[:, None, None], (2, *blood_grid.shape)
        )
        transit_length = case.transit_length or cfg.scatterers.blood_transit_length

        # What the statistical tier assumes for this flow, and what its draws have.
        frames_model = speckle_frames(cfg, operator, cbv, velocity, blood_grid)
        assumed = _pooled_relative_variance(expected, frames_model.blood)
        tier_rng = tree.rng("speckle_statistical", case_index)
        tier_draws = np.stack(
            [
                statistical_power_doppler(
                    operator, cbv_cells, 0.0, frames_model.blood, frames_model.noise, tier_rng
                )
                for _ in range(n_tier_images)
            ]
        )
        default_frames = speckle_frames(default_cfg, operator, cbv, velocity, blood_grid)

        intensity = np.empty((case.n_realisations, sequence.n_frames, *pixel_grid.shape))
        images = np.empty((case.n_realisations, *pixel_grid.shape))
        for draw in range(case.n_realisations):
            blood = blood_scatterers(
                cbv,
                velocity,
                blood_grid,
                n_scatterers,
                tree.rng("blood_scatterers", case_index, draw),
                transit_length,
            )
            frames = acquire_frames(bank, bank, blood, None, sequence, pixel_grid, cfg, tree)
            intensity[draw] = np.abs(frames) ** 2
            images[draw] = power_doppler(frames)
        frame_powers.append(intensity)

        measured = _relative_variance(images)
        corrected = measured - density_bound
        ratios.append(measured / assumed)
        corrected_ratios.append(corrected / assumed)
        values.update(
            {
                f"{name}_speed_mm_s": 1.0e3 * speed,
                f"{name}_flow_angle_deg": case.angle_deg,
                f"{name}_scatterer_path_mm": 1.0e3 * transit_length,
                f"{name}_n_realisations": float(case.n_realisations),
                f"{name}_relative_variance": measured,
                f"{name}_relative_variance_standard_error": _jackknife_error(images),
                f"{name}_relative_variance_density_corrected": corrected,
                f"{name}_independent_frames": 1.0 / measured,
                f"{name}_assumed_relative_variance": assumed,
                f"{name}_statistical_tier_sampled_relative_variance": _relative_variance(
                    tier_draws
                ),
                f"{name}_ratio": ratios[-1],
                f"{name}_ratio_density_corrected": corrected_ratios[-1],
                f"{name}_travel_wavelengths": duration * speed / lam,
                f"{name}_model_independent_frames": 1.0 / assumed,
                f"{name}_model_independent_frames_default_ensemble": 1.0
                / _pooled_relative_variance(expected, default_frames.blood),
            }
        )
        lines.append(
            f"{name} ({1.0e3 * speed:g} mm/s at {case.angle_deg:g} deg, "
            f"{duration * speed / lam:.2f} wavelengths travelled, "
            f"{1.0e3 * transit_length:g} mm paths, {case.n_realisations} draws): "
            f"relative variance {measured:.3f} "
            f"+- {values[f'{name}_relative_variance_standard_error']:.3f}, "
            f"{1.0 / measured:.2f} independent frames (model {1.0 / assumed:.2f}), "
            f"assumed {assumed:.4f}, ratio {ratios[-1]:.2f} "
            f"(density-corrected {corrected_ratios[-1]:.2f})"
        )

    single_frame = np.concatenate(frame_powers)
    centre = np.array([[0.0, depth]])
    lateral_length = float(
        lateral_correlation_length(
            transducer, sequence, cfg.beamforming, centre, setup.freqs, C_REF
        )[0]
    )
    axial_length = axial_correlation_length(transducer, setup.freqs, C_REF)
    values.update(
        {
            "max_ratio": max(ratios),
            "min_ratio": min(ratios),
            "max_ratio_density_corrected": max(corrected_ratios),
            "n_frames": float(sequence.n_frames),
            "ensemble_duration_s": duration,
            "default_n_frames": float(default_sequence.n_frames),
            "default_ensemble_duration_s": default_duration,
            "single_frame_relative_variance": _relative_variance(single_frame),
            "single_frame_relative_variance_standard_error": _jackknife_error(single_frame),
            "finite_density_bound": density_bound,
            "scatterers_per_psf_area": SCATTERER_DENSITY * psf_area,
            "lateral_correlation_length_mm": 1.0e3 * lateral_length,
            "axial_correlation_length_mm": 1.0e3 * axial_length,
        }
    )
    note = (
        "; ".join(lines)
        + f"; ensemble {sequence.n_frames} frames = {duration:g} s (default "
        f"{default_sequence.n_frames} frames = {default_duration:g} s); single-frame relative "
        f"variance {values['single_frame_relative_variance']:.3f} "
        f"+- {values['single_frame_relative_variance_standard_error']:.3f} against "
        f"1 + {density_bound:.3f} expected for {SCATTERER_DENSITY * psf_area:.1f} scatterers "
        "per PSF area; assumed values from fus.speckle.speckle_frames (correlation lengths "
        f"of the resolution cell {1.0e3 * lateral_length:.2f} mm lateral, "
        f"{1.0e3 * axial_length:.2f} mm axial)"
    )
    low, high = RATIO_RANGE
    report(
        EVAL_ID,
        values,
        passed=min(ratios) >= low and max(corrected_ratios) <= high,
        note=note,
    )
