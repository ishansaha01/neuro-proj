"""Typed configuration of the reconstruction and evaluation stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from neurosim.common.config import field
from neurosim.ct.config import HuMappingConfig
from neurosim.recon.synthetic import SyntheticConfig


@dataclass(frozen=True)
class DpsConfig:
    """Sampler and guidance settings of diffusion posterior sampling (`recon.dps`)."""

    n_steps: int = field(
        30,
        ge=1,
        doc="Denoiser evaluations per sample. Chung et al. (2023) use all 1000 ancestral "
        "steps; a few tens is a run-time choice whose bias is described in recon.dps.",
    )
    eta: float = field(
        1.0,
        ge=0.0,
        le=1.0,
        doc="Noise scale of the reverse step: 1 = ancestral (as in Chung et al. 2023), "
        "0 = deterministic DDIM.",
    )
    guidance_scale: float = field(
        1.0,
        ge=0.0,
        doc="Multiplier of the likelihood score; 1 is the weight the DPS approximation "
        "prescribes (recon.dps), 0 switches the data off.",
    )
    max_step_ratio: float = field(
        1.0,
        gt=0.0,
        doc="Trust region: the guidance correction of an image is at most this multiple of "
        "the expected length of the unguided step. Tuning choice; assumed, to verify.",
    )
    n_samples: int = field(
        4, ge=1, doc="Posterior samples drawn per trial (the calibration metric needs >= 2)."
    )


@dataclass(frozen=True)
class DeconvolutionConfig:
    """Classical regularised deconvolution without a learned prior (`recon.baselines`)."""

    n_iterations: int = field(
        10,
        ge=1,
        doc="Multiplicative updates from a flat start. Stopping early is the regulariser, as "
        "is usual for Richardson-Lucy. Tuning choice: 10 gave the highest CBV correlation "
        "on a synthetic development scene (5 to 1000 tried); assumed, to verify on real scenes.",
    )


@dataclass(frozen=True)
class CalibrationConfig:
    """Blind estimation of the skull phase screen from a scene's trial set (`recon.calibrate`)."""

    sampler: DpsConfig = field(
        default_factory=lambda: DpsConfig(n_steps=30, n_samples=4),
        doc="Sampler of the image iterates. `n_samples` is the number of image chains that "
        "share one screen estimate within a candidate run.",
    )
    n_candidates: int = field(
        2,
        ge=1,
        doc="Independent blind runs; the one with the best final data fit is kept "
        "(best-of-N, after Raghu et al. 2025). Small because every run costs about "
        "n_updates * lbfgs_iterations operator gradients.",
    )
    n_updates: int = field(
        3,
        ge=1,
        doc="Reverse steps at which the screen is updated, evenly spaced from the first step "
        "to `update_until`; the image iterate is updated at every step.",
    )
    update_until: float = field(
        0.4,
        ge=0.0,
        le=1.0,
        doc="Fraction of the reverse process after which the screen is frozen. On synthetic "
        "development scenes updates made later, when the image iterate has adapted to the "
        "current screen, increased the delay error again; tuning choice, to verify.",
    )
    delay_fit_shape: float = field(
        3.0,
        gt=0.0,
        doc="Upper limit of the gamma shape assumed in the delay updates. The denoised image "
        "the delays are fitted to is itself uncertain, which the likelihood of the "
        "time-averaged image (per-pixel shape times n_trials, tens to hundreds) ignores; a low "
        "shape stands in for that uncertainty and lets the Gaussian prior hold the delays. "
        "Tuning choice: on seven synthetic scenes the ratio of calibrated to starting delay "
        "error had median 0.90 and maximum 4.1 without a limit and median 0.91 and maximum "
        "1.4 with 3. The amplitude update is not limited: with the limit it stopped short "
        "on weak-signal scenes of the demo dataset. Assumed, to verify.",
    )
    lbfgs_iterations: int = field(
        8, ge=1, doc="L-BFGS iterations of one screen update; run-time and tuning choice."
    )
    n_pixels: int = field(
        128,
        ge=1,
        doc="Measurement pixels in the objective of the screen updates, a random subset "
        "fixed per scene (all pixels if the image has fewer); the image iterate is always "
        "guided by all pixels. Run-time choice: an operator gradient costs in proportion.",
    )
    source_refinement: int = field(
        1,
        ge=1,
        doc="Sub-cell source points per pixel side of the forward model used during "
        "calibration. Lower than the scene's value is an approximation made for run time "
        "(cost grows with its square); the final operator H(phi_hat) uses "
        "ReconConfig.source_refinement.",
    )
    delay_prior_sigma: float = field(
        100.0e-9,
        gt=0.0,
        doc="Standard deviation of the Gaussian prior of each element's delay about the CT "
        "screen [s]; assumed, to verify against fitted screens of simulated skulls.",
    )
    common_log_amplitude_prior_sigma: float = field(
        1.5,
        gt=0.0,
        doc="Standard deviation of the prior of the log amplitude common to all elements "
        "[Np]. Wide because the CT prior's absorption is about twice the truth in dB "
        "(ROADMAP D-014c); assumed, to verify.",
    )
    coarse_modes: int = field(
        4,
        ge=1,
        doc="Delay modes (smoothest first) in the second screen update; the first update "
        "moves the common amplitude only, and the number of delay modes then grows "
        "geometrically to `max_delay_modes` at the last update (coarse-to-fine guidance, "
        "after Raghu et al. 2025).",
    )
    max_delay_modes: int = field(
        4,
        ge=1,
        doc="Largest number of delay modes estimated. On synthetic development scenes "
        "rougher modes (6 and 8 were tried) followed the image error, not the screen; "
        "tuning choice, to verify. "
        "Of the amplitudes only the common factor is estimated: per-element amplitudes "
        "absorbed image error on the development scene and stay at the CT values.",
    )
    initial_spread: float = field(
        0.0,
        ge=0.0,
        doc="Candidates after the first start from the CT screen plus a draw from the prior "
        "(on the estimated modes) scaled by this factor; 0 = all start at the CT screen and "
        "differ by their noise only.",
    )
    gauge: Literal["mean", "mean_tilt"] = field(
        "mean_tilt",
        doc="Delay directions the data cannot fix and that are therefore kept at the CT "
        "values: the common delay ('mean'), or the common delay and the linear tilt across "
        "the aperture ('mean_tilt'). See recon.calibrate.",
    )

    def __post_init__(self) -> None:
        if self.n_updates > self.sampler.n_steps:
            raise ValueError("n_updates cannot exceed sampler.n_steps")
        if self.coarse_modes > self.max_delay_modes:
            raise ValueError("coarse_modes cannot exceed max_delay_modes")


@dataclass(frozen=True)
class SkullModelConfig:
    """The family of CT-derived skull models and its prior (`recon.skull_model`)."""

    hu_knot_spacing: float = field(
        50.0,
        gt=0.0,
        doc="Largest spacing of the knots of the CT-number axis of the path-length "
        "histograms [HU]. The slowness of the HU map is interpolated linearly between "
        "knots; at 50 HU the error is below 1e-4 of the slowness (under 1 ns through a "
        "skull). Numerical choice.",
    )
    lateral_knot_spacing: float = field(
        2.0e-3,
        gt=0.0,
        doc="Largest spacing of the lateral knots of the bone path-length histogram [m]; "
        "the lateral modes are linear between knots. Below the 3 mm correlation length "
        "of the head model's hidden sound-speed component (head.config); numerical choice.",
    )
    n_lateral_modes: int = field(
        16,
        ge=1,
        doc="Cosine modes of extra slowness across the plane inside the CT bone, smoothest "
        "first (half-periods from the plane width down to width / n). Modelling choice: "
        "how much of the skull's CT-invisible sound speed they can represent is measured "
        "by the experiment (recon.medium_experiment).",
    )
    lateral_sound_speed_sigma: float = field(
        100.0,
        gt=0.0,
        doc="Prior standard deviation of the sound-speed change a single lateral mode makes "
        "at `lateral_reference_speed` [m/s]. Webb et al. 2018 (IEEE TUFFC 65:1111) report a "
        "mean 95 % confidence interval of +-198 m/s on skull velocity predicted from HU, "
        "about 100 m/s standard deviation; how that splits over lateral modes is assumed, "
        "to verify.",
    )
    lateral_reference_speed: float = field(
        2500.0,
        gt=0.0,
        doc="Sound speed at which `lateral_sound_speed_sigma` is converted to slowness, "
        "sigma_s = sigma_c / c^2 [m/s]: a mid-range skull value between the end points of "
        "the HU map (1500 and 2900 m/s); assumed, to verify.",
    )
    mapping_prior: HuMappingConfig = field(
        default_factory=HuMappingConfig,
        doc="Ranges of the HU-map constants (uniform prior): `c_bone_range`, "
        "`c_water_range` and `hu_bone_relative_range` of the CT package.",
    )


@dataclass(frozen=True)
class FocusConfig:
    """Calibration of the skull model by re-beamforming (`recon.focus`)."""

    n_pixels: int = field(
        96,
        ge=1,
        doc="Calibration pixels: a random subset of the brain pixels, fixed per scene (all "
        "of them if the brain has fewer). Run-time choice: one criterion gradient costs "
        "about 10 ms per pixel at 128 elements, 136 frequencies and 1024 blood cells.",
    )
    pixels_per_pass: int = field(
        32,
        ge=1,
        doc="Calibration pixels whose criterion terms are differentiated together; the "
        "gradient is accumulated over passes. Memory choice (a pass keeps about 8 MB per "
        "pixel at 136 frequencies and 1024 blood cells); the result does not depend on it "
        "beyond rounding.",
    )
    source_refinement: int = field(
        1,
        ge=1,
        doc="Blood source points per pixel side in the emulated criterion "
        "(recon.scanner); 1 is an approximation made for run time, the scored images use "
        "MediumReconConfig.source_refinement.",
    )
    criterion_power: float = field(
        1.0,
        gt=0.0,
        doc="Exponent g of the criterion (1 / g) log mean_p r_p^g of the pixel signal-to-"
        "floor ratios r_p: 1 is the mean ratio (the speckle-brightness criterion of Nock, "
        "Trahey & Smith 1989 on power Doppler), 2 the image sharpness of Muller & Buffington "
        "1974. Choice explained in recon.focus.",
    )
    mode_stages: tuple[int, ...] = field(
        (0, 2, 4, 8, 16),
        ge=0,
        doc="Number of lateral modes optimised in each stage, coarse to fine; the HU-map "
        "constants are optimised in every stage. The last entry must not exceed "
        "SkullModelConfig.n_lateral_modes. Tuning choice; assumed, to verify.",
    )
    iterations_per_stage: int = field(
        8, ge=1, doc="L-BFGS iterations of one stage; run-time and tuning choice."
    )
    prior_weight: float = field(
        0.01,
        ge=0.0,
        doc="Weight of the Gaussian log prior of the lateral coefficients against the "
        "criterion (a log of a mean ratio, not a log likelihood, so the two have no common "
        "scale). 0.01: one prior standard deviation on one mode costs 0.005, far less than "
        "the criterion gains of tenths that focusing brings, so the prior only holds the "
        "directions the criterion does not see. Tuning choice; assumed, to verify.",
    )

    def __post_init__(self) -> None:
        if list(self.mode_stages) != sorted(self.mode_stages) or not self.mode_stages:
            raise ValueError("mode_stages must be a non-empty ascending tuple")


@dataclass(frozen=True)
class MediumReconConfig:
    """The medium-space skull model, its calibration and the condition-mean reconstruction."""

    skull_model: SkullModelConfig = field(
        default_factory=SkullModelConfig, doc="The family of CT-derived skull models."
    )
    focus: FocusConfig = field(
        default_factory=FocusConfig, doc="Calibration of the family by re-beamforming."
    )
    dps: DpsConfig = field(
        default_factory=lambda: DpsConfig(n_steps=25, n_samples=4),
        doc="Posterior sampling of the two condition means of a scene (rest and task).",
    )
    fold_samples: int = field(
        1,
        ge=1,
        doc="Posterior samples per condition mean of a training fold, for the matched-"
        "filter template of the DPS rows; run-time choice.",
    )
    reference_cbv: Optional[float] = field(
        None,
        gt=0.0,
        doc="Blood volume fraction assumed for the brain on average when the common "
        "amplitude of a skull model is matched (recon.medium_recon). None: the scene's "
        "`imaging.noise.cbv_reference`, the nominal blood volume the noise level is "
        "specified against (0.03). A prior assumption on the CBV scale.",
    )
    dice_fraction: float = field(
        0.5,
        gt=0.0,
        lt=1.0,
        doc="A pixel of an estimated activation map counts as activated above this "
        "fraction of the map's maximum inside the brain (half maximum).",
    )


@dataclass(frozen=True)
class MetricsConfig:
    """Settings of the evaluation metrics (`recon.metrics`)."""

    n_folds: int = field(5, ge=2, doc="Cross-validation folds of the decoder.")
    n_components: int = field(
        10,
        ge=1,
        doc="Principal components of the PCA + LDA decoder (Norman et al. 2021 keep the "
        "components explaining 95 % of the variance; a fixed number is used here because "
        "a scene has only a few tens of trials); assumed, to verify.",
    )
    t_threshold: float = field(
        3.0,
        doc="Welch t statistic above which a pixel counts as activated (about p < 0.003 one-"
        "sided for 40 trials, uncorrected); assumed.",
    )
    activation_threshold: float = field(
        0.5, gt=0.0, lt=1.0, doc="True activation pattern above which a pixel is activated."
    )
    reference_threshold: float = field(
        0.01,
        gt=0.0,
        lt=1.0,
        doc="True activation pattern below which a brain pixel is a reference pixel of the "
        "contrast-to-noise ratio.",
    )
    coverage_levels: tuple[float, ...] = field(
        (0.5, 0.8, 0.9), gt=0.0, lt=1.0, doc="Nominal levels of the posterior intervals."
    )


@dataclass(frozen=True)
class ReconConfig:
    """Everything `recon.experiment` needs besides the dataset, the prior and the seed."""

    source_refinement: int = field(
        2,
        ge=1,
        doc="Sub-cell source points per pixel side of the phase-screen operator H(phi); must "
        "equal data.config.RunConfig.source_refinement of the dataset.",
    )
    dps: DpsConfig = field(default_factory=DpsConfig, doc="Per-trial posterior sampling.")
    calibration: CalibrationConfig = field(
        default_factory=CalibrationConfig, doc="Per-scene screen calibration."
    )
    deconvolution: DeconvolutionConfig = field(
        default_factory=DeconvolutionConfig, doc="Classical baseline."
    )
    metrics: MetricsConfig = field(default_factory=MetricsConfig, doc="Evaluation metrics.")
    medium: MediumReconConfig = field(
        default_factory=MediumReconConfig,
        doc="The second design: medium-space skull model calibrated by re-beamforming, and "
        "reconstruction of condition means (recon.medium_experiment).",
    )
    num_threads: int = field(
        1,
        ge=1,
        doc="Torch intra-op threads; results are bit-reproducible for a fixed seed and "
        "thread count (recon.seeding.configure_torch).",
    )
    synthetic: Optional[SyntheticConfig] = field(
        None,
        doc="Settings of the synthetic phase-screen stand-in dataset "
        "(recon.synthetic.write_synthetic_dataset); unused when a real dataset is given.",
    )
