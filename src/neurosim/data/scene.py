"""Build the static part of a run: one skull, one probe placement, one imaging setup.

`build_scene` chains the finished packages (docs/ARCHITECTURE.md, section 7):

    head -> imaging plane -> fine plane medium -> skull-effect switches / skull removed
         -> Green's banks (true skull, no skull) -> CT prior medium -> its model bank
         -> beamformer per key of `data.types.BEAMFORMERS` -> power Doppler operator,
            noise floor and independent-frame counts per key -> phase screens.

Beamformer keys and their banks (`beamformer_banks`): the channel data of
`skull_das`, `skull_ct` and `skull_true` come from the true-skull bank and those
of `clean_das` from the no-skull bank. The model the beamformer assumes is the
homogeneous bank for `skull_das` and `clean_das` (conventional delay-and-sum at
C_REF), the CT-derived model bank for `skull_ct`, and a model bank through the
true medium for `skull_true` (`_model_backend`: the data's own backend for `ray`
and `bent_ray`, `bent_ray` for `kwave`). With the `kwave` backend the data then
contain reverberation and later arrivals that the model ignores: the "true-skull
correction" baseline stays a ray-model correction, as cheap as the CT one, and
is not the exact inverse of the simulated skull. With `ray` or `bent_ray` the
model equals the bank that made the data.

Normalisation (`fus.config`, `fus.noise`): the expected power Doppler image of
key `k` is `operators[k] @ (cbv * cell_area) + noise_floor[k]`, with
`noise_floor[k] = electronic noise variance * Beamformer.noise_gain` and the
same noise variance for every key. The speckle about that mean is described by
`effective_frames[k]` and `noise_frames` (`speckle_frame_counts`, `fus.speckle`).

Seed streams: those of `head.generate.generate_head`, `ct_photon_noise` and
(for a sampled HU mapping) `ct_hu_mapping`. Nothing here draws from the
ultrasound noise or speckle streams.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from neurosim.common.config import config_hash, to_dict
from neurosim.common.grid import Grid2D, PlaneSpec, sample_plane_map
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig, medium_grid, trial_fidelity
from neurosim.data.ct_prior import ct_plane_hu, ct_prior_medium, prior_hu_mapping
from neurosim.data.ct_prior import estimate_cost as estimate_ct_prior_cost
from neurosim.data.types import (
    BEAMFORMERS,
    CLEAN_DAS,
    SKULL_CT,
    SKULL_DAS,
    SKULL_TRUE,
    Scene,
)
from neurosim.fus.acquire import estimate_cost as estimate_acquisition_cost
from neurosim.fus.backends import BackendConfig, compute_bank
from neurosim.fus.backends import estimate_cost as estimate_bank_cost
from neurosim.fus.backends.phantoms import Material, slab_medium
from neurosim.fus.beamform import make_beamformer
from neurosim.fus.config import BeamformingConfig, ImagingConfig, ImagingSetup, imaging_setup
from neurosim.fus.noise import electronic_noise_variance
from neurosim.fus.operators import pd_operator, psf_operator
from neurosim.fus.screen import fit_phase_screen
from neurosim.fus.speckle import speckle_frames
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.ablations import apply_skull_effects, remove_skull
from neurosim.head.config import HeadConfig
from neurosim.head.generate import estimate_memory_bytes, estimate_runtime_seconds, generate_head
from neurosim.head.materials import brain_mask, power_law_prefactor, skull_mask
from neurosim.head.plane import default_plane, sample_plane
from neurosim.head.types import HeadVolume, PlaneMedium, Tissue

logger = logging.getLogger(__name__)

# Wall time per matrix element of one statistical power Doppler image (`H @ cbv` and the
# gamma draw), single-threaded on the 2-core development machine (2026-11-17), used only
# by `estimate_cost`: 0.64 ms measured for a 1024 x 1024 and 9.5 ms for a 4096 x 4096
# float64 operator.
_SECONDS_PER_OPERATOR_ELEMENT = 6.0e-10
# Resident memory of the interpreter with the simulator's libraries loaded, before any
# array of a run exists; measured on the development machine (2026-11-17): 0.29 GiB.
_BASELINE_BYTES = 300 * 2**20
_COMPLEX64_BYTES = 8
_FLOAT32_BYTES = 4


@dataclass(frozen=True)
class SceneExtras:
    """What `build_scene` computes besides the `Scene` contract.

    Needed to simulate trials at scatterer fidelity, and to save or plot the media.

    Attributes:
        medium: the true plane medium on the fine grid, skull-effect switches applied
            (the medium `Scene.bank_skull` was computed for).
        medium_ct: the CT-derived plane medium on the same grid.
        model_true: model bank through `medium` (`_model_backend`), the model of the
            `skull_true` beamformer (`Scene.bank_skull` itself unless the backend is `kwave`).
        blood_velocity: in-plane blood velocity on the scene's pixel grid [m/s],
            float32, shape (2, nz, nx), components (vx, vz).
    """

    medium: PlaneMedium
    medium_ct: PlaneMedium
    model_true: GreensBank
    blood_velocity: np.ndarray


def probe_plane(head: HeadVolume, cfg: RunConfig) -> PlaneSpec:
    """Imaging plane of the run: `head.plane.default_plane` moved by `cfg.probe`.

    The array centre is shifted by `lateral_shift` along the array axis, then
    the array is turned by `tilt_deg` about that centre within the plane.
    """
    plane = default_plane(head, cfg.head)
    u_lateral = np.asarray(plane.u_lateral)
    u_depth = np.asarray(plane.u_depth)
    tilt = np.deg2rad(cfg.probe.tilt_deg)
    origin = np.asarray(plane.origin) + cfg.probe.lateral_shift * u_lateral
    return PlaneSpec(
        origin=tuple(origin),
        u_lateral=tuple(np.cos(tilt) * u_lateral - np.sin(tilt) * u_depth),
        u_depth=tuple(np.cos(tilt) * u_depth + np.sin(tilt) * u_lateral),
    )


def _check_array_outside_head(medium: PlaneMedium, transducer: Transducer) -> None:
    """Raise if any element centre of the array (z = 0) lies inside the head."""
    x = transducer.element_x()
    face = np.stack((x, np.zeros_like(x)), axis=-1)
    if np.any(sample_plane_map(medium.labels, medium.grid, face, order=0) != Tissue.BACKGROUND):
        raise ValueError(
            "the array face touches the head: increase head.probe.standoff or reduce the "
            "probe tilt and shift"
        )


def skull_sound_speed_r2(truth: PlaneMedium, estimate: PlaneMedium) -> float:
    """R^2 (squared Pearson correlation) of an estimated against the true skull sound speed.

    Taken over the true-skull pixels of the common grid of the two media (the
    quantity of eval E-DATA-001, after Webb et al. 2018).
    """
    in_skull = skull_mask(truth.labels)
    true_speed = truth.sound_speed[in_skull].astype(np.float64)
    estimated_speed = estimate.sound_speed[in_skull].astype(np.float64)
    return float(np.corrcoef(estimated_speed, true_speed)[0, 1] ** 2)


def beamformer_banks(
    bank_skull: GreensBank, bank_clean: GreensBank, bank_ct: GreensBank, model_true: GreensBank
) -> dict[str, tuple[GreensBank, GreensBank]]:
    """`(medium bank, model bank)` of every beamformer key (module docstring)."""
    homogeneous = GreensBank.homogeneous(
        bank_skull.transducer, bank_skull.roi, bank_skull.freqs, bank_skull.c_ref
    )
    return {
        SKULL_DAS: (bank_skull, homogeneous),
        SKULL_CT: (bank_skull, bank_ct),
        SKULL_TRUE: (bank_skull, model_true),
        CLEAN_DAS: (bank_clean, homogeneous),
    }


def power_doppler_operator(
    bank: GreensBank,
    model_bank: GreensBank,
    setup: ImagingSetup,
    beamforming: BeamformingConfig,
    refinement: int,
) -> np.ndarray:
    """Power Doppler operator from CBV cells to pixels, both on the pixel grid.

    `H[p, s]` is `|A|^2` (`fus.operators`) of pixel `p` for a unit scatterer in
    cell `s`, averaged over `refinement` x `refinement` points spread evenly
    over the cell (the midpoint rule for blood filling the cell uniformly);
    `refinement = 1` samples the cell centre only.

    Returns:
        float32 array, shape (n_pix, n_pix).
    """
    grid = setup.pixel_grid
    fractions = (np.arange(refinement) + 0.5) / refinement - 0.5  # of a cell, about its centre
    operator = np.zeros((grid.shape[0] * grid.shape[1],) * 2, dtype=np.float32)
    for depth_fraction in fractions:
        for lateral_fraction in fractions:
            sources = Grid2D(
                shape=grid.shape,
                spacing=grid.spacing,
                origin=(
                    grid.origin[0] + depth_fraction * grid.spacing,
                    grid.origin[1] + lateral_fraction * grid.spacing,
                ),
            )
            operator += pd_operator(
                psf_operator(bank, grid, sources, model_bank, setup.sequence, beamforming)
            )
    operator /= np.float32(refinement**2)
    return operator


def speckle_frame_counts(
    imaging: ImagingConfig,
    pixel_grid: Grid2D,
    operators: dict[str, np.ndarray],
    cbv_baseline: np.ndarray,
    blood_velocity: np.ndarray,
) -> tuple[dict[str, np.ndarray], float]:
    """`Scene.effective_frames` and `Scene.noise_frames` for a set of operators.

    `fus.speckle.speckle_frames` once per beamformer key, with the baseline CBV
    and the blood velocity on the pixel grid (the source cells of the operators).

    Args:
        imaging: imaging configuration of the scene.
        pixel_grid: grid of the CBV cells and of the image.
        operators: power Doppler operator per beamformer key, (n_pix, n_pix) each.
        cbv_baseline: resting CBV fraction, shape (nz, nx).
        blood_velocity: in-plane blood velocity [m/s], shape (2, nz, nx), (vx, vz).

    Returns:
        `(effective_frames, noise_frames)`: independent blood frames per key, float32,
        shape (n_pix,) each, and the independent noise frames.
    """
    counts = {
        key: speckle_frames(imaging, operator, cbv_baseline, blood_velocity, pixel_grid)
        for key, operator in operators.items()
    }
    (noise_frames,) = {count.noise for count in counts.values()}
    return {key: count.blood for key, count in counts.items()}, noise_frames


def _model_backend(cfg: RunConfig) -> BackendConfig:
    """Backend of the correction models: the CT-derived bank and the true-skull model.

    A correction model must be cheap, so it is never the full-wave solver: with the
    `kwave` backend the models use `bent_ray` (the most faithful ray tier); with `ray`
    or `bent_ray` they use the same backend as the data.
    """
    name = "bent_ray" if cfg.backend.backend == "kwave" else cfg.backend.backend
    return dataclasses.replace(cfg.backend, backend=name)


def _grid_summary(grid: Grid2D) -> dict[str, Any]:
    return {"shape": list(grid.shape), "spacing": grid.spacing, "origin": list(grid.origin)}


def build_scene_with_extras(cfg: RunConfig, tree: SeedTree) -> tuple[Scene, SceneExtras]:
    """Build the scene of a run and the by-products the `Scene` contract does not carry.

    Args:
        cfg: the run configuration.
        tree: seed tree of the run.

    Raises:
        ValueError: if the fine medium grid leaves the head volume
            (`head.plane.sample_plane`), the array face touches the head, or a
            backend rejects the geometry.
    """
    setup = imaging_setup(cfg.imaging)
    grid = medium_grid(cfg)
    cost = estimate_cost(cfg)
    logger.info("building scene; estimated cost: %s", cost)

    head = generate_head(cfg.head, tree)
    plane = probe_plane(head, cfg)
    truth = sample_plane(head, plane, grid)
    _check_array_outside_head(truth, setup.transducer)
    on_pixels = sample_plane(head, plane, setup.pixel_grid)
    ct, medium_ct = ct_prior_medium(head, plane, grid, cfg.ct, cfg.ct_prior, tree)
    del head  # the largest object of the run; nothing below needs the 3D volume

    medium = apply_skull_effects(truth, cfg.effects)
    bank_arguments = (setup.transducer, setup.roi, setup.freqs)
    bank_skull = compute_bank(medium, *bank_arguments, cfg.backend)
    bank_clean = compute_bank(remove_skull(truth), *bank_arguments, cfg.backend)
    bank_ct = compute_bank(medium_ct, *bank_arguments, _model_backend(cfg))
    if cfg.backend.backend == "kwave":
        model_true = compute_bank(medium, *bank_arguments, _model_backend(cfg))
    else:
        model_true = bank_skull

    noise_variance = (
        electronic_noise_variance(bank_skull, setup.sequence, cfg.imaging.noise)
        if cfg.imaging.noise.electronic_noise
        else 0.0
    )
    operators, noise_floor = {}, {}
    banks = beamformer_banks(bank_skull, bank_clean, bank_ct, model_true)
    for key, (bank, model_bank) in banks.items():
        operators[key] = power_doppler_operator(
            bank, model_bank, setup, cfg.imaging.beamforming, cfg.source_refinement
        )
        beamformer = make_beamformer(
            setup.pixel_grid, model_bank, setup.sequence, cfg.imaging.beamforming
        )
        noise_floor[key] = (noise_variance * beamformer.noise_gain).reshape(-1).astype(np.float32)

    effective_frames, noise_frames = speckle_frame_counts(
        cfg.imaging,
        setup.pixel_grid,
        operators,
        on_pixels.cbv_baseline,
        on_pixels.blood_velocity,
    )

    mapping = prior_hu_mapping(cfg.ct, cfg.ct_prior, tree)
    meta = {
        "config_hash": config_hash(cfg),
        "root_seed": tree.root_seed,
        "backend": cfg.backend.backend,
        "skull_true_model": model_true.backend,
        "model_backend": to_dict(_model_backend(cfg)),
        "effects": to_dict(cfg.effects),
        "medium_grid": _grid_summary(grid),
        "cost": cost,
        "ct": {
            "kvp": cfg.ct.kvp,
            "mas": cfg.ct.mas,
            "kernel": cfg.ct.kernel,
            "photon_noise": cfg.ct.photon_noise,
            "slab_shape": list(ct.grid.shape),
            "slab_spacing": ct.grid.spacing,
            "mu_water": ct.meta["mu_water"],
            "mu_air": ct.meta["mu_air"],
            "hu_mapping": dataclasses.asdict(mapping),
            "alpha_power": cfg.ct.hu_mapping.alpha_power,
            "bone_threshold_hu": cfg.ct.hu_mapping.bone_threshold_hu,
        },
        "ct_sound_speed_r2": skull_sound_speed_r2(truth, medium_ct),
    }
    scene = Scene(
        imaging=cfg.imaging,
        plane=plane,
        pixel_grid=setup.pixel_grid,
        cell_area=setup.pixel_grid.spacing**2,
        cbv_baseline=on_pixels.cbv_baseline,
        activation=on_pixels.activation,
        brain_mask=brain_mask(on_pixels.labels),
        bank_skull=bank_skull,
        bank_clean=bank_clean,
        bank_ct=bank_ct,
        screen_ct=fit_phase_screen(bank_ct),
        screen_true=fit_phase_screen(bank_skull),
        operators=operators,
        noise_floor=noise_floor,
        effective_frames=effective_frames,
        noise_frames=noise_frames,
        meta=meta,
        ct_hu=ct_plane_hu(ct, grid, plane),
        medium_grid=grid,
    )
    extras = SceneExtras(
        medium=medium,
        medium_ct=medium_ct,
        model_true=model_true,
        blood_velocity=on_pixels.blood_velocity,
    )
    return scene, extras


def build_scene(cfg: RunConfig, tree: SeedTree) -> Scene:
    """Build the scene of a run (module docstring).

    Args:
        cfg: the run configuration.
        tree: seed tree of the run.
    """
    return build_scene_with_extras(cfg, tree)[0]


def _nominal_medium(head_cfg: HeadConfig, grid: Grid2D) -> PlaneMedium:
    """A layered stand-in for the plane medium, for cost estimates made before a head exists.

    Brain tissue with a flat bone slab of the median skull thickness under the
    median scalp. The slab has the upper bound of the skull sound speed, which
    sets the time step of the full-wave backend, and the largest bone absorption.
    """
    materials = head_cfg.materials
    y = materials.alpha_power
    grey, skull = materials.grey, materials.skull
    background = Material(
        sound_speed=grey.sound_speed,
        density=grey.density,
        attenuation=float(power_law_prefactor(grey.absorption, grey.absorption_frequency, y)),
    )
    bone = Material(
        sound_speed=head_cfg.microstructure.sound_speed_max,
        density=skull.rho_bone,
        attenuation=float(
            power_law_prefactor(skull.absorption_max, skull.absorption_frequency, y)
        ),
    )
    return slab_medium(
        grid,
        background,
        bone,
        z_top=head_cfg.probe.standoff + head_cfg.anatomy.scalp_thickness,
        thickness=head_cfg.anatomy.skull_thickness_vault,
        alpha_power=y,
    )


def estimate_cost(cfg: RunConfig) -> dict[str, Any]:
    """Aggregate cost estimate of one run, from the estimates of the packages.

    The bank estimates use a nominal layered medium on the run's fine grid
    (`_nominal_medium`), since the head does not exist yet; for the `kwave`
    backend this also checks the array placement before anything is simulated.

    Returns:
        dict with one entry per stage, each the estimate of the package that owns it:
        `head` (`memory_bytes`, `time_s`), `ct_prior` (`data.ct_prior.estimate_cost`),
        `bank` (one bank of the configured backend; `fus.backends.estimate_cost`),
        `bank_ray` (one ray bank), `operator` (one power Doppler operator at
        `source_refinement = 1`; `fus.acquire.estimate_cost`) and `trial_image` (one
        image for one beamformer at the trial fidelity); the counts `n_backend_banks`,
        `n_ray_banks`, `n_operator_passes` and `n_images`; and the totals `time_s`
        (single thread), `memory_bytes` (peak resident memory, interpreter included) and
        `bundle_bytes` (size of the saved run).
    """
    setup = imaging_setup(cfg.imaging)
    grid = medium_grid(cfg)
    z, x = grid.axes()
    nominal = _nominal_medium(cfg.head, grid)
    bank_arguments = (nominal, setup.transducer, setup.roi, setup.freqs)
    n_pix = setup.pixel_grid.shape[0] * setup.pixel_grid.shape[1]
    n_keys = len(BEAMFORMERS)
    fidelity = trial_fidelity(cfg)

    head = {
        "memory_bytes": estimate_memory_bytes(cfg.head),
        "time_s": estimate_runtime_seconds(cfg.head),
    }
    ct_prior = estimate_ct_prior_cost(
        cfg.head.grid.spacing, cfg.ct, cfg.ct_prior, (x[0], x[-1]), (z[0], z[-1])
    )
    bank = estimate_bank_cost(*bank_arguments, cfg.backend)
    bank_ray = estimate_bank_cost(*bank_arguments, _model_backend(cfg))
    operator = estimate_acquisition_cost(
        dataclasses.replace(cfg.imaging, fidelity="statistical"), n_pix
    )
    if fidelity == "statistical":
        trial_image = {
            "time_s": _SECONDS_PER_OPERATOR_ELEMENT * n_pix**2,
            "memory_bytes": 2 * _FLOAT32_BYTES * n_pix**2,
        }
    else:
        noise = cfg.imaging.noise
        moving_tissue = noise.tissue_clutter and noise.tissue_motion
        n_scatterers = cfg.imaging.scatterers.n_blood + (
            cfg.imaging.scatterers.n_tissue if moving_tissue else 0
        )
        trial_image = estimate_acquisition_cost(
            dataclasses.replace(cfg.imaging, fidelity="scatterer"), n_scatterers
        )

    n_backend_banks = 2
    n_ray_banks = 2 if cfg.backend.backend == "kwave" else 1
    n_operator_passes = n_keys * cfg.source_refinement**2
    n_images = cfg.trials.n_images
    n_roi = setup.roi.shape[0] * setup.roi.shape[1]
    bank_bytes = _COMPLEX64_BYTES * setup.freqs.size * setup.transducer.n_elements * n_roi
    operators_bytes = n_keys * _FLOAT32_BYTES * n_pix**2
    images_bytes = (n_keys + 1) * _FLOAT32_BYTES * n_images * n_pix

    time_s = (
        head["time_s"]
        + ct_prior["time_s"]
        + n_backend_banks * bank["run_time_s"]
        + n_ray_banks * bank_ray["run_time_s"]
        + n_operator_passes * operator["time_s"]
        + n_keys * n_images * trial_image["time_s"]
    )
    # The head is released before the banks are computed. Besides the computed banks the
    # homogeneous model bank of the two conventional beamformers is held.
    memory_with_head = head["memory_bytes"] + ct_prior["memory_bytes"]
    memory_after_head = (
        (n_backend_banks + n_ray_banks + 1) * bank_bytes
        + operators_bytes
        + max(bank["memory_bytes"], operator["memory_bytes"], trial_image["memory_bytes"])
    )
    return {
        "head": head,
        "ct_prior": ct_prior,
        "bank": bank,
        "bank_ray": bank_ray,
        "operator": operator,
        "trial_image": trial_image,
        "n_backend_banks": n_backend_banks,
        "n_ray_banks": n_ray_banks,
        "n_operator_passes": n_operator_passes,
        "n_images": n_images,
        "time_s": float(time_s),
        "memory_bytes": int(_BASELINE_BYTES + max(memory_with_head, memory_after_head)),
        "bundle_bytes": int(3 * bank_bytes + operators_bytes + images_bytes),
    }
