"""The time-varying part of a run: CBV ground truth and power Doppler image series.

Image `i` is taken at `t_i = i * trials.image_interval` along the head's task
design. Its CBV is `head.cbv.cbv_at(cbv_baseline, activation, a(t_i))` on the
scene's pixel grid, and one power Doppler image is produced per beamformer key
of `data.types.BEAMFORMERS`.

Fidelity "statistical": `fus.acquire.statistical_power_doppler` with the
scene's operator, noise floor and independent blood frames of the key
(`Scene.effective_frames`) and the scene's noise frames. The speckle of image `i` and key
number `k` comes from the stream `("speckle_statistical", i, k)`, so any image
can be regenerated alone. The three skull keys therefore get independent
speckle, although a scanner would beamform one set of channel data three ways.

Fidelity "scatterer": `fus.acquire.acquire_power_doppler`, once per key. Blood
scatterers of image `i` are drawn from `("blood_scatterers", i)` for that
image's CBV; the tissue scatterers (`tissue_scatterers`), the tissue motion and
the electronic noise of each frame (`us_electronic_noise`, absolute frame
index `round(t_i * frame_rate) + n`) are shared by the four keys, so the three
skull keys are the same channel data beamformed three ways and the clean run is
the same acquisition with the skull removed.

Response amplitude: block `b` of the task design responds with the gain
`exp(amplitude_jitter * n_b)`, `n_b` standard normal from the stream
`("cbv", 0)` (the stream `"cbv"` without sub-index is the head's vascular
texture).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig, trial_fidelity
from neurosim.data.scene import SceneExtras, beamformer_banks
from neurosim.data.types import BEAMFORMERS, Scene, TrialSet
from neurosim.fus.acquire import acquire_power_doppler, statistical_power_doppler
from neurosim.fus.config import imaging_setup
from neurosim.fus.scatterers import blood_scatterers, tissue_amplitude, tissue_scatterers
from neurosim.head.cbv import cbv_at, task_labels, task_timecourse

# Sub-index of the "cbv" stream that draws the block gains (module docstring).
_BLOCK_GAIN_SUB = 0


def image_times(cfg: RunConfig) -> np.ndarray:
    """Start time of every power Doppler image [s], float64, shape (n_images,)."""
    return np.arange(cfg.trials.n_images) * cfg.trials.image_interval


def task_amplitude(cfg: RunConfig, tree: SeedTree, times: np.ndarray) -> np.ndarray:
    """Fractional CBV change a(t) at the activation centre, with block-to-block jitter.

    The sum over task blocks of `head.cbv.task_timecourse` of the single block,
    each scaled by its gain (module docstring).

    Args:
        cfg: the run configuration (`head.task` and `trials.amplitude_jitter`).
        tree: seed tree of the run.
        times: times [s], shape (n,).

    Returns:
        a(t), dimensionless, float64, shape (n,).
    """
    task = cfg.head.task
    gains = np.ones(len(task.onsets))
    if cfg.trials.amplitude_jitter > 0.0:
        normal = tree.rng("cbv", _BLOCK_GAIN_SUB).standard_normal(len(task.onsets))
        gains = np.exp(cfg.trials.amplitude_jitter * normal)
    amplitude = np.zeros(np.shape(times))
    for onset, gain in zip(task.onsets, gains):
        amplitude += gain * task_timecourse(replace(task, onsets=(onset,)), times)
    return amplitude


def _statistical_images(scene: Scene, cbv: np.ndarray, tree: SeedTree) -> dict[str, np.ndarray]:
    shape = scene.pixel_grid.shape
    power_doppler = {}
    for key_number, key in enumerate(BEAMFORMERS):
        operator = scene.operators[key].astype(np.float64)
        power_doppler[key] = np.stack(
            [
                statistical_power_doppler(
                    operator,
                    image_cbv * scene.cell_area,
                    scene.noise_floor[key],
                    scene.effective_frames[key],
                    scene.noise_frames,
                    tree.rng("speckle_statistical", image, key_number),
                ).reshape(shape)
                for image, image_cbv in enumerate(cbv)
            ]
        ).astype(np.float32)
    return power_doppler


def _scatterer_images(
    scene: Scene, extras: SceneExtras, cbv: np.ndarray, times: np.ndarray, tree: SeedTree
) -> dict[str, np.ndarray]:
    imaging = scene.imaging
    setup = imaging_setup(imaging)
    banks = beamformer_banks(scene.bank_skull, scene.bank_clean, scene.bank_ct, extras.model_true)
    tissue = tissue_scatterers(
        setup.roi,
        imaging.scatterers.n_tissue,
        tissue_amplitude(setup.roi, imaging.scatterers.n_tissue, imaging.noise),
        tree.rng("tissue_scatterers"),
    )
    velocity = extras.blood_velocity.astype(np.float64)
    power_doppler = {key: np.empty(cbv.shape, dtype=np.float32) for key in BEAMFORMERS}
    for image, (image_cbv, time) in enumerate(zip(cbv, times)):
        blood = blood_scatterers(
            image_cbv,
            velocity,
            scene.pixel_grid,
            imaging.scatterers.n_blood,
            tree.rng("blood_scatterers", image),
            imaging.scatterers.blood_transit_length,
        )
        frame_offset = round(time * setup.sequence.frame_rate)
        for key, (bank, model_bank) in banks.items():
            power_doppler[key][image] = acquire_power_doppler(
                bank,
                model_bank,
                blood,
                tissue,
                setup.sequence,
                scene.pixel_grid,
                imaging,
                tree,
                frame_offset,
            )
    return power_doppler


def simulate_trials(
    scene: Scene, cfg: RunConfig, tree: SeedTree, extras: SceneExtras | None = None
) -> TrialSet:
    """Simulate the image series of a scene (module docstring).

    Args:
        scene: the static part of the run, from `data.scene.build_scene` with `cfg`.
        cfg: the run configuration (`head.task`, `trials`, and the fidelity).
        tree: seed tree of the run.
        extras: by-products of `data.scene.build_scene_with_extras`; required at
            scatterer fidelity (blood velocity and the `skull_true` model bank),
            unused at statistical fidelity.

    Returns:
        the `TrialSet` of `cfg.trials.n_images` images.
    """
    fidelity = trial_fidelity(cfg)
    times = image_times(cfg)
    amplitude = task_amplitude(cfg, tree, times)
    cbv = cbv_at(scene.cbv_baseline, scene.activation, amplitude)
    if fidelity == "statistical":
        power_doppler = _statistical_images(scene, cbv, tree)
    else:
        if extras is None:
            raise ValueError(
                "scatterer fidelity needs the `extras` of data.scene.build_scene_with_extras"
            )
        power_doppler = _scatterer_images(scene, extras, cbv, times, tree)
    return TrialSet(
        time=times,
        amplitude=amplitude.astype(np.float32),
        label=task_labels(cfg.head.task, times),
        cbv=cbv,
        power_doppler=power_doppler,
        fidelity=fidelity,
    )
