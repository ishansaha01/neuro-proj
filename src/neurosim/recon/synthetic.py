"""Synthetic stand-in scenes whose skull is exactly a phase screen.

For developing and testing `recon` without the full pipeline, and for evals that
need a problem without model mismatch. Everything is built from the same fus and
data functions the pipeline uses; only the skull is replaced:

- true skull: a random smooth `PhaseScreen` on the homogeneous reference bank
  (`fus.screen.apply_phase_screen`);
- CT-derived skull: the true screen plus a smooth random error and an amplitude
  bias (the CT prior is imperfect, ROADMAP D-002 and D-014c);
- no skull: the homogeneous bank.

The CBV ground truth and the blood velocity (which sets the speckle statistics,
`data.scene.speckle_frame_counts`) are the head model's, sampled on the pixel grid
(`head.plane.sample_plane`), and the image series is `data.trials.simulate_trials`.
A synthetic scene is not a simulation of a skull: there is no refraction, no
position dependence of the aberration and no reverberation, so the phase-screen
model of `recon.forward` is exact for it. `Scene.meta["synthetic"]` is True.

Seed streams, besides those of the head and of the trials: the true screen comes
from `("skull_microstructure", 1)` and the CT error from `("hidden_soundspeed", 1)`
(the part of the skull the CT cannot see); the head itself uses these two
streams without a sub-index.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from neurosim.common.config import config_hash, field
from neurosim.common.constants import C_REF
from neurosim.common.io import provenance
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig, trial_fidelity
from neurosim.data.io import save_run
from neurosim.data.scene import (
    beamformer_banks,
    power_doppler_operator,
    probe_plane,
    speckle_frame_counts,
)
from neurosim.data.sweep import INDEX_NAME, scene_file
from neurosim.data.trials import simulate_trials
from neurosim.data.types import PhaseScreen, Scene, TrialSet
from neurosim.fus.beamform import make_beamformer
from neurosim.fus.config import imaging_setup
from neurosim.fus.noise import electronic_noise_variance
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.types import GreensBank
from neurosim.head.generate import generate_head
from neurosim.head.materials import brain_mask
from neurosim.head.plane import sample_plane

# Sub-index of the two skull streams used for the screens (module docstring).
_SCREEN_SUB = 1


@dataclass(frozen=True)
class ScreenSkullConfig:
    """Statistics of the synthetic true screen and of the CT-derived estimate of it."""

    delay_rms: float = field(
        80.0e-9,
        ge=0.0,
        doc="Rms of the true per-element delay about its mean [s]. 80 ns is 0.16 period at "
        "2 MHz; assumed, to verify against screens fitted to simulated skulls.",
    )
    correlation_elements: float = field(
        6.0,
        gt=0.0,
        doc="Correlation length of the screen across the array [elements] (standard "
        "deviation of the Gaussian smoothing kernel); assumed, to verify.",
    )
    mean_amplitude: float = field(
        0.4,
        gt=0.0,
        doc="Geometric-mean one-way amplitude of the true screen (0.4 is -8 dB); assumed.",
    )
    log_amplitude_rms: float = field(
        0.2, ge=0.0, doc="Rms of the true log amplitude about its mean [Np]; assumed."
    )
    ct_delay_error_rms: float = field(
        50.0e-9,
        ge=0.0,
        doc="Rms of the delay error of the CT-derived screen [s]; assumed, to verify "
        "(CT explains only part of the sound-speed variance, Webb et al. 2018).",
    )
    ct_log_amplitude_error_rms: float = field(
        0.15, ge=0.0, doc="Rms of the log-amplitude error of the CT-derived screen [Np]; assumed."
    )
    ct_log_amplitude_bias: float = field(
        -0.3,
        doc="Mean log-amplitude error of the CT-derived screen [Np]; negative because the CT "
        "prior overestimates bone absorption (ROADMAP D-014c). The size is assumed.",
    )


@dataclass(frozen=True)
class SyntheticConfig:
    """A synthetic phase-screen dataset: the run settings that apply and the screen statistics."""

    run: RunConfig = field(
        doc="`head`, `imaging`, `probe`, `source_refinement` and `trials` are used as in the "
        "pipeline; `backend`, `ct`, `ct_prior`, `effects` and the medium grid are not used."
    )
    skull: ScreenSkullConfig = field(default_factory=ScreenSkullConfig, doc="Screen statistics.")
    n_scenes: int = field(8, ge=1, doc="Number of scenes of `write_synthetic_dataset`.")


def _smooth_profile(rng: np.random.Generator, n: int, correlation: float, rms: float) -> np.ndarray:
    """Zero-mean smooth random profile across `n` elements with the given rms."""
    if rms == 0.0:
        return np.zeros(n)
    profile = gaussian_filter1d(rng.standard_normal(n), correlation, mode="reflect")
    profile -= profile.mean()
    return profile * (rms / np.sqrt(np.mean(profile**2)))


def draw_screens(
    skull: ScreenSkullConfig, n_elements: int, tree: SeedTree
) -> tuple[PhaseScreen, PhaseScreen]:
    """Draw the true screen and the CT-derived estimate of it.

    Returns `(screen_true, screen_ct)`, each with float64 arrays of shape (n_elements,).
    """
    rng_true = tree.rng("skull_microstructure", _SCREEN_SUB)
    rng_ct = tree.rng("hidden_soundspeed", _SCREEN_SUB)
    correlation = skull.correlation_elements
    delays = _smooth_profile(rng_true, n_elements, correlation, skull.delay_rms)
    log_amplitudes = np.log(skull.mean_amplitude) + _smooth_profile(
        rng_true, n_elements, correlation, skull.log_amplitude_rms
    )
    ct_delays = delays + _smooth_profile(rng_ct, n_elements, correlation, skull.ct_delay_error_rms)
    ct_log_amplitudes = (
        log_amplitudes
        + skull.ct_log_amplitude_bias
        + _smooth_profile(rng_ct, n_elements, correlation, skull.ct_log_amplitude_error_rms)
    )
    return (
        PhaseScreen(delays=delays, amplitudes=np.exp(log_amplitudes)),
        PhaseScreen(delays=ct_delays, amplitudes=np.exp(ct_log_amplitudes)),
    )


def synthetic_scene(cfg: SyntheticConfig, tree: SeedTree) -> Scene:
    """Build one synthetic scene (module docstring).

    cfg: run settings and screen statistics.
    tree: seed tree of the scene.
    """
    run = cfg.run
    setup = imaging_setup(run.imaging)
    head = generate_head(run.head, tree)
    plane = probe_plane(head, run)
    on_pixels = sample_plane(head, plane, setup.pixel_grid)
    del head

    homogeneous = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)
    screen_true, screen_ct = draw_screens(cfg.skull, setup.transducer.n_elements, tree)
    bank_skull = apply_phase_screen(homogeneous, screen_true.delays, screen_true.amplitudes)
    bank_ct = apply_phase_screen(homogeneous, screen_ct.delays, screen_ct.amplitudes)

    noise = run.imaging.noise
    noise_variance = (
        electronic_noise_variance(bank_skull, setup.sequence, noise)
        if noise.electronic_noise
        else 0.0
    )
    operators, noise_floor = {}, {}
    banks = beamformer_banks(bank_skull, homogeneous, bank_ct, bank_skull)
    for key, (bank, model_bank) in banks.items():
        operators[key] = power_doppler_operator(
            bank, model_bank, setup, run.imaging.beamforming, run.source_refinement
        )
        beamformer = make_beamformer(
            setup.pixel_grid, model_bank, setup.sequence, run.imaging.beamforming
        )
        noise_floor[key] = (noise_variance * beamformer.noise_gain).reshape(-1).astype(np.float32)
    effective_frames, noise_frames = speckle_frame_counts(
        run.imaging,
        setup.pixel_grid,
        operators,
        on_pixels.cbv_baseline,
        on_pixels.blood_velocity,
    )

    return Scene(
        imaging=run.imaging,
        plane=plane,
        pixel_grid=setup.pixel_grid,
        cell_area=setup.pixel_grid.spacing**2,
        cbv_baseline=on_pixels.cbv_baseline,
        activation=on_pixels.activation,
        brain_mask=brain_mask(on_pixels.labels),
        bank_skull=bank_skull,
        bank_clean=homogeneous,
        bank_ct=bank_ct,
        screen_ct=screen_ct,
        screen_true=screen_true,
        operators=operators,
        noise_floor=noise_floor,
        effective_frames=effective_frames,
        noise_frames=noise_frames,
        meta={
            "synthetic": True,
            "config_hash": config_hash(cfg),
            "root_seed": tree.root_seed,
            "source_refinement": run.source_refinement,
        },
    )


def synthetic_run(cfg: SyntheticConfig, tree: SeedTree) -> tuple[Scene, TrialSet]:
    """One synthetic scene and its image series (`data.trials.simulate_trials`)."""
    if trial_fidelity(cfg.run) != "statistical":
        raise ValueError("synthetic scenes produce statistical-fidelity trials only")
    scene = synthetic_scene(cfg, tree)
    return scene, simulate_trials(scene, cfg.run, tree)


def write_synthetic_dataset(
    cfg: SyntheticConfig, root_seed: int, out_dir: str | os.PathLike[str]
) -> None:
    """Write `cfg.n_scenes` synthetic runs in the layout of `data.sweep.run_sweep`.

    Scene `i` uses the seed subtree `SeedTree(root_seed).child(i)`. The index file
    carries ``"synthetic": true`` so the dataset cannot be mistaken for a simulated one;
    `data.sweep.load_dataset` reads it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = SeedTree(root_seed)
    scenes = []
    for index in range(cfg.n_scenes):
        scene, trials = synthetic_run(cfg, tree.child(index))
        scene = dataclasses.replace(scene, meta={**scene.meta, "sweep": {"index": index}})
        save_run(out_dir / scene_file(index), scene, trials, cfg.run, root_seed)
        scenes.append({"index": index, "file": scene_file(index), "seed_path": [index]})
    index_data = {
        **provenance(cfg, root_seed),
        "synthetic": True,
        "n_scenes": cfg.n_scenes,
        "scenes": scenes,
    }
    (out_dir / INDEX_NAME).write_text(
        json.dumps(index_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
