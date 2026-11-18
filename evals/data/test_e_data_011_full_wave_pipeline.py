"""E-DATA-011: with the full-wave backend, the pipeline's ray model describes the same skull.

A toy scene (8 elements at 0.5 MHz on the 2 mm head, medium grid of a quarter
pitch) is built with the `kwave` backend. `bank_skull` is then a full-wave
bank, while the model of the `skull_true` beamformer is a ray bank through the
same medium (`data.scene`). Both are reduced to per-element delays with
`fus.screen.fit_phase_screen` and compared. The eval also exercises the array
placement the full-wave backend demands of `data.config.medium_grid`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.scene import build_scene_with_extras
from neurosim.fus.config import imaging_setup
from neurosim.fus.screen import fit_phase_screen

EVAL_ID = "E-DATA-011"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
CENTER_FREQUENCY = 0.5e6  # [Hz]
N_ELEMENTS = 8
CELLS_PER_PITCH = 4  # medium grid of 0.385 mm: 5 points per wavelength at the top of the band
MAX_DELAY_DIFFERENCE = 0.15e-6  # [s]


@eval_case(
    id=EVAL_ID,
    claim=(
        "built with the kwave backend, the scene's full-wave skull bank and the ray-model bank "
        "that serves as the true-skull beamforming model (straight rays when this eval was "
        "written; bent rays since ROADMAP D-021) give the same per-element skull delays"
    ),
    reference=(
        "Two independent solvers on one medium: k-Wave (full wave) against the ray model, "
        "both reduced by the phase slope over frequency (extends E-FUS-005 from a slab phantom "
        "to a generated head inside the pipeline)"
    ),
    metric=(
        "max_delay_difference_us: largest |delay of the full-wave screen - delay of the ray "
        "screen| over the elements [us]; mean_delay_us of each is reported as a diagnostic"
    ),
    threshold=(
        "< 0.15 us, 7.5 % of the 2 us period (set from the solvers' own error budgets: the "
        "full-wave plan bounds its phase error at 0.4 rad at the top frequency, 0.08 us, and "
        "straight rays are first-order accurate in the sound-speed contrast, a few percent of "
        "the 1.5 us skull delay. Stated plainly: a trial run of this toy scene, which gave "
        "0.08 us, had been seen when the bar was written)"
    ),
    seeds=(0,),
)
def test_e_data_011_full_wave_pipeline() -> None:
    base = from_dict(RunConfig, load_yaml(CONFIG))
    imaging = replace(
        base.imaging,
        transducer=replace(
            base.imaging.transducer, n_elements=N_ELEMENTS, center_frequency=CENTER_FREQUENCY
        ),
        frequencies=replace(base.imaging.frequencies, n_frequencies=20),
        roi=replace(base.imaging.roi, z_range=(0.015, 0.027), spacing=3.0e-3),
        pixels=replace(base.imaging.pixels, shape=(4, 4), spacing=3.0e-3),
    )
    # Coarse-grid solver settings, as in the unit tests of the full-wave backend.
    kwave = replace(
        base.backend.kwave,
        cfl=0.3,
        pml_size=10,
        pml_search=4,
        reverberation_tail=10.0e-6,
        reference_margin=8,
    )
    cfg = replace(
        base,
        imaging=imaging,
        backend=replace(base.backend, backend="kwave", kwave=kwave),
        medium_spacing=imaging_setup(imaging).transducer.pitch / CELLS_PER_PITCH,
        source_refinement=1,
    )
    scene, extras = build_scene_with_extras(cfg, SeedTree(REGISTRY[EVAL_ID].seeds[0]))
    assert scene.bank_skull.backend == scene.bank_clean.backend == "kwave"
    assert scene.bank_ct.backend == extras.model_true.backend == "bent_ray"
    assert scene.bank_skull.static_fmc is not None

    full_wave = scene.screen_true.delays
    ray = fit_phase_screen(extras.model_true).delays
    difference = float(np.abs(full_wave - ray).max())
    report(
        EVAL_ID,
        {
            "max_delay_difference_us": difference * 1.0e6,
            "mean_delay_us_full_wave": float(full_wave.mean() * 1.0e6),
            "mean_delay_us_ray": float(ray.mean() * 1.0e6),
        },
        passed=difference < MAX_DELAY_DIFFERENCE,
    )
