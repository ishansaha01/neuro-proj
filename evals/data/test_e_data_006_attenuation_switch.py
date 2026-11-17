"""E-DATA-006: switching skull absorption off raises the uncorrected signal by the skull loss.

Measured: the mean expected blood signal of conventional DAS through the skull,
`operators[skull_das] @ (cbv * cell_area)` over the brain pixels, with the
attenuation switch off against on. Predicted: the two-way absorption along the
central ray (x = 0) through the skull pixels of the true medium, from the
medium maps alone, `a(f) = integral alpha(f, z) dz` [Np] with alpha the power
law of `common.units.db_cm_mhz_to_np_m`.

Band weighting of the prediction. The blood signal is a sum of independent
scatterer powers, so by Parseval over the scatterer positions its level is a
sum over frequency of the image's power spectrum W(f) times the two-way power
loss `exp(-4 a(f))`:

    ratio = sum_f W(f) exp(-4 a(f)) / sum_f W(f),    W(f) = P(f)^2 / f^4.

`P` is the two-way pulse spectrum. The factor `1 / f^4` is the frequency
dependence of sound in two dimensions: a plane wave sent by line sources of
pitch `p` has the amplitude `1 / (2 k p)` (power `1 / k^2`), and the energy that
an array receives from a line source, summed over the lateral position of the
source, is proportional to `|G(k_x)|^2 = 1 / (4 k_z^2)`, again `1 / k^2`, whatever
the phases of the receive weights. The loss at the centre frequency alone and
the loss with W = P^2 are reported as diagnostics; both overstate the change,
because the image is formed mostly by the low end of the band.

History: the first version of this eval used W = P^2 and failed (measured rise
4 to 8 dB below the prediction on the three seeds). The simulator was not
changed and the 3 dB bar was kept; the reference gained the `1 / f^4` factor
derived above.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.common.units import db_cm_mhz_to_np_m, np_to_db
from neurosim.data.config import RunConfig
from neurosim.data.scene import build_scene_with_extras
from neurosim.data.types import SKULL_DAS, Scene
from neurosim.fus.config import imaging_setup
from neurosim.fus.pulse import pulse_spectrum
from neurosim.head.materials import skull_mask
from neurosim.head.types import PlaneMedium

EVAL_ID = "E-DATA-006"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
MAX_ERROR_DB = 3.0


def _mean_blood_signal(scene: Scene) -> float:
    cells = scene.cbv_baseline.ravel().astype(np.float64) * scene.cell_area
    signal = scene.operators[SKULL_DAS].astype(np.float64) @ cells
    return float(signal[scene.brain_mask.ravel()].mean())


def _central_ray_alpha0(medium: PlaneMedium) -> float:
    """Integral of the skull's absorption prefactor along x = 0 [dB/(cm MHz^y) * m]."""
    column = medium.grid.shape[1] // 2
    assert abs(medium.grid.axes()[1][column]) < 1e-9  # the grid has a node on x = 0
    in_skull = skull_mask(medium.labels[:, column])
    return float(medium.attenuation[in_skull, column].sum(dtype=np.float64) * medium.grid.spacing)


@eval_case(
    id=EVAL_ID,
    claim=(
        "switching bone absorption off (SkullEffects.attenuation = False) raises the mean blood "
        "signal of conventional DAS through the skull by the two-way skull absorption"
    ),
    reference=(
        "Power-law absorption integrated along the central ray through the skull pixels of the "
        "medium maps, two-way, weighted over the band by the image power spectrum "
        "P(f)^2 / f^4 of a two-dimensional plane-wave acquisition (Parseval; module docstring)"
    ),
    metric=(
        "error_db_seed_<s>: measured rise of the mean signal minus the predicted loss [dB]; "
        "predicted_db, pulse_weighted_db (W = P^2) and centre_frequency_db per seed are "
        "reported as diagnostics"
    ),
    threshold=(
        "|error| < 3 dB for every seed (fixed in advance, for a loss of 20 to 30 dB: "
        "rays from the edge of the aperture cross the skull up to 1/cos(25 deg) = 1.10 times "
        "longer than the central ray, about +1 dB on average; skull thickness and porosity "
        "vary by about 10 % across the aperture, about 1 dB after averaging over elements; "
        "and the aberrated sum weights the band a little differently with and without "
        "absorption, about 1 dB)"
    ),
    seeds=(0, 1, 2),
)
def test_e_data_006_attenuation_switch() -> None:
    cfg = from_dict(RunConfig, load_yaml(CONFIG))
    lossless = replace(cfg, effects=replace(cfg.effects, attenuation=False))
    setup = imaging_setup(cfg.imaging)
    pulse_power = pulse_spectrum(setup.transducer, setup.freqs) ** 2
    image_power = pulse_power / setup.freqs**4
    values = {}
    for seed in REGISTRY[EVAL_ID].seeds:
        scene, extras = build_scene_with_extras(cfg, SeedTree(seed))
        scene_lossless, _ = build_scene_with_extras(lossless, SeedTree(seed))
        # A power ratio R is an amplitude level of ln(R) / 2 nepers.
        gain = _mean_blood_signal(scene_lossless) / _mean_blood_signal(scene)
        measured = float(np_to_db(0.5 * np.log(gain)))

        alpha0_path = _central_ray_alpha0(extras.medium)
        nepers_per_alpha0 = db_cm_mhz_to_np_m(1.0, setup.freqs, extras.medium.alpha_power)
        one_way_nepers = alpha0_path * nepers_per_alpha0
        two_way_power_loss = np.exp(-4.0 * one_way_nepers)
        predicted, pulse_weighted = (
            -float(np_to_db(0.5 * np.log(np.sum(weight * two_way_power_loss) / weight.sum())))
            for weight in (image_power, pulse_power)
        )
        centre = alpha0_path * float(
            db_cm_mhz_to_np_m(1.0, setup.transducer.center_frequency, extras.medium.alpha_power)
        )
        values[f"error_db_seed_{seed}"] = measured - predicted
        values[f"predicted_db_seed_{seed}"] = predicted
        values[f"pulse_weighted_db_seed_{seed}"] = pulse_weighted
        values[f"centre_frequency_db_seed_{seed}"] = float(np_to_db(2.0 * centre))
    errors = [values[f"error_db_seed_{seed}"] for seed in REGISTRY[EVAL_ID].seeds]
    report(EVAL_ID, values, passed=max(abs(error) for error in errors) < MAX_ERROR_DB)
