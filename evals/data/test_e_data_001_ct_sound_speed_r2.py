"""E-DATA-001: CT-derived sound speed explains 23-53 % of the true skull sound-speed variance."""

from __future__ import annotations

import numpy as np

from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.head.materials import skull_mask

EVAL_ID = "E-DATA-001"
R2_BAND = (0.23, 0.53)


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Pearson correlation of two samples."""
    return float(np.corrcoef(a.astype(np.float64), b.astype(np.float64))[0, 1] ** 2)


@eval_case(
    id=EVAL_ID,
    claim="At the default settings (0.5 mm box head, 120 kVp CT, ramp kernel, nominal HU "
    "mapping) the CT-derived sound speed explains between 23 % and 53 % of the variance of "
    "the true sound speed over the skull pixels of the imaging plane: the CT is an "
    "imperfect prior of the measured size (README; ROADMAP Phase 3 and D-002)",
    reference="Webb et al. 2018, IEEE TUFFC 65(7):1111-1124, abstract: 'CT is able to "
    "account for 23%-53% of the variation in velocity in the human skull' (91 skull "
    "fragments, 80 CT scans)",
    metric="R^2 (squared Pearson correlation) between the CT-derived and the true sound "
    "speed over the true-skull pixels of the 0.25 mm plane grid, per seed [-]",
    threshold="Every seed inside [0.23, 0.53]: the published range itself, fixed by the "
    "README and ROADMAP before the CT prior was built; no margin is added because the "
    "range already spans Webb's energies and reconstruction kernels",
    seeds=(0, 1, 2),
)
def test_e_data_001_ct_sound_speed_r2(plane_prior):
    values = {}
    for seed in REGISTRY[EVAL_ID].seeds:
        prior = plane_prior(seed)
        in_skull = skull_mask(prior.truth.labels)
        true_speed = prior.truth.sound_speed[in_skull]
        ct_speed = prior.ct_medium.sound_speed[in_skull]
        values[f"r2_seed_{seed}"] = _r2(ct_speed, true_speed)
        # Diagnostics (not part of the verdict): what the true porosity of the same
        # pixels explains, the mean offset of the CT-derived speed, and the sample size.
        values[f"porosity_r2_seed_{seed}"] = _r2(prior.truth.porosity[in_skull], true_speed)
        values[f"mean_speed_error_m_s_seed_{seed}"] = float(
            ct_speed.astype(np.float64).mean() - true_speed.astype(np.float64).mean()
        )
        values[f"skull_pixels_seed_{seed}"] = float(np.count_nonzero(in_skull))
    r2 = [values[f"r2_seed_{seed}"] for seed in REGISTRY[EVAL_ID].seeds]
    report(EVAL_ID, values, passed=all(R2_BAND[0] <= value <= R2_BAND[1] for value in r2))
