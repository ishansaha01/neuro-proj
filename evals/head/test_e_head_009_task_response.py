"""E-HEAD-009: the task time course is the block design convolved with the HRF."""

from __future__ import annotations

import numpy as np
from scipy import stats

from neurosim.common.evalkit import eval_case, report
from neurosim.head.cbv import hrf, task_labels, task_timecourse
from neurosim.head.config import TaskConfig

EVAL_ID = "E-HEAD-009"
TIME_STEP = 0.002  # s
DURATION = 160.0  # s
TOLERANCE = 2.0e-3


def reference_hrf(t: np.ndarray) -> np.ndarray:
    """Canonical double-gamma HRF from scipy's gamma densities (shapes 6 and 16, ratio 1/6)."""
    return stats.gamma.pdf(t, 6.0) - stats.gamma.pdf(t, 16.0) / 6.0


@eval_case(
    id=EVAL_ID,
    claim="a(t) equals the block design convolved with the canonical double-gamma HRF, "
    "scaled so that a long block peaks at the configured fractional change",
    reference="Independent numerical convolution (2 ms steps) of the 0/1 block design "
    "with the double-gamma function built from scipy.stats.gamma (SPM defaults: shapes 6 "
    "and 16, undershoot ratio 1/6), normalised to its own largest step response",
    metric="Largest |a(t) - reference| over 160 s, relative to the configured peak; and "
    "the largest |hrf - reference| after scaling the reference to unit peak",
    threshold="both < 2e-3: the rectangle-rule convolution error is of the order of the "
    "2 ms step over the 1-2 s rise time, i.e. about 1e-3. Fixed before the first run",
    seeds=(0,),
)
def test_e_head_009_task_response():
    task = TaskConfig(onsets=(10.0, 80.0), block_duration=40.0, peak_fractional_change=0.2)
    t = np.arange(0.0, DURATION, TIME_STEP)
    kernel = reference_hrf(t)
    step_response = np.cumsum(kernel) * TIME_STEP
    design = task_labels(task, t).astype(np.float64)
    reference = np.convolve(design, kernel)[: t.size] * TIME_STEP / step_response.max()
    timecourse_error = float(
        np.max(np.abs(task_timecourse(task, t) / task.peak_fractional_change - reference))
    )
    hrf_error = float(np.max(np.abs(hrf(t) - kernel / kernel.max())))
    report(
        EVAL_ID,
        {"timecourse_error": timecourse_error, "hrf_error": hrf_error},
        passed=timecourse_error < TOLERANCE and hrf_error < TOLERANCE,
    )
