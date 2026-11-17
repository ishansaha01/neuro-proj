"""E-FUS-018: electronic noise reaches the image with the power the propagation formula gives.

White channel noise of variance v per (frequency, angle, element) sample is
multiplied by the beamformer weights and summed, so the image noise power is
`v * sum |weights|^2` (variance of a sum of independent terms). That sum is
`Beamformer.noise_gain`; it is the noise floor of the statistical tier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.fus.beamform import beamform, make_beamformer
from neurosim.fus.config import BeamformingConfig, ImagingConfig, imaging_setup
from neurosim.fus.noise import electronic_noise, electronic_noise_variance
from neurosim.fus.types import GreensBank

EVAL_ID = "E-FUS-018"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
N_FRAMES = 400
MAX_MEAN_ERROR = 0.03


@eval_case(
    id=EVAL_ID,
    claim=(
        "with electronic noise only, the mean image power equals the channel noise variance "
        "times the sum of the squared beamformer weights"
    ),
    reference=(
        "Variance of a weighted sum of independent zero-mean variables: "
        "E|sum_i w_i n_i|^2 = v * sum_i |w_i|^2"
    ),
    metric=(
        "mean_error: sum over pixels of the mean simulated noise power / sum of the predicted "
        "floor, minus 1; pixel_rms: rms over pixels of simulated / predicted - 1, in units of "
        "1/sqrt(number of frames)"
    ),
    threshold=(
        "|mean_error| < 0.03 and pixel_rms < 1.5 (fixed in advance: with 400 frames a pixel's "
        "mean power has a standard error of 5 %, so pixel_rms should be 1; the image holds "
        "some tens of independent noise cells, so the image mean has a standard error below "
        "1 %; an f-number of 1.5 with a Hann window makes the weights differ from pixel to pixel)"
    ),
    seeds=(0,),
)
def test_e_fus_018_noise_floor() -> None:
    cfg = from_dict(ImagingConfig, load_yaml(CONFIG))
    setup = imaging_setup(cfg)
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    bank = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)
    beamformer = make_beamformer(
        setup.pixel_grid, bank, setup.sequence, BeamformingConfig(f_number=1.5, window="hann")
    )
    variance = electronic_noise_variance(bank, setup.sequence, cfg.noise)
    shape = (setup.freqs.shape[0], len(setup.sequence.angles_deg), setup.transducer.n_elements)

    power = np.zeros(setup.pixel_grid.shape)
    for frame in range(N_FRAMES):
        power += np.abs(beamform(beamformer, electronic_noise(tree, frame, shape, variance))) ** 2
    power /= N_FRAMES
    predicted = variance * beamformer.noise_gain

    values = {
        "mean_error": float(power.sum() / predicted.sum() - 1.0),
        "pixel_rms": float(np.sqrt(np.mean((power / predicted - 1.0) ** 2) * N_FRAMES)),
    }
    report(
        EVAL_ID,
        values,
        passed=abs(values["mean_error"]) < MAX_MEAN_ERROR and values["pixel_rms"] < 1.5,
    )
