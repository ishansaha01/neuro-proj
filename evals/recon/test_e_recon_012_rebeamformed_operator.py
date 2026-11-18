"""E-RECON-012: the torch weights of a skull-model member reproduce `psf_operator`.

The reference path gives `fus.operators.psf_operator` the member's bank as the
model bank: `fus.beamform.make_beamformer` interpolates the bank, forms the
weights in numpy and contracts them with the true medium. The path under test
builds the weights in torch from the exported factors and the family's
fixed-path travel times (`recon.rebeamform.model_weights`) and contracts them
with the true medium's factors (`recon.forward`), as `recon.scanner` does.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.fus.beamform import make_beamformer
from neurosim.fus.config import imaging_setup
from neurosim.fus.operators import operator_factors, psf_operator
from neurosim.recon.config import SkullModelConfig
from neurosim.recon.family_scene import bent_ray_case, draw_member
from neurosim.recon.forward import psf_matrix, screen_model_from_factors
from neurosim.recon.rebeamform import model_weights, weight_model
from neurosim.recon.skull_model import delay_only, fermat_bank, parameter_tensors

EVAL_ID = "E-RECON-012"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
# The small scene's medium is 21 mm wide: 8 lateral modes instead of the default 16.
SKULL_MODEL = SkullModelConfig(n_lateral_modes=8)
LATERAL_SPEED_SIGMA = 40.0  # m/s per lateral mode of the drawn member
MAX_RELATIVE_ERROR = 1.0e-4


@eval_case(
    id=EVAL_ID,
    claim="The beamformer weights of a skull-model member built in torch (recon.rebeamform), "
    "contracted with the true medium's factors, equal the point-spread operator "
    "fus.operators.psf_operator(true bank, model_bank = the member's bank), and their noise "
    "gain equals that of fus.beamform.make_beamformer for that bank",
    reference="fus.operators.psf_operator and fus.beamform.make_beamformer (numpy) with the "
    "member's fixed-path bank, delays only (recon.skull_model.delay_only of fermat_bank, times "
    "a common amplitude of 0.7), as the model bank and the scene's true bank as the medium",
    metric="max over the operator (max |A_torch - A_psf| / max |A_psf|) and the noise gain "
    "(max relative difference) on the small bent-ray scene (48 elements, 56 frequencies, "
    "16 x 16 pixels and sources) for a member drawn from the prior (constants in their "
    "ranges, 40 m/s per lateral mode)",
    threshold="< 1e-4 (fixed in advance: both paths sum the same complex64 terms; the torch "
    "path rounds the reference Green's function and the phases to single precision before "
    "the unit phasors are formed, the numpy path after, which gives differences of order "
    "1e-6; a wrong interpolation, sign, normalisation or frequency dependence gives errors "
    "of order 0.1 to 1)",
    seeds=(3,),
)
def test_e_recon_012_rebeamformed_operator():
    seed = REGISTRY[EVAL_ID].seeds[0]
    scene, _, family = bent_ray_case(from_dict(RunConfig, load_yaml(CONFIG)), seed, SKULL_MODEL)
    setup = imaging_setup(scene.imaging)
    member = draw_member(family, SeedTree(seed), LATERAL_SPEED_SIGMA, 12)
    member = dataclasses.replace(member, common_amplitude=0.7)
    bank = delay_only(fermat_bank(family, member))
    bank = dataclasses.replace(bank, B=bank.B * np.float32(member.common_amplitude))
    arguments = (setup.sequence, scene.imaging.beamforming)

    reference = psf_operator(scene.bank_skull, scene.pixel_grid, scene.pixel_grid, bank, *arguments)
    beamformer = make_beamformer(scene.pixel_grid, bank, *arguments)

    weights = model_weights(
        weight_model(family, scene.pixel_grid, *arguments),
        *parameter_tensors(family, member),
        common_amplitude=member.common_amplitude,
    )
    factors = operator_factors(
        scene.bank_skull,
        scene.pixel_grid,
        scene.pixel_grid.points().reshape(-1, 2),
        bank,
        *arguments,
        beamformer,
    )
    medium = dataclasses.replace(
        screen_model_from_factors(factors, 1),
        tx_weights=weights.tx_phase,
        rx_weights=weights.rx_phase / weights.unit_response,
    )
    no_screen = torch.zeros(medium.n_elements, dtype=torch.float64)
    assembled = psf_matrix(medium, no_screen, no_screen).numpy()
    noise_gain = (weights.squared_weights / weights.unit_response.square()).numpy()

    operator_error = float(np.abs(assembled - reference).max() / np.abs(reference).max())
    expected_gain = beamformer.noise_gain.reshape(-1)
    gain_error = float(np.abs(noise_gain / expected_gain - 1.0).max())
    nominal = psf_operator(
        scene.bank_skull, scene.pixel_grid, scene.pixel_grid, delay_only(scene.bank_ct), *arguments
    )
    effect = float(np.abs(nominal - reference).max() / np.abs(reference).max())
    report(
        EVAL_ID,
        {
            "operator_relative_error": operator_error,
            "noise_gain_relative_error": gain_error,
            "operator_change_from_nominal_member": effect,
        },
        passed=max(operator_error, gain_error) < MAX_RELATIVE_ERROR,
        note="operator_change_from_nominal_member is the same comparison against the operator "
        "of the nominal CT bank (delays only): the size of the effect the member's parameters, "
        "and the common amplitude, have",
    )
