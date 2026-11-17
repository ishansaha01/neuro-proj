"""E-RECON-005: the torch assembly of A(phi) equals `fus.operators.psf_operator` on a screen bank.

The reference path never sees the screen as a separate factor: the screen is
multiplied into the bank (`fus.screen.apply_phase_screen`), the bank is
interpolated to the source points and `psf_operator` contracts it with the
beamformer weights in numpy. The path under test takes the factors of the
*homogeneous* bank (`fus.operators.operator_factors`) and inserts the screen as an
elementwise factor between the matrix products, in torch (`recon.forward`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.fus.config import ImagingConfig, imaging_setup
from neurosim.fus.operators import operator_factors, psf_operator
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.types import GreensBank
from neurosim.recon.forward import psf_matrix, screen_model_from_factors, screen_tensors
from neurosim.recon.synthetic import ScreenSkullConfig, draw_screens

EVAL_ID = "E-RECON-005"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
MAX_RELATIVE_ERROR = 1.0e-5


@eval_case(
    id=EVAL_ID,
    claim="The point-spread operator A(phi) assembled in torch from the exported factors of "
    "the homogeneous bank, with the phase screen as an elementwise factor, equals "
    "fus.operators.psf_operator applied to the bank with the screen multiplied in",
    reference="fus.operators.psf_operator on fus.screen.apply_phase_screen(homogeneous bank): "
    "the numpy path of E-FUS-013, in which the screen is part of the bank",
    metric="max |A_torch - A_psf| / max |A_psf| over all pixels and sources (48 elements, 56 "
    "frequencies, 32 x 32 pixels, 13 x 13 sources; screen of 80 ns rms delay and 0.2 Np rms "
    "log amplitude about 0.4)",
    threshold="< 1e-5 (fixed in advance: both paths sum the same nf * n_angles * n_elements "
    "complex64 terms in a different order, which gives rounding differences of order 1e-6, "
    "as in E-FUS-013; a wrong sign, a missing transmit or receive factor or a wrong "
    "frequency dependence of the screen gives errors of order 1)",
    seeds=(0,),
)
def test_e_recon_005_screen_operator_assembly():
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    cfg = from_dict(ImagingConfig, load_yaml(CONFIG))
    setup = imaging_setup(cfg)
    homogeneous = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)
    screen, _ = draw_screens(ScreenSkullConfig(), setup.transducer.n_elements, tree)

    reference = psf_operator(
        apply_phase_screen(homogeneous, screen.delays, screen.amplitudes),
        setup.pixel_grid,
        setup.roi,
        homogeneous,
        setup.sequence,
        cfg.beamforming,
    )
    factors = operator_factors(
        homogeneous,
        setup.pixel_grid,
        setup.roi.points().reshape(-1, 2),
        homogeneous,
        setup.sequence,
        cfg.beamforming,
    )
    assembled = psf_matrix(screen_model_from_factors(factors, 1), *screen_tensors(screen)).numpy()

    error = float(np.abs(assembled - reference).max() / np.abs(reference).max())
    unscreened = float(
        np.abs(
            psf_matrix(
                screen_model_from_factors(factors, 1), *(0.0 * t for t in screen_tensors(screen))
            ).numpy()
            - reference
        ).max()
        / np.abs(reference).max()
    )
    report(
        EVAL_ID,
        {"relative_error": error, "relative_error_without_screen": unscreened},
        passed=error < MAX_RELATIVE_ERROR,
        note="relative_error_without_screen is the same comparison with the screen left out of "
        "the torch assembly: the size of the effect being checked",
    )
