"""E-FUS-013: a column of the point-spread operator is the image of a scatterer at that node."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.fus.beamform import beamform, make_beamformer
from neurosim.fus.config import BeamformingConfig, ImagingConfig, imaging_setup
from neurosim.fus.operators import psf_operator
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.synthesis import channel_spectra
from neurosim.fus.types import GreensBank, Scatterers

EVAL_ID = "E-FUS-013"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
DELAY_RMS_PERIODS = 0.15  # rms element delay of the phase screen [periods at the centre frequency]
AMPLITUDE_RANGE = (0.7, 1.3)  # element amplitudes of the phase screen
N_COLUMNS = 12  # source nodes compared
MAX_RELATIVE_ERROR = 1e-5


@eval_case(
    id=EVAL_ID,
    claim=(
        "each column of psf_operator equals the image that channel_spectra followed by "
        "delay-and-sum gives for a unit scatterer at that source node"
    ),
    reference=(
        "The explicit path (channel_spectra, then beamform) for one scatterer at a time, on a "
        "phase-screen bank beamformed with the homogeneous model and an f-number of 1.5"
    ),
    metric=(
        "largest |operator column - explicit image| / max|explicit image| over 12 source nodes"
    ),
    threshold=(
        "< 1e-5 (fixed in advance: both paths sum the same nf * n_angles * n_elements terms in "
        "complex64 in a different order, which gives rounding differences of order 1e-6)"
    ),
    seeds=(0,),
)
def test_e_fus_013_operator_consistency() -> None:
    setup = imaging_setup(from_dict(ImagingConfig, load_yaml(CONFIG)))
    rng = SeedTree(REGISTRY[EVAL_ID].seeds[0]).rng("skull_microstructure")
    water = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)
    n_elements = setup.transducer.n_elements
    bank = apply_phase_screen(
        water,
        rng.normal(0.0, DELAY_RMS_PERIODS / setup.transducer.center_frequency, n_elements),
        rng.uniform(*AMPLITUDE_RANGE, n_elements),
    )
    beamforming = BeamformingConfig(f_number=1.5, window="hann")
    source_grid = setup.roi

    operator = psf_operator(bank, setup.pixel_grid, source_grid, water, setup.sequence, beamforming)
    beamformer = make_beamformer(setup.pixel_grid, water, setup.sequence, beamforming)
    sources = source_grid.points().reshape(-1, 2)
    columns = rng.choice(sources.shape[0], size=N_COLUMNS, replace=False)
    worst = 0.0
    for column in columns:
        scatterer = Scatterers(xz=sources[column : column + 1], amplitude=np.ones(1))
        image = beamform(beamformer, channel_spectra(bank, scatterer, setup.sequence)).reshape(-1)
        worst = max(worst, float(np.abs(operator[:, column] - image).max() / np.abs(image).max()))

    report(EVAL_ID, worst, passed=worst < MAX_RELATIVE_ERROR)
