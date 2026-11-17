"""E-FUS-014: the statistical tier (H @ cbv with gamma speckle) matches the scatterer tier.

Blood only, no noise: many independent draws of the blood scatterers, one frame
each. The ensemble mean of |IQ|^2 is compared with `H @ (cbv * cell_area)` and
the spread of |IQ|^2 with the gamma model of `statistical_power_doppler`
(shape 1 for one frame: variance = mean^2; shape R for the mean of R frames:
relative standard deviation 1/sqrt(R)).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.fus.beamform import beamform, make_beamformer
from neurosim.fus.config import ImagingConfig, imaging_setup
from neurosim.fus.operators import pd_operator, psf_operator
from neurosim.fus.scatterers import blood_scatterers
from neurosim.fus.synthesis import channel_spectra
from neurosim.fus.types import GreensBank

EVAL_ID = "E-FUS-014"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
REGION_X = (-0.003, 0.003)  # blood-filled region, lateral [m]
REGION_Z = (0.027, 0.033)  # blood-filled region, depth [m]
SOURCE_SPACING = 0.25e-3  # CBV grid [m]: a third of a wavelength, well below the PSF width
PIXEL_SPACING = 0.4e-3  # [m]
N_SCATTERERS = 1200  # about 30 per resolution cell (1 mm^2) in the 36 mm^2 region
N_REALISATIONS = 300
MAX_MEAN_ERROR = 0.03
MAX_VARIANCE_ERROR = 0.10


def _cbv_map(grid: Grid2D) -> np.ndarray:
    """A smooth CBV map: 2 % background with a 6 % Gaussian blob off centre."""
    points = grid.points()
    distance2 = (points[..., 0] - 0.001) ** 2 + (points[..., 1] - 0.0295) ** 2
    return 0.02 + 0.04 * np.exp(-0.5 * distance2 / 0.0012**2)


@eval_case(
    id=EVAL_ID,
    claim=(
        "for blood with random positions the ensemble mean of the scatterer-tier power equals "
        "H @ (cbv * cell_area), and its fluctuations follow the gamma speckle model"
    ),
    reference=(
        "Statistics of fully developed speckle: one frame's power is exponentially distributed "
        "(variance = mean^2), the mean of R independent frames is gamma distributed with shape R "
        "(Goodman, Statistical Optics, 1985); expected power of independent scatterers is the "
        "sum of their individual powers"
    ),
    metric=(
        "mean_error: sum over pixels of the ensemble-mean power / sum of H @ cbv, minus 1; "
        "variance_error: pixel average of var(power) / mean(power)^2, minus 1; "
        "pixel_rms: rms over pixels of ensemble mean / (H @ cbv) - 1, in units of 1/sqrt(R)"
    ),
    threshold=(
        "|mean_error| < 0.03, |variance_error| < 0.10, pixel_rms < 2 (fixed in advance, R = 300 "
        "realisations and about 30 independent speckle cells: the mean has a standard error of "
        "about 1 %; the normalised variance has a standard error of about 3 % and is biased up "
        "by about 1/30 by the finite number of scatterers per resolution cell; a gamma "
        "distributed pixel mean has pixel_rms = 1)"
    ),
    seeds=(0,),
)
def test_e_fus_014_statistical_tier() -> None:
    cfg = from_dict(ImagingConfig, load_yaml(CONFIG))
    setup = imaging_setup(cfg)
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    bank = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)
    source_grid = Grid2D.from_extent(REGION_X, REGION_Z, SOURCE_SPACING)
    pixel_grid = Grid2D.from_extent(REGION_X, REGION_Z, PIXEL_SPACING)
    cbv = _cbv_map(source_grid)
    no_flow = np.zeros((2, *source_grid.shape))

    operator = pd_operator(psf_operator(bank, pixel_grid, source_grid, bank, setup.sequence))
    expected = operator.astype(np.float64) @ (cbv.reshape(-1) * source_grid.spacing**2)

    beamformer = make_beamformer(pixel_grid, bank, setup.sequence)
    power_sum = np.zeros(pixel_grid.shape)
    power2_sum = np.zeros(pixel_grid.shape)
    for realisation in range(N_REALISATIONS):
        blood = blood_scatterers(
            cbv,
            no_flow,
            source_grid,
            N_SCATTERERS,
            tree.rng("blood_scatterers", realisation),
            cfg.scatterers.blood_transit_length,
        )
        frame = beamform(beamformer, channel_spectra(bank, blood.scatterers(), setup.sequence))
        power = np.abs(frame) ** 2
        power_sum += power
        power2_sum += power**2
    mean = power_sum / N_REALISATIONS
    variance = power2_sum / N_REALISATIONS - mean**2

    expected = expected.reshape(pixel_grid.shape)
    values = {
        "mean_error": float(mean.sum() / expected.sum() - 1.0),
        "variance_error": float(np.mean(variance / mean**2) - 1.0),
        "pixel_rms": float(np.sqrt(np.mean((mean / expected - 1.0) ** 2) * N_REALISATIONS)),
    }
    report(
        EVAL_ID,
        values,
        passed=abs(values["mean_error"]) < MAX_MEAN_ERROR
        and abs(values["variance_error"]) < MAX_VARIANCE_ERROR
        and values["pixel_rms"] < 2.0,
    )
