"""E-FUS-011: scatterers at known positions appear at those positions in the image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import eval_case, report
from neurosim.fus.beamform import beamform, make_beamformer
from neurosim.fus.config import ImagingConfig, imaging_setup
from neurosim.fus.synthesis import channel_spectra
from neurosim.fus.types import GreensBank, Scatterers

EVAL_ID = "E-FUS-011"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
# Scatterer positions (x, z) [m]: off the pixel lattice, spread over the ROI of the small config.
POSITIONS = (
    (0.00013, 0.03008),
    (-0.00431, 0.02622),
    (0.00367, 0.03389),
    (0.00458, 0.02571),
    (-0.00224, 0.03417),
)
MAX_ERROR_PIXELS = 0.5


def _vertex_offset(below: float, centre: float, above: float) -> float:
    """Offset of the vertex of the parabola through three equally spaced samples [samples]."""
    return 0.5 * (below - above) / (below - 2.0 * centre + above)


@eval_case(
    id=EVAL_ID,
    claim="a point scatterer in water is imaged at its true position",
    reference="The known scatterer positions (ground truth of the simulation input)",
    metric=(
        "largest distance along x or z between the image peak (parabolic fit through the "
        "peak pixel and its neighbours) and the true position, over five scatterers [pixels]"
    ),
    threshold=(
        "< 0.5 pixel (fixed in advance: the pixel is half a wavelength, below the resolution, "
        "so a correct beamformer must put the peak in the right pixel)"
    ),
    seeds=(0,),
)
def test_e_fus_011_localisation() -> None:
    setup = imaging_setup(from_dict(ImagingConfig, load_yaml(CONFIG)))
    bank = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)
    beamformer = make_beamformer(setup.pixel_grid, bank, setup.sequence)
    z_axis, x_axis = setup.pixel_grid.axes()
    spacing = setup.pixel_grid.spacing

    worst = 0.0
    for x_true, z_true in POSITIONS:
        scatterer = Scatterers(xz=np.array([[x_true, z_true]]), amplitude=np.ones(1))
        envelope = np.abs(beamform(beamformer, channel_spectra(bank, scatterer, setup.sequence)))
        iz, ix = np.unravel_index(np.argmax(envelope), envelope.shape)
        z_peak = z_axis[iz] + spacing * _vertex_offset(*envelope[iz - 1 : iz + 2, ix])
        x_peak = x_axis[ix] + spacing * _vertex_offset(*envelope[iz, ix - 1 : ix + 2])
        error = max(abs(x_peak - x_true), abs(z_peak - z_true)) / spacing
        worst = max(worst, error)

    report(EVAL_ID, worst, passed=worst < MAX_ERROR_PIXELS)
