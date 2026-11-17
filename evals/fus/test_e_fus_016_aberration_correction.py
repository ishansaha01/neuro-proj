"""E-FUS-016: beamforming with the true model bank undoes a phase screen; conventional DAS does not.

A near-field phase screen gives every element an independent random delay.
Conventional delay-and-sum (homogeneous model) then adds its terms with random
phase errors; for Gaussian errors of standard deviation s radians the coherent
sum keeps `exp(-s^2 / 2)` of its amplitude (Ruze 1966, Proc. IEEE 54(4):633-640),
on transmit and again on receive: a two-way loss of `8.69 s^2` dB, 7.7 dB here.

The beamformer is normalised to unit response for a scatterer in its model
medium (`fus.beamform`). With the true screen as the model the peak is
therefore restored exactly, although conjugating phases cannot give back the
strength the screen takes from the transmitted wave at the scatterer: with a
plain average of the phase-conjugated terms the corrected peak would stay 1-3 dB
below its value in water for this screen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.common.units import np_to_db, wavelength
from neurosim.fus.beamform import das
from neurosim.fus.config import ImagingConfig, imaging_setup
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.synthesis import channel_spectra
from neurosim.fus.types import GreensBank, Scatterers

EVAL_ID = "E-FUS-016"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
DELAY_RMS_PERIODS = 0.15  # rms element delay [periods at the centre frequency]
AMPLITUDE_RANGE = (0.8, 1.2)  # element amplitude factors of the screen
SCATTERER = (0.0011, 0.0303)  # (x, z) [m], off the pixel lattice
HALF_WIDTH = 0.004  # half size of the imaged square around the ROI centre [m]
MAX_CORRECTED_ERROR_DB = 1.0
MIN_CONVENTIONAL_LOSS_DB = 3.0


def _peak_db(image: np.ndarray, reference: float) -> float:
    return float(np_to_db(np.log(np.abs(image).max() / reference)))


@eval_case(
    id=EVAL_ID,
    claim=(
        "for a phase screen with independent element delays of 0.15 period rms, beamforming "
        "with the true bank as model restores the peak of a point scatterer to within 1 dB of "
        "its value in water, while conventional delay-and-sum loses at least 3 dB"
    ),
    reference=(
        "The same scatterer imaged in water, and the coherence loss of a sum with Gaussian "
        "phase errors, exp(-s^2/2) per pass (Ruze 1966): 8.69 s^2 = 7.7 dB two-way for "
        "s = 2 pi * 0.15 rad"
    ),
    metric=(
        "corrected_db: largest |image peak with the true model / peak in water| over the seeds "
        "[dB]; conventional_loss_db: smallest loss of the image peak with the homogeneous model "
        "over the seeds [dB]"
    ),
    threshold=(
        "corrected_db < 1 and conventional_loss_db >= 3 (fixed in advance: the 1 dB bar is the "
        "claim; the coherent part alone would lose 7.7 dB, and the incoherent part of the sum "
        "and the search for the image maximum can win back part of it, so 3 dB is required "
        "for every seed)"
    ),
    seeds=(0, 1, 2, 3),
)
def test_e_fus_016_aberration_correction() -> None:
    setup = imaging_setup(from_dict(ImagingConfig, load_yaml(CONFIG)))
    transducer, sequence = setup.transducer, setup.sequence
    water = GreensBank.homogeneous(transducer, setup.roi, setup.freqs, C_REF)
    centre_z = float(np.mean(setup.roi.axes()[0]))
    pixel_grid = Grid2D.from_extent(
        (-HALF_WIDTH, HALF_WIDTH),
        (centre_z - HALF_WIDTH, centre_z + HALF_WIDTH),
        float(wavelength(C_REF, transducer.center_frequency)) / 4.0,
    )
    scatterer = Scatterers(xz=np.array([SCATTERER]), amplitude=np.ones(1))
    water_image = das(channel_spectra(water, scatterer, sequence), pixel_grid, water, sequence)
    in_water = np.abs(water_image).max()

    corrected, conventional = [], []
    for seed in REGISTRY[EVAL_ID].seeds:
        rng = SeedTree(seed).rng("skull_microstructure")
        screen = apply_phase_screen(
            water,
            rng.normal(0.0, DELAY_RMS_PERIODS / transducer.center_frequency, transducer.n_elements),
            rng.uniform(*AMPLITUDE_RANGE, transducer.n_elements),
        )
        spectra = channel_spectra(screen, scatterer, sequence)
        corrected.append(abs(_peak_db(das(spectra, pixel_grid, screen, sequence), in_water)))
        conventional.append(-_peak_db(das(spectra, pixel_grid, water, sequence), in_water))

    values = {"corrected_db": max(corrected), "conventional_loss_db": min(conventional)}
    report(
        EVAL_ID,
        values,
        passed=values["corrected_db"] < MAX_CORRECTED_ERROR_DB
        and values["conventional_loss_db"] >= MIN_CONVENTIONAL_LOSS_DB,
        note="conventional loss per seed [dB]: "
        + ", ".join(f"{loss:.1f}" for loss in conventional),
    )
