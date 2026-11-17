"""E-FUS-012: the SVD clutter filter separates moving blood from static tissue.

The filter multiplies the space-time matrix X (n_frames x n_pix) by a temporal
projector Pi that it derives from X itself. Once X is given the filter is
linear, so its effect on the blood and tissue parts of X can be told apart: Pi
is recovered from the filter's input and output (`Pi = X_filtered X^+`, X has
full row rank) and applied to the blood-only and tissue-only frames.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.common.units import np_to_db
from neurosim.fus.acquire import acquire_frames
from neurosim.fus.config import ImagingConfig, imaging_setup
from neurosim.fus.doppler import svd_clutter_filter
from neurosim.fus.scatterers import blood_scatterers, tissue_amplitude, tissue_scatterers
from neurosim.fus.types import GreensBank

EVAL_ID = "E-FUS-012"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
TISSUE_TO_BLOOD_DB = 40.0
BLOOD_VELOCITY = (0.005, 0.015)  # (vx, vz) [m/s]: 15 mm/s axially is a 39 Hz Doppler shift at 2 MHz
N_DISCARD = 1  # static tissue is one singular component
MIN_GAIN_DB = 50.0
MIN_RETAINED = 0.8


def _power(frames: np.ndarray) -> float:
    return float(np.sum(np.abs(frames) ** 2))


def _power_ratio_db(ratio: float) -> float:
    """Power ratio in decibels (half the amplitude level of the same ratio)."""
    return 0.5 * float(np_to_db(np.log(ratio)))


@eval_case(
    id=EVAL_ID,
    claim=(
        "with static tissue 40 dB above blood, removing one singular component raises the "
        "blood-to-clutter ratio by at least 50 dB while keeping at least 80 % of the blood power"
    ),
    reference=(
        "Blood-only and tissue-only frames of the same acquisition (linearity of the simulation), "
        "and the rank-one structure of a static signal in the space-time matrix (Demene et al. "
        "2015, IEEE TMI 34(11):2271-2285)"
    ),
    metric=(
        "gain_db: blood-to-clutter power ratio after / before the filter [dB]; "
        "retained: blood power after / before the filter"
    ),
    threshold=(
        "gain_db >= 50 and retained >= 0.8 (fixed in advance: a static signal is exactly one "
        "singular component, so the residual clutter must fall below the blood, 40 dB, with a "
        "further 10 dB because blood leaks into the tissue component only through its "
        "projection on one of many spatial modes; one of 40 temporal components is removed and "
        "blood moving over the ensemble spreads over many of them, so at least 80 % must remain)"
    ),
    seeds=(0,),
)
def test_e_fus_012_svd_clutter_filter() -> None:
    base = from_dict(ImagingConfig, load_yaml(CONFIG))
    cfg = replace(
        base,
        noise=replace(
            base.noise,
            electronic_noise=False,
            tissue_motion=False,
            tissue_to_blood_db=TISSUE_TO_BLOOD_DB,
        ),
    )
    setup = imaging_setup(cfg)
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    bank = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)
    cbv = np.full(setup.roi.shape, cfg.noise.cbv_reference)
    velocity = np.broadcast_to(np.array(BLOOD_VELOCITY)[:, None, None], (2, *setup.roi.shape))
    blood = blood_scatterers(
        cbv,
        velocity,
        setup.roi,
        cfg.scatterers.n_blood,
        tree.rng("blood_scatterers"),
        cfg.scatterers.blood_transit_length,
    )
    tissue = tissue_scatterers(
        setup.roi,
        cfg.scatterers.n_tissue,
        tissue_amplitude(setup.roi, cfg.scatterers.n_tissue, cfg.noise),
        tree.rng("tissue_scatterers"),
    )

    def acquire(tissue_part):
        return acquire_frames(
            bank, bank, blood, tissue_part, setup.sequence, setup.pixel_grid, cfg, tree
        )

    mixed = acquire(tissue)
    blood_only = acquire(None)
    tissue_only = mixed - blood_only
    filtered = svd_clutter_filter(mixed, N_DISCARD)

    n_frames = mixed.shape[0]
    projector = filtered.reshape(n_frames, -1) @ np.linalg.pinv(mixed.reshape(n_frames, -1))
    blood_after = _power(projector @ blood_only.reshape(n_frames, -1))
    clutter_after = _power(projector @ tissue_only.reshape(n_frames, -1))
    ratio_before = _power(blood_only) / _power(tissue_only)
    gain_db = _power_ratio_db(blood_after / clutter_after / ratio_before)
    retained = blood_after / _power(blood_only)

    report(
        EVAL_ID,
        {"gain_db": gain_db, "retained": retained},
        passed=gain_db >= MIN_GAIN_DB and retained >= MIN_RETAINED,
        note=f"blood-to-clutter ratio before the filter: {_power_ratio_db(ratio_before):.1f} dB",
    )
