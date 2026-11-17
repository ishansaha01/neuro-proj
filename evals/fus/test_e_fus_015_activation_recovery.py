"""E-FUS-015: without a skull, a CBV increase in a patch appears as the same relative increase
in power Doppler there, and nowhere else.

Full scatterer chain in water: moving blood, static tissue 40 dB above it, SVD
clutter filter, power Doppler. Baseline and activated CBV maps are imaged with
independent blood scatterers and averaged over several ensembles. Electronic
noise is off, because a noise floor adds the same power to both images and
would lower the ratio by a known but separate amount.

Blood flows in a random direction at 5 to 15 mm/s in every ROI cell, and frames
are 100 ms apart, so a scatterer travels 0.5 to 1.5 mm between frames, about
the length of its path, and each frame is a nearly independent speckle realisation. The SVD
filter does not depend on the frame rate. (With one velocity everywhere the
speckle pattern only slides between frames and an ensemble holds two or three
independent realisations, not 40.)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.constants import C_REF
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.fus.acquire import acquire_power_doppler
from neurosim.fus.config import ImagingConfig, imaging_setup
from neurosim.fus.scatterers import blood_scatterers, tissue_amplitude, tissue_scatterers
from neurosim.fus.types import GreensBank

EVAL_ID = "E-FUS-015"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fus_small.yaml"
CBV_INCREASE = 0.40  # relative CBV change in the patch
PATCH_X = (-0.0065, -0.0005)  # activated patch: left half of the ROI [m]
CORE_X = (-0.0050, -0.0020)  # patch interior, 1.5 mm (one lateral PSF width) from its edge [m]
CONTROL_X = (0.0015, 0.0050)  # outside the patch, 2 mm from its edge [m]
CORE_Z = (0.0265, 0.0335)  # depth range of both measurement regions [m]
BLOOD_SPEED_RANGE = (0.005, 0.015)  # [m/s]
FRAME_RATE = 10.0  # [Hz]
N_BLOOD = 500  # the expected power does not depend on the number of scatterers
N_ENSEMBLES = 8
TOLERANCE = 0.06


def _region(grid, x_range: tuple[float, float]) -> np.ndarray:
    """Mask of the pixels inside a lateral range and the common depth range."""
    points = grid.points()
    return (
        (points[..., 0] >= x_range[0])
        & (points[..., 0] <= x_range[1])
        & (points[..., 1] >= CORE_Z[0])
        & (points[..., 1] <= CORE_Z[1])
    )


@eval_case(
    id=EVAL_ID,
    claim=(
        "in water, raising CBV by 40 % in a patch raises the power Doppler signal in the patch "
        "by 40 % and leaves it unchanged outside the patch"
    ),
    reference=(
        "Linearity of blood backscatter power in the number of scatterers: power Doppler is "
        "proportional to CBV (Mace et al. 2011, Nature Methods 8:662-664, the principle of fUS)"
    ),
    metric=(
        "patch_change: mean power Doppler in the patch interior, activated / baseline, minus 1; "
        "control_change: the same in a region outside the patch"
    ),
    threshold=(
        "|patch_change - 0.40| < 0.06 and |control_change| < 0.06 (fixed in advance: each "
        "region holds about 20 resolution cells, with 8 ensembles of 40 nearly independent "
        "frames, so the ratio of two independent images has a standard error of "
        "sqrt(2 / (20 * 40 * 8)) = 1.8 %; 0.06 is three standard errors plus leakage through "
        "the PSF sidelobes)"
    ),
    seeds=(0,),
)
def test_e_fus_015_activation_recovery() -> None:
    base = from_dict(ImagingConfig, load_yaml(CONFIG))
    cfg = replace(
        base,
        sequence=replace(base.sequence, frame_rate=FRAME_RATE),
        doppler=replace(base.doppler, cutoff_rule="fixed", n_discard=1),
        noise=replace(base.noise, electronic_noise=False, tissue_motion=False),
    )
    setup = imaging_setup(cfg)
    tree = SeedTree(REGISTRY[EVAL_ID].seeds[0])
    bank = GreensBank.homogeneous(setup.transducer, setup.roi, setup.freqs, C_REF)

    roi_x = setup.roi.points()[..., 0]
    baseline = np.full(setup.roi.shape, cfg.noise.cbv_reference)
    in_patch = (roi_x >= PATCH_X[0]) & (roi_x <= PATCH_X[1])
    activated = baseline * (1.0 + CBV_INCREASE * in_patch)
    flow_rng = tree.rng("cbv")
    speed = flow_rng.uniform(*BLOOD_SPEED_RANGE, size=setup.roi.shape)
    direction = flow_rng.uniform(0.0, 2.0 * np.pi, size=setup.roi.shape)
    velocity = speed * np.stack((np.cos(direction), np.sin(direction)))
    tissue = tissue_scatterers(
        setup.roi,
        cfg.scatterers.n_tissue,
        tissue_amplitude(setup.roi, cfg.scatterers.n_tissue, cfg.noise),
        tree.rng("tissue_scatterers"),
    )

    images = []
    for condition, cbv in enumerate((baseline, activated)):
        total = np.zeros(setup.pixel_grid.shape)
        for ensemble in range(N_ENSEMBLES):
            blood = blood_scatterers(
                cbv,
                velocity,
                setup.roi,
                N_BLOOD,
                tree.rng("blood_scatterers", condition, ensemble),
                cfg.scatterers.blood_transit_length,
            )
            total += acquire_power_doppler(
                bank, bank, blood, tissue, setup.sequence, setup.pixel_grid, cfg, tree
            )
        images.append(total / N_ENSEMBLES)
    power_baseline, power_activated = images

    core = _region(setup.pixel_grid, CORE_X)
    control = _region(setup.pixel_grid, CONTROL_X)
    values = {
        "patch_change": float(power_activated[core].mean() / power_baseline[core].mean() - 1.0),
        "control_change": float(
            power_activated[control].mean() / power_baseline[control].mean() - 1.0
        ),
    }
    report(
        EVAL_ID,
        values,
        passed=abs(values["patch_change"] - CBV_INCREASE) < TOLERANCE
        and abs(values["control_change"]) < TOLERANCE,
    )
