"""Stand-in scenes whose true skull is a member of their own CT skull-model family.

For evals of the calibration without model mismatch (E-RECON-013, E-RECON-014):
take a real scene, draw a member of its family (`recon.skull_model`) and make
that member's bank, traced by the real backend, the scene's true bank. The data
then come from a medium the family contains, so what is measured is whether the
criterion identifies the parameters, not how well the family represents a skull.
The fixed-path shortcut of the calibration is still an approximation of such a
truth (the true bank is re-traced), as it is for a real skull.

Like `recon.synthetic`, this module creates ground truth; it is not part of
the reconstruction.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.scene import build_scene
from neurosim.data.trials import simulate_trials
from neurosim.data.types import CLEAN_DAS, Scene, TrialSet
from neurosim.recon.config import SkullModelConfig
from neurosim.recon.skull_model import SkullFamily, SkullParameters, build_family, model_bank

# Stream of the member draw (`common.seeding.STREAMS`): the HU-map constants are what
# `ct.acoustics.sample_hu_mapping` draws from this stream for a sampled CT prior.
_STREAM = "ct_hu_mapping"


def bent_ray_case(
    cfg: RunConfig, seed: int, skull_model: SkullModelConfig
) -> tuple[Scene, TrialSet, SkullFamily]:
    """A scene, its trials and its skull-model family, with the `bent_ray` backend.

    cfg: a run configuration; its backend choice is replaced by `bent_ray`, which
        the family needs (the backend's settings are kept).
    seed: root seed of the run.
    skull_model: settings of the family.
    """
    cfg = dataclasses.replace(cfg, backend=dataclasses.replace(cfg.backend, backend="bent_ray"))
    tree = SeedTree(seed)
    scene = build_scene(cfg, tree)
    return scene, simulate_trials(scene, cfg, tree), build_family(scene, skull_model)


def draw_member(
    family: SkullFamily, tree: SeedTree, lateral_speed_sigma: float, *sub: int
) -> SkullParameters:
    """Draw a member of a family: constants from their prior ranges, Gaussian lateral modes.

    family: the skull-model family of a scene.
    tree: seed tree; the draw comes from the stream `("ct_hu_mapping", *sub)`.
    lateral_speed_sigma: standard deviation of the sound-speed change of each
        lateral mode at the family's reference speed [m/s].
    """
    prior = family.cfg.mapping_prior
    rng = tree.rng(_STREAM, *sub)
    c_bone = rng.uniform(*prior.c_bone_range)
    c_water = rng.uniform(*prior.c_water_range)
    hu_bone = family.nominal.hu_bone * rng.uniform(*prior.hu_bone_relative_range)
    sigma = lateral_speed_sigma / family.cfg.lateral_reference_speed**2
    return SkullParameters(
        c_bone=float(c_bone),
        c_water=float(c_water),
        hu_bone=float(hu_bone),
        lateral=sigma * rng.standard_normal(family.n_lateral_modes),
    )


def with_member_skull(scene: Scene, family: SkullFamily, parameters: SkullParameters) -> Scene:
    """A copy of a scene whose true skull bank is the given member of its family.

    `bank_skull` becomes `recon.skull_model.model_bank(family, parameters)` (one
    trace of the real backend). The operators of the three skull beamformers
    belonged to the scene's original skull and are dropped (only `clean_das`
    stays), so nothing can use them by mistake; a `TrialSet` of the original
    scene keeps its true CBV, but its skull images no longer match.
    """
    return dataclasses.replace(
        scene,
        bank_skull=model_bank(family, parameters),
        operators={CLEAN_DAS: scene.operators[CLEAN_DAS]},
    )
