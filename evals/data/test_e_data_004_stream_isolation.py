"""E-DATA-004: end to end, a seed stream changes only the part of the run that draws from it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neurosim.common.config import from_dict, load_yaml
from neurosim.common.evalkit import REGISTRY, eval_case, report
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.scene import SceneExtras, build_scene_with_extras
from neurosim.data.trials import simulate_trials
from neurosim.data.types import BEAMFORMERS, SKULL_CT, Scene, TrialSet

EVAL_ID = "E-DATA-004"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run_small.yaml"
OTHER_ROOT = 12345  # root seed of the replacement streams
ULTRASOUND_STREAMS = ("us_electronic_noise", "speckle_statistical")
CT_STREAMS = ("ct_photon_noise",)


class _SwappedTree(SeedTree):
    """A seed tree whose named streams come from another root seed."""

    def __init__(self, root_seed: int, swapped: tuple[str, ...], other_root_seed: int):
        super().__init__(root_seed)
        self._swapped = swapped
        self._other = SeedTree(other_root_seed)

    def rng(self, name: str, *sub: int) -> np.random.Generator:
        if name in self._swapped:
            return self._other.rng(name, *sub)
        return super().rng(name, *sub)


def _quantities(scene: Scene, extras: SceneExtras, trials: dict[str, TrialSet]) -> dict:
    """Every array of a run by name, grouped by what it derives from."""
    head = {
        "cbv_baseline": scene.cbv_baseline,
        "activation": scene.activation,
        "brain_mask": scene.brain_mask,
        "plane": np.asarray(scene.plane.origin + scene.plane.u_lateral + scene.plane.u_depth),
        "medium.sound_speed": extras.medium.sound_speed,
        "medium.density": extras.medium.density,
        "medium.attenuation": extras.medium.attenuation,
        "blood_velocity": extras.blood_velocity,
        "bank_skull": scene.bank_skull.B,
        "bank_clean": scene.bank_clean.B,
        "screen_true.delays": scene.screen_true.delays,
        "screen_true.amplitudes": scene.screen_true.amplitudes,
    }
    ct = {
        "medium_ct.sound_speed": extras.medium_ct.sound_speed,
        "bank_ct": scene.bank_ct.B,
        "screen_ct.delays": scene.screen_ct.delays,
        "screen_ct.amplitudes": scene.screen_ct.amplitudes,
        "ct_sound_speed_r2": np.asarray(scene.meta["ct_sound_speed_r2"]),
    }
    images = {}
    for key in BEAMFORMERS:
        group = ct if key == SKULL_CT else head
        group[f"operator[{key}]"] = scene.operators[key]
        group[f"noise_floor[{key}]"] = scene.noise_floor[key]
        for fidelity, trial_set in trials.items():
            images[f"{fidelity}.power_doppler[{key}]"] = trial_set.power_doppler[key]
    truth = {}
    for fidelity, trial_set in trials.items():
        for name in ("time", "amplitude", "label", "cbv"):
            truth[f"{fidelity}.{name}"] = getattr(trial_set, name)
    return {"head": head, "ct": ct, "images": images, "truth": truth}


def _changed(reference: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> set[str]:
    return {name for name in reference if not np.array_equal(reference[name], other[name])}


@eval_case(
    id=EVAL_ID,
    claim=(
        "through the whole pipeline, replacing only the ultrasound noise streams "
        "(us_electronic_noise, speckle_statistical) leaves head, media, CT prior, banks, "
        "screens, operators, noise floors and the CBV series bit-identical and changes every "
        "power Doppler image (both fidelities); replacing only ct_photon_noise changes the "
        "CT-derived quantities (CT medium, CT bank and screen, the skull_ct operator, floor and "
        "images, the CT R^2) and nothing else"
    ),
    reference=(
        "Bitwise comparison of runs from the same root seed whose named streams are taken "
        "from another root seed (ROADMAP D-004)"
    ),
    metric=(
        "unexpected_changes: number of quantities that changed although they do not derive "
        "from the replaced streams; missing_changes: number of quantities that derive from "
        "the replaced streams and did not change; summed over the two replacements"
    ),
    threshold=(
        "unexpected_changes = 0 and missing_changes = 0 (fixed in advance: stream isolation "
        "is exact; a missing change would mean a stream is not used where the design says)"
    ),
    seeds=(7,),
)
def test_e_data_004_stream_isolation() -> None:
    base = from_dict(RunConfig, load_yaml(CONFIG))
    # A short ensemble and few scatterers keep the scatterer-fidelity image cheap.
    imaging = replace(
        base.imaging,
        sequence=replace(base.imaging.sequence, n_frames=8),
        doppler=replace(base.imaging.doppler, n_discard=1),
        scatterers=replace(base.imaging.scatterers, n_blood=200, n_tissue=200),
    )
    cfg = replace(base, imaging=imaging, source_refinement=1)
    scatterer_cfg = replace(cfg, trials=replace(cfg.trials, n_images=1, fidelity="scatterer"))
    seed = REGISTRY[EVAL_ID].seeds[0]

    def run(tree: SeedTree) -> dict:
        scene, extras = build_scene_with_extras(cfg, tree)
        trials = {
            "statistical": simulate_trials(scene, cfg, tree, extras),
            "scatterer": simulate_trials(scene, scatterer_cfg, tree, extras),
        }
        return _quantities(scene, extras, trials)

    reference = run(SeedTree(seed))
    ultrasound = run(_SwappedTree(seed, ULTRASOUND_STREAMS, OTHER_ROOT))
    ct = run(_SwappedTree(seed, CT_STREAMS, OTHER_ROOT))

    unexpected: set[str] = set()
    missing: set[str] = set()
    # Ultrasound streams: only the images change, and all of them do.
    for group in ("head", "ct", "truth"):
        unexpected |= _changed(reference[group], ultrasound[group])
    missing |= set(reference["images"]) - _changed(reference["images"], ultrasound["images"])
    # CT photon noise: the CT group and the skull_ct images change, nothing else.
    for group in ("head", "truth"):
        unexpected |= _changed(reference[group], ct[group])
    missing |= set(reference["ct"]) - _changed(reference["ct"], ct["ct"])
    ct_images = {name for name in reference["images"] if SKULL_CT in name}
    changed_images = _changed(reference["images"], ct["images"])
    unexpected |= changed_images - ct_images
    missing |= ct_images - changed_images

    report(
        EVAL_ID,
        {
            "unexpected_changes": float(len(unexpected)),
            "missing_changes": float(len(missing)),
            "quantities_compared": float(sum(len(group) for group in reference.values())),
        },
        passed=not unexpected and not missing,
        note=f"unexpected: {sorted(unexpected)}; missing: {sorted(missing)}",
    )
