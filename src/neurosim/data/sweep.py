"""Sweep driver: many scenes from one base config, written as a paired dataset.

Scene `i` of a sweep is one run of `data.pipeline` with the seed subtree
`tree.child(i)`: every scene has its own head (skull seed), CT noise and
ultrasound noise. What else varies is drawn from the scene's `dataset_sweep`
stream, one sub-index per parameter (`_PARAMETER_STREAMS`), so switching one
kind of variation on or off never changes the draws of another. A scene
depends on the sweep config, the root seed and its index only, never on which
other scenes are generated.

The dataset is a directory: one run bundle per scene (`data.io.save_run`) and
`index.json`, which lists every scene's file, seed and varied parameters and is
a function of the sweep config and the root seed alone.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurosim.common.config import config_hash, field, to_dict
from neurosim.common.io import provenance
from neurosim.common.seeding import SeedTree
from neurosim.data.config import RunConfig
from neurosim.data.io import load_run, save_run
from neurosim.data.pipeline import run_tree
from neurosim.data.types import Scene, TrialSet
from neurosim.head.anatomy import unit_vector
from neurosim.head.types import SkullEffects

INDEX_NAME = "index.json"
# Suffix of a bundle that is still being written; it is renamed when complete.
_PARTIAL_SUFFIX = ".partial"
# Sub-index of the `dataset_sweep` stream that draws each varied parameter.
_PARAMETER_STREAMS = {
    "center_frequency": 0,
    "probe_shift": 1,
    "probe_tilt": 2,
    "target_direction": 3,
    "effects": 4,
    "electronic_snr": 5,
}


@dataclass(frozen=True)
class SweepConfig:
    """A base run and what varies from scene to scene. The skull seed always varies."""

    base: RunConfig = field(doc="Run configuration every scene starts from.")
    n_scenes: int = field(ge=1, doc="Number of scenes (scene indices 0 .. n_scenes - 1).")
    center_frequencies: tuple[float, ...] = field(
        (),
        gt=0.0,
        doc="Pulse centre frequencies to choose from, uniformly [Hz]; empty = the base value. "
        "The element pitch follows the wavelength, so the aperture changes with it.",
    )
    probe_shift_jitter: float = field(
        0.0,
        ge=0.0,
        doc="Largest lateral shift of the probe [m]; `probe.lateral_shift` is the base value "
        "plus a uniform draw in +- this.",
    )
    probe_tilt_jitter_deg: float = field(
        0.0,
        ge=0.0,
        lt=45.0,
        doc="Largest in-plane tilt of the probe [degrees]; `probe.tilt_deg` is the base value "
        "plus a uniform draw in +- this.",
    )
    target_direction_jitter_deg: float = field(
        0.0,
        ge=0.0,
        le=90.0,
        doc="Largest angle between the scene's and the base `head.cbv.target_direction` "
        "[degrees]; the direction is uniform over that spherical cap. It moves the probe, "
        "the activated site and the head box to another part of the skull.",
    )
    effects: tuple[SkullEffects, ...] = field(
        (),
        doc="Skull-effect switch combinations to choose from, uniformly; empty = the base value.",
    )
    electronic_snr_db: tuple[float, ...] = field(
        (),
        doc="Electronic noise levels `imaging.noise.electronic_snr_db` to choose from, "
        "uniformly [dB]; empty = the base value.",
    )

    def __post_init__(self) -> None:
        # Building the scene config of every frequency runs the imaging validation
        # (range aliasing, pixel grid) for each of them.
        for frequency in self.center_frequencies:
            _with_center_frequency(self.base, frequency)


def _with_center_frequency(cfg: RunConfig, frequency: float) -> RunConfig:
    transducer = dataclasses.replace(cfg.imaging.transducer, center_frequency=frequency)
    return dataclasses.replace(
        cfg, imaging=dataclasses.replace(cfg.imaging, transducer=transducer)
    )


def _direction_in_cap(
    direction: tuple[float, float, float], max_angle_deg: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    """A unit vector uniformly distributed over the spherical cap around `direction`."""
    axis = unit_vector(direction)
    cos_limit = np.cos(np.deg2rad(max_angle_deg))
    cos_angle = rng.uniform(cos_limit, 1.0)
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    sin_angle = np.sqrt(1.0 - cos_angle**2)
    drawn = cos_angle * axis + sin_angle * (np.cos(azimuth) * first + np.sin(azimuth) * second)
    return tuple(float(component) for component in drawn / np.linalg.norm(drawn))


def scene_config(cfg: SweepConfig, tree: SeedTree, index: int) -> tuple[RunConfig, dict[str, Any]]:
    """Run configuration of one scene and the parameters that were drawn for it.

    Args:
        cfg: the sweep configuration.
        tree: root seed tree of the sweep.
        index: scene index.

    Returns:
        `(run config, varied)`; `varied` maps the name of every varied parameter
        to its value in this scene, as plain data.
    """
    scene_tree = tree.child(index)

    def rng(parameter: str) -> np.random.Generator:
        """The scene's generator for one varied parameter."""
        return scene_tree.rng("dataset_sweep", _PARAMETER_STREAMS[parameter])

    run = cfg.base
    varied: dict[str, Any] = {}
    if cfg.center_frequencies:
        frequency = float(rng("center_frequency").choice(cfg.center_frequencies))
        run = _with_center_frequency(run, frequency)
        varied["center_frequency"] = frequency
    if cfg.probe_shift_jitter > 0.0 or cfg.probe_tilt_jitter_deg > 0.0:
        shift = run.probe.lateral_shift + rng("probe_shift").uniform(-1.0, 1.0) * (
            cfg.probe_shift_jitter
        )
        tilt = run.probe.tilt_deg + rng("probe_tilt").uniform(-1.0, 1.0) * (
            cfg.probe_tilt_jitter_deg
        )
        probe = dataclasses.replace(run.probe, lateral_shift=float(shift), tilt_deg=float(tilt))
        run = dataclasses.replace(run, probe=probe)
        varied["probe"] = to_dict(probe)
    if cfg.target_direction_jitter_deg > 0.0:
        direction = _direction_in_cap(
            run.head.cbv.target_direction,
            cfg.target_direction_jitter_deg,
            rng("target_direction"),
        )
        cbv = dataclasses.replace(run.head.cbv, target_direction=direction)
        run = dataclasses.replace(run, head=dataclasses.replace(run.head, cbv=cbv))
        varied["target_direction"] = list(direction)
    if cfg.effects:
        effects = cfg.effects[int(rng("effects").integers(len(cfg.effects)))]
        run = dataclasses.replace(run, effects=effects)
        varied["effects"] = to_dict(effects)
    if cfg.electronic_snr_db:
        snr = float(rng("electronic_snr").choice(cfg.electronic_snr_db))
        noise = dataclasses.replace(run.imaging.noise, electronic_snr_db=snr)
        run = dataclasses.replace(run, imaging=dataclasses.replace(run.imaging, noise=noise))
        varied["electronic_snr_db"] = snr
    return run, varied


def scene_file(index: int) -> str:
    """File name of the bundle of scene `index` inside the dataset directory."""
    return f"scene_{index:04d}.h5"


def sweep_index(cfg: SweepConfig, root_seed: int) -> dict[str, Any]:
    """The index of a sweep: provenance and, per scene, file, seed and varied parameters.

    A function of the sweep config and the root seed only. Scene `i` is the run
    of its `config_hash` with the seed subtree `SeedTree(root_seed).child(i)`.
    """
    tree = SeedTree(root_seed)
    scenes = []
    for index in range(cfg.n_scenes):
        run, varied = scene_config(cfg, tree, index)
        scenes.append(
            {
                "index": index,
                "file": scene_file(index),
                "root_seed": root_seed,
                "seed_path": [index],
                "config_hash": config_hash(run),
                "varied": varied,
            }
        )
    return {**provenance(cfg, root_seed), "n_scenes": cfg.n_scenes, "scenes": scenes}


def _existing_run_matches(path: Path, entry: dict[str, Any]) -> bool:
    """True if `path` holds the complete run of an index entry; False if it must be (re)made.

    Raises:
        ValueError: if the file loads but belongs to another config, seed or scene.
    """
    if not path.exists():
        return False
    try:
        scene, _ = load_run(path)
    except (OSError, KeyError):
        # Not a readable run bundle (for example cut short by a crash): generate it again.
        return False
    found = (
        scene.meta["config_hash"],
        scene.meta["root_seed"],
        scene.meta.get("sweep", {}).get("index"),
    )
    expected = (entry["config_hash"], entry["root_seed"], entry["index"])
    if found != expected:
        raise ValueError(
            f"{path} holds a run of another sweep (config hash, root seed, scene index) = "
            f"{found}, expected {expected}; use an empty output directory"
        )
    return True


def _generate_scene(
    cfg: SweepConfig, tree: SeedTree, entry: dict[str, Any], path: Path
) -> None:
    """Run one scene and write its bundle; nothing of the run outlives the call."""
    run, varied = scene_config(cfg, tree, entry["index"])
    scene, trials = run_tree(run, tree.child(entry["index"]))
    sweep_meta = {"index": entry["index"], "seed_path": entry["seed_path"], "varied": varied}
    scene = dataclasses.replace(scene, meta={**scene.meta, "sweep": sweep_meta})
    partial = path.with_name(path.name + _PARTIAL_SUFFIX)
    save_run(partial, scene, trials, run, tree.root_seed)
    os.replace(partial, path)


def run_sweep(
    cfg: SweepConfig,
    root_seed: int,
    out_dir: str | os.PathLike[str],
    scenes: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Generate the scenes of a sweep into a dataset directory.

    Writes `index.json` first, then one bundle per scene. A scene whose bundle
    already exists and loads is skipped, so an interrupted sweep is resumed by
    calling this again; a bundle is written under a temporary name and renamed
    when complete. One scene is held in memory at a time.

    Args:
        cfg: the sweep configuration.
        root_seed: root seed of the sweep.
        out_dir: dataset directory (created if missing).
        scenes: scene indices to generate; None = all. The index always lists
            every scene of the sweep.

    Returns:
        the index (`sweep_index`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = sweep_index(cfg, root_seed)
    (out_dir / INDEX_NAME).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tree = SeedTree(root_seed)
    for scene_index in range(cfg.n_scenes) if scenes is None else scenes:
        entry = index["scenes"][scene_index]
        path = out_dir / entry["file"]
        if not _existing_run_matches(path, entry):
            _generate_scene(cfg, tree, entry, path)
    return index


def load_dataset(out_dir: str | os.PathLike[str]) -> Iterator[tuple[Scene, TrialSet]]:
    """Iterate over the runs of a dataset directory, in scene order.

    Raises:
        FileNotFoundError: when a scene listed in the index has no bundle yet.
    """
    out_dir = Path(out_dir)
    index = json.loads((out_dir / INDEX_NAME).read_text(encoding="utf-8"))
    for entry in index["scenes"]:
        path = out_dir / entry["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path}: scene {entry['index']} has not been generated")
        yield load_run(path)
