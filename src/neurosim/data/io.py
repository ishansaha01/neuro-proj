"""Save and load one run (a `Scene` and its `TrialSet`) as one HDF5 bundle with provenance.

Layout of the bundle (`common.io.save_bundle`):

- ``scene/cbv_baseline``, ``scene/activation``, ``scene/brain_mask``;
- ``scene/<bank>/...`` for ``bank_skull``, ``bank_clean`` and ``bank_ct``, each
  with the datasets of `fus.io.bank_to_bundle`; the bank's attributes are the
  bundle attribute ``<bank>``;
- ``scene/<screen>/delays`` and ``scene/<screen>/amplitudes`` for ``screen_ct``
  and ``screen_true``;
- ``scene/operators/<key>`` and ``scene/noise_floor/<key>`` per beamformer key;
- ``trials/time``, ``trials/amplitude``, ``trials/label``, ``trials/cbv`` and
  ``trials/power_doppler/<key>``;
- attributes: the provenance of the run (`common.io.provenance`), ``imaging``
  (the `ImagingConfig` as plain data), ``plane``, ``pixel_grid``, ``cell_area``,
  ``effective_frames``, ``fidelity`` and ``meta`` (`Scene.meta`).

The fine plane media of a run are not part of the `Scene` contract. When they
are kept, they are written with `head.io.save_plane_medium` next to the run
bundle, under the names of `media_paths`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from neurosim.common.config import from_dict, to_dict
from neurosim.common.grid import Grid2D, PlaneSpec
from neurosim.common.io import load_bundle, provenance, save_bundle
from neurosim.data.config import RunConfig
from neurosim.data.types import BEAMFORMERS, PhaseScreen, Scene, TrialSet
from neurosim.fus.config import ImagingConfig
from neurosim.fus.io import bank_from_bundle, bank_to_bundle

_BANKS: tuple[str, ...] = ("bank_skull", "bank_clean", "bank_ct")
_SCREENS: tuple[str, ...] = ("screen_ct", "screen_true")
_SCENE_MAPS: tuple[str, ...] = ("cbv_baseline", "activation", "brain_mask")
_TRIAL_SERIES: tuple[str, ...] = ("time", "amplitude", "label", "cbv")


def media_paths(run_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Files of the true and the CT-derived plane medium that belong to a run bundle.

    Returns ``(<stem>_medium_true.h5, <stem>_medium_ct.h5)`` in the directory of `run_path`.
    """
    run_path = Path(run_path)
    return (
        run_path.with_name(f"{run_path.stem}_medium_true.h5"),
        run_path.with_name(f"{run_path.stem}_medium_ct.h5"),
    )


def save_run(
    path: str | os.PathLike[str], scene: Scene, trials: TrialSet, cfg: RunConfig, root_seed: int
) -> None:
    """Write a run to one HDF5 bundle (module docstring).

    Args:
        path: output file.
        scene: static part of the run.
        trials: image series of the run.
        cfg: the configuration that produced the run (stored as provenance).
        root_seed: root seed of the `SeedTree` that produced it.
    """
    arrays: dict[str, np.ndarray] = {
        f"scene/{name}": getattr(scene, name) for name in _SCENE_MAPS
    }
    attrs: dict[str, Any] = provenance(cfg, root_seed)
    for name in _BANKS:
        bank_arrays, attrs[name] = bank_to_bundle(getattr(scene, name))
        arrays.update({f"scene/{name}/{key}": value for key, value in bank_arrays.items()})
    for name in _SCREENS:
        screen = getattr(scene, name)
        arrays[f"scene/{name}/delays"] = screen.delays
        arrays[f"scene/{name}/amplitudes"] = screen.amplitudes
    for key in BEAMFORMERS:
        arrays[f"scene/operators/{key}"] = scene.operators[key]
        arrays[f"scene/noise_floor/{key}"] = scene.noise_floor[key]
        arrays[f"trials/power_doppler/{key}"] = trials.power_doppler[key]
    arrays.update({f"trials/{name}": getattr(trials, name) for name in _TRIAL_SERIES})
    attrs.update(
        imaging=to_dict(scene.imaging),
        plane={
            "origin": list(scene.plane.origin),
            "u_lateral": list(scene.plane.u_lateral),
            "u_depth": list(scene.plane.u_depth),
        },
        pixel_grid={
            "shape": list(scene.pixel_grid.shape),
            "spacing": scene.pixel_grid.spacing,
            "origin": list(scene.pixel_grid.origin),
        },
        cell_area=scene.cell_area,
        effective_frames=scene.effective_frames,
        fidelity=trials.fidelity,
        meta=scene.meta,
    )
    save_bundle(path, arrays, attrs)


def load_run(path: str | os.PathLike[str]) -> tuple[Scene, TrialSet]:
    """Read a run written by `save_run`.

    Returns the `Scene` and the `TrialSet`. The provenance attributes of the
    bundle (config YAML, git commit, package version) are not part of either;
    read them with `common.io.load_bundle`.
    """
    arrays, attrs = load_bundle(path)

    def group(prefix: str) -> dict[str, np.ndarray]:
        """The datasets under a name prefix, keyed by the rest of their name."""
        return {
            name[len(prefix) :]: value for name, value in arrays.items() if name.startswith(prefix)
        }

    banks = {name: bank_from_bundle(group(f"scene/{name}/"), attrs[name]) for name in _BANKS}
    screens = {
        name: PhaseScreen(
            delays=arrays[f"scene/{name}/delays"], amplitudes=arrays[f"scene/{name}/amplitudes"]
        )
        for name in _SCREENS
    }
    scene = Scene(
        imaging=from_dict(ImagingConfig, attrs["imaging"]),
        plane=PlaneSpec(**{name: tuple(value) for name, value in attrs["plane"].items()}),
        pixel_grid=Grid2D(
            shape=tuple(attrs["pixel_grid"]["shape"]),
            spacing=attrs["pixel_grid"]["spacing"],
            origin=tuple(attrs["pixel_grid"]["origin"]),
        ),
        cell_area=attrs["cell_area"],
        effective_frames=attrs["effective_frames"],
        operators={key: arrays[f"scene/operators/{key}"] for key in BEAMFORMERS},
        noise_floor={key: arrays[f"scene/noise_floor/{key}"] for key in BEAMFORMERS},
        meta=attrs["meta"],
        **{name: arrays[f"scene/{name}"] for name in _SCENE_MAPS},
        **banks,
        **screens,
    )
    trials = TrialSet(
        power_doppler={key: arrays[f"trials/power_doppler/{key}"] for key in BEAMFORMERS},
        fidelity=attrs["fidelity"],
        **{name: arrays[f"trials/{name}"] for name in _TRIAL_SERIES},
    )
    return scene, trials
