"""Byte-deterministic HDF5 bundles and run provenance.

A bundle is one HDF5 file holding named arrays plus a flat set of attributes.
Writing the same arrays and attributes always gives the same file bytes, so a
file hash identifies a result (eval E-COMMON-001).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from neurosim import __version__
from neurosim.common.config import config_hash, dump_yaml

# Fixed HDF5 format bounds, so the file layout does not follow the installed library.
_LIBVER = ("earliest", "v110")
# Attribute listing the names of the attributes that are stored as JSON text.
_JSON_ATTRS = "__json_attrs__"
# numpy dtype kinds a bundle can hold: bool, signed/unsigned integer, float, complex.
_ARRAY_KINDS = "biufc"
# Directory of the installed `neurosim` package; `git_commit` reports the repository tracking it.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
# `git_commit` when the commit cannot be determined, and its suffix for a modified work tree.
_UNKNOWN_COMMIT = "unknown"
_DIRTY_SUFFIX = "-dirty"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _plain(value: Any, name: str) -> Any:
    """Return ``value`` built from plain Python types only, or raise ``TypeError``."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"attribute {name!r}: nested keys must be strings")
        return {key: _plain(item, f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{name}[{i}]") for i, item in enumerate(value)]
    raise TypeError(f"attribute {name!r}: unsupported type {type(value).__name__}")


def save_bundle(
    path: str | os.PathLike[str],
    arrays: Mapping[str, np.ndarray],
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write arrays and attributes to an HDF5 file, byte-deterministically.

    Parameters
    ----------
    path:
        Output file; parent directories are created, an existing file is replaced.
    arrays:
        Dataset name -> array (bool, integer, float or complex dtype, stored
        as given, uncompressed and contiguous). A ``/`` in a name makes groups.
    attrs:
        Attribute name -> ``str``, ``bool``, ``int`` or ``float``, stored as
        native HDF5 attributes; or a nested ``dict`` / ``list`` / ``tuple`` of
        those (and ``None``), stored as canonical JSON text. Tuples load back
        as lists.

    The same arrays and attributes give identical file bytes, whatever the
    order of the mappings: no timestamps are written, names are written in
    sorted order and the HDF5 format version is fixed.
    """
    plain_attrs = {name: _plain(value, name) for name, value in (attrs or {}).items()}
    if _JSON_ATTRS in plain_attrs:
        raise ValueError(f"attribute name {_JSON_ATTRS!r} is reserved")
    for name, value in plain_attrs.items():
        if value is None:
            raise TypeError(f"attribute {name!r}: None is only allowed inside a nested value")
    data = {}
    for name, array in arrays.items():
        data[name] = np.asarray(array)
        if data[name].dtype.kind not in _ARRAY_KINDS:
            raise TypeError(f"array {name!r}: unsupported dtype {data[name].dtype}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", libver=_LIBVER) as file:
        for name in sorted(data):
            file.create_dataset(name, data=data[name], track_times=False)
        json_names = []
        for name in sorted(plain_attrs):
            value = plain_attrs[name]
            if isinstance(value, (dict, list)):
                json_names.append(name)
                value = _canonical_json(value)
            file.attrs[name] = value
        file.attrs[_JSON_ATTRS] = _canonical_json(json_names)


def load_bundle(path: str | os.PathLike[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a bundle written by `save_bundle`.

    Returns ``(arrays, attrs)``: arrays keyed by their full name (``group/name``)
    with the stored dtype, and attributes as plain Python values.
    """
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as file:

        def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if isinstance(item, h5py.Dataset):
                arrays[name] = np.asarray(item[()])

        file.visititems(collect)
        attrs = _read_attrs(file)
    return arrays, attrs


def load_attrs(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read only the attributes of a bundle written by `save_bundle`, without its arrays.

    Returns the attributes as plain Python values, as `load_bundle` does. The
    cost does not grow with the size of the stored arrays.
    """
    with h5py.File(path, "r") as file:
        return _read_attrs(file)


def _read_attrs(file: h5py.File) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    json_names = set(json.loads(file.attrs[_JSON_ATTRS]))
    for name, value in file.attrs.items():
        if name == _JSON_ATTRS:
            continue
        if name in json_names:
            attrs[name] = json.loads(value)
        else:
            attrs[name] = value.item() if isinstance(value, np.generic) else value
    return attrs


def _git(directory: Path, *arguments: str) -> str | None:
    """Standard output of a read-only git command run in `directory`; None if it fails.

    ``--no-optional-locks`` keeps the command from writing to the repository
    (``git status`` would otherwise refresh the index).
    """
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError:
        # git is not installed, or the directory does not exist.
        return None
    return result.stdout if result.returncode == 0 else None


def git_commit() -> str:
    """Return the commit of the repository that holds this package's source.

    The value is ``git rev-parse HEAD`` of the work tree that tracks the
    package directory, with the suffix ``-dirty`` when any tracked file of
    that work tree has uncommitted changes (``git status --porcelain
    --untracked-files=no`` is not empty; untracked files do not count).

    Returns ``"unknown"`` when git is not installed, when the package is not
    inside a git work tree, or when the work tree around it does not track the
    package (a non-editable install into a ``site-packages`` directory that
    happens to lie inside some other repository).
    """
    located = _git(_PACKAGE_DIR, "rev-parse", "--show-toplevel", "HEAD")
    if located is None:
        return _UNKNOWN_COMMIT
    toplevel, commit = located.splitlines()
    # `--show-toplevel` is the work tree around the package directory; it is the
    # package's own repository only if it also tracks the package source.
    if not _PACKAGE_DIR.is_relative_to(Path(toplevel).resolve()):
        return _UNKNOWN_COMMIT
    if _git(_PACKAGE_DIR, "ls-files", "--error-unmatch", "--", "__init__.py") is None:
        return _UNKNOWN_COMMIT
    changes = _git(Path(toplevel), "status", "--porcelain", "--untracked-files=no")
    if changes is None:
        return _UNKNOWN_COMMIT
    return commit + _DIRTY_SUFFIX if changes.strip() else commit


def provenance(config_obj: Any, root_seed: int) -> dict:
    """Return what is needed to reproduce an output; store it with every output.

    Parameters
    ----------
    config_obj:
        The run's config dataclass instance.
    root_seed:
        Root seed of the run's `SeedTree`.

    Returns a dict with keys ``config_yaml`` (canonical YAML text),
    ``config_hash`` (its SHA-256), ``root_seed``, ``git_commit`` (`git_commit`:
    the commit hash, ``-dirty`` appended when tracked files had uncommitted
    changes, or ``"unknown"``) and ``neurosim_version``, usable directly as
    bundle attributes.
    """
    return {
        "config_yaml": dump_yaml(config_obj),
        "config_hash": config_hash(config_obj),
        "root_seed": int(root_seed),
        "git_commit": git_commit(),
        "neurosim_version": __version__,
    }
