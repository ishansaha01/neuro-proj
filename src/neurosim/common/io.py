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
    attrs: dict[str, Any] = {}
    with h5py.File(path, "r") as file:

        def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if isinstance(item, h5py.Dataset):
                arrays[name] = np.asarray(item[()])

        file.visititems(collect)
        json_names = set(json.loads(file.attrs[_JSON_ATTRS]))
        for name, value in file.attrs.items():
            if name == _JSON_ATTRS:
                continue
            if name in json_names:
                attrs[name] = json.loads(value)
            else:
                attrs[name] = value.item() if isinstance(value, np.generic) else value
    return arrays, attrs


def git_commit() -> str:
    """Return ``git rev-parse HEAD`` of the repository holding this package.

    Returns ``"unknown"`` when git is not installed or the package is not
    inside a git work tree.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def provenance(config_obj: Any, root_seed: int) -> dict:
    """Return what is needed to reproduce an output; store it with every output.

    Parameters
    ----------
    config_obj:
        The run's config dataclass instance.
    root_seed:
        Root seed of the run's `SeedTree`.

    Returns a dict with keys ``config_yaml`` (canonical YAML text),
    ``config_hash`` (its SHA-256), ``root_seed``, ``git_commit`` and
    ``neurosim_version``, usable directly as bundle attributes.
    """
    return {
        "config_yaml": dump_yaml(config_obj),
        "config_hash": config_hash(config_obj),
        "root_seed": int(root_seed),
        "git_commit": git_commit(),
        "neurosim_version": __version__,
    }
