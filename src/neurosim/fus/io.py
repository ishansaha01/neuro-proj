"""Save and load a Green's bank as an HDF5 bundle, alone or as part of a larger bundle.

Layout (`bank_to_bundle`): datasets ``B``, ``freqs`` and, when the bank has it,
``static_fmc``; attributes ``roi_shape``, ``roi_spacing``, ``roi_origin``,
``c_ref``, ``backend``, ``transducer`` (the fields of `fus.types.Transducer`)
and ``meta``. `neurosim.data.io` stores several banks in one run bundle with
the same layout under a name prefix.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.common.io import load_bundle, save_bundle
from neurosim.fus.types import GreensBank, Transducer

_STATIC_FMC = "static_fmc"


def bank_to_bundle(bank: GreensBank) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Arrays and attributes that hold a bank (module docstring)."""
    arrays = {"B": bank.B, "freqs": bank.freqs}
    if bank.static_fmc is not None:
        arrays[_STATIC_FMC] = bank.static_fmc
    attrs = {
        "roi_shape": list(bank.roi.shape),
        "roi_spacing": bank.roi.spacing,
        "roi_origin": list(bank.roi.origin),
        "c_ref": bank.c_ref,
        "backend": bank.backend,
        "transducer": dataclasses.asdict(bank.transducer),
        "meta": dict(bank.meta),
    }
    return arrays, attrs


def bank_from_bundle(arrays: Mapping[str, np.ndarray], attrs: Mapping[str, Any]) -> GreensBank:
    """Rebuild a bank from the arrays and attributes of `bank_to_bundle`."""
    return GreensBank(
        freqs=arrays["freqs"],
        roi=Grid2D(
            shape=tuple(attrs["roi_shape"]),
            spacing=attrs["roi_spacing"],
            origin=tuple(attrs["roi_origin"]),
        ),
        B=arrays["B"],
        c_ref=attrs["c_ref"],
        transducer=Transducer(**attrs["transducer"]),
        backend=attrs["backend"],
        static_fmc=arrays.get(_STATIC_FMC),
        meta=dict(attrs["meta"]),
    )


def save_bank(
    path: str | os.PathLike[str], bank: GreensBank, provenance: Mapping[str, Any]
) -> None:
    """Write a bank to its own HDF5 bundle.

    Args:
        path: output file.
        bank: the bank to store.
        provenance: attributes that say where the bank comes from
            (`common.io.provenance` of the config and seed that produced it).
    """
    arrays, attrs = bank_to_bundle(bank)
    save_bundle(path, arrays, {**provenance, **attrs})


def load_bank(path: str | os.PathLike[str]) -> GreensBank:
    """Read a bank written by `save_bank`."""
    return bank_from_bundle(*load_bundle(path))
