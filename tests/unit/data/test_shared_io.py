"""fus.io (bank bundles) and the plane-medium part of head.io, the loaders shared by the scripts.

They live with the data tests because `data.io` builds the run bundle on them.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.common.io import load_bundle
from neurosim.fus.io import bank_from_bundle, bank_to_bundle, load_bank, save_bank
from neurosim.fus.types import GreensBank, Transducer
from neurosim.head.io import load_plane_medium, save_plane_medium
from neurosim.head.types import PlaneMedium


def _bank(with_fmc: bool) -> GreensBank:
    rng = np.random.default_rng(1)
    transducer = Transducer(
        n_elements=3,
        pitch=0.385e-3,
        element_width=0.3465e-3,
        center_frequency=2.0e6,
        fractional_bandwidth=0.6,
    )
    roi = Grid2D.from_extent((-0.001, 0.001), (0.020, 0.023), 1.0e-3)
    shape = (4, 3, *roi.shape)
    values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fmc = (rng.standard_normal((4, 3, 3)) + 0j).astype(np.complex64) if with_fmc else None
    return GreensBank(
        freqs=np.linspace(1.5e6, 2.5e6, 4),
        roi=roi,
        B=values.astype(np.complex64),
        c_ref=C_REF,
        transducer=transducer,
        backend="kwave" if with_fmc else "ray",
        static_fmc=fmc,
        meta={"config": {"step_fraction": 1.0}, "ray_step_m": 1.0e-4},
    )


def _assert_same_bank(loaded: GreensBank, bank: GreensBank) -> None:
    assert loaded.roi == bank.roi and loaded.transducer == bank.transducer
    assert loaded.c_ref == bank.c_ref and loaded.backend == bank.backend
    assert loaded.meta == bank.meta
    assert loaded.B.dtype == np.complex64 and loaded.freqs.dtype == np.float64
    np.testing.assert_array_equal(loaded.B, bank.B)
    np.testing.assert_array_equal(loaded.freqs, bank.freqs)
    if bank.static_fmc is None:
        assert loaded.static_fmc is None
    else:
        np.testing.assert_array_equal(loaded.static_fmc, bank.static_fmc)


def test_bank_round_trip_with_and_without_static_echoes(tmp_path):
    for with_fmc in (False, True):
        bank = _bank(with_fmc)
        path = tmp_path / f"bank_{with_fmc}.h5"
        save_bank(path, bank, {"root_seed": 4})
        _assert_same_bank(load_bank(path), bank)
        arrays, attrs = load_bundle(path)
        assert attrs["root_seed"] == 4 and ("static_fmc" in arrays) == with_fmc


def test_bank_bundle_parts_can_be_nested_in_a_larger_bundle():
    bank = _bank(with_fmc=False)
    arrays, attrs = bank_to_bundle(bank)
    assert set(arrays) == {"B", "freqs"}
    prefixed = {f"scene/bank/{name}": value for name, value in arrays.items()}
    restored = {name.removeprefix("scene/bank/"): value for name, value in prefixed.items()}
    _assert_same_bank(bank_from_bundle(restored, attrs), bank)


def test_plane_medium_round_trip(tmp_path, small_run):
    _, extras, _ = small_run
    path = tmp_path / "medium.h5"
    save_plane_medium(path, extras.medium, {"root_seed": 3})
    loaded = load_plane_medium(path)
    assert load_bundle(path)[1]["root_seed"] == 3
    for spec in dataclasses.fields(PlaneMedium):
        stored, original = getattr(loaded, spec.name), getattr(extras.medium, spec.name)
        if isinstance(original, np.ndarray):
            assert stored.dtype == original.dtype, spec.name
            np.testing.assert_array_equal(stored, original)
        else:
            assert stored == original, spec.name


def test_saved_files_are_byte_deterministic(tmp_path, small_run):
    _, extras, _ = small_run
    for name, save, item in (
        ("bank", save_bank, _bank(with_fmc=True)),
        ("medium", save_plane_medium, extras.medium_ct),
    ):
        first, second = tmp_path / f"{name}_a.h5", tmp_path / f"{name}_b.h5"
        save(first, item, {"root_seed": 1})
        save(second, item, {"root_seed": 1})
        assert first.read_bytes() == second.read_bytes()
