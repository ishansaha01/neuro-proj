import dataclasses
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.special import hankel1

from neurosim.common.constants import C_REF
from neurosim.common.grid import Grid2D
from neurosim.fus.backends import BackendConfig, compute_bank, fullwave
from neurosim.fus.backends.phantoms import Material, homogeneous_medium, slab_medium
from neurosim.fus.io import save_bank
from neurosim.fus.reference import reference_greens
from neurosim.fus.types import Transducer

FREQS = np.linspace(0.3e6, 0.7e6, 5)
WATER = Material(sound_speed=1480.0, density=1200.0)


def kwave(settings) -> BackendConfig:
    return BackendConfig(backend="kwave", kwave=settings)


@pytest.fixture(scope="module")
def water_bank():
    """Bank of homogeneous water (not the reference medium) for three node-wide elements."""
    from neurosim.fus.backends.config import KWaveConfig

    spacing = 0.3e-3
    grid = Grid2D.from_extent((-6.0e-3, 6.0e-3), (0.0, 12.0e-3), spacing)
    transducer = Transducer(3, 4 * spacing, spacing, 0.5e6, 0.8)
    roi = Grid2D.from_extent((-3.0e-3, 3.0e-3), (6.0e-3, 9.0e-3), 1.5e-3)
    settings = KWaveConfig(
        cfl=0.3, pml_size=8, pml_search=4, reverberation_tail=2.0e-6, reference_margin=6
    )
    medium = homogeneous_medium(grid, WATER, alpha_power=1.5)
    bank = compute_bank(medium, transducer, roi, FREQS, kwave(settings))
    return medium, transducer, roi, settings, bank


def test_bank_contract_fields_and_meta(water_bank):
    _, transducer, roi, _, bank = water_bank
    assert bank.backend == "kwave" and bank.c_ref == C_REF and bank.roi == roi
    assert bank.B.shape == (5, 3, 3, 5) and bank.B.dtype == np.complex64
    assert bank.static_fmc.shape == (5, 3, 3) and bank.static_fmc.dtype == np.complex64
    assert np.all(np.isfinite(bank.B)) and np.all(np.isfinite(bank.static_fmc))
    meta = bank.meta
    assert meta["roi_snap_distance_m"] < 1e-9
    assert meta["element_width_on_grid_m"] == pytest.approx(0.3e-3)
    assert meta["time_steps"] == meta["cost_estimate"]["time_steps"]
    assert meta["source_factor_max_relative_deviation"] < 0.02
    assert meta["tail_level_db"] < -40.0
    assert meta["config"]["cfl"] == 0.3
    # Nothing measured on the clock is stored (it would differ from run to run).
    assert "run_time_s" not in meta


def test_every_element_matches_the_analytic_greens_ratio(water_bank):
    _, transducer, roi, _, bank = water_bank
    points = roi.points()
    distance = np.hypot(
        points[None, ..., 0] - transducer.element_x()[:, None, None], points[None, ..., 1]
    )
    k = 2.0 * np.pi * FREQS[:, None, None, None]
    analytic = hankel1(0, k / WATER.sound_speed * distance) / hankel1(0, k / C_REF * distance)
    # Off-centre elements exercise the shifted and mirrored reference.
    np.testing.assert_allclose(bank.B, analytic, rtol=5e-3)
    # exp(-i w t): water is slower than the reference, so the phase grows with frequency.
    assert np.all(np.diff(np.unwrap(np.angle(bank.B), axis=0), axis=0) > 0.0)


def test_static_fmc_is_symmetric_in_a_mirror_symmetric_medium(water_bank):
    bank = water_bank[-1]
    scale = np.abs(bank.static_fmc).max()
    fmc, tolerance = bank.static_fmc, 2e-3 * scale
    np.testing.assert_allclose(fmc, fmc.transpose(0, 2, 1), atol=tolerance)
    np.testing.assert_allclose(fmc[:, 0, 0], fmc[:, 2, 2], atol=tolerance)


def test_reference_medium_gives_unit_bank_and_no_echo(grid, soft, transducer, roi, fast_kwave):
    medium = homogeneous_medium(grid, soft, alpha_power=1.5)
    bank = compute_bank(medium, transducer, roi, FREQS, kwave(fast_kwave))
    np.testing.assert_allclose(bank.B, 1.0, atol=2e-3)
    direct = np.abs(reference_greens(transducer, np.array([[0.0, 4 * 0.3e-3]]), FREQS, C_REF)).max()
    assert np.abs(bank.static_fmc).max() < 5e-3 * direct


def test_the_same_bank_computed_twice_is_saved_as_identical_bytes(
    grid, soft, bone, transducer, roi, fast_kwave, spacing, tmp_path, caplog
):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    provenance = {"root_seed": 0, "config_hash": "fixed"}
    for name in ("first.h5", "second.h5"):
        with caplog.at_level(logging.INFO, logger="neurosim.fus.backends"):
            bank = compute_bank(medium, transducer, roi, FREQS, kwave(fast_kwave))
        save_bank(tmp_path / name, bank, provenance)
    assert (tmp_path / "first.h5").read_bytes() == (tmp_path / "second.h5").read_bytes()
    # The measured run time goes to the log instead of the stored metadata.
    measured = [r.getMessage() for r in caplog.records if "computed in" in r.getMessage()]
    assert len(measured) == 2


def test_worker_processes_are_spawned_with_single_threaded_libraries(monkeypatch):
    start_methods = []

    class RecordingExecutor(ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            start_methods.append(kwargs["mp_context"].get_start_method())
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(fullwave, "ProcessPoolExecutor", RecordingExecutor)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    names = fullwave._THREAD_VARIABLES
    assert {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"} <= set(names)

    # Each worker reports the thread variables of its own environment.
    in_workers = list(fullwave._map_in_worker_processes(os.getenv, names, workers=2))

    assert start_methods == ["spawn"]
    assert in_workers == ["1"] * len(names)
    # The environment of this process is as it was.
    assert os.environ["OMP_NUM_THREADS"] == "3" and "OPENBLAS_NUM_THREADS" not in os.environ


def test_worker_processes_give_the_same_bank(water_bank):
    medium, transducer, roi, settings, serial = water_bank
    parallel = compute_bank(
        medium, transducer, roi, FREQS, kwave(dataclasses.replace(settings, n_workers=2))
    )
    np.testing.assert_array_equal(parallel.B, serial.B)
    np.testing.assert_array_equal(parallel.static_fmc, serial.static_fmc)


def test_single_precision_agrees_with_double(water_bank):
    medium, transducer, roi, settings, double = water_bank
    single = compute_bank(
        medium, transducer, roi, FREQS, kwave(dataclasses.replace(settings, dtype="float32"))
    )
    np.testing.assert_allclose(single.B, double.B, rtol=1e-3)


def test_elements_centred_between_nodes(grid, fast_kwave, spacing):
    # Two elements, 3 cells apart and 2 nodes wide: centres half-way between nodes.
    transducer = Transducer(2, 3 * spacing, 2 * spacing, 0.5e6, 0.8)
    roi = Grid2D.from_extent((-3.0e-3, 3.0e-3), (6.0e-3, 9.0e-3), 3.0e-3)
    medium = homogeneous_medium(grid, WATER, alpha_power=1.5)
    bank = compute_bank(medium, transducer, roi, FREQS, kwave(fast_kwave))
    # Each element is two line sources on its nodes, in the medium and in the reference.
    analytic = reference_greens(
        transducer, roi.points(), FREQS, WATER.sound_speed, n_sub=2
    ) / reference_greens(transducer, roi.points(), FREQS, C_REF, n_sub=2)
    np.testing.assert_allclose(bank.B, analytic, rtol=5e-3)


def test_weak_reference_is_refused(grid, soft, transducer, roi, fast_kwave):
    medium = homogeneous_medium(grid, soft, alpha_power=1.5)
    strict = dataclasses.replace(fast_kwave, min_reference_level_db=-0.5)
    with pytest.raises(ValueError, match="ill-conditioned"):
        compute_bank(medium, transducer, roi, FREQS, kwave(strict))


def test_cost_is_logged_before_running_and_a_ringing_tail_is_reported(
    grid, soft, bone, transducer, roi, fast_kwave, spacing, caplog
):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.5)
    # A limit far below the reverberation left after 2 us, so that the warning must fire.
    strict = dataclasses.replace(fast_kwave, max_tail_level_db=-150.0)
    with caplog.at_level(logging.INFO, logger="neurosim.fus.backends"):
        bank = compute_bank(medium, transducer, roi, FREQS, kwave(strict))
    messages = [record.getMessage() for record in caplog.records]
    assert "estimated cost" in messages[0] and "'n_simulations': 4" in messages[0]
    assert any("increase kwave.reverberation_tail" in message for message in messages)
    assert -150.0 < bank.meta["tail_level_db"] < 0.0
