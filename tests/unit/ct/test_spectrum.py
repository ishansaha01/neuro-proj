"""xray_spectrum: shapes, units and the way settings scale it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurosim.ct.spectrum import xray_spectrum


def _mean_energy(cfg) -> float:
    energies, fluence = xray_spectrum(cfg)
    return float(energies @ fluence / fluence.sum())


def test_spectrum_shape_and_support(small_cfg):
    energies, fluence = xray_spectrum(small_cfg)
    assert energies.shape == fluence.shape and energies.ndim == 1
    assert np.all(fluence > 0.0)
    assert np.all(np.diff(energies) == pytest.approx(small_cfg.energy_bin_kev))
    assert energies.max() < small_cfg.kvp


def test_spectrum_is_the_spekpy_fluence_per_square_metre(small_cfg):
    import spekpy

    model = spekpy.Spek(
        kvp=small_cfg.kvp,
        th=small_cfg.anode_angle_deg,
        dk=small_cfg.energy_bin_kev,
        z=100.0 * small_cfg.geometry.source_detector_distance,
        mas=1.0,
    )
    model.filter("Al", small_cfg.filtration_al_mm)
    _, fluence = xray_spectrum(small_cfg)
    # get_flu: total photons per cm^2 per mAs at z.
    assert fluence.sum() == pytest.approx(1.0e4 * model.get_flu(), rel=1e-9)


def test_fluence_follows_the_inverse_square_law(small_cfg):
    near = small_cfg.geometry
    geometry = dataclasses.replace(
        near,
        source_detector_distance=2.0 * near.source_detector_distance,
        detector_pitch=2.0 * near.detector_pitch,
    )
    far = dataclasses.replace(small_cfg, geometry=geometry)
    assert xray_spectrum(far)[1].sum() == pytest.approx(
        xray_spectrum(small_cfg)[1].sum() / 4.0, rel=1e-6
    )


def test_more_filtration_hardens_and_weakens_the_beam(small_cfg):
    hard = dataclasses.replace(small_cfg, filtration_al_mm=small_cfg.filtration_al_mm + 4.0)
    assert _mean_energy(hard) > _mean_energy(small_cfg)
    assert xray_spectrum(hard)[1].sum() < xray_spectrum(small_cfg)[1].sum()


def test_monoenergetic_spectrum_keeps_the_fluence(small_cfg):
    mono = dataclasses.replace(small_cfg, monoenergetic_kev=60.0)
    energies, fluence = xray_spectrum(mono)
    assert energies.tolist() == [60.0]
    assert fluence[0] == pytest.approx(xray_spectrum(small_cfg)[1].sum())


def test_returned_arrays_are_copies(small_cfg):
    energies, fluence = xray_spectrum(small_cfg)
    fluence[:] = 0.0
    assert xray_spectrum(small_cfg)[1].sum() > 0.0
