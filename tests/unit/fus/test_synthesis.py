"""Channel-data synthesis."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.fus import synthesis
from neurosim.fus.pulse import pulse_spectrum
from neurosim.fus.reference import reference_greens
from neurosim.fus.screen import apply_phase_screen
from neurosim.fus.synthesis import (
    channel_spectra,
    static_clutter_spectra,
    transmit_delays,
    transmit_field,
    transmit_phasors,
)
from neurosim.fus.types import Scatterers


def _scatterers(roi, n: int, seed: int = 0) -> Scatterers:
    rng = np.random.default_rng(seed)
    z, x = roi.axes()
    xz = np.stack((rng.uniform(x[0], x[-1], n), rng.uniform(z[0], z[-1], n)), axis=1)
    return Scatterers(xz=xz, amplitude=rng.normal(size=n) + 1j * rng.normal(size=n))


def test_transmit_delays(tiny, water):
    delays = transmit_delays(tiny.transducer, tiny.sequence, water.c_ref)
    assert delays.shape == (3, 16)
    np.testing.assert_allclose(delays.min(axis=1), 0.0, atol=1e-18)
    np.testing.assert_allclose(delays[1], 0.0, atol=1e-18)  # 0 degrees: all elements together
    np.testing.assert_allclose(delays[0], delays[2][::-1], rtol=1e-12)  # -5 and +5 degrees mirror
    slope = np.diff(delays[2]) * water.c_ref / tiny.transducer.pitch
    np.testing.assert_allclose(slope, np.sin(np.deg2rad(5.0)), rtol=1e-9)


def test_transmit_phasors_follow_the_sign_convention(tiny, water):
    """A delay tau multiplies the spectrum by exp(+i w tau)."""
    phasors = transmit_phasors(water, tiny.sequence)
    delays = transmit_delays(tiny.transducer, tiny.sequence, water.c_ref)
    assert phasors.shape == (28, 3, 16)
    np.testing.assert_allclose(
        np.angle(phasors[5, 2, 3] / phasors[5, 2, 2]),
        2 * np.pi * tiny.freqs[5] * (delays[2, 3] - delays[2, 2]),
        rtol=1e-9,
    )


def test_channel_spectra_match_the_defining_sum(tiny, water):
    bank = apply_phase_screen(water, np.linspace(-1e-7, 1e-7, 16), np.linspace(0.8, 1.2, 16))
    scatterers = _scatterers(tiny.roi, 4)
    greens = reference_greens(
        tiny.transducer, scatterers.xz, tiny.freqs, water.c_ref
    ) * bank.interp(scatterers.xz)
    transmit = np.einsum("fae,fek->fak", transmit_phasors(bank, tiny.sequence), greens)
    expected = np.einsum(
        "f,k,fak,fjk->faj",
        pulse_spectrum(tiny.transducer, tiny.freqs),
        scatterers.amplitude,
        transmit,
        greens,
    )
    spectra = channel_spectra(bank, scatterers, tiny.sequence)
    assert spectra.shape == (28, 3, 16) and spectra.dtype == np.complex128
    np.testing.assert_allclose(spectra, expected, rtol=1e-5, atol=1e-9 * np.abs(expected).max())


def test_transmit_field_shape_and_value(tiny, water):
    points = tiny.pixel_grid.points()
    field = transmit_field(water, tiny.sequence, points)
    assert field.shape == (28, 3, 8, 8)
    greens = reference_greens(tiny.transducer, points, tiny.freqs, water.c_ref)
    expected = np.einsum("fae,fezx->fazx", transmit_phasors(water, tiny.sequence), greens)
    np.testing.assert_allclose(field, expected, rtol=1e-6)


def test_superposition_and_chunking(tiny, water, monkeypatch):
    scatterers = _scatterers(tiny.roi, 11)
    whole = channel_spectra(water, scatterers, tiny.sequence)
    first = Scatterers(scatterers.xz[:4], scatterers.amplitude[:4])
    rest = Scatterers(scatterers.xz[4:], scatterers.amplitude[4:])
    parts = channel_spectra(water, first, tiny.sequence) + channel_spectra(
        water, rest, tiny.sequence
    )
    np.testing.assert_allclose(whole, parts, rtol=1e-10, atol=1e-12 * np.abs(whole).max())
    monkeypatch.setattr(synthesis, "_CHUNK_VALUES", 3 * 28 * 16)  # three scatterers per pass
    np.testing.assert_allclose(
        channel_spectra(water, scatterers, tiny.sequence),
        whole,
        rtol=1e-10,
        atol=1e-12 * np.abs(whole).max(),
    )


def test_no_scatterers_gives_zeros(tiny, water):
    empty = Scatterers(xz=np.zeros((0, 2)), amplitude=np.zeros(0))
    assert not channel_spectra(water, empty, tiny.sequence).any()


def test_mismatched_amplitudes_raise(tiny, water):
    with pytest.raises(ValueError, match="amplitudes"):
        channel_spectra(water, Scatterers(np.array([[0.0, 0.022]]), np.ones(2)), tiny.sequence)


def test_scatterer_outside_roi_raises(tiny, water):
    with pytest.raises(ValueError, match="outside the bank ROI"):
        channel_spectra(water, Scatterers(np.array([[0.0, 0.030]]), np.ones(1)), tiny.sequence)


def test_static_clutter_is_the_delayed_sum_over_transmit_elements(tiny, water):
    rng = np.random.default_rng(3)
    fmc = (rng.normal(size=(28, 16, 16)) + 1j * rng.normal(size=(28, 16, 16))).astype(np.complex64)
    bank = replace(water, static_fmc=fmc)
    expected = np.einsum(
        "f,fae,fej->faj",
        pulse_spectrum(tiny.transducer, tiny.freqs),
        transmit_phasors(bank, tiny.sequence),
        fmc,
    )
    np.testing.assert_allclose(static_clutter_spectra(bank, tiny.sequence), expected, rtol=1e-6)
    with pytest.raises(ValueError, match="static_fmc"):
        static_clutter_spectra(water, tiny.sequence)
