"""Data acquisition at the two fidelities of docs/ARCHITECTURE.md, section 5.

`scatterer`: explicit blood and tissue scatterers, frame by frame
(`acquire_frames`, `acquire_power_doppler`). `statistical`: the expected image
`H @ cbv` plus a noise floor, with gamma speckle (`statistical_power_doppler`).
Both use the normalisation of `fus.config`, so they can be compared directly
(eval E-FUS-014).

Within a frame all transmits see the same scatterer positions (motion between
the angles of one compound frame is neglected), and frame `n` is taken at
`t = n / frame_rate`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from neurosim.common.grid import Grid2D
from neurosim.common.seeding import SeedTree
from neurosim.fus.beamform import beamform, make_beamformer
from neurosim.fus.config import ImagingConfig, pixel_grid_from_config
from neurosim.fus.doppler import choose_svd_cutoff, power_doppler, svd_clutter_filter
from neurosim.fus.greens import check_same_sampling
from neurosim.fus.noise import electronic_noise, electronic_noise_variance, tissue_displacement
from neurosim.fus.scatterers import BloodState, advance
from neurosim.fus.synthesis import channel_spectra, static_clutter_spectra
from neurosim.fus.types import GreensBank, PlaneWaveSequence, Scatterers
from neurosim.fus.window import check_time_window

# Point sources per element in `reference_greens` (its default).
_N_SUB = 5
# Measured single-thread costs on the development machine (Intel Xeon 2.1 GHz), used only
# by `estimate_cost`. assumed, to verify on the machine that runs the job.
_SECONDS_PER_GREENS_VALUE = 9.0e-9  # per (frequency, point source, scatterer)
_SECONDS_PER_BANK_VALUE = 2.5e-8  # per (frequency, element, scatterer): interpolating B, products
_SECONDS_PER_COMPLEX_FLOP = 2.5e-10  # complex64 multiply-add inside BLAS matrix products
# Real floating-point operations per item, for the reported flop count.
_FLOPS_PER_GREENS_VALUE = 40
_FLOPS_PER_COMPLEX_FLOP = 8


def acquire_frames(
    bank: GreensBank,
    model_bank: GreensBank,
    blood_state: BloodState,
    tissue: Optional[Scatterers],
    sequence: PlaneWaveSequence,
    pixel_grid: Grid2D,
    cfg: ImagingConfig,
    tree: SeedTree,
    frame_offset: int = 0,
) -> np.ndarray:
    """Beamformed IQ frames of one Doppler ensemble at scatterer fidelity.

    Frame `n = frame_offset + i` is the sum of

    - blood: `blood_state` advanced to `t = n / frame_rate`;
    - tissue (`cfg.noise.tissue_clutter`): the given scatterers;
    - rigid micro-motion (`cfg.noise.tissue_motion`): tissue and blood are both
      displaced by `noise.tissue_displacement(t)`, since vessels move with the tissue;
    - static clutter (`cfg.noise.static_clutter`, if the bank has `static_fmc`),
      identical in every frame;
    - electronic noise (`cfg.noise.electronic_noise`), stream
      `us_electronic_noise` with sub-index `n`.

    Frame `n` depends only on `n`, not on which other frames are simulated. The
    parts that do not change from frame to frame are beamformed once, and the
    noise is beamformed separately and added in the image, so the noise-free
    part of a frame is unchanged by the noise to rounding error.

    Args:
        bank: the true medium.
        model_bank: the medium the beamformer assumes.
        blood_state: blood scatterers at t = 0.
        tissue: tissue scatterers, or None for no tissue.
        sequence: steering angles, frame rate and number of frames.
        pixel_grid: image grid.
        cfg: imaging configuration (`noise` and `beamforming` are used).
        tree: seed tree of the run.
        frame_offset: absolute index of the first frame.

    Returns:
        complex128 array, shape (n_frames, nz, nx).
    """
    check_same_sampling(bank, model_bank)
    check_time_window(bank.freqs, bank.transducer, sequence, bank.roi, bank.c_ref)
    beamformer = make_beamformer(pixel_grid, model_bank, sequence, cfg.beamforming)
    noise = cfg.noise
    use_tissue = noise.tissue_clutter and tissue is not None
    moving = noise.tissue_motion and noise.tissue_motion_rms > 0.0
    frame_index = frame_offset + np.arange(sequence.n_frames)
    times = frame_index / sequence.frame_rate
    displacement = (
        tissue_displacement(tree, noise, times) if moving else np.zeros((times.shape[0], 2))
    )

    spectra_shape = (bank.freqs.shape[0], len(sequence.angles_deg), bank.transducer.n_elements)
    static_spectra = np.zeros(spectra_shape, dtype=np.complex128)
    if noise.static_clutter and bank.static_fmc is not None:
        static_spectra += noise.static_clutter_scale * static_clutter_spectra(bank, sequence)
    if use_tissue and not moving:
        static_spectra += channel_spectra(bank, tissue, sequence)
    static_image = beamform(beamformer, static_spectra)
    noise_variance = (
        electronic_noise_variance(bank, sequence, noise) if noise.electronic_noise else 0.0
    )

    frames = np.empty((sequence.n_frames, *pixel_grid.shape), dtype=np.complex128)
    for i, (n, t) in enumerate(zip(frame_index, times)):
        blood = advance(blood_state, t).scatterers()
        xz, amplitude = blood.xz, blood.amplitude
        if use_tissue and moving:
            xz = np.concatenate((xz, tissue.xz))
            amplitude = np.concatenate((amplitude, tissue.amplitude))
        moved = Scatterers(xz=xz + displacement[i], amplitude=amplitude)
        frames[i] = static_image + beamform(beamformer, channel_spectra(bank, moved, sequence))
        if noise.electronic_noise:
            frames[i] += beamform(
                beamformer, electronic_noise(tree, int(n), spectra_shape, noise_variance)
            )
    return frames


def acquire_power_doppler(
    bank: GreensBank,
    model_bank: GreensBank,
    blood_state: BloodState,
    tissue: Optional[Scatterers],
    sequence: PlaneWaveSequence,
    pixel_grid: Grid2D,
    cfg: ImagingConfig,
    tree: SeedTree,
    frame_offset: int = 0,
) -> np.ndarray:
    """Power Doppler image of one ensemble: frames -> SVD clutter filter -> mean power.

    Arguments as for `acquire_frames`; the cutoff follows `cfg.doppler`.

    Returns:
        float64 array, shape (nz, nx).
    """
    frames = acquire_frames(
        bank, model_bank, blood_state, tissue, sequence, pixel_grid, cfg, tree, frame_offset
    )
    return power_doppler(svd_clutter_filter(frames, choose_svd_cutoff(frames, cfg)))


def statistical_power_doppler(
    H: np.ndarray,
    cbv_cells: np.ndarray,
    noise_floor: np.ndarray | float,
    n_frames_effective: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Power Doppler image at statistical fidelity.

    The expected image is `H @ cbv_cells + noise_floor`. A pixel of one frame is
    a circular complex Gaussian (many scatterers with random phases, plus noise),
    so its power is exponentially distributed about that mean; the average of
    `n` independent frames is gamma distributed with shape `n` and the same mean
    (Goodman, Statistical Optics, 1985, on the sum of independent speckle
    intensities).

    Not modelled: the spatial correlation of speckle between neighbouring pixels
    (every pixel is drawn independently), the loss of blood and noise power in
    the SVD filter, and the different number of independent frames of blood
    (slow decorrelation) and noise (new in every frame).

    Args:
        H: power Doppler operator (`operators.pd_operator`), shape (n_pix, n_src).
        cbv_cells: backscatter of each source cell, `cbv * cell_area` [m^2], shape (n_src,)
            or the shape of the source grid.
        noise_floor: image noise power, scalar or shape (n_pix,) / (nz, nx):
            `electronic noise variance * Beamformer.noise_gain` (`fus.noise`).
        n_frames_effective: independent speckle realisations averaged in the image.
        rng: generator of the `speckle_statistical` stream.

    Returns:
        float64 array, shape (n_pix,).
    """
    floor = np.broadcast_to(np.asarray(noise_floor, dtype=np.float64).reshape(-1), (H.shape[0],))
    expected = H @ np.asarray(cbv_cells, dtype=np.float64).reshape(-1) + floor
    return expected * rng.gamma(n_frames_effective, 1.0 / n_frames_effective, size=expected.shape)


def estimate_cost(cfg: ImagingConfig, n_scatterers: int) -> dict[str, float]:
    """Rough cost of producing one power Doppler image, to print before a long run.

    Args:
        cfg: imaging configuration; `cfg.fidelity` selects what is estimated.
        n_scatterers: "scatterer" fidelity: scatterers synthesised per frame (blood,
            plus tissue when it moves). "statistical" fidelity: source cells of the
            CBV grid.

    Returns:
        dict with `n_frames`, `flops` (real floating-point operations), `time_s`
        (single thread) and `memory_bytes` (peak, largest arrays only).
    """
    nf = cfg.frequencies.n_frequencies
    n_elements = cfg.transducer.n_elements
    n_angles = len(cfg.sequence.angles_deg)
    n_pix = int(np.prod(pixel_grid_from_config(cfg).shape))
    weights_bytes = 8 * nf * (n_elements + n_angles) * n_pix  # complex64 beamformer weights

    def greens(n_points: int) -> tuple[float, float]:
        """(flops, seconds) to evaluate the medium Green's function at `n_points` points."""
        values = nf * n_elements * n_points
        return (
            _FLOPS_PER_GREENS_VALUE * _N_SUB * values,
            (_SECONDS_PER_GREENS_VALUE * _N_SUB + _SECONDS_PER_BANK_VALUE) * values,
        )

    setup_flops, setup_time = greens(n_pix)
    if cfg.fidelity == "statistical":
        source_flops, source_time = greens(n_scatterers)
        products = nf * n_pix * n_scatterers * (n_elements + n_angles)
        return {
            "n_frames": 0,
            "flops": setup_flops + source_flops + _FLOPS_PER_COMPLEX_FLOP * products,
            "time_s": setup_time + source_time + _SECONDS_PER_COMPLEX_FLOP * products,
            "memory_bytes": weights_bytes + 4 * 8 * n_pix * n_scatterers,
        }
    n_frames = cfg.sequence.n_frames
    frame_flops, frame_time = greens(n_scatterers)
    das_passes = 2 if cfg.noise.electronic_noise else 1
    products = nf * n_angles * n_elements * (2 * n_scatterers + das_passes * n_pix)
    return {
        "n_frames": n_frames,
        "flops": setup_flops + n_frames * (frame_flops + _FLOPS_PER_COMPLEX_FLOP * products),
        "time_s": setup_time + n_frames * (frame_time + _SECONDS_PER_COMPLEX_FLOP * products),
        "memory_bytes": weights_bytes + 16 * n_frames * n_pix + 3 * 16 * 2_000_000,
    }
