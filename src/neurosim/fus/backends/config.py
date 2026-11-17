"""Typed settings of the Green's-bank backends (docs/ARCHITECTURE.md, section 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from neurosim.common.config import field

_MEBIBYTE = 1024**2


@dataclass(frozen=True)
class RayConfig:
    """Settings of the straight-ray backend."""

    step_fraction: float = field(
        1.0,
        gt=0.0,
        le=1.0,
        doc="Upper bound on the ray sampling step, as a fraction of the medium grid spacing dx "
        "[dimensionless]. Per interface crossed, the travel-time error of the midpoint rule on "
        "the bilinear slowness map is at most step^2 / (8 dx) times the slowness jump, i.e. "
        "dx / 8 of path at 1.0 (about 3 ns for bone in soft tissue at dx = 0.08 mm). Modelling "
        "choice: 1.0 keeps a 128-element bank at tens of seconds.",
    )
    max_chunk_bytes: int = field(
        128 * _MEBIBYTE,
        gt=0,
        doc="Memory budget for the ray sample coordinates of one chunk of rays [bytes]. "
        "Numerical setting, no effect on the result.",
    )
    seconds_per_sample: float = field(
        2.5e-7,
        gt=0.0,
        doc="Run-time model: seconds per ray sample (three medium lookups) on one core. "
        "Measured on the 2-core development machine.",
    )
    seconds_per_bank_value: float = field(
        1.0e-7,
        gt=0.0,
        doc="Run-time model: seconds per bank value (one complex exponential) on one core. "
        "Measured on the 2-core development machine.",
    )


@dataclass(frozen=True)
class KWaveConfig:
    """Settings of the full-wave backend (k-wave-python, pure-Python k-space solver)."""

    min_points_per_wavelength: float = field(
        3.0,
        ge=2.0,
        doc="Smallest allowed number of grid points per wavelength at the top frequency and the "
        "slowest sound speed; a coarser medium grid is refused. 3 gives < 1 % error in the "
        "transmission coefficient of a sound-speed step (Treeby, Jaros, Rendell & Cox 2012, "
        "J. Acoust. Soc. Am. 131(6):4324-4336, sec. V B); 2 is the Nyquist limit.",
    )
    cfl: float = field(
        0.1,
        gt=0.0,
        lt=0.45,
        doc="Courant number c_max dt / dx [dimensionless]. Upper bound: the 2D pseudospectral "
        "stability limit 2 / (pi sqrt(2)) = 0.45 used by k-Wave's own check. Default 0.1 is "
        "assumed, to verify: k-Wave's default is 0.3, but the phase error of the k-space "
        "scheme in soft tissue next to bone grows with cfl^2 (see `phase_error_estimate`).",
    )
    pml_size: int = field(
        20,
        ge=4,
        doc="Smallest thickness of the perfectly matched layer [grid points]; the layer is "
        "placed outside the medium grid and may be thickened by up to `pml_search` points "
        "per axis to give FFT-friendly sizes. Every grid axis must be longer than twice this. "
        "20 is k-Wave's default.",
    )
    pml_search: int = field(
        12,
        ge=1,
        doc="Extra layer thickness that may be added to make the grid sizes FFT-friendly "
        "[grid points]. Numerical setting.",
    )
    pml_alpha: float = field(
        2.0,
        gt=0.0,
        doc="Absorption of the perfectly matched layer [Np per grid point]. k-Wave's default.",
    )
    dtype: Literal["float32", "float64"] = field(
        "float64",
        doc="Precision of the solver fields. float32 halves the memory; float64 is the "
        "accuracy reference.",
    )
    reverberation_tail: float = field(
        20.0e-6,
        ge=0.0,
        doc="Simulated time after the direct arrival at the farthest ROI point [s], so that "
        "reverberation inside the skull has decayed. 20 us is 4 to 5 round trips of a 6 mm "
        "bone layer at 2800 m/s (4.3 us each), each of which leaves about 0.3 of the amplitude "
        "for a bone-like impedance step; assumed, to verify against `meta['tail_level_db']`.",
    )
    pulse_edge_level_db: float = field(
        6.0,
        gt=0.0,
        doc="Level of the drive-pulse amplitude spectrum at the lowest and highest requested "
        "frequency, below its peak [dB]. The pulse cancels in the bank; modelling choice.",
    )
    pulse_max_cycles: float = field(
        2.0,
        gt=0.0,
        doc="Upper bound on the Gaussian envelope standard deviation of the drive pulse "
        "[carrier periods], which keeps the pulse short for a narrow frequency list. "
        "Modelling choice.",
    )
    min_reference_level_db: float = field(
        -40.0,
        lt=0.0,
        doc="Lowest allowed reference pressure spectrum at any used point, relative to the "
        "strongest point at the same frequency [dB]; below it the ratio G_medium / G_ref is "
        "refused. Assumed, to verify: set near the level of boundary-layer reflections.",
    )
    max_tail_level_db: float = field(
        -40.0,
        lt=0.0,
        doc="A warning is logged when the pressure in the last 5 % of a recording exceeds this "
        "level relative to the recording's peak [dB] (reverberation tail too short). Assumed.",
    )
    reference_margin: int = field(
        16,
        ge=0,
        doc="Grid points between the outermost source or sensor column and the absorbing "
        "layer in the reference simulation [grid points]. Numerical setting.",
    )
    max_recording_bytes: int = field(
        1024 * _MEBIBYTE,
        gt=0,
        doc="Memory budget for the recorded time series of one simulation [bytes]. The "
        "reference recording is split over several simulations to respect it.",
    )
    model_dispersion: bool = field(
        False,
        doc="Include the dispersion term of k-Wave's power-law absorption. Default False: the "
        "sound_speed map is the phase speed in the imaging band, whereas the dispersion term "
        "takes it as the low-frequency limit and adds 2 alpha0 c0^y tan(pi y / 2) k^(y-1) on "
        "top. For a bone-like slab (2800 m/s, 8 dB/(cm MHz^y)) simulated at 0.5 MHz the phase "
        "speed then comes out 21 % above the map for y = 1.1 (E-FUS-008), and the loss 2 to "
        "6 % below alpha0 f^y for y = 1.5 (E-FUS-003). Without the term the phase speed stays "
        "on the map and the loss follows the power law to about 1 %, at the price of a "
        "non-causal medium. True is refused when alpha_power is within 0.05 of 1, where the "
        "term is singular.",
    )
    n_workers: int = field(
        1,
        ge=1,
        doc="Number of worker processes; one simulation per array element is distributed "
        "over them.",
    )
    seconds_per_point_step: float = field(
        1.6e-8,
        gt=0.0,
        doc="Run-time model: seconds per grid point per time step per log2(grid points) on one "
        "core, float64, lossless medium. Measured on the 2-core development machine: 0.17 s per "
        "step at 547 000 points (1.6e-8); smaller grids are faster per point (1.0e-8 at 80 000).",
    )


@dataclass(frozen=True)
class BackendConfig:
    """Which backend computes the Green's bank, and its settings."""

    backend: Literal["analytic", "ray", "kwave"] = field(
        "ray", doc="Backend name (docs/ARCHITECTURE.md, section 5)."
    )
    ray: RayConfig = field(default_factory=RayConfig, doc="Settings of the `ray` backend.")
    kwave: KWaveConfig = field(
        default_factory=KWaveConfig, doc="Settings of the `kwave` backend."
    )
