# Roadmap

This file is the project's working plan and its record of every change. It has three parts:

1. **[Milestones](#1-milestones)**: what we are building, phase by phase, with checkboxes.
2. **[Design decisions](#2-design-decisions)**: numbered decisions (D-xxx), each with its reason and status. Change a decision by adding a new entry that supersedes the old one, not by editing history.
3. **[Change log](#3-change-log)**: one entry per change to the repo, newest first: what changed, why, and which files.

**Rules for every change**
- Each change gets a change-log entry here in the same commit.
- Each simulation output must be reproducible from `(config file, seed, code version)` alone.
- Each physical constant in the code names its source (paper, table, or "assumed, to verify").
- No hidden randomness: every random draw goes through the seed tree (D-004).

---

## 1. Milestones

**Wednesday goal (18 Nov 2026):** a complete synthetic simulation with a 3D ground-truth head, a fUS sensor model and a CT scanner model, all driven by one config and one seed, ready to produce data.

### Phase 0 — Project setup *(Mon 16 Nov)*
- [x] Repo created and cloned; proposal page 1 added
- [x] README.md and ROADMAP.md
- [ ] Python toolchain: this Mac only has system Python 3.9. Install `uv` and pin Python 3.11+ (needs your OK, since it downloads an installer)
- [x] `pyproject.toml` with pinned dependencies: numpy, scipy, pyyaml, h5py or zarr, matplotlib, k-wave-python, scikit-image; optional jax and jwave
- [x] Package skeleton `src/neurosim/` with `common/` (seed tree, units, config schema, I/O)
- [x] Determinism test: same config and seed give byte-identical outputs

### Phase 1 — Ground-truth head *(Tue 17 Nov)*
Output: one HDF5/Zarr file per head holding 3D voxel maps on a common grid (default 0.5 mm isotropic), plus the config that made it.

- [ ] **Anatomy (procedural):** nested ellipsoids for scalp, skull, CSF and brain. Smooth random deformation (Gaussian random field) of each surface, so every seed gives a different head shape. Skull thickness map with realistic regional variation, thinner at the temporal window.
- [ ] **Skull microstructure:** three-layer skull (outer table, diploë, inner table). Porosity field φ(x) that is low in the tables and high in the diploë, with spatially correlated random texture.
- [ ] **Material mapping:** derive every physical property from the microstructure (D-002):
  - acoustic: sound speed c, density ρ, attenuation α (porosity mixing law, Aubry et al., 2003);
  - X-ray: volume fractions of soft tissue, water and bone mineral → μ(E) (used in Phase 3).
  - Hidden variation: add a component of sound speed that the CT cannot see, tuned so CT explains about 23–53% of sound-speed variance (Webb et al., 2018). This is what makes the CT an *imperfect* prior.
- [ ] **CBV ground truth:** baseline vascular density map (higher in grey matter); a target region (default: a cortical patch, analogous to the parietal/motor areas decoded in Norman et al., 2021); task design → hemodynamic response function → ΔCBV(t) time series in the target. Blood speed field for the Doppler model.
- [ ] **Real-data skull path (optional):** load skull geometry from MUG500+ (meshes and segmentations) or HU volumes from CQ500, so procedural and real skulls share one interface.
- [ ] **Generative skull path (stretch):** SkullGAN (Naftchi-Ardebili et al., 2023) generates 2D skull CT slices; check whether weights are public and whether it can feed 2D imaging planes.
- [ ] Sanity checks: layer thicknesses, property ranges and CT-vs-sound-speed R² checked against literature values; orthogonal-slice figures.

### Phase 2 — fUS sensor model *(Tue 17 – Wed 18 Nov)*
Output: per run, raw channel data (RF), beamformed IQ frames, a power Doppler image series, and the matching no-skull run.

- [ ] **Imaging plane:** take a 2D plane through the 3D head under the probe (D-003). The probe position and angle are config parameters.
- [ ] **Transducer:** 128-element linear array; centre frequency 2 MHz (sweep 1–3 MHz); pitch and element width set from wavelength; Gaussian-modulated pulse with set bandwidth; element directivity.
- [ ] **Transmit sequence:** 7–11 plane waves over ±10°, coherent compounding; PRF and compounded frame rate set by imaging depth; Doppler ensemble length (e.g. a few hundred compounded frames per CBV image).
- [ ] **Scatterers:** static tissue scatterers (speckle) plus moving blood scatterers whose number density follows CBV and which move with the blood speed field. Blood is a weak scatterer compared with tissue.
- [ ] **Acoustic solver:** k-wave-python, pseudo-spectral, with heterogeneous c, ρ and power-law absorption (D-001). The same medium without the skull gives the paired clean run.
- [ ] **Electronic noise:** additive, seeded, set by an SNR parameter.
- [ ] **Processing chain:** delay-and-sum beamforming (assuming the brain's speed of sound, as a real scanner does), compounding, SVD clutter filter (Demené et al., 2015), power Doppler → CBV image.
- [ ] **Noise ablation switches:** each skull effect on or off on its own, for example a homogeneous skull with the same mean properties (no aberration), zero bone absorption (no attenuation), matched impedance (no reverberation).
- [ ] **Fast path (from the proposal):** precompute the skull operator once per element and angle, then synthesize the echoes for any blood configuration linearly (Born approximation). Check against the full simulation on a small case. Estimate memory before building.
- [ ] Sanity checks: point-spread function in water against theory; arrival-time shifts through a homogeneous slab against analytic values; attenuation against α·f·d; activation recovered in the no-skull case.

### Phase 3 — CT scanner model *(Tue 17 – Wed 18 Nov)*
Output: simulated CT volume in HU from the same head, plus the scan settings used.

- [ ] **Source:** polyenergetic spectrum for a given kVp and filtration (SpekPy), default 120 kVp.
- [ ] **Attenuation:** μ(E) per voxel from material fractions and mass-attenuation tables (NIST XCOM) for soft tissue, water and cortical bone.
- [ ] **Projection:** fan-beam geometry per slice to start (3D cone beam later), ray-driven line integrals, polyenergetic Beer–Lambert, so beam hardening appears without extra modelling.
- [ ] **Detector:** Poisson photon noise for a given tube current (mAs) and detector size; optional electronic noise.
- [ ] **Reconstruction:** filtered back-projection with a choice of kernels (ramp, Shepp–Logan, Hann); calibration to HU with water and air.
- [ ] **HU → acoustics:** the clinical linear HU-to-density/sound-speed map (Marquet et al., 2009), with constants randomized within published error bounds (Webb et al., 2018). This is the CT prior the reconstruction will use.
- [ ] Sanity checks: water ≈ 0 HU and air ≈ −1000 HU; cortical bone HU in the published range; HU shifts with kVp and kernel in the direction Webb et al. report; R² of CT-derived vs true sound speed in the 23–53% band.

### Phase 4 — Dataset generation *(after Wednesday)*
- [ ] Sweep driver: grid or random sweeps over skull seed, frequency, probe pose, target region and noise switches.
- [ ] Paired dataset format: clean CBV, corrupted CBV, raw channel data, CT volume, ground-truth skull, and task labels per frame.
- [ ] Brain activity from real task recordings (Rabut et al., 2024; Caltech primate data) instead of the synthetic HRF, so every frame carries a real label.

### Phase 5 — Reconstruction and evaluation *(after Wednesday)*
- [ ] Baselines: no skull (upper bound); delay-and-sum with SVD filtering; CT-derived delay correction; true-skull correction.
- [ ] Diffusion prior on clean CBV images.
- [ ] Known-skull DPS, then uncertain-skull BlindDPS, then raw-channel guidance (stretch).
- [ ] Metrics: decoding accuracy, activation-map overlap, CBV correlation, contrast, uncertainty calibration, runtime.
- [ ] Robustness: elastic solver with shear waves, perturbed HU maps, a different solver; then real transcranial data.

---

## 2. Design decisions

**D-001 — Acoustic solver: k-wave-python first, j-Wave as an option.** *Proposed, 16 Nov.*
The proposal names k-Wave, the standard for transcranial simulation, and k-wave-python ships a C++ OpenMP binary for Apple Silicon. j-Wave is a JAX port of k-Wave and is differentiable, which will matter later when the reconstruction needs gradients through the physics (raw-channel guidance). The solver sits behind one interface so we can switch.

**D-002 — One latent microstructure drives both CT and acoustics.** *Proposed, 16 Nov.*
If we set sound speed straight from HU, the CT would be a perfect prior and the main result would be meaningless. Instead, the ground truth is porosity plus composition, and both the X-ray attenuation and the acoustic properties are computed from it. A hidden component of sound speed is added that X-rays cannot see, tuned so CT explains 23–53% of sound-speed variance (Webb et al., 2018). The CT model then has to recover the skull through a realistic, imperfect measurement.

**D-003 — Anatomy in 3D, ultrasound in 2D planes.** *Proposed, 16 Nov.*
At 2 MHz a full-wave grid needs about 6 points per wavelength at the top of the band, roughly 0.08 mm spacing. A full 3D head at that spacing has on the order of 10⁹ voxels, which is not practical on a laptop (Apple M5 Pro, 24 GB). A 2D plane through the head, about 100 mm deep by 60 mm wide, is about 10⁶ grid points, which runs in seconds to minutes. A linear array also images a single plane, which is how real fUS works, and the proposal's simulation is 2D. The head stays 3D so that probe placement, out-of-plane skull shape and the CT are all real 3D objects. 3D acoustic runs on small cropped volumes are a later option.

**D-004 — Determinism through a seed tree.** *Proposed, 16 Nov.*
One root seed per run, split with `numpy.random.SeedSequence.spawn` into named streams: `anatomy`, `skull_microstructure`, `hidden_soundspeed`, `cbv`, `tissue_scatterers`, `blood_scatterers`, `us_electronic_noise`, `ct_photon_noise`. Changing one stream, for example the ultrasound noise, leaves every other part of the run bit-identical. Every output stores its config, its seed and the git commit.

**D-005 — Every noise effect is a switch.** *Proposed, 16 Nov.*
Aberration, attenuation, reverberation, tissue clutter, electronic noise and CT photon noise can each be turned on or off and scaled from the config. This allows clean ablations: how much of the image loss comes from each effect.

---

## 3. Change log

### 2026-11-17 — fUS imaging chain: synthesis, beamforming, Doppler, operators (Phase 2)
- **What:** Added the imaging chain on top of `GreensBank`: typed `ImagingConfig` with range-aliasing validation (`fus/config.py`, `fus/window.py`); Gaussian two-way pulse and frequency grid (`fus/pulse.py`); medium Green's function G = B * G_ref (`fus/greens.py`); Born channel synthesis and static skull clutter (`fus/synthesis.py`); delay-and-sum as phase conjugation against a model bank, with f-number and Hann options and an analytic noise gain (`fus/beamform.py`); SVD clutter filter, cutoff rules and power Doppler (`fus/doppler.py`); `psf_operator` and `pd_operator` built from the same factors as synthesis and DAS (`fus/operators.py`); blood scatterers (density proportional to CBV) and tissue scatterers (`fus/scatterers.py`); electronic noise and rigid tissue micro-motion on sub-indexed streams (`fus/noise.py`); phase-screen banks (`fus/screen.py`); `acquire_frames`, `acquire_power_doppler`, `statistical_power_doppler`, `estimate_cost` (`fus/acquire.py`); `scripts/run_fus.py`; `configs/fus_default.yaml` (128 elements, 2 MHz, 9 angles over ±10°) and `configs/fus_small.yaml`. `fus/reference.py` now uses the large-argument Hankel expansion where k r >= 20 (equal to `hankel1` to 1e-11, about 20 times faster).
- **Why:** Phase 2 (transmit sequence, scatterers, electronic noise, processing chain, fast path) and both data fidelities of D-007.
- **Design notes (see D-011, D-013):** `das` and `psf_operator` take a model bank instead of a delay table. The beamformer is normalised to unit scatterer gain under its model, so a corrected beamformer restores the peak by construction and pays for it in noise gain where the model field is weak. Blood scatterers carry random signs so the mean density does not reflect coherently, and move on re-entrant paths so density stays proportional to CBV. Static skull clutter is folded into the ROI time window, not time-gated (an upper bound on what a gated scanner sees).
- **Limits:** Channel data exist only as spectra; there is no time-domain RF output. The default config costs roughly 20 s per frame with 12 000 scatterers, about an hour per 200-frame power Doppler image, so sweeps must use the statistical fidelity. The statistical tier draws pixels independently (no spatial speckle correlation) and does not model SVD loss. No eval yet checks `static_fmc` clutter or tissue micro-motion against an independent reference.
- **Files:** `src/neurosim/fus/{config,pulse,window,greens,synthesis,beamform,doppler,operators,scatterers,noise,screen,acquire,reference}.py`, `scripts/run_fus.py`, `configs/fus_default.yaml`, `configs/fus_small.yaml`, `tests/unit/fus/*`, `evals/fus/test_e_fus_010..018_*.py`, `docs/ARCHITECTURE.md`.
- **Evals added (all passing):** E-FUS-010 PSF widths against diffraction theory (within 3 %); E-FUS-011 localisation (0.19 pixel); E-FUS-012 SVD clutter filter (61 dB gain, 97 % of blood kept); E-FUS-013 operator against the explicit path (3e-7); E-FUS-014 statistical against scatterer tier (mean +0.3 %); E-FUS-015 activation recovery in water (+40.6 % for +40 % CBV); E-FUS-016 aberration correction with the true bank (0.1 dB from water; conventional DAS loses at least 5.7 dB); E-FUS-017 determinism and noise-stream isolation; E-FUS-018 noise-floor formula. Two evals failed on their first run and passed with unchanged thresholds after fixes: E-FUS-014 (all-positive blood amplitudes made the mean density a coherent reflector; fixed in the simulator) and E-FUS-015 (the eval's uniform flow field gave only 2–3 independent speckle realisations; fixed in the eval set-up). Unit tests: 98 passing.

### 2026-11-17 — Green's-bank backends (Phase 2, acoustic solver)
- **What:** Added `fus/backends` with `compute_bank` and `estimate_cost` behind one interface and three backends. `analytic`: the reference medium, B = 1. `ray`: straight rays with travel time, power-law absorption, flux-normalised interface transmission and a paraxial spreading factor; exposes `ray_travel_times`. `kwave`: k-wave-python 0.6.2 pure-Python solver, one simulation per element, referenced to the same solver in the homogeneous reference medium (shifted and mirrored across elements), direct DFT at the requested frequencies, and `static_fmc` from medium-minus-reference recordings at the array. Typed `BackendConfig`, layered test phantoms, and `scripts/compute_bank.py`.
- **Why:** The bank B = G_medium / G_ref is the only interface between the skull acoustics and the imaging chain (D-006, D-007).
- **Decisions (see D-008, D-012):** k-Wave's dispersion term is off by default. With it on, bone-like absorption at y = 1.1 raises the simulated phase speed about 21 % above the sound-speed map and the loss falls below the power law, so the reference tier would not simulate the medium that `PlaneMedium` describes. Default CFL is 0.1 for the reference tier because of the k-space phase error next to bone.
- **Limits:** Full-resolution full-wave is far beyond a laptop: about 10^6 grid points and 23 000 steps is an extrapolated 2 to 2.5 hours per element on one core, roughly 300 core-hours for 128 elements. The `ray` backend has no refraction and no reverberation. The ratio G_medium / G_ref is ill-conditioned in the directivity nulls of elements wider than a wavelength (the k-Wave backend raises there).
- **Files:** `src/neurosim/fus/backends/*`, `scripts/compute_bank.py`, `tests/unit/fus/backends/*`, `evals/fus/test_e_fus_001..008_*.py`.
- **Evals added (all passing):** E-FUS-001 k-Wave against the analytic Green's ratio (1.1e-4); E-FUS-002 slab delay (ray 0.40 %, k-Wave 0.21 %); E-FUS-003 slab attenuation (ray 0.08 %, k-Wave 1.0 %); E-FUS-004 interface transmission loss; E-FUS-005 ray against k-Wave phase (0.107 rad rms); E-FUS-006 reverberation switch (−21.4 dB against −20 dB, a thin margin); E-FUS-007 reciprocity; E-FUS-008 absorption keeps the sound speed. **E-FUS-003 first failed** at 5.65 % against 5 % with the dispersion term on; the threshold was kept and the backend default changed as described above. Unit tests: 79 passing.

### 2026-11-17 — CT-derived skull in the imaging plane; shared constants
- **What:** Added `neurosim.data.ct_prior`. The head's basis-material fractions are resampled onto a thin slab aligned with the imaging plane (a tilted-gantry acquisition; voxels outside the head volume are air) and scanned with `ct.simulate_ct_fractions`, so the central CT slice is the imaging plane. `ct_plane_hu` interpolates the slab onto the fine acoustic grid and `ct_plane_medium` applies `ct.acoustics.hu_to_acoustic` to give a `PlaneMedium` (bone label where HU >= 300, CT-estimated porosity, no hidden sound speed, no CBV). Nominal or sampled HU mapping; cost estimate; `configs/ct_prior_default.yaml`. Removed duplication: constants typed in both `head` and `ct` moved to `common/constants.py` (NIST ICRU-44 densities 1060 and 1920 kg/m^3, the Aubry 2003 sound-speed end points and absorption exponent, the default absorption power-law exponent 1.1).
- **Why:** Phase 3 "HU -> acoustics" and the CT-derived skull prior of the reconstruction (D-002); scanning every axial slice an oblique plane crosses would take tens of minutes. One source of truth for constants (CLAUDE.md section 2).
- **Limits:** The slab is scanned as an isolated block about 6 cm across, so photon noise is about 4 HU and beam hardening is weaker than in a head scan; the CT noise switch therefore barely affects this prior. The returned `CtVolume` is in the slab frame (stored in `meta["slab_frame"]`). CT-derived skull speed is 68–88 m/s low on average.
- **Files:** `src/neurosim/data/ct_prior.py`, `configs/ct_prior_default.yaml`, `tests/unit/data/test_ct_prior.py`, `evals/data/`, `src/neurosim/common/constants.py`, `src/neurosim/head/materials.py`, `src/neurosim/ct/attenuation.py`, `src/neurosim/ct/config.py`, `docs/ARCHITECTURE.md`.
- **Evals:** E-DATA-001 **passing**, at the low end of the band: R^2 of CT-derived against true skull sound speed is 0.29, 0.25 and 0.26 for seeds 0–2 (band 0.23–0.53). The loss from the latent 0.5 is mostly spatial resolution (0.5 mm voxels and CT blur against 1–1.5 mm tables), not noise, kVp or mapping constants. E-DATA-002 **failing and left failing**: the largest skull-interface error along the central line is 0.54 mm against the 0.5 mm threshold. It is not a registration error (the skull centre agrees within 0.07 mm): thresholding blurred CT at 300 HU makes the bone label about 0.9 mm too thick. Head and CT evals re-run after the constant move, values identical. Unit tests: 34 new, all passing.

### 2026-11-17 — CT scanner model (Phase 3)
- **What:** Added `neurosim.ct`: SpekPy tungsten spectrum (kVp, Al filtration, mAs to photons per detector element per view); basis-material attenuation from xraydb with NIST compositions (air, water, ICRU-44 soft tissue and cortical bone); a vectorised ray-driven fan-beam projector (equiangular detector, chunked over views); polyenergetic Beer–Lambert on the sinogram with an energy-integrating detector, so beam hardening appears by itself; Poisson photon noise per slice from `("ct_photon_noise", iz)`; a water beam-hardening correction; fan-beam filtered back-projection (Kak & Slaney ch. 3) with ramp, Shepp–Logan and Hann kernels; HU calibration from a water and air scan at the same settings; the Aubry 2003 HU to (sound speed, density, absorption) map with nominal and randomised constants (`ct_hu_mapping` stream); a cost estimate; HDF5 I/O with provenance; `scripts/run_ct.py`; `configs/ct_default.yaml` (clinical geometry, 8 mm slab) and `configs/ct_small.yaml`.
- **Why:** Phase 3. The CT-derived skull is the deliberately imperfect prior for reconstruction (D-002); kVp and kernel are adjustable because both shift the HU-to-sound-speed relation (Webb 2018).
- **Limits:** No bow-tie filter, heel effect, scatter, electronic noise or detector blur; slices are infinitely thin. The isocentre is fixed at head-frame x = y = 0 and a box-region head is scanned as an isolated block. The nominal Aubry 2003 absorption gives a CT-derived skull absorption about twice the head's ground truth (configurable; see D-014). The Marquet 2009 constants could not be checked against the paper.
- **Files:** `src/neurosim/ct/{config,spectrum,attenuation,projector,recon,detector,calibration,simulate,acoustics,io,phantoms}.py`, `scripts/run_ct.py`, `configs/ct_default.yaml`, `configs/ct_small.yaml`, `tests/unit/ct/`, `evals/ct/`.
- **Evals added (all passing):** E-CT-001 HU calibration (water +0.2 HU, air −998 HU); E-CT-002 attenuation against NIST; E-CT-003 projector chords; E-CT-004 monoenergetic FBP; E-CT-005 cortical-bone HU (1624 HU at 120 kVp); E-CT-006 kVp dependence (80:140 kVp ratio 1.44, against 1.42–1.53 in Webb 2018); E-CT-007 noise scaling; E-CT-008 kernel effect; E-CT-009 determinism and slice-noise isolation; E-CT-010 beam hardening and its correction; E-CT-011 porosity recovery. **E-CT-003 was redesigned after failing:** on a 2 mm grid it gave 1.85 % against the 1 % threshold, from the pixel representation of a sharp edge for grazing rays, not from the projector. The threshold was kept and the eval moved to the 0.5 mm production pixel size (0.39 %). Unit tests: 83 passing.

### 2026-11-17 — Ground-truth head package (Phase 1)
- **What:** Added `neurosim.head`: a procedural, seeded 3D head. Anatomy is a deformed-ellipsoid scalp with six strictly positive smooth thickness maps (radial signed-distance fields of the seven surfaces; skull about 6.5 mm at the vault and about 3.5 mm at the temporal windows; outer table, diploë, inner table). Skull porosity is a three-layer profile times a correlated texture. `hidden_c` is a bounded correlated field whose amplitude is solved per volume so that porosity explains `porosity_r2` (default 0.5) of skull sound-speed variance. `materials.py` derives sound speed, density, absorption (Aubry 2003 mixing law; soft tissue from the Aubry 2022 benchmark table) and CT basis-material fractions from `(labels, porosity, hidden_c)` only. CBV ground truth: grey/white baseline with a procedural vessel texture, a cortical activation patch, a blood velocity field, and a block design convolved with a double-gamma HRF. Plane sampling interpolates the latent fields and re-derives the acoustics. Four skull-effect ablations. HDF5 save/load with provenance, `scripts/generate_head.py`, `scripts/plot_head.py`, and default (80 mm box at 0.5 mm) and small (64 mm box at 2 mm) configs.
- **Why:** Phase 1; single source of truth for the ultrasound medium and the CT (D-002), with switchable skull effects (D-005).
- **Choices to confirm (see D-014):** bone and pore densities follow NIST ICRU-44 (1920 and 1060 kg/m^3, as the CT uses) instead of Aubry 2003 (2100 and 1000), so the acoustic density equals the mass the CT sees; absorption end points (1.4 and 11.2 dB/cm at 0.5 MHz) are fitted to the Aubry 2022 benchmark values, not Aubry 2003's 0.2 and 8 dB/mm; the hidden sound-speed component needed for `porosity_r2 = 0.5` has a standard deviation of about 220–250 m/s.
- **Limits:** `generate_head` returns `GeneratedHead` (a `HeadVolume` plus its material parameters). `sample_plane` rejects planes that leave the head volume; the default box holds a plane about 50 mm wide by 55 mm deep. The cortex is a smooth 2.5 mm shell without gyri, so the activation in a plane is a thin arc. Porosity, `hidden_c` and CBV textures are tied to the voxel grid (anatomy is not). The optional MUG500+, CQ500 and SkullGAN skull paths are not built.
- **Files:** `src/neurosim/head/{config,materials,fields,anatomy,microstructure,vasculature,cbv,generate,plane,ablations,io}.py`, `configs/head_default.yaml`, `configs/head_small.yaml`, `scripts/generate_head.py`, `scripts/plot_head.py`, `tests/unit/head/`, `evals/head/`.
- **Evals added (all passing):** E-HEAD-001 skull thickness and layer proportions; E-HEAD-002 property ranges; E-HEAD-003 porosity R^2; E-HEAD-004 plane resolution independence; E-HEAD-005 determinism and cbv-stream isolation; E-HEAD-006 ablation signatures; E-HEAD-007 X-ray/acoustic mass consistency; E-HEAD-008 grey/white CBV fractions; E-HEAD-009 task response against numerical convolution. E-HEAD-002 failed on its first run (skull sound speed 1499.9999 m/s from float32 rounding); fixed in code, threshold unchanged. Unit tests: 85 passing.

### 2026-11-17 — Phase 0: shared core, packaging and eval harness
- **What:** Added `docs/ARCHITECTURE.md` (module boundaries and data contracts) and the contract type files `src/neurosim/{head,fus,ct}/types.py` and `src/neurosim/fus/reference.py`. Added `pyproject.toml` (setuptools, src layout, dependencies pinned with `==`, extras `recon` and `dev`), `.gitignore`, the `neurosim` package skeleton and `neurosim.common`: `seeding` (`SeedTree`: named, order-independent streams on `SeedSequence` spawn keys, with child subtrees), `units`, `constants`, `config` (strict frozen-dataclass loading, canonical YAML, config hash), `grid` (`Grid3D`, `Grid2D`, `PlaneSpec`, volume and plane sampling), `io` (byte-deterministic HDF5 bundles with provenance) and `evalkit` (`@eval_case`, `report`). `scripts/build_registry.py` generates `evals/REGISTRY.md` from the eval decorators and the latest results, so the registry is never typed by hand.
- **Why:** One source of truth for randomness, units, constants, grids, config validation and output format before the head, fUS and CT packages are built on top (D-004; CLAUDE.md sections 1 and 2).
- **Notes:** Built with Python 3.11 through `uv`; the package supports Python 3.10 and later. Water and air constants use a 20 °C reference state; `C_WATER` and `RHO_WATER` are marked "assumed, to verify" (see open items). Seed sub-indices are limited to below 2^32.
- **Files:** `docs/ARCHITECTURE.md`, `pyproject.toml`, `.gitignore`, `src/neurosim/**/__init__.py`, `src/neurosim/common/*`, `src/neurosim/{head,fus,ct}/types.py`, `src/neurosim/fus/reference.py`, `scripts/{run_evals,build_registry}.py`, `evals/common/*`, `evals/REGISTRY.md`, `tests/unit/common/*`, `tests/unit/scripts/*`.
- **Evals added:** E-COMMON-001 (bundle writes are byte-identical across processes; passing), E-COMMON-002 (a seed stream depends only on root seed, name and sub-index; passing). Unit tests: 100 passing.

### 2026-11-16 — Add CLAUDE.md
- **What:** Added `CLAUDE.md` with working rules for the repo: distinct physics evals with a registry (`evals/REGISTRY.md`, one claim per eval, thresholds fixed in advance), separate from unit tests; clean, non-duplicated code (single source of truth for constants, units, grids and randomness); grounding every number in a source; and logging every change here.
- **Why:** To keep the simulator trustworthy and the code base clean as it grows.
- **Files:** `CLAUDE.md`, `ROADMAP.md`

### 2026-11-16 — Add README and ROADMAP
- **What:** Added `README.md`, a project overview drawn from the full 3-page proposal (motivation, the three skull effects, the method, the three stages, the planned layout and key references). Added `ROADMAP.md` with phases 0–5, design decisions D-001 to D-005 and this change log.
- **Why:** To set the plan for the Wednesday goal (head, fUS sensor and CT scanner simulation) and to keep a structured record of every later change.
- **Note:** The proposal attributes the "23–53% of sound-speed variability" figure to Drainville et al. (2023). That figure comes from Webb et al. (2018); Drainville et al. cite a figure of about 50%. The README cites Webb et al.
- **Files:** `README.md`, `ROADMAP.md`

### 2026-11-16 — Add project proposal
- **What:** Added page 1 of the COS 597Y project proposal.
- **Files:** `project-proposal.pdf` (commit `eb474b3`)
