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
- [ ] `pyproject.toml` with pinned dependencies: numpy, scipy, pyyaml, h5py or zarr, matplotlib, k-wave-python, scikit-image; optional jax and jwave
- [ ] Package skeleton `src/neurosim/` with `common/` (seed tree, units, config schema, I/O)
- [ ] Determinism test: same config and seed give byte-identical outputs

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

### 2026-11-16 — Add README and ROADMAP
- **What:** Added `README.md`, a project overview drawn from the full 3-page proposal (motivation, the three skull effects, the method, the three stages, the planned layout and key references). Added `ROADMAP.md` with phases 0–5, design decisions D-001 to D-005 and this change log.
- **Why:** To set the plan for the Wednesday goal (head, fUS sensor and CT scanner simulation) and to keep a structured record of every later change.
- **Note:** The proposal attributes the "23–53% of sound-speed variability" figure to Drainville et al. (2023). That figure comes from Webb et al. (2018); Drainville et al. cite a figure of about 50%. The README cites Webb et al.
- **Files:** `README.md`, `ROADMAP.md`

### 2026-11-16 — Add project proposal
- **What:** Added page 1 of the COS 597Y project proposal.
- **Files:** `project-proposal.pdf` (commit `eb474b3`)
