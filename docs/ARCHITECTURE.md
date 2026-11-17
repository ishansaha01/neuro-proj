# Architecture and interface contracts

This file fixes the module boundaries and the data contracts between them. Code in
`src/neurosim/` must match it. Change a contract here first (and log it in
`ROADMAP.md`), then change the code.

Read `README.md`, `ROADMAP.md` and `CLAUDE.md` first. The rules in `CLAUDE.md`
(evals separate from unit tests, one source of truth, seeded randomness, every
number sourced) apply to everything below.

## 1. Conventions

- **Units:** SI inside the code (m, s, Hz, kg/m^3, m/s, Pa). The one exception is
  acoustic attenuation, stored as the power-law prefactor `alpha0` in
  dB/(cm·MHz^y) with one global exponent `y` (k-Wave's convention). All
  conversions live in `common/units.py`.
- **Head frame (3D):** right-handed, origin at the head centre, metres.
  `x` = left to right, `y` = posterior to anterior, `z` = inferior to superior.
  Volumes are C-ordered arrays indexed `[ix, iy, iz]`.
- **Imaging-plane frame (2D):** `x` = lateral along the array (0 at the array
  centre), `z` = depth (0 at the array face, positive into the head), metres.
  Plane maps are indexed `[iz, ix]` (rows are depth).
- **dtype:** float32 for stored fields, complex64 for stored spectra, float64
  inside solvers where accuracy needs it.
- **Time convention for spectra:** `exp(-i ω t)`; a delay `τ` multiplies a
  spectrum by `exp(+i ω τ)`.
- **Randomness:** only through `common.seeding.SeedTree`. No global `np.random`
  calls, no unseeded generators, no `torch.manual_seed` outside `recon/`'s single
  seeding helper (which derives its seed from the tree).
- **Purity:** physics functions take arrays and parameters and return arrays.
  File I/O, plotting and config loading stay in `scripts/` and `common/io.py`.

## 2. Package layout and ownership

```
src/neurosim/
  common/    seeding, units, constants, config machinery, grids, I/O, eval harness
  head/      3D anatomy, skull microstructure, material maps, CBV signal
  fus/       transducer, Green's-function backends, synthesis, beamforming, Doppler
  ct/        X-ray spectrum, attenuation, projector, reconstruction, HU -> acoustics
  data/      pipeline (one config + one seed -> one run), sweep driver, dataset format
  recon/     baselines, diffusion prior, DPS / BlindDPS, metrics
configs/     YAML, one file = one reproducible run
scripts/     thin command-line entry points
tests/unit/  fast code checks, mirrors the package layout
evals/       physics checks, one claim each, registered through the eval harness
```

Dependency direction (no cycles): `common` <- `head` <- {`fus`, `ct`} <- `data`
<- `recon`. `fus` and `ct` may import `head.types` and `head.materials` only.

## 3. `common`

| Module | Contract |
|---|---|
| `seeding.py` | `STREAMS`: an append-only tuple of stream names. `SeedTree(root_seed)`; `tree.rng(name, *sub) -> np.random.Generator`; `tree.seed_int(name, *sub) -> int`. A stream depends only on `(root_seed, name, sub)`, never on call order. |
| `units.py` | Named conversion functions only, e.g. `db_cm_mhz_to_np_m(alpha0, f_hz, y)`, `np_to_db`, `hu_from_mu(mu, mu_water, mu_air)`. |
| `constants.py` | Universal constants (water, air, reference sound speed `C_REF`) each with a source comment. Names of CT basis materials `BASIS_MATERIALS`. |
| `config.py` | `load_yaml(path)`, `from_dict(cls, d)` (strict: unknown key, wrong type or out-of-range value raises `ConfigError`), `to_dict(obj)`, `config_hash(obj)`. Config classes are frozen dataclasses; ranges come from field metadata. |
| `grid.py` | `Grid3D(shape, spacing, origin)`, `Grid2D(shape, spacing, origin)`, `PlaneSpec(origin, u_lateral, u_depth)` (plane embedded in the head frame), `plane_points(plane, grid2d) -> (nz, nx, 3)`, `sample_volume(volume, grid3d, points, order)` (trilinear or nearest; used by head, fus and ct). |
| `io.py` | `save_bundle(path, arrays, attrs)` / `load_bundle(path)` on HDF5, byte-deterministic (no timestamps, fixed chunking). `provenance(config, seed)` returns config YAML, config hash, root seed, git commit, package version; every output stores it. |
| `evalkit.py` | `@eval_case(id, claim, reference, metric, threshold, seeds)` decorator and `report(eval_id, value, passed, note)`; results go to `evals/results/<id>.json`. `scripts/build_registry.py` regenerates `evals/REGISTRY.md` from the decorators and the latest results, so the registry is never typed by hand. |

Streams (append only): `anatomy`, `skull_microstructure`, `hidden_soundspeed`,
`cbv`, `tissue_scatterers`, `blood_scatterers`, `us_electronic_noise`,
`ct_photon_noise`, `tissue_motion`, `ct_hu_mapping`, `dataset_sweep`,
`speckle_statistical`, `diffusion_training`, `recon_sampling`.

## 4. `head`

`head/types.py` holds the contract types (see the file). In short:

- `HeadVolume`: `grid`, tissue `labels`, signed-distance fields of every tissue
  surface (`sdf`), `porosity`, `hidden_c` (the part of sound speed X-rays cannot
  see), acoustic maps (`sound_speed`, `density`, `attenuation`), CT basis-material
  volume fractions (`material_fractions`), and the CBV model (`cbv_baseline`,
  `activation`, `blood_velocity`). `generate_head` returns `GeneratedHead`, a
  `HeadVolume` that also carries the `MaterialParams` it was built with, so
  `sample_plane` can apply the same mapping.
- `PlaneMedium`: the same quantities sampled on a fine `Grid2D` in one imaging
  plane.

Rules:

1. **One latent microstructure (D-002).** Acoustic and X-ray properties are both
   pure functions of `(labels, porosity, hidden_c)` in `head/materials.py`.
   `hidden_c` enters the acoustic map only.
2. **Resolution independence.** `sample_plane(head, plane, grid2d)` interpolates
   the *latent* fields (`sdf`, `porosity`, `hidden_c`) to the fine grid, thresholds
   the SDFs into labels, then applies the same material mapping. It never
   interpolates sound speed directly.
3. **CBV signal.** `cbv(x, t) = cbv_baseline(x) * (1 + a(t) * activation(x))`,
   with `a(t)` = task design convolved with a hemodynamic response function.
4. **Skull-effect switches (D-005)** are pure transforms of a `PlaneMedium` in
   `head/ablations.py`: homogeneous skull (no aberration from heterogeneity),
   zero bone absorption, matched impedance (no reverberation), and `remove_skull`
   (the paired clean run, D-010).

## 5. `fus`: three-tier forward model (D-006, D-007)

Blood and brain-tissue scatterers are weak, so the received signal is linear in
them (Born approximation) while the skull is treated exactly. Everything the
skull does is captured once per skull in a **Green's bank**:

```
B[f, e, r] = G_medium(r, e, f) / G_ref(r, e, f)
```

the ratio of the one-way transfer function from array element `e` to ROI point
`r` in the real medium to that in a homogeneous reference (`C_REF`, no
absorption). `B` varies slowly in `r`, so it is stored on a coarse ROI grid and
interpolated; `G_ref` is analytic (2D: Hankel function) and evaluated exactly at
any point. By reciprocity the same `B` serves transmit and receive.

Backends behind one interface `compute_bank(medium, transducer, roi, freqs, cfg) -> GreensBank`:

| Backend | Physics | Cost | Use |
|---|---|---|---|
| `analytic` | Homogeneous medium, `B = 1` | none | water evals, upper bound |
| `ray` | Straight rays through the medium: travel-time and absorption integrals, interface transmission loss. No refraction, no reverberation | seconds | development, large sweeps, CT-derived correction |
| `kwave` | Full wave (k-wave-python, pure-Python k-space backend): aberration, absorption, refraction, reverberation | one simulation per element | reference tier |

Nothing outside `fus/backends/` imports a solver.

On top of a bank (`fus/types.py` has the types):

1. **Channel synthesis.** `channel_spectra(bank, scatterers, sequence)` returns
   frequency-domain channel data `[f, angle, element]` for a set of point
   scatterers (position, complex amplitude). A plane-wave transmit is a delayed
   sum of element transmits.
2. **Beamforming.** `das(spectra, pixel_grid, model_bank, sequence)` is
   delay-and-sum with coherent compounding, written as phase conjugation against
   a *model* bank and normalised to unit scatterer gain under that model. With
   the homogeneous bank it is conventional DAS at `C_REF` (as a real scanner
   does); with a `ray` bank through a given sound-speed map (true skull or
   CT-derived skull) it is aberration-corrected beamforming (D-011).
3. **Doppler.** Frames -> SVD clutter filter (Demené 2015) -> power Doppler.
4. **Operators.** `psf_operator(bank, pixel_grid, source_grid, model_bank, sequence) -> A`
   (complex, `[n_pix, n_src]`) is built from 1 and 2 applied to unit scatterers, so
   there is no second implementation. For blood with random phases the expected
   power Doppler image is linear in CBV: `E[PD] = H @ cbv`, `H = |A|^2`
   (`pd_operator`). `H` is the image-domain forward model used by `recon`.

Two fidelities produce data, chosen in the config:

- `scatterer`: explicit moving blood scatterers and tissue scatterers, frame by
  frame, plus static skull clutter and electronic noise. Physically explicit; slow.
- `statistical`: `PD = H @ cbv` with gamma speckle (shape set by the ensemble
  length) and a noise floor. Fast; validated against `scatterer` by an eval.

## 6. `ct`

`simulate_ct(head, cfg, tree) -> CtVolume` (HU on a `Grid3D`, for a slab of
slices around the imaging plane or the whole head):

polyenergetic spectrum (SpekPy) -> per-voxel `μ(E)` from `material_fractions`
and elemental mass-attenuation data (xraydb, cross-checked against NIST XCOM
values) -> fan-beam ray-driven projection of each basis material (line integrals
are energy-independent, so each material is projected once) -> polyenergetic
Beer–Lambert on the sinogram -> Poisson noise -> filtered back-projection with a
choice of kernel -> HU calibration with water and air.

`hu_to_acoustic(hu, mapping) -> (sound_speed, density, attenuation)` is the
clinical linear map (Aubry 2003, Marquet 2009). `sample_hu_mapping(tree, bounds)`
draws its constants within published error bounds (Webb 2018). This CT-derived
skull is the prior used by `recon`.

## 7. `data` and `recon`

- `data.pipeline.run(config) -> Run`: head -> plane -> banks (skull and
  no-skull) -> fUS data (both) -> CT -> CT-derived skull. One config, one seed,
  one bundle on disk.
- `data.sweep`: grids or random sweeps over skull seed, frequency, probe pose,
  target region and noise switches; writes the paired dataset.
- `recon`: baselines (uncorrected DAS, CT-corrected DAS, true-skull DAS, no
  skull), a small diffusion prior on clean CBV images (PyTorch, D-009), DPS with
  known `H`, BlindDPS with skull parameters (per-element delay and amplitude)
  initialised from the CT, and the metrics in the README.

## 8. Scale

Development and tests run on 2 cores and 7 GB. Unit tests for one package finish
in under a minute; one eval finishes in a few minutes at most. Every simulation
entry point estimates grid size, memory and run time before it starts and prints
the estimate (`CLAUDE.md` section 3). Full-resolution runs are launched by the user.
