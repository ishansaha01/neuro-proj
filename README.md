# neuro-proj — Skull-Aware Transcranial fUS Reconstruction

**COS 597Y course project** · Ishan Saha, Sanketh Kamath, Rishi Kannan

Non-invasive brain–computer interfaces need a way to read brain activity without opening the skull. **Functional ultrasound (fUS)** images changes in cerebral blood volume (CBV) by sending ultrafast plane waves into the brain and measuring the echoes off moving red blood cells (Macé et al., 2011). It works well through a skull window or a sonolucent implant (Rabut et al., 2024). Through intact adult skull, though, the signal is corrupted by three effects:

| Effect | What the skull does | What it does to the image |
|---|---|---|
| **Aberration** | Bone thickness and sound speed vary, so wavefronts arrive at the array at the wrong times | Blurred, displaced images |
| **Attenuation** | Bone absorbs and scatters sound, more so at higher frequency | Weak signal; forces lower frequency, so lower resolution |
| **Reverberation** | Sound bounces inside the bone and static tissue | Strong clutter that swamps the weak blood signal |

**Our idea:** treat transcranial fUS as an inverse problem. A **CT scan** gives a strong but imperfect prior on the skull's acoustic properties. A **diffusion model** trained on clean CBV images gives a prior on what brain signals look like. **Diffusion posterior sampling** (Chung et al., 2023), extended to the blind case where the skull is only partly known, combines the two to recover the clean brain signal.

CT is a good prior, but not a complete one. Across 80 CT scans of 91 skull fragments, Hounsfield units explained only **23–53% of the variation in sound speed**, and the relationship changed with X-ray energy and reconstruction method (Webb et al., 2018). So the skull must be treated as *uncertain*, never as exactly known.

---

## Project structure: three stages

### Stage 1 — Synthetic simulation *(current focus, target: Wed 18 Nov 2026)*

A deterministic, physics-grounded simulator that produces paired data. It has three linked parts, all driven by one config file and one seed:

1. **Ground-truth head (3D).** Scalp, skull (outer table, diploë, inner table), CSF and brain. The skull carries a microstructure (porosity) field. Both the acoustic properties and the X-ray properties are derived from that field, so the CT is correlated with the true sound speed but does not determine it. The brain carries a **CBV ground truth**: baseline vascular density, plus a task-evoked activation in a chosen target region, driven by a hemodynamic response.
2. **fUS sensor.** A 128-element linear array (2 MHz by default, sweep 1–3 MHz) sends 7–11 tilted plane waves (±10°) that are compounded per frame. Full-wave acoustic simulation through the skull is followed by delay-and-sum beamforming, SVD clutter filtering (Demené et al., 2015) and power Doppler. Every run is also simulated **without the skull**, which gives the clean paired target.
3. **CT scanner.** A polyenergetic X-ray source (kVp, filtration), Beer–Lambert attenuation through the same head, photon (Poisson) noise, and filtered back-projection to Hounsfield units. Energy and reconstruction kernel can be changed, because both shift the HU-to-sound-speed relationship.

Every noise source is explicit, seeded and switchable, so any experiment can be replayed exactly or varied one parameter at a time.

### Stage 2 — Dataset generation
Sweep skulls, frequencies, target regions and noise settings to produce paired clean and skull-corrupted fUS data, plus the matching simulated CT. The proposal's speed-up applies here: the skull is static and blood is a weak scatterer. So we can simulate each skull once per element and transmit angle, then produce the received signals for any blood configuration with linear operations.

### Stage 3 — Reconstruction and evaluation
- **Priors:** a diffusion model on clean CBV images, and a CT-derived skull model whose HU mapping constants are randomized within published error bounds (Marquet et al., 2009; Webb et al., 2018).
- **Static background model:** the skull and tissue are fixed while blood changes, so clutter and reverberation are low-rank over time.
- **Joint posterior sampling (BlindDPS):** sample the CBV signal together with a low-dimensional skull correction (per-element delays and amplitudes) initialized from the CT. Use coarse-to-fine guidance and best-of-N selection, following CryoBoltz (Raghu et al., 2025).
- **Staging:** known skull, then uncertain skull (main result), then guidance on raw channel data (stretch goal).
- **Baselines:** no-skull simulation (upper bound); standard beamforming with SVD filtering; CT-derived delay correction; correction with the true skull.
- **Metrics:** decoding accuracy of task activity from the reconstruction (Norman et al., 2021), activation-map overlap, CBV correlation, contrast, calibration of uncertainty, and runtime.
- **Robustness:** train on a fluid model and test on an elastic (shear-wave) solver, perturbed HU mappings and a different solver. Then real transcranial data (sheep, head phantoms on Zenodo).

See **[ROADMAP.md](ROADMAP.md)** for the detailed plan, the design decisions and a log of every change.

---

## Status (17 Nov 2026)

All three stages exist as working, tested code on the `build-out` branch. The current state of every physics check is in **[evals/REGISTRY.md](evals/REGISTRY.md)** (generated; failing evals stay listed as failing), and every design choice and known limit is in **[ROADMAP.md](ROADMAP.md)**.

What the simulator says so far (8 simulated scenes, 120 images each):

- **The skull is the problem it is supposed to be.** Without correction the image has almost no relation to the true blood-volume map (correlation 0.07, against 0.90 with no skull).
- **A CT-derived skull model helps, and calibrating it against the ultrasound data helps more.** Beamforming with CT-derived delays gives 0.44; after calibrating the CT model by re-beamforming, 0.61; with the true delays, 0.72. Inverting on top (deconvolution or diffusion posterior sampling) adds a few points to each.
- **The functional signal is not recovered by anything yet, not even without a skull.** With realistic speckle one image is close to a single speckle look, and decoding is at chance for every method. This is the most important open problem.

Things to know before trusting a number:

- **The calibration is not yet safe.** Its focus criterion is a brightness, and on scenes built to test it, it is not maximised by the true skull model (E-RECON-013 and 014 fail).
- **Data come from a refraction-aware ray model (`bent_ray`), not full-wave simulation**, which at full resolution is not a laptop job (about 300 core-hours). It keeps only the first arrival: no reverberation.
- **Several constants and defaults are marked `assumed, to verify`**, and decisions D-011, D-014 and D-017 in the ROADMAP need the team's confirmation. The noise level (D-017) in particular decides whether there is any signal to work with.
- **The diffusion prior is small and trained only on this simulator's procedural CBV maps.**

## Getting started

Python 3.11 or later. From the repository root:

```
# with conda (Windows, macOS, Linux)
conda create -n neurosim python=3.11
conda activate neurosim
pip install -e ".[recon,dev]"

# or with uv
uv venv --python 3.11
uv pip install -e ".[recon,dev]"
```

Check the install (about 10 seconds), then run the physics evals (several minutes):

```
python -m pytest tests/unit -q
python scripts/run_evals.py
python scripts/build_registry.py
```

One reproducible run: head, both ultrasound runs (with and without skull), CT and the CT-derived skull, from one config and one seed. Every script prints a cost estimate before it starts.

```
python scripts/run_pipeline.py --config configs/run_demo.yaml --seed 0 --out outputs/demo/run.h5
python scripts/plot_run.py --run outputs/demo/run.h5 --out outputs/demo/run.png
```

A paired dataset, the prior and the reconstruction experiment:

```
python scripts/make_dataset.py --config configs/sweep_demo.yaml --seed 0 --out outputs/demo_sweep
python scripts/make_prior_dataset.py --config configs/prior_default.yaml --seed 0 --out outputs/prior/dataset.h5
python scripts/train_prior.py --config configs/prior_default.yaml --data outputs/prior/dataset.h5 --seed 0 --out outputs/prior/prior.pt
python scripts/run_experiment.py --dataset outputs/demo_sweep --prior outputs/prior/prior.pt --config configs/recon_default.yaml --seed 0 --out outputs/experiment_v2
python scripts/plot_experiment.py --results outputs/experiment_v2 --out outputs/experiment_v2/figure.png
```

`run_experiment.py` runs the current design (CT skull model calibrated by re-beamforming); add `--design screen` for the first design (per-element phase screen), kept for comparison. Single parts can be run alone: `scripts/generate_head.py`, `scripts/plot_head.py`, `scripts/run_ct.py`, `scripts/compute_bank.py`, `scripts/run_fus.py`. `outputs/` is git-ignored.

## Repository layout

```
neuro-proj/
├── README.md               project overview (this file)
├── ROADMAP.md              plan, design decisions, change log
├── CLAUDE.md               working rules: evals, code quality, rigor
├── docs/ARCHITECTURE.md    module boundaries and data contracts
├── project-proposal.pdf    proposal, page 1
├── pyproject.toml          package and pinned dependencies
├── configs/                YAML configs; one file = one reproducible run
├── src/neurosim/
│   ├── common/             seeding, units, constants, config, grids, I/O, eval harness
│   ├── head/               3D anatomy, skull microstructure, material maps, CBV signal
│   ├── fus/                transducer, Green's-bank backends, synthesis, beamforming, Doppler
│   ├── ct/                 X-ray spectrum, attenuation, projector, reconstruction, HU to acoustics
│   ├── data/               pipeline, CT-derived skull prior, sweeps, dataset format
│   └── recon/              diffusion prior, baselines, posterior sampling, calibration, metrics
├── scripts/                command-line entry points
├── tests/unit/             code checks (fast)
└── evals/                  physics checks, one claim each; REGISTRY.md is generated
```

## Key references

- Macé et al. (2011). *Functional ultrasound imaging of the brain.* Nature Methods 8(8):662–664.
- Demené et al. (2015). *Spatiotemporal clutter filtering of ultrafast ultrasound data highly increases Doppler and fUltrasound sensitivity.* IEEE TMI 34(11):2271–2285.
- Norman et al. (2021). *Single-trial decoding of movement intentions using functional ultrasound neuroimaging.* Neuron.
- Rabut et al. (2024). *Functional ultrasound imaging of adult human neural activity through an acoustically transparent cranial prosthetic.* Science Translational Medicine. doi:10.1126/scitranslmed.adj3143
- Aubry et al. (2003). *Experimental demonstration of noninvasive transskull adaptive focusing based on prior computed tomography scans.* JASA 113(1):84–93.
- Marquet et al. (2009). *Non-invasive transcranial ultrasound therapy based on a 3D CT scan: protocol validation and in vitro results.* Phys. Med. Biol. 54(9):2597–2613.
- Webb et al. (2018). *Measurements of the relationship between CT Hounsfield units and acoustic velocity and how it changes with photon energy and reconstruction method.* IEEE TUFFC 65(7):1111–1124.
- Drainville et al. (2023). *A simulation study on the sensitivity of transcranial ray-tracing ultrasound modeling to skull properties.* JASA.
- Chung et al. (2023). *Diffusion posterior sampling for general noisy inverse problems.* ICLR.
- Raghu et al. (2025). *CryoBoltz: multiscale guidance of protein structure prediction with heterogeneous cryo-EM data.* NeurIPS. arXiv:2506.04490
- Naftchi-Ardebili et al. (2023). *SkullGAN: Synthetic skull CT generation with generative adversarial networks.* arXiv:2308.00206
- Li et al. (2021). *MUG500+: Database of 500 high-resolution healthy human skulls and 29 craniotomy skulls and implants.* Data in Brief.
- Stanziola et al. (2023). *j-Wave: an open-source differentiable wave simulator.* SoftwareX.
- Treeby & Cox (2010). *k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields.* J. Biomed. Opt. (k-wave-python is its Python port.)
