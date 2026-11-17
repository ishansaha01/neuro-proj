# CLAUDE.md

Guidance for Claude (and anyone else) working in this repo. Read [README.md](README.md) for what the project is and [ROADMAP.md](ROADMAP.md) for the plan, the design decisions and the change log.

**Project in one line:** a deterministic, physics-grounded simulator of a 3D head, a functional ultrasound (fUS) sensor and a CT scanner. It produces paired clean and skull-corrupted data for skull-aware fUS reconstruction with diffusion posterior sampling.

---

## 1. Evals: every claim gets its own distinct check

The simulator is only useful if we can trust it. Each piece of physics we model must be backed by an eval that checks it against something independent of our code: an analytic solution, a published measurement or a conservation law.

**Evals are separate from unit tests.**
- `tests/unit/`: does the code do what it says (shapes, units, seeding, I/O, config parsing). Fast, run on every change.
- `evals/`: is the physics right. Each eval compares simulator output to an outside reference and reports a number against a threshold.

**Each eval checks exactly one claim.** Before adding one, check `evals/REGISTRY.md` to make sure no existing eval already covers that claim. If one does, extend it instead of adding a near-copy. Two evals that would fail for the same reason are one eval.

**Every eval has a registry entry** in `evals/REGISTRY.md` with these fields:

| Field | Example |
|---|---|
| ID | `E-FUS-003` (area prefix: `HEAD`, `FUS`, `CT`, `DATA`, `RECON`) |
| Claim | Plane-wave arrival time through a homogeneous slab matches the analytic delay |
| Reference | Analytic: Δt = d·(1/c_slab − 1/c_water) |
| Metric | Maximum absolute error in arrival time |
| Threshold | < 1 sample, or < 2% of Δt |
| Seeds | Fixed seed list, so the result is reproducible |
| Status | passing / failing / not built |

**Rules for evals**
- Set the threshold before looking at the result, and write down why that threshold. Never loosen a threshold to make an eval pass; if it has to change, record the reason in the ROADMAP change log.
- Test the simplest case that isolates the claim: water before tissue, a flat slab before a skull, one scatterer before many.
- Include ablation evals: turning a noise effect off must remove its signature (for example, a homogeneous skull removes aberration but keeps attenuation).
- Include determinism evals: same config and seed give byte-identical output, and changing one seed stream changes only that stream's part of the output.
- An eval that fails stays in the registry as failing. Report it; do not delete it or hide it.
- When a phase is finished, every claim in that phase's ROADMAP checklist maps to at least one passing eval.

---

## 2. Clean, non-duplicated code

**One source of truth for everything.**
- Physical constants and material properties live in one module (`src/neurosim/common/constants.py` or a per-area `materials.py`), each with its source. Never retype a constant in another file; import it.
- Unit conversions (dB/cm/MHz ↔ Np/m, HU ↔ μ, and so on) live in `common/units.py`. Never convert inline.
- Grid and geometry handling lives in one place and is reused by the head, fUS and CT code.
- Randomness only comes from the seed tree in `common/seeding.py` (ROADMAP D-004). No `np.random.*` global calls, no unseeded generators.

**Before writing a function, look for one that already does it.** Search the codebase first. If two places need slightly different versions, write one function with a parameter, not two copies. If you find duplication, remove it in the same change and say so in the change log.

**Structure**
- Keep modules small and single-purpose, following the layout in the README: `head/`, `fus/`, `ct/`, `common/`.
- Pure functions for the physics: arrays and parameters in, arrays out. Keep file I/O, plotting and config loading at the edges (`scripts/`).
- Solver backends (k-wave-python, j-Wave) sit behind one interface, so the rest of the code never imports a solver directly.
- Every config value is typed and validated at load time; no magic numbers in the code that should be in the config.
- Type hints on public functions; docstrings state units for every physical argument and return value (for example `c: sound speed [m/s]`).
- Name things for what they are physically (`sound_speed`, `porosity`, `cbv`), not `arr2` or `tmp`.

**Do not**
- Leave dead code, commented-out blocks or unused parameters.
- Add an abstraction for a case that does not exist yet.
- Catch exceptions just to hide them.

---

## 3. Be thorough

- **Ground every number.** Each physical constant or modelling choice names its source in a comment or docstring: paper and table, dataset, or `# assumed, to verify`. Open a ROADMAP item for every "to verify".
- **Check the literature before inventing.** If a paper or established tool already models something (k-Wave, SpekPy, NIST XCOM, SVD clutter filtering), use or follow it, and cite it.
- **Verify before claiming.** Do not report a feature as done until its evals pass. Report failures and skipped steps plainly, with the output.
- **Watch the scale.** Before running a simulation, estimate its grid size, memory and run time, and note the estimate. This machine is an Apple M5 Pro with 24 GB of RAM and no CUDA.
- **Ask when a choice is the user's.** Scientific choices that change results (target brain region, frequencies, which skull source) go to the user or into a ROADMAP design decision, not into a silent default.

---

## 4. Record every change

- Every change to the repo gets an entry in the ROADMAP change log in the same commit: what changed, why, which files, and which evals it added or affected.
- A change to a design decision gets a new D-xxx entry that supersedes the old one; old entries are not rewritten.
- Commit messages are short and say what changed and why. One logical change per commit.
- Every simulation output stores its config, its seed and the git commit that produced it.
