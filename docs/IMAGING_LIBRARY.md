# The PWM imaging library (LIB-40, 5 Oct 2026)

**Goal (owner, 4–5 Oct 2026):** every imaging system in the PWM registries, and every new modality a user designs,
is in the library. "Design" covers the full scope of three papers:
- the *Finite Primitive Basis* paper: 11 primitives, a typed DAG, 5 carriers, 4 fidelity levels, the extension
  protocol;
- *System Design*: the eight-field `spec.md`, the C1–C6 compiler, three gates, system elements with source, optics,
  detector, calibration and cost;
- *PWM Flagship*: the triad gates as design specifications, and purpose-conditioned selection over eight adequacy
  dimensions.

Design is not just masks, sampling and geometry.

## What is here
- `pwm_mode/imaging/catalog.json`: all **170** registry modalities.
  - Each is an eight-field `spec.md` record: modality, carrier, geometry, object, forward_model, noise, target,
    system_elements.
  - Each also records its forward-model templates and whether they run, its linked genesis principle, a validation
    level and its gaps.
  - Plus **46** genesis imaging-family principles that are not in the registry, marked `principle_only`.
  - Rebuild: `python -m pwm_mode.imaging.catalog --build`.
- `pwm_mode/imaging/design.py`: checks a new design and runs, in order:
  - the spec.md fields;
  - C1 DAG and registered primitives;
  - C2 canonical chain, with matches to validated chains and novelty;
  - C3 bounds (≤ 20 nodes, depth ≤ 10);
  - C4 nonlinear families;
  - C5 adjoint test on the linear part;
  - C6 representation error against a reference;
  - Gate 1: compression ratio, forced null space, exact rank when small;
  - Gate 2: SNR from source photons, QE, read noise and dark current;
  - Gate 3: sensitivity at each declared tolerance, and which modes no blind calibration can recover;
  - the eight adequacy dimensions, only when supplied.

  Missing inputs are reported as **unassessed**, never as passed. `submit_design` files a passing design as a
  pending L2 submission; only `Library.accept` by a named validator other than the submitter adds it.
- `pwm_mode/imaging/server.py`: MCP tools `imaging_search`, `imaging_modality`, `imaging_primitives`,
  `imaging_design_check` and `imaging_design_submit`. Separate from `mcp_server.py` so it merges cleanly.
- `tests/test_imaging.py`: 7 tests.

## Coverage today (honest)
| Level | Count | Meaning |
|---|---|---|
| registered | 170 | element chain and forward equation in `modalities.yaml` |
| runs | 170 | a graph template compiles and runs (145 before the primitive-name aliases, 165 before IMG-01 repaired the last five) |
| canonical | 28 | chain is in the 31-entry validated decomposition registry |
| real_data | 11 | hardware validation on real data (flagship paper) |

Gaps:
- No template fails to compile or run. That is a weaker claim than "correct": the five sampled-Fourier templates
  (`eht_imaging`, `insar`, `radio_astronomy`, `radio_interferometry`, `xray_crystallography`) are now one `mri_kspace`
  node (FFT plus a random k-space mask with a fully sampled centre, 25 % rate) and complex Gaussian noise. That is a
  generic random-mask Fourier sampler, not a baseline (u,v) track, a SAR phase model or a diffraction geometry, and its
  `adjoint` is the zero-filled inverse FFT, off by exactly H*W in the dot-product test (xfail in `tests/test_imaging.py`;
  the MRI template has the same property). Making them physically specific is part of IMG-05/IMG-07.
- Until pwm_core carries `results/IMG-01/pwm_core_templates.patch`, this repo repairs those five templates at load time
  (`core.repair_template`) and maps three unregistered names with `PRIMITIVE_ALIASES`. Both become no-ops once patched.
- **54** modalities have no source element, so source and excitation design cannot be expressed for them.
- **No** modality records calibration tolerances, so Gate 3 can only be assessed when a design supplies them.
- 21 registry modalities have no linked principle.
- The 46 principle-only entries need curation: some are imaging (DIC, DEXA, fNIRS, STEM, 4D-STEM, STEM-EDX, LiDAR,
  X-ray NDT, PAUT, phase contrast, THz-TDS, phase retrieval); some are not (EM solvers, room and structural
  acoustics).
- Quantitative reconstruction benchmarks exist for about 39 modalities, and real data for about 12.

## Tasks that can run on other machines (heavy work: abraham / WSL, not agent-prod)
- **IMG-01 (done here, upstream pending):** the 5 failing templates are repaired; the patch that renames the template ids
  in `pwm_core` is `results/IMG-01/pwm_core_templates.patch`, to be applied by someone with write access to
  Physics_World_Model. See `results/IMG-01.md`.
- **IMG-02:** add source elements to the 54 source-less modalities, from the literature, with units.
- **IMG-03:** record calibration parameters **with tolerances** per modality (Gate 3). Start from genesis
  `mismatch_parameters` and the flagship tables (e.g. CASSI mask < 0.25 px, CT centre of rotation ~2 px).
- **IMG-04:** curate the 46 principle-only entries. Add registry entries and templates for the imaging ones; mark
  the rest out of scope.
- **IMG-05:** extend validated canonical chains (C2/C6) beyond 28 toward all 170 running modalities, with a
  reference forward model per modality.
- **IMG-06:** quantitative reconstruction benchmarks for the about 130 modalities at level `runs`, by category, each
  with a strong fixed baseline (LIB-33 showed the library loses to a well-chosen fixed setting in most families).
- **IMG-07:** C5 adjoint tests for all 170 running templates in CI.
- **IMG-08:** source and sample design records (paper A2): measured source spectra, contrast-agent size
  distributions, bleaching rates, each with conditions and date.
