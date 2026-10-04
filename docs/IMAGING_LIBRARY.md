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
  - Plus the curation of the 46 genesis imaging-family principles that were not linked to the registry (IMG-04,
    `pwm_mode/imaging/curation.json`): 14 linked to a registry modality, 2 proposed as new modalities
    (`proposed_modalities`), 30 marked out of scope with a reason (`out_of_scope`). `principle_only` is now empty.
    Details and caveats: `docs/LIB-40-IMG-04.md`.
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
- `tests/test_imaging.py`: 7 tests; `tests/test_imaging_sources.py`: 5 tests for the drafted sources.

## Coverage today (honest)
| Level | Count | Meaning |
|---|---|---|
| registered | 170 | element chain and forward equation in `modalities.yaml` |
| runs | 170 | a graph template compiles and runs (145 before the primitive-name aliases, 165 before IMG-01 repaired the last five) |
| canonical | 28 | chain is in the 31-entry validated decomposition registry |
| real_data | 11 | hardware validation on real data (flagship paper) |

Gaps:
- 5 templates still fail: `eht_imaging`, `insar`, `radio_astronomy`, `radio_interferometry`,
  `xray_crystallography`.
- **54** modalities had no source element in the registry. IMG-02 (branch `lib/LIB-40-img02-sources`) adds one for each, in
  `pwm_mode/imaging/sources.py`, at validation level **drafted**: the radiation or particle, a registered source primitive,
  numeric parameters with units and constraints, written from general instrument knowledge and **not checked** against a
  paper, datasheet or measurement (`verified_against` is null). 116 modalities keep their registry source
  (`source_validation: registry`); the 54 are `source_validation: drafted` and carry a gap saying so. Source primitives do not
  take these values (only `strength`, or `beam_current_na` for the electron beam), and Gate 2 `photons_per_pixel` stays unset.
  Catalog carriers that disagree with the real source are noted per element (`carrier_note`, e.g. eddy_current, maldi_msi,
  sims, atom_probe, neutron_diffraction, nirs_brain, impedance_tomo, magnetic_particle, afm, mfm, desi); 7 modalities
  (cars, libs, pump_probe, raman_imaging, shg, srs, streak_camera) also have a placeholder `[400, 700]` geometry range
  that does not contain the source wavelength, flagged as a gap and not changed.
- **No** modality records calibration tolerances, so Gate 3 can only be assessed when a design supplies them.
- 21 registry modalities have no linked principle.
- The 46 principle-only entries are curated (IMG-04), but 26 of the title-matched registry links are still
  `fuzzy_unverified` and many look wrong (e.g. muon tomography linked to electron tomography); see
  `docs/LIB-40-IMG-04.md`.
- Quantitative reconstruction benchmarks exist for about 39 modalities, and real data for about 12.

## Tasks for other machines ("LIB-40 IMG-0x"; Abr has an unrelated IMG-01) (heavy work: abraham / WSL, not agent-prod)
- **LIB-40 IMG-01 (done here, upstream pending):** the 5 failing templates are repaired; the patch that renames the template ids
  in `pwm_core` is `results/IMG-01/pwm_core_templates.patch`, to be applied by someone with write access to
  Physics_World_Model. See `results/IMG-01.md`.
- **LIB-40 IMG-02:** add source elements to the 54 source-less modalities, from the literature, with units.
- **LIB-40 IMG-03:** record calibration parameters **with tolerances** per modality (Gate 3). Start from genesis
  `mismatch_parameters` and the flagship tables (e.g. CASSI mask < 0.25 px, CT centre of rotation ~2 px).
- **LIB-40 IMG-04 (done on `lib/LIB-40-img04-genesis`):** curate the 46 principle-only entries. Add registry entries and
  templates for the imaging ones; mark the rest out of scope. What is left upstream: the registry entries are
  proposals (pwm_core is read-only here), and 3 imaging principles have no honest template for lack of primitives.
- **LIB-40 IMG-05:** extend validated canonical chains (C2/C6) beyond 28 toward all 165 running modalities, with a
  reference forward model per modality.
- **LIB-40 IMG-06:** quantitative reconstruction benchmarks for the about 130 modalities at level `runs`, by category, each
  with a strong fixed baseline (LIB-33 showed the library loses to a well-chosen fixed setting in most families).
- **LIB-40 IMG-07:** C5 adjoint tests for all 170 running templates in CI.
- **LIB-40 IMG-08:** source and sample design records (paper A2): measured source spectra, contrast-agent size
  distributions, bleaching rates, each with conditions and date.

## The four lifecycle functions, for every modality (5 Oct 2026)
Owner: every modality must have design, simulate, reconstruct and calibrate. `pwm_mode/imaging/lifecycle.py` gives
each registered modality all four, built from its own forward-model template. They are exposed as MCP tools
(`imaging_lifecycle_design`, `imaging_simulate`, `imaging_reconstruct`, `imaging_calibrate`).
- **design (Q4):** the design check on the modality's own spec.md record and graph.
- **simulate (Q3):** the template's forward model, with noise, plus its clean linear output.
- **reconstruct (Q1):** regularised least squares through the linear part, with a **verified adjoint**.
  - It uses the primitives' own adjoint when that passes the dot-product test.
  - Otherwise it repairs the adjoint: a fitted constant, or the exact transpose by probing (objects up to 8192
    entries).
  - The adjoint used is reported. Nonlinear stages are not inverted.
- **calibrate (Q2):** a one-at-a-time grid search over the template's learnable parameters, the `mismatch_db`
  parameters the graph uses, and the gain.
  - Blind mode uses the data residual; reference mode uses a known object.
  - Blind mode refuses gain (absorbable, paper A1).

Smoke audit on 7 core modalities (synthetic object; `lifecycle_audit.json`):
- **reconstruct:** 7/7.
  - Relative error: CT 0.4% (probed adjoint), MRI 0.8% (probed), widefield 0.2%, lensless 2.9%.
  - Underdetermined: CASSI 75%, SPC 93%, ultrasound 67%. Plain Tikhonov is not a good solver for these.
- **calibrate:** 6/7 recover a shifted parameter with a reference.
- **design:** 4/7 pass. CT, MRI and ultrasound fail C5 because of upstream adjoint bugs.
- **all four:** 4/7.

Upstream problems found:
- `mri_kspace` adjoint is off by a constant.
- `ct_radon` adjoint is not the transpose.
- The ultrasound acoustic chain adjoint is wrong.
- Ultrasound `propagate.speed_of_sound` has **no effect** on the output.
- The CT template declares **no** calibratable parameter (no centre of rotation, no geometry), so only gain can be
  calibrated.

The full 170-modality audit is heavy (probing builds a dense transpose) and goes to a remote machine:
`python -m pwm_mode.imaging.lifecycle`.
- **LIB-40 IMG-09:** run the full four-function audit on all 170 modalities (remote machine) and commit
  `lifecycle_audit.json`.
- **LIB-40 IMG-10:** fix the failing adjoints and the dead learnable parameters in `pwm_core` (as a patch), and add
  geometry calibration parameters to templates that have none (CT centre of rotation, detector offset, ...).
