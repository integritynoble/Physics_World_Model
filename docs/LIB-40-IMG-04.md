# LIB-40 IMG-04: curation of the 46 principle-only genesis entries

Branch `lib/LIB-40-img04-genesis`, from `lib/LIB-40-imaging-catalog` (d93fdf2). Not pushed. Decisions: `pwm_mode/imaging/curation.json`
(applied by `catalog.build()`; see Reproduce). Numbers below are from the rebuilt
`pwm_mode/imaging/catalog.json` on this branch, with pwm_core 3899e328 and the genesis l1 files at the default `GENESIS_SRC`.
No engine was run; the build compiles templates and takes about a minute. `tests/test_imaging.py`: 10 passed.

## Result

| decision | count | what it means |
|---|---|---|
| imaging, linked to an existing registry modality | 14 | `genesis_principle` and carrier set by hand (`link_source: curated`); gaps recorded |
| imaging, no registry counterpart | 2 | `proposed_modalities`: `xray_phase_contrast` (L1-413), `thz_tds` (L1-415), both level `registered`, no template |
| out of scope | 30 | one reason each in `out_of_scope`; 4 are marked borderline for the owner (below) |

`principle_only` is now empty. The registry still has 170 modalities, 165 at level runs (the test invariant); proposals are not
counted in them. Correcting the links freed 3 more principles (L1-234 CDI kept as a second principle of `phase_retrieval`;
L1-307 XAS and L1-420 acoustic BEM marked out of scope).

## The main finding: most of the 12 were already in the registry

Of the 13 imaging principles (your 12 plus flash LiDAR), 11 already had a registry modality that the title matcher missed or mislinked
(dic, dexa, nirs_brain, stem, electron_diffraction = 4D-STEM, edx_mapping, lidar, xray_ndt, ultrasonic_phased_array,
phase_retrieval, plus flash_lidar). Examples of the old links: fNIRS to EELS (carrier electron), DEXA to XAS, STEM to TEM,
X-ray NDT to medical radiography, MRI to sonar, CT to industrial CT. So IMG-04 became mostly relinking, not new entries.
Only X-ray phase contrast and THz-TDS have no counterpart. L1-005 (CT) and L1-006 (MRI) also map to existing `ct` and `mri`
(real_data; the level is the paper's, unchanged by the link).

## Templates: three proposals, five refusals

pwm_core is read-only here, so a "registry entry" is a proposal in `curation.json` for upstream. A template was proposed only
if every non-noise node implements the modality's named operation (I read each primitive's source). Level `runs`, fidelity
"nominal", unvalidated:

- `xray_ndt_v2`: xray_source, beer_lambert, photon_sensor, noise (monoenergetic transmission; no scatter or beam hardening).
- `nirs_v2`: photon_source, scatter_model (gaussian blur), beer_lambert, photon_sensor, noise (not a diffusion solver).
- `stem_v3`: conv2d probe, scan_trajectory (4096 points), electron_detector_sensor, noise (incoherent only).

All three compile, run, depend on the object, and pass C1, C3, C4, C5 (and so structural). C2: novel composition. Gates 1-3 and
adequacy are not run (the registry holds no photon budget or tolerances).

No honest template, gap recorded: dic (no shear primitive), edx_mapping (energy_resolving_detector ignores its channels:
returns x*qe), flash_lidar (spad_tof_sensor ignores dead time), paut (beamform_delay averages over axis 0 with no delays),
xray phase contrast (thin_object_phase is an amplitude attenuation, not a phase shift), THz-TDS (no pulse or time axis).

## Two findings about the existing registry

1. **Templates that ignore the object.** `electron_beam_source` returns `current * ones`, so any chain starting with it
   gives an output independent of x. Five modalities at level runs have only such templates (sem, tem, electron_tomography,
   stem, doppler_ultrasound; `runs_but_ignore_object: 5`). I added the gap and the per-template field `depends_on_object`,
   but did not lower their level (the level is "compiles and finite", and lowering would break `runs >= 165`). The
   owner should decide whether "runs" should require object dependence. My first stem_v3 draft had this bug; C5 caught it.
2. **Stubs.** 81 of the 165 running templates reduce to the single canonical step C (a convolution or propagation stub).
   I have not audited each; the count is from `canonical_chain` in `catalog.json`.

## Links I did not fix

149 title links were made by fuzzy matching. After curation, 114 are `name_match` (one title contains the other; not verified),
14 curated, and 26 `fuzzy_unverified`. Many of the 26 look wrong, e.g. muon_tomo, seismic_tomo and neutron_tomo to Electron
Tomography; atom_probe and ocean_acoustic_tomo to Photoacoustic Tomography; impedance_tomo to OCT; afm to TIRF; pet and spect to
other tomographies. They still feed `object`, `target` and `carrier` of those entries. Fixing them is a separate job (suggest IMG-09).

Also: 44 of the 46 principles carry placeholder text for forward_model and world_state_x ("World state x for ..."), so
the eight-field records for the new proposals have almost no physics content from genesis.

## Calls for the owner

- Borderline, marked out of scope: L1-283 guided-wave testing (imaging only as tomography), L1-284 digital image correlation
  (measurement on formed images), L1-424 acoustic beamforming (imaging as an acoustic camera). If any is in, it needs a registry entry.
- L1-415 THz-TDS is spectroscopy unless raster-scanned; kept as a proposal on your say.
- `electron_diffraction` stands in for 4D-STEM, but its template is single-pattern; a proper 4D-STEM entry is needed upstream.
- L1-234 and L1-406 both match `phase_retrieval`; L1-406 is primary as you listed it.

## Reproduce

```
export PWM_CORE_SRC=$PWD/Physics_World_Model/packages/pwm_core
python -m pwm_mode.imaging.catalog --build
PYTHONPATH=. python -m pytest tests/test_imaging.py -q
```
