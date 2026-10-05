# LIB-40 IMG-10 part 2: dead learnable parameters and geometry calibration parameters

Branch `lib/LIB-40-img10-params`, base `lib/LIB-40-imaging-catalog` as in my bundle (de6024a; the `lifecycle.py` commit
is not in that bundle, so the Python side only adds `sensitivity.py` and a test and touches nothing in `lifecycle.py`).
pwm_core: Physics_World_Model master `3899e328`. Public/synthetic data only; the templates' own 64 x 64 inputs, one
process, nice 19. No engine run.

Deliverables (all in this directory):

| file | what |
|---|---|
| `pwm_core_params.patch` | the pwm_core patch, `git apply -p1` on plain `3899e328` (checked) |
| `pwm_core_params_on_img01.patch` | the same patch rebased on `3899e328` + tina1's IMG-01 template patch (checked); needed because both edit the shared `propagation_2d` / `radon_2d` anchors around lines 4006 and 4123 of `graph_templates.yaml` (a fuzz-2 `patch` also works) |
| `sensitivity.json` | the sensitivity scan before and after, per parameter |
| `../../pwm_mode/imaging/sensitivity.py`, `../../tests/test_imaging_params.py` | the finder and its test (the test skips on an unpatched pwm_core) |

## Result in one table

| | before | after |
|---|---|---|
| templates compiled / run | 197 / 192 | 197 / 192 (the same 5 IMG-01 templates fail to compile on plain 3899e328) |
| parameters tested (every `params` and `learnable` entry) | 917 | 904 |
| **dead parameters** (no effect on forward or adjoint output) | **84** (29 learnable) | **0** (0 learnable) |
| shape-tied parameters (`H`, `W`, ...: perturbing them breaks the template, so they act structurally) | 37 | 37 |
| conditionally inactive (reported separately, see below) | not separated | 8 (7 + 1) |
| templates whose output ignores the object | 6 | 1 (`doppler_ultrasound_graph_v2`, see below) |

pwm_core's own tests on the patched tree on top of IMG-01 (`test_graph_adjoint`, `test_graph_compiler`,
`test_graph_equivalence`, `test_golden_runs`, `test_ci_enforcement`, `test_compute_benchmarks`, `test_fc_primitives`,
new `test_graph_params`): all pass (2 skips). On plain `3899e328` the compiler tests already fail
for the alias names (`propagation_2d`, `radon_2d`, `fft_2d`, `poisson_gaussian_sensor_noise`), before and after my patch;
IMG-01 fixes that. `tests/test_imaging.py` still passes (7 tests).

## Method (`pwm_mode/imaging/sensitivity.py`)

For every template and every node, each entry of `params` and of `learnable` is perturbed (bool flip; int +1 / -1, or
x1.5, /2, /10 for larger values; float x1.37, x0.7, x5, x0.1, x20; 1.0 or 0.1 if the value is 0; numeric lists x1.37 and
x0.7). Several magnitudes are used so that saturated or sub-pixel parameters are not called dead. For each perturbed copy
the compiled chain is run on a fixed random object, and **also its adjoint applied to a fixed measurement-shaped array
through the linear nodes**, so adjoint-only parameters count. A parameter is **dead** if the largest relative change over
all perturbations, in forward and adjoint, is below 1e-12; **acts** otherwise (an adjoint that raises because the
parameter is tied to the operator shape also counts as acting). A dead parameter is then retried with a 3-D object and
with the node's other zero-valued parameters set to 0.5, and is reported as `needs_axis` or `gated` if that makes it act.
Outputs are checked to be deterministic (all 192 are) and non-degenerate (none is constant). A static AST check lists
names the primitive source reads and never uses.

Limits: one fixed random object and one measurement array, one operating point per template; a parameter that acts only
for special inputs or only at some other nominal value could still be missed. Timeouts (20 s per forward) can occur under
the machine load (the 5-minute load was 35 to 90); the three templates that timed out in one run were rerun alone and are
in `sensitivity.json` (`after`).

## (a) Dead parameters: what each group became

"wire" = the primitive now reads it; "rename" = the template name was wrong, the primitive reads another name;
"unmodelled" = no primitive uses it, so it is no longer a parameter. Removed names are not lost: they are kept under
`metadata.unmodelled_params` of the template as `node.param: value`, with the value they had.

| group (templates) | decision |
|---|---|
| `acoustic_propagation.speed_of_sound` (ultrasound, doppler_ultrasound, elastography, photoacoustic) | **wire**: a dilation of the time axis of the received signal by `speed_of_sound / nominal_speed` (linear interpolation matrix `R`; the adjoint applies `R^T`, exact); identity at the nominal speed; new template param `nominal_speed` (1540 m/s; 1500 for photoacoustic); learnable |
| `coded_mask.density` (10 coded-mask templates; shared anchor) | **wire**: `mask = rand > 1 - density`, identical to before at the default 0.5 |
| `propagation_2d.distance_m` (11 templates; shared anchor) | **rename** to `distance` (SI metres, 0.001, the old 1.0 m was never read); learnable; the output is unchanged |
| `fresnel_prop.wavelength_um, pixel_size_um, z_um` (compressive_holography) | **rename** to `wavelength` 5.32e-07, `pixel_size` 5e-06, `distance` 1e-04, all in metres (the old values would have been microns and wrong in metres); `distance` learnable. Output changes |
| `spectral_dispersion.step` (7 templates; shared anchor) | **rename** to `disp_step` (2.0); but see "conditional" |
| `elastic_wave_model.shear_speed_m_per_s` | **rename** to `wave_speed` 3.0 (the primitive's default was 1.5, so the gain doubles) |
| `yield_model.quantum_yield` (fluorescence_microscopy) | **rename** to `yield_coeff` 0.7 (the primitive default was 0.3) |
| `electron_beam_source`: the source ignored the object (`current * ones_like(x)`) | **fix**: `current * x`. The five electron templates (sem, tem, electron_tomography, stem, cryo_em) were object-independent; they now depend on the object |
| `electron_detector_sensor.dqe` (cryo_em) | **wire**: a multiplier in forward and adjoint, default 1.0; the template's 0.85 now applies. Output changes |
| `sequence_block.flip_angle_deg` (fmri, mrs) | **wire**: weight x `sin(flip)`, 1 at the default 90 degrees |
| `objective_lens.vignetting` (cacti) | **wire**: radial `1 - v r^2` factor, exact adjoint, default 0 |
| `thin_object_phase.sigma` (tem) | acts after the beam-source fix (it was dead only because the object never reached it) |
| `frame_integration.T` (cassi) | not a defect: the first scan called it dead because the perturbed adjoint raised; the tool now counts that as acting |
| **unmodelled**, moved to `metadata.unmodelled_params`: `dmd_pattern_sequence.sampling_rate`, `doppler_estimator.prf_hz, wall_filter_cutoff`, `elastic_wave_model.push_duration_us`, `electron_beam_source.accelerating_voltage_kv, voltage_kv`, `energy_resolving_detector.n_channels, energy_range_ev`, `evanescent_decay.penetration_depth`, `fiber_bundle_sensor.core_pitch`, `fluoro_temporal_integrator.decay`, `gaussian_splatting_stub.n_views`, `volume_rendering_stub.n_views`, `interference.carrier_freq`, `photon_sensor.offset`, `shutter_integration.clock_offset`, `ray_trace.focal_length`, `reciprocal_space_geometry.detector_distance`, `sar_backprojection.swath_width`, `spad_tof_sensor.dead_time_ns, n_bins`, `structured_light_projector.n_patterns`, `thin_object_phase.thickness_nm`, `track_detector_sensor.n_layers`, `vessel_flow_contrast.n_frames, threshold`, `tof_gate.n_bins` | **remove** from `params` / `learnable`. Each is a physical quantity the primitive does not model; calibrating it would be a no-op, and Gate 3 would be told it can be recovered. If a primitive later models one, move it back |

### Conditionally inactive (kept, not dead)

* `spectral_dispersion.disp_step` in the 7 templates cup, ftir_imaging, ghost_imaging, hdr_imaging, multispectral_sat,
  ocean_color, streak_camera: their `x_shape` is 2-D, so dispersion has no spectral axis to act on and is a no-op in
  these templates. The name is now right and it acts as soon as the object has a spectral axis (`needs_axis`).
  Giving the 7 templates a spectral object is a structural change, not done here.
* `imbalanced_response.disp_axis` in cassi: acts only if the gain-imbalance parameter `a2` is non-zero (`gated`).

### Findings left as they are

* `doppler_ultrasound_graph_v2`: the output does not depend on the object (the estimator returns zeros for a 2-D input).
  Its two parameters (`prf_hz`, `wall_filter_cutoff`) were moved to unmodelled; the estimator itself is untouched.
* `tirf` and `octa` need a 3-D object for their z / time parameters to matter (they act in the 3-D retry).
* `interference` is the identity in the sequential chain, so `carrier_freq` was moved to unmodelled.
* The 18 primitive-level names that the source reads and does not use (static scan in `sensitivity.json`) are listed there;
  those that appear in no template (`collimator_model.septal_thickness`, `doppler_estimator.prf`, `n_ensembles`,
  `elastic_wave_model.frequency`, `fluoro_temporal_integrator.n_frames`, `scan_trajectory.scan_type`,
  `sequence_block.sequence_type`) were not touched.
* Pre-existing bug fixed on the way: `subpixel_shift_2d(arr, dx, dy)` was called with `(mask_dy, mask_dx)` in the SPC and
  temporal-mask paths, so `mask_dx` shifted vertically. The three calls I touched or added now pass `(dx, dy)`. SPC with
  a non-zero `mask_dx` or `mask_dy` changes; at 0 nothing changes. (The other `subpixel_shift_2d` calls of the file,
  `warp_shift`, were not examined.)

## (b) Geometry calibration parameters added

All are identity at their default (0), act on forward and adjoint, and the adjoint is the exact transpose (dot test in
`tests/test_graph_params.py`). `learnable` lists them, and `mismatch_db.yaml` `graph_param_map` lets the mismatch sampler
inject them.

| parameter (same name as mismatch_db, for IMG-03) | where | meaning | before |
|---|---|---|---|
| `detector_offset` (px) | `ct_radon`, in ct_graph_v1, ct_graph_v2, cbct_graph_v2 and the radon anchor shared by 7 more templates (brachytherapy_img, cryo_et, digital_breast_tomo, industrial_ct, portal_imaging, proton_therapy_img, xray_ndt) | centre-of-rotation / detector shift, sub-pixel by linear interpolation of the detector axis | only `gain` was calibratable |
| `angle_offset` (rad) | same nodes | rotation of the projection angles (mismatch_db name `projection_angle_offset`, mapped in `graph_param_map`) | none |
| `speed_of_sound` (m/s) | `acoustic_propagation`: ultrasound, doppler_ultrasound, elastography, photoacoustic | see (a) | dead |
| `mask_dx`, `mask_dy` (px) | `coded_mask` and `temporal_mask`: cassi `modulate`, cacti `coded_aperture` | sub-pixel mask misregistration relative to the detector | none (spc already has them) |
| `distance` (m) | `propagation_2d`/`angular_spectrum` anchor (11 templates), compressive_holography `fresnel_prop` | propagation distance | dead name `distance_m` / `z_um` |
| `disp_step` | `spectral_dispersion` (7 templates; needs a spectral axis) | dispersion per band | dead name `step` |

`graph_param_map` entries added: `ct`, `cbct`: `detector_offset: radon.detector_offset`; `photoacoustic`, `ultrasound`:
`speed_of_sound: propagate.speed_of_sound`; `cassi`: `mask_dx`, `mask_dy` to `modulate.*`; `cacti` (new map): `mask_dx`,
`mask_dy` to `coded_aperture.*`. (The sampler also injects a name by scanning every node for a parameter of that name,
so the names work even without a map entry.)

### Names for tina1 / IMG-03 (calibration tolerances)

Use the mismatch_db names above: `detector_offset` (px; the flagship's CT "centre of rotation ~ 2 px"), `angle_offset`
(rad; mismatch_db `projection_angle_offset`), `speed_of_sound` (m/s), `mask_dx` / `mask_dy` (px; the flagship's CASSI
"mask < 0.25 px"), `distance` (m), `disp_step`, `gain`. The graph node that carries each is the "node.param" in the table
(`radon.detector_offset`, `propagate.speed_of_sound`, `modulate.mask_dx`, `coded_aperture.mask_dx`). I could not see
tina1's IMG-03 branch from here (no such branch in my bundle), so the agreement is through this file and the commit message.
If IMG-03 uses another name for any of these, one of us renames; the node-level names are what the code reads.

### Geometry parameters NOT added (honest list)

SPECT `center_of_rotation`, neutron_tomo `rotation_axis_offset`, ptychography probe position, sonar `speed_of_sound`,
and any geometry of the cone-beam templates (`cone_beam_radon` already reads `detector_offset` in its own code and was
not touched). Each needs its own
primitive change; the sensitivity test will flag any of them that is added as a no-op.

## Output-changing edits (everything else is the identity at the defaults)

* elastography: `wave_speed` 3.0 instead of the default 1.5, so the elastic gain doubles;
* fluorescence_microscopy: yield 0.7 instead of the default 0.3;
* cryo_em: `dqe` 0.85 now applies;
* compressive_holography: wavelength, pixel size, distance now take effect, in SI units;
* the five electron templates depend on the object now (before they ignored it);
* cacti and fmri/mrs: the `vignetting` and `flip_angle_deg` values in the templates are the identity defaults, so no change;
* SPC with a non-zero `mask_dx` / `mask_dy` (the axis fix above).

## What this does not show

The test shows that a parameter acts, not that its model is right: the new shifts are linear interpolation on the grid, the
speed-of-sound dilation is a time-axis scaling and not a ray-traced delay, and `detector_offset` shifts the whole
sinogram, which is the right model for a rotation-axis shift in parallel beam only. No reconstruction or calibration
recovery was run, so whether a calibration can recover these parameters from data is not claimed here (that is IMG-03 /
Gate 3).
