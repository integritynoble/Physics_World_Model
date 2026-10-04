# IMG-01: repair the 5 failing imaging templates, prepare the pwm_core aliases

Branch `lib/LIB-40-img01-templates`, from `lib/LIB-40-imaging-catalog` (d93fdf2). pwm_core: Physics_World_Model master
3899e328, read-only here (tarball sha256 `a270f3cc...9c99`). Public code only, no data, no spend. Nothing was pushed.

## Result
- All 5 templates (`eht_imaging`, `insar`, `radio_astronomy`, `radio_interferometry`, `xray_crystallography`) now compile
  and run. Catalog: **170 of 170** registry modalities run (was 165). Only those 5 entries changed (registered -> runs).
- All 197 graph templates in pwm_core compile and run, on the unpatched tree (with this repo's overlay) and on the
  patched tree (with no overlay and no alias needed).
- The patch `results/IMG-01/pwm_core_templates.patch` (one file, `contrib/graph_templates.yaml`, 13 insertions, 18
  deletions) applies cleanly to a fresh extraction of 3899e328 and yields the same bytes as the edit it was cut from.
  I cannot push to Physics_World_Model, so someone with write access must apply it: from the repo root,
  `git apply -p1 results/IMG-01/pwm_core_templates.patch`.

## Cause
All five templates were the same chain `fft_2d -> subsample(acceleration=4) -> gaussian`, and neither `fft_2d` nor
`subsample` is a registered primitive (`GraphCompilationError: Node 'subsample' references unknown primitive_id`).
`fft_2d` occurs in no other template, and `subsample` in no other.

## What I changed, and one decision that departs from the item text
The item listed four aliases: `propagation_2d`, `radon_2d`, `fft_2d`, `poisson_gaussian_sensor_noise`.

| name | templates | action in the patch |
|---|---|---|
| `propagation_2d` | 11 | renamed to `angular_spectrum` (identical to the alias already applied here) |
| `radon_2d` | 7 | renamed to `ct_radon` |
| `poisson_gaussian_sensor_noise` | 2 | renamed to `poisson_gaussian_sensor` |
| `fft_2d` | 5 (the failing ones) | **not aliased**; removed by rewriting the 5 templates |

Why `fft_2d` is not aliased: this repo had mapped it to `fourier_relay`, but `fourier_relay` is FFT, a transfer function,
then IFFT, so it is a propagation operator and not a Fourier transform. The alias could not have fixed these templates
anyway (they also need `subsample`). I removed it from `PRIMITIVE_ALIASES` so it cannot substitute silently.

I chose renaming the template ids over registering aliases in `PRIMITIVE_REGISTRY` because pwm_core's own tests
parametrise over `PRIMITIVE_REGISTRY.keys()` (`tests/test_gate0_ir_types.py:136,157`), so alias keys would change what they
iterate over. The ids sit behind YAML anchors, so the whole change is four one-line renames and one block.

The 5 templates become `mri_kspace` (FFT, then a random k-space mask with a fully sampled centre block; `sampling_rate =
1/acceleration = 0.25`, seed 42, as in the MRI template) followed by `complex_gaussian_sensor` (sigma 0.01, seed 0). The
old noise node, `gaussian`, casts complex data to real (numpy `ComplexWarning`) and would have dropped the imaginary part of
the visibilities; the test checks the output stays complex with non-zero imaginary spread.

In this repo (pwm_core is read-only): `pwm_mode/imaging/core.py` gets `repair_template`, applied inside `contrib()` to the
five ids and only while a template still has a `subsample` node, so it is a no-op on a patched pwm_core.
`catalog._run` records `repaired_by`. `contrib()` also now closes its file.

## Evidence (commands run from the repo root, `nice -n 19`, `OMP_NUM_THREADS=1`, one process)
`PWM_CORE_SRC=<tree>/packages/pwm_core PYTHONPATH=. python -W ignore results/IMG-01/check_templates.py OUT.json`

| tree | `graph_templates.yaml` sha256 | run | aliases applied | repaired | evidence |
|---|---|---|---|---|---|
| 3899e328 unpatched | `0a1cce42...be8` | 197/197 | 20 | 5 | `templates_unpatched_core.json` |
| 3899e328 + patch | `544009d3...5a2` | 197/197 | 0 | 0 | `templates_patched_core.json` |

The sha256 of the forward output on `rng(0).random(x_shape)` is identical for all 197 templates between the two trees: the
renames change nothing, and the overlay and the patch produce the same operator for the five.

pwm_core's own tests on `test_graph_compiler.py`, `test_gate0_ir_types.py`, `test_graph_equivalence.py` and
`test_canonical_decompositions.py`, run on each tree: **131 failures on the unpatched tree, 6 on the patched tree**, no
failure that is new on the patched tree. The 125 fixed are 5 tests (`test_compile_template`, `test_serialization_roundtrip`, and three node-tag tests) on each of the
25 templates that used the unregistered names (11 + 7 + 2 + 5). The remaining 6 are all in `test_canonical_decompositions.py` (counts of canonical primitives; unrelated;
lists in `pwm_core_tests_pristine_failures.txt` and `pwm_core_tests_patched_failures.txt`).

`tests/test_imaging.py`: 15 passed and 1 expected failure on the unpatched tree (the 7 original tests, one changed to
`runs == 170`, plus 8 new); 14 passed, 1 skipped, 1 expected failure on the patched tree (the patch-apply test skips
because there is nothing left to patch). The new tests cover each of the 5 templates, the alias set, the no-op repair,
and that the `.patch` applies to an unpatched copy and leaves only registered primitive ids.

## What this does not show
- "Runs" means compiles and gives finite output. It does not mean the physics is right. A radio interferometer samples
  (u,v) along baseline tracks, InSAR forms phase from two passes, crystallography measures intensities (the phase is lost),
  and EHT has closure phases. All five are now the same generic random-mask Fourier sampler. They stay at level `runs`.
- **Finding for IMG-07:** `mri_kspace.adjoint` is the unit-normalised inverse FFT (the zero-filled reconstruction), so the
  adjoint dot-product test fails by exactly H*W = 4096 (`Re<Ax,y> = 1573.9`, `<x,A'y> = 0.3843`). The MRI template has the same
  property. I did not change it, because that adjoint is the zero-filled baseline. It is an `xfail(strict)` test, which will
  turn red when pwm_core fixes it, so the mark must then be removed.
- The patch was tested only against pwm_core 3899e328 and with the tests listed above, not the full pwm_core suite.
- I did not change the sampling-rate semantics: with the fully sampled centre block (1/8 of each axis), the actual fraction
  of k-space kept is 0.304 (measured on the 64 x 64 mask), not 1/acceleration = 0.25.
