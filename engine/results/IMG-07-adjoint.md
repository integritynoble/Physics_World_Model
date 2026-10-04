# IMG-07: C5 adjoint tests for the running templates of the imaging library

Branch `lib/LIB-40-img07-adjoint` (from `lib/LIB-40-imaging-catalog` d93fdf2). pwm_core: Physics_World_Model master 3899e328 (read-only, not changed). Validator: verifier1; the library session built the catalog and the checker.

## Result

**Of the 192 templates that compile, 156 pass the dot-product test and 36 fail. The other 5 of the 197 templates do not compile** (unknown primitive `subsample`). The IMG-07 text says "165 running templates"; 165 is the number of running *modalities*. There are 192 running templates, and all are tested.

| | templates | modalities (running: 165) |
| --- | --- | --- |
| pass | 156 | 131 modalities have every template passing |
| fail | 36 | 34 modalities have at least one failing template (27 have no passing template, 7 have one passing and one failing) |
| do not compile | 5 | the 5 `registered`-level modalities: eht_imaging, insar, radio_astronomy, radio_interferometry, xray_crystallography |

13 of the 39 `canonical` or `real_data` modalities are affected: spc, ct, mri, ultrasound, cbct, cryo_em, compressive_holography (real_data); photoacoustic, pet, spect, sem, tem, elastography (canonical). CT, MRI, PET, SPECT, ultrasound, SEM and TEM are all in this list.

A pass means: the adjoint of every maximal run of consecutive linear primitives in that template is the transpose to float64 round-off (worst passing relative error 2.6e-14). It does not mean the physics is right (a wrong operator can have an exact adjoint), and it says nothing about the non-linear primitives (noise, Beer-Lambert, detection), which a dot test cannot cover.

## What is tested

`pwm_mode/imaging/adjoint.py`, `tests/test_imaging_adjoint.py`, full results in `img07_adjoint.json` (this folder; regenerate with `PWM_CORE_SRC=… python -m pwm_mode.imaging.adjoint --out img07_adjoint.json`, under a minute).

* Each template is compiled with `core.compile_graph` after the 4 primitive-name aliases, and its `forward_plan` is cut into maximal runs of consecutive `is_linear` primitives. Each run is tested as its own linear operator at the shape and dtype it really receives (taken from a forward pass of a seeded input). All 192 templates end in a non-linear primitive (166 only at the end, 26 also in the middle), so none is all-linear and pwm_core's own `GraphOperator.check_adjoint` (all-linear graphs only) never runs on them.
* Test: `<Ax, y> = <x, A^T y>` with x and y standard normal (complex if the operator's input or output is complex), 3 draws, seed 20261005. Real input gives the real inner product on the real part; complex input the complex one. Relative error `|lhs - rhs| / max(|lhs|, |rhs|)`, tolerance **1e-8**. The choice does not matter: every passing run is below 3e-14 and every failing one above 0.14, so any tolerance from 1e-12 to 1e-2 gives the same 156/36 split (pinned by `test_tolerance_is_not_what_decides`).
* A failing run is diagnosed: linearity (additivity, response to zero, repeatability), a dot test of each primitive on its own, the ratio rhs/lhs (a constant ratio is a pure normalization error), the error under the real inner product (complex data), and the error after reshaping when only the shape is wrong.
* The test file has two kinds of tests. `test_dot_product[template]` must pass for 156 templates; the 36 known failures are strict xfails with the primitive recorded, so a fix shows as XPASS (red) and a template failing in a different place is caught by `test_known_failure_is_where_recorded`. Two tests check the checker itself (accepts an exact adjoint; rejects a wrong scale, a 0.1% error, an affine map, a shape error, a missing adjoint, a missing conjugate, a dropped imaginary part); four pin the explanations below at primitive level. Result here: 214 passed, 36 xfailed, with `tests/test_imaging.py` included (19 s, one process, nice 19).

## Why the 36 fail (by primitive)

Causes were found by reading the primitive code in `pwm_core/graph/primitives.py` and then confirmed numerically. The fixes are suggestions; pwm_core was not touched.

| Primitive | Templates | Cause | Evidence | Suggested fix |
| --- | --- | --- | --- | --- |
| `ct_radon` | 10: ct_graph_v1, ct_graph_v2, cbct_graph_v2, brachytherapy_img, cryo_et, digital_breast_tomo, industrial_ct, portal_imaging, proton_therapy_img, xray_ndt | The exact system matrix is built only when H*W <= 1024 (up to 32x32). The templates use 64x64, which takes the fallback "rotation-based backprojection (approximate)", per the code's own comment. | with 16 angles: exact at 16x16 and 32x32 (4e-16, 2e-15), 0.15 at 40x40, 0.48 at 64x64; in the templates the run error is 0.98 and the rhs/lhs ratio about 19, not constant | build the matrix for any size (sparse), or transpose the interpolation exactly |
| `mri_kspace` | 5: mri_graph_v1, mri_graph_v2, diffusion_mri, fmri, mrs | Forward is the unnormalised `fft2`; adjoint is `ifft2`, which carries 1/(H*W). The adjoint is 1/4096 of the true one. | rhs/lhs is constant at 2.44e-4 = 1/4096 (1/256 at 16x16), error 1.0 | adjoint = H*W*ifft2, or `norm="ortho"` in both |
| `sequence_block` (with mri_kspace) | fmri, mrs | Casts its complex k-space input to float64 and drops the imaginary part (numpy ComplexWarning), so the primitive is not complex-linear and discards half the k-space data. Forward and adjoint are consistent only under the real inner product. | error 1.0 under the complex product, 6e-15 under the real one | keep the complex dtype |
| `electron_beam_source` | 5: sem, tem, electron_tomography, stem, cryo_em | A source (Pattern B) that ignores x: forward is `current * ones_like(x)`. That is an offset, not a linear map, although `is_linear` is True. | response to zero 0.91, additivity error 0.09 | treat as a constant source term, or set `is_linear` False |
| `acoustic_propagation` | 4: photoacoustic, doppler_ultrasound, elastography, ultrasound | Forward rotates and sums; the "adjoint" tiles and rotates back (time-reversal approximation), not the transpose, and also divides by n_angles. | per-node error 0.98 at 64x64, 0.90 at 16x16 | explicit transpose |
| `emission_projection` | 2: pet, spect | Same pattern as `ct_radon` (rotate-and-sum forward, tile-and-rotate "adjoint"). | per-node error 0.31 (64x64), 0.62 (16x16) | explicit transpose |
| `sar_backprojection` | 1: sar | Same pattern. | per-node error 0.67 (64x64), 0.62 (16x16) | explicit transpose |
| `motion_warp` | 2: light_field (dx=dy=0.5), integral (0.3) | The adjoint is the shift by the negative amount. Exact for integer shifts (panorama_graph_v1 uses 2 and 1 and passes at 4e-16), not for fractional ones with `mode="constant"`. | 0.41 at 0.5, 0.84 at 0.3, 0 at 0, 1, 2 | exact transposed interpolation, or integer shifts only |
| `photon_sensor` | spc_graph_v2; compressive_holography_graph_v2 | In spc: `dark_current=0.01` makes it affine (the adjoint ignores it); this is the only template with a non-zero dark current. In compressive holography: the input is complex (from `fresnel_prop`) and it casts to float64, like `sequence_block`. | spc: response to zero 0.02, error 0.067; holography: 0.99 complex, 3e-15 real | treat the offset separately; keep dtype |
| `interference` | compressive_holography | `is_linear` is True but the adjoint raises `NotImplementedError("… non-linear interference")`; its `forward_multi` is `|s + r|^2`. The flag and the behaviour contradict each other. | adjoint not implemented | set `is_linear` False |
| `frame_integration` | panorama_graph_v2 | Adjoint ignores `axis` and stacks `T` copies on the last axis. The template has axis 0, T 1, x_shape 64x64: forward gives (64,), adjoint gives (64, 1) instead of (64, 64). | shape error | use x's own shape and axis |
| `temporal_mask` | flim_graph_v2 | With a 2D x and T=1 the forward is `x * mask`, but the adjoint always adds a trailing T axis: (64, 64, 1). Numerically correct: error 1e-14 after reshaping. | shape error only | squeeze when T=1 |
| `fluoro_temporal_integrator` | 2: fluoroscopy, angiography | Forward on 2D x is a copy (motion_sigma 0), the adjoint always stacks `n_frames` (8) copies: (8, 64, 64). | shape error | follow x's dimension |
| `beamform_delay` | sonar_graph_v2; also in ultrasound_graph_v2, where the shapes happen to line up and it shows as a value mismatch | Forward is the mean over axis 0 (64 rows); the adjoint tiles `n_elements` (32) copies and omits the 1/64. | shape error (32 vs 64 rows) | tile to x.shape[0] and divide by it |

The shape errors (4 rows above, 5 templates) are failures of the adjoint contract, not of the mathematics in every case: `temporal_mask` is exact once reshaped. I count them as failing because a solver that feeds the adjoint output back as an image gets the wrong shape.

## The 5 templates that do not compile

eht_imaging, insar, radio_astronomy, radio_interferometry and xray_crystallography use a node with primitive_id `subsample`, which is not in the 101-primitive registry. They are the 5 `registered`-level modalities in the catalog and cannot be tested. I did not pick an equivalent primitive for them; that is a modelling decision for the library or pwm_core.

## Comparison with the design checker's own C5 (`design.py`)

`design.check_design` runs its own C5. On the 192 compiling templates it gives the same verdict as the per-run test on 189 and a different one on 3. The 26 templates with a non-linear primitive in the middle (where `design._linear_part` strips it and tests the remaining hybrid chain) give identical verdicts: 18 pass, 8 fail. The three differences are weaknesses of the checker's test, not of the templates:

* **ptychography_graph_v1** (exact adjoint, design says fail, 1.5% error). The output is complex and the checker draws a *real* y, so `<Ax, y>` has an imaginary part that the transpose cannot match. With the real part, the error is 2e-16.
* **sar_graph_v2** (approximate adjoint, design says pass at 2.5e-5). x and y are uniform on [0, 1), so the sums are dominated by the mean and a roughly adjoint pair looks exact. Over five draws the error is 2.5e-5 to 8e-4 with uniform inputs and 0.05 to 0.26 with zero-mean inputs. It passes the 1e-4 limit only on the seed it uses.
* **flim_graph_v2** (adjoint of the wrong shape, design says pass). `np.vdot` flattens both arguments, so (64, 64, 1) against (64, 64) goes unnoticed.

I did not change `design.py`. I recommend: zero-mean inputs, a complex y for a complex output, a shape check, and a tighter tolerance; the checker would then give the same 156/36 split as here. Until then C5 can pass a design that has an approximate adjoint (sar) and fail an exact one (ptychography).

## Limits of this check

* The template is tested as shipped: `x_shape` and parameters from `graph_templates.yaml`. A primitive that is exact at a small size (`ct_radon` up to 32x32) fails at the template's size; the opposite case is not excluded, since other sizes were not scanned.
* The plan is the compiler's topologically sorted `forward_plan`, applied in sequence. Edges and branches are flattened, and multi-input nodes (`forward_multi`, e.g. in `interference`) are not exercised.
* One seed and 3 draws per run. With errors separated by 12 orders of magnitude between passes and failures this does not affect the verdicts, but rare conditions (a draw with a near-zero inner product) are not scanned.
* For a run with real input the test uses the real part of the inner product (complex y when the output is complex). That is the right test for a real-to-complex operator such as ptychography, but it does not check the imaginary part of the products; that is checked only when the input of a run is complex.
* Parameter values at which a primitive changes behaviour (other `n_channels`, `motion_sigma` > 0) are not scanned.
* I did not check whether any solver in the engine uses these adjoints, so the effect of the 36 failures on reconstructions is not assessed.
* Running the tests without `PWM_CORE_SRC` uses the repo's default path, which is the owner's tree (`/home/spiritai/...`). I set it to my own extraction of the pinned tarball.
