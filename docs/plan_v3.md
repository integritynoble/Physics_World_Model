# PWM v3.1 Roadmap: Physics-Correct Graph Execution

**Previous milestone:** v3.0 (2026-02-11) — see `v3_implementation_status.md`
**This document:** Forward-looking roadmap for v3.1 hardening

---

## Diagnosis of v3.0 gaps

v3.0 established the canonical chain (Source→Elements→Sensor→Noise) but left several
structural and physics-fidelity issues open. This roadmap addresses 10 work items
organized into 4 implementation waves.

| # | Gap | Severity | Root cause |
|---|-----|----------|------------|
| 1 | Single-input primitives only | High | `PrimitiveOp.forward(x)` has no fan-in path |
| 2 | No physics_subrole on elements | Medium | Validator lumps all non-source/sensor/noise as "element" |
| 3 | NoiseModel = Objective (conflated) | High | `_infer_objective_from_noise()` hardcodes NLL from primitive_id |
| 4 | Operator correction is unstructured | Medium | `provided_operator` replaces entire A, no graph node |
| 5 | Exactly-1 sensor enforced | Medium | Multi-coil MRI / multi-channel sensors impossible |
| 6 | CT/PA/NeRF/3DGS templates wrong | High | Missing Beer-Lambert, acoustic propagation, volume rendering |
| 7 | PhysicsTier unused | Low | Enum exists but no selection policy or tier-tagged primitives |
| 8 | BeliefState ignores ParamSpecs | High | `build_belief_from_graph` uses naive ±2x, ignores node metadata |
| 9 | Prompt parser returns untyped output | Medium | `ParsedPrompt` is a plain class, not `ExperimentSpec` |
| 10 | Plan doc = status doc (mixed) | Low | Retrospective claims mixed with roadmap items |

---

## Work items

### R1: Multi-input graph nodes; Source/Interaction consume x explicitly

**Problem.** Every primitive has `forward(x) -> ndarray`. Nodes with multiple incoming
edges (fan-in) — e.g., interference (signal + reference), SIM (scene × pattern),
or interaction nodes receiving both a field and a modulation — cannot be expressed.
Source primitives blindly compute `x * strength` without documenting that `x` is
the scene under illumination.

**Changes.**

`graph/ir_types.py`:
- Add `PortSpec(StrictBaseModel)`: `name: str`, `tensor_spec: Optional[TensorSpec]`, `required: bool = True`
- Add `input_ports: List[PortSpec]` and `output_ports: List[PortSpec]` to `NodeTags`
  (both default to single anonymous port for backward compat)

`graph/primitives.py` — `BasePrimitive`:
- Add `_n_inputs: int = 1` class attribute
- Add `forward_multi(inputs: Dict[str, np.ndarray]) -> np.ndarray` method with
  default impl: `return self.forward(next(iter(inputs.values())))`
- Source primitives: docstring update — `forward(x)` receives the scene/object,
  returns illuminated field `strength * x` (no signature change needed since source
  is always first in chain and receives the pipeline input)
- New multi-input primitives (for R6): `Interference(forward_multi({"signal": s, "reference": r}))`

`graph/compiler.py`:
- During topo-sort binding, count incoming edges per node
- Validate `len(incoming_edges) == prim._n_inputs` (or ≤ if optional ports)
- Pass `_n_inputs` into compiled plan metadata

`graph/executor.py`:
- Replace linear `for node_id, prim in forward_plan` with topological DAG execution:
  - Maintain `outputs: Dict[str, ndarray]`
  - For each node in topo order, gather predecessors' outputs
  - If `prim._n_inputs == 1`: call `prim.forward(predecessor_output)`
  - If `prim._n_inputs > 1`: call `prim.forward_multi(gathered_dict)`
- `_StrippedGraphOp` similarly updated

**Tests.** `tests/test_multi_input.py`:
- 2-input interference node compiles and executes
- Source node correctly receives x and applies illumination
- Backward compat: all existing single-input graphs still work
- Adjoint of multi-input graph raises clear error

**Lines:** ~180 new/modified

---

### R2: physics_subrole for element nodes; require Interaction/Transduction

**Problem.** `NodeRole` has `transport` and `interaction` but the canonical validator
treats all non-source/sensor/noise nodes identically as "elements." There is no way
to require that a photoacoustic graph has a carrier-transition node or that CT has
a domain-change node. Element nodes are an undifferentiated bag.

**Changes.**

`graph/ir_types.py`:
- Add `PhysicsSubrole(str, Enum)`:
  - `propagation` — free-space propagation (Fresnel, angular spectrum, acoustic wave)
  - `modulation` — coded mask, DMD, SIM pattern
  - `sampling` — Radon, k-space, random mask
  - `interaction` — carrier-type transition (photon→acoustic, photon→electron)
  - `transduction` — domain change within same carrier (intensity→log-intensity)
  - `encoding` — temporal/spectral encoding (FLIM, spectral dispersion)
  - `relay` — Fourier relay, identity propagation segment

`graph/graph_spec.py` — `GraphNode`:
- Add `physics_subrole: Optional[PhysicsSubrole] = None`

`graph/primitives.py`:
- Add `_physics_subrole: Optional[str] = None` class attribute to `BasePrimitive`
- Set on all 40 primitives:
  - `FresnelProp`, `AngularSpectrum` → `propagation`
  - `CodedMask`, `DMDPattern`, `SIMPattern` → `modulation`
  - `CTRadon`, `MRIKspace`, `RandomMask` → `sampling`
  - `SpectralDispersion`, `TemporalMask` → `encoding`
  - New `BeerLambert`, `OpticalAbsorption` → `transduction` / `interaction`
  - `FourierRelay` (R7) → `relay`
  - `RayTrace`, `Conv2d`, `Conv3d` → `propagation`

`graph/canonical.py`:
- Add `metadata.carrier_transitions` support: list of `"carrier_a->carrier_b"` strings
- If present, require at least one node with `physics_subrole in (interaction, transduction)`
  whose `carrier_type` matches the transition
- Warn (don't reject) if carrier_transitions not declared but source/sensor carrier types differ

`graph/compiler.py`:
- During tag derivation, populate `physics_subrole` from `node.physics_subrole`
  or primitive class `_physics_subrole`

**Tests.** `tests/test_subrole.py`:
- Photoacoustic graph with carrier_transitions enforced → needs interaction node
- Photoacoustic graph without interaction node → rejected
- All 26 v2 templates still compile (backward compat: carrier_transitions is optional)
- Subrole populated in compiled tags

**Lines:** ~200 new/modified

---

### R3: Separate NoiseModel from Objective; objective overridable

**Problem.** Noise is triple-duty: (1) samples noise in Mode S via primitive `forward()`,
(2) its `primitive_id` is mapped to an NLL kind in `_infer_objective_from_noise()`,
(3) each noise primitive has a `likelihood()` method that is never called.
The user cannot override the objective (e.g., use Huber loss for robustness even when
noise is Poisson-Gaussian) without bypassing the entire inference.

**Changes.**

New file `objectives/noise_model.py`:
- `NoiseModelSpec(StrictBaseModel)`: `kind: str`, `params: Dict[str, Any]`
  (extracted from noise primitive params)
- `NoiseModel(ABC)`:
  - `sample(y_clean: ndarray, rng: np.random.Generator) -> ndarray` (Mode S)
  - `log_likelihood(y: ndarray, y_clean: ndarray) -> float` (scoring)
  - `default_objective() -> ObjectiveSpec` (infer default NLL for Mode I)
- Concrete implementations:
  - `PoissonGaussianNoiseModel(peak_photons, read_sigma)`
  - `ComplexGaussianNoiseModel(sigma)`
  - `PoissonOnlyNoiseModel(peak_photons)`
  - `GaussianNoiseModel(sigma)` — for acoustic/electronic noise
- `NOISE_MODEL_REGISTRY: Dict[str, Type[NoiseModel]]`
- `build_noise_model(spec: NoiseModelSpec) -> NoiseModel`

`graph/primitives.py`:
- Noise primitives (`PoissonGaussianSensorNoise`, etc.) delegate to `NoiseModel.sample()`
  internally instead of reimplementing sampling logic
- Remove duplicated `likelihood()` methods (now on `NoiseModel`)

`graph/executor.py`:
- `GraphExecutor.__init__`: build `NoiseModel` from noise primitive params
- `_simulate()`: `noise_model.sample(y_clean, rng)` instead of `prim.forward(y_clean)`
- `_invert()`: `config.objective_spec or noise_model.default_objective()` — user can
  override with any objective (Huber, Tukey, custom) regardless of noise type
- Remove `_infer_objective_from_noise()` (replaced by `noise_model.default_objective()`)

`graph/graph_spec.py`:
- Deprecate `OperatorGraphSpec.noise_model: Optional[NoiseSpec]` (redundant with noise node)

**Tests.** `tests/test_noise_model.py`:
- PoissonGaussian sampling produces correct statistics (mean, variance check)
- `default_objective()` returns expected NLL kind
- User can override objective: Huber loss on Poisson-Gaussian noise graph
- NoiseModel and Objective are independently constructible
- `build_noise_model` round-trip from spec

**Lines:** ~300 new/modified

---

### R4: OperatorCorrectionNode (structured correction)

**Problem.** `ExecutionConfig.provided_operator` replaces the entire graph-derived A
with an external operator. This is all-or-nothing — there is no way to express
"the graph model A is approximately correct but needs a per-pixel gain correction"
or "add a learned residual." Structured correction is essential for model-measurement
mismatch beyond simple parameter calibration.

**Changes.**

`graph/ir_types.py`:
- Add `correction` to `NodeRole` enum
- Add `CorrectionKind(str, Enum)`: `affine`, `residual`, `lut`, `field_map`

`graph/primitives.py`:
- New primitive family (role=`correction`):
  - `AffineCorrectionNode`: per-element `y_corrected = gain * y + offset`
    - params: `gain: ndarray` (or scalar), `offset: ndarray` (or scalar)
    - `_n_inputs = 1`, `_node_role = "correction"`, `_is_linear = True`
    - Learnable: gain, offset
  - `ResidualCorrectionNode`: additive residual `y_corrected = y + R(theta)`
    - params: `residual: ndarray` (dense or sparse)
    - `_node_role = "correction"`
  - `FieldMapCorrectionNode`: multiplicative field map (e.g., B0 inhomogeneity in MRI)
    - params: `field_map: ndarray`
    - `_node_role = "correction"`
- Register all in PRIMITIVE_REGISTRY

`graph/canonical.py`:
- Allow 0 or 1 correction nodes in the canonical chain
- Valid placement: between last element and sensor, or between sensor and noise
- Correction node does not count as an "element" for the ≥1 element rule

`graph/executor.py`:
- `_StrippedGraphOp`: include correction nodes in the forward/adjoint chain
  (strip only noise, keep correction)
- Deprecate `ExecutionConfig.provided_operator` in favor of graph-based correction
  (keep working for backward compat but log deprecation warning)

`graph/graph_spec.py`:
- `GraphNode` already supports `role` and `learnable`; correction nodes use these

**Tests.** `tests/test_correction_node.py`:
- Affine correction compiles and runs forward/adjoint
- Residual correction compiles
- Correction node placement validated by canonical.py
- Mode I with affine correction produces different recon than without
- Backward compat: `provided_operator` still works (with deprecation warning)

**Lines:** ~250 new/modified

---

### R5: Multi-channel sensors in one SensorNode

**Problem.** The canonical validator enforces exactly 1 sensor node with no way to
model multi-channel detection (MRI multi-coil, RGB camera, polarimetric sensor).
The current `CoilSensor` primitive applies a single coil sensitivity; real MRI
has 8–32 coils producing parallel k-space data.

**Changes.**

`graph/primitives.py`:
- `PhotonSensor`: add `n_channels: int = 1`, `channel_responses: Optional[List[float]]`
  - If `n_channels > 1`: `forward(x)` returns `(n_channels, *x.shape)` array
    with per-channel QE/gain applied
- `CoilSensor`: add `n_coils: int = 1`, `sensitivity_maps: Optional[ndarray]`
  - If `n_coils > 1`: `forward(x)` returns `(n_coils, *x.shape)` with per-coil
    sensitivity multiplication
  - `adjoint(y)`: sum over coils with conjugate sensitivity
- `GenericSensor`: add `n_channels: int = 1`

`graph/canonical.py`:
- Still requires exactly 1 SensorNode (multi-channel is internal to the node)
- No validator changes needed

`graph/executor.py`:
- Noise application: if sensor output has channel dimension, apply noise per-channel
  independently
- `_StrippedGraphOp`: handle shape change from multi-channel sensor

`contrib/graph_templates.yaml`:
- `mri_graph_v2`: update `coil_sensor` params to include `n_coils: 8` (was 1 implicit)
- Add metadata `y_shape: [8, 64, 64]` reflecting multi-coil output

**Tests.** `tests/test_multi_channel.py`:
- PhotonSensor with n_channels=3 produces (3, H, W) output
- CoilSensor with n_coils=8 produces (8, H, W) complex output
- CoilSensor adjoint reduces back to (H, W)
- Single-channel sensors still produce (H, W) — no regression
- MRI v2 template compiles with n_coils=8

**Lines:** ~200 new/modified

---

### R6: Fix CT, Photoacoustic, NeRF/3DGS templates

**Problem.** Four templates have physically wrong forward models:

| Template | Issue |
|----------|-------|
| CT | Missing Beer-Lambert law; Radon acts on attenuation directly but detector measures transmitted photon counts I = I₀·exp(−∫μ dl), not μ itself |
| Photoacoustic | Uses `conv2d` (Gaussian blur) as "acoustic propagation"; carrier transition photon→acoustic is absent; noise is Poisson (should be Gaussian for acoustic) |
| NeRF | `ray_trace` primitive is `scipy.ndimage.zoom` (magnification warp); no volume rendering, transmittance, or multi-view geometry |
| 3DGS | Same `ray_trace` issue; no Gaussian splatting projection or alpha compositing |

**Changes.**

`graph/primitives.py` — new primitives:
- `BeerLambert` (subrole: `transduction`, tier: `tier1_approx`):
  - `forward(sinogram)`: `I_0 * np.exp(-sinogram)` — models photon transmission
  - `adjoint(y)`: `-I_0 * np.exp(-sinogram_cached) * y` (derivative of exp)
  - params: `I_0: float` (source intensity)
  - `_is_linear = False` (exponential is nonlinear)
- `OpticalAbsorption` (subrole: `interaction`, carrier: `photon→acoustic`):
  - `forward(fluence)`: `grueneisen * mu_a * fluence` — converts absorbed light to
    initial pressure via Grüneisen parameter
  - `_n_inputs = 1`, `_is_linear = True` (linear in fluence)
  - params: `grueneisen: float`, `mu_a: float`
- `AcousticPropagation` (subrole: `propagation`, carrier: `acoustic`):
  - `forward(p0)`: acoustic wave propagation operator (simplified: circular Radon or
    k-space propagation model for initial implementation)
  - `adjoint(y)`: time-reversal operator
  - params: `speed_of_sound: float`, `n_sensors: int`, `dt: float`
  - `_physics_tier = tier1_approx`
- `VolumeRenderingStub` (subrole: `propagation`, tier: `tier3_learned`):
  - `forward(x)`: stub that documents the interface — input is a density/color volume,
    output is rendered 2D image. Initial impl: simple maximum intensity projection.
  - Raises `NotImplementedError("Full volume rendering requires PyTorch/JAX backend")`
    if `full_mode=True`
  - params: `n_views: int`, `render_mode: str` (mip/quadrature)
- `GaussianSplattingStub` (subrole: `propagation`, tier: `tier3_learned`):
  - `forward(x)`: stub with correct interface — input is Gaussian params,
    output is rendered 2D image. Initial impl: weighted sum of 2D Gaussians.
  - params: `n_views: int`, `image_size: List[int]`

`contrib/graph_templates.yaml` — rewrite 4 v2 templates:

**CT v2 (corrected):**
```
xray_source → ct_radon (sampling) → beer_lambert (transduction)
→ photon_sensor → poisson_only_sensor
```
- `metadata.carrier_transitions: []` (stays photon throughout, transduction is
  intensity↔log-intensity domain)
- `beer_lambert.I_0 = 10000.0` (source photon budget)
- Reconstruction note: solver operates on log-transformed data
  `y_log = -log(y_noisy / I_0)` which linearizes to `y_log ≈ Radon(μ) + noise`

**Photoacoustic v2 (corrected):**
```
photon_source → optical_absorption (interaction: photon→acoustic)
→ acoustic_propagation (propagation) → transducer_sensor
→ gaussian_sensor_noise
```
- `metadata.carrier_transitions: ["photon->acoustic"]`
- Noise is `gaussian_sensor_noise` (thermal + electronic), not Poisson
- New noise primitive `GaussianSensorNoise` (if not already registered,
  alias of `ComplexGaussianSensorNoise` restricted to real domain)

**NeRF v2 (corrected):**
```
generic_source → volume_rendering_stub (tier3_learned)
→ generic_sensor → gaussian_sensor_noise
```
- Remove meaningless `ray_trace` (ndimage.zoom)
- Mark as tier3 in metadata; reconstruction requires learned backend

**3DGS v2 (corrected):**
```
generic_source → gaussian_splatting_stub (tier3_learned)
→ generic_sensor → gaussian_sensor_noise
```
- Remove `ray_trace + conv2d` chain
- Mark as tier3 in metadata

**Tests.** Update `tests/test_canonical_chain.py`:
- CT v2 compiles with beer_lambert node
- Photoacoustic v2 compiles with interaction node; carrier_transition validated
- NeRF/3DGS v2 compile with stub primitives
- CT forward: `exp(-radon(x))` produces correct attenuation curve
- All 26 v2 templates still compile (no regression)

**Lines:** ~500 new/modified

---

### R7: TierPolicy (budget→tier selection) + Fourier relay + Maxwell stub

**Problem.** `PhysicsTier` enum is defined but never populated on any primitive
or used for tier selection. There is no policy for choosing physics fidelity
based on compute budget. No Fourier-domain relay primitive for tier-1 propagation.
No interface for tier-2 full-wave solvers.

**Changes.**

New file `graph/tier_policy.py`:
- `TierBudget(StrictBaseModel)`: `max_seconds: float`, `max_memory_mb: float`,
  `accuracy: str` (low/medium/high/maximum)
- `TierPolicy`:
  - `select_tier(modality: str, budget: TierBudget) -> PhysicsTier`
  - Rule table:
    - `accuracy == "low"` or `max_seconds < 0.1` → `tier0_geometry`
    - `accuracy == "medium"` or `max_seconds < 1.0` → `tier1_approx`
    - `accuracy == "high"` or `max_seconds < 60.0` → `tier2_full`
    - `accuracy == "maximum"` → `tier3_learned`
  - Per-modality overrides: CT always ≥ tier1, MRI always ≥ tier1, NeRF/3DGS always tier3
- `suggest_primitives(modality: str, tier: PhysicsTier) -> List[str]`:
  returns recommended primitive_ids for the given modality at the selected tier

`graph/primitives.py`:
- Set `_physics_tier` class attribute on all 40+ primitives:
  - `Identity`, `SumAxis` → `tier0_geometry`
  - `Conv2d`, `Conv3d`, `FresnelProp`, `AngularSpectrum`, `CTRadon` → `tier1_approx`
  - `BeerLambert`, `AcousticPropagation` → `tier1_approx`
  - `VolumeRenderingStub`, `GaussianSplattingStub` → `tier3_learned`
- New `FourierRelay` primitive (subrole: `relay`, tier: `tier1_approx`):
  - `forward(x)`: FFT → multiply by transfer function H(f) → IFFT
  - params: `transfer_function: str` ("free_space"/"low_pass"/"band_pass"),
    `wavelength_m: float`, `propagation_distance_m: float`
  - `_is_linear = True`
  - Adjoint: conjugate transfer function
- New `MaxwellInterface` primitive (subrole: `propagation`, tier: `tier2_full`):
  - `forward(x)`: raises `NotImplementedError(
    "Maxwell solver (FDTD/BPM) not yet integrated. "
    "Set up via MaxwellInterface.configure(backend='meep'|'tidy3d'|'custom')")`
  - Defines interface: input = source field (complex, [H, W] or [H, W, D]),
    output = propagated field
  - params: `backend: str`, `grid_spacing_m: float`, `n_steps: int`
  - `_is_linear = True` (Maxwell equations are linear)

`graph/compiler.py`:
- During tag derivation, populate `NodeTags.physics_tier` from primitive `_physics_tier`

**Tests.** `tests/test_tier_policy.py`:
- `select_tier("widefield", budget(0.05s))` returns `tier0_geometry`
- `select_tier("widefield", budget(0.5s))` returns `tier1_approx`
- `select_tier("ct", budget(0.01s))` returns `tier1_approx` (CT override)
- `select_tier("nerf", budget(100s))` returns `tier3_learned`
- `FourierRelay` compiles + forward produces correct shape
- `FourierRelay` adjoint passes dot-product test `<Ax, y> ≈ <x, A^T y>`
- `MaxwellInterface` raises `NotImplementedError` with helpful message
- Tier-switch test: replace `conv2d` with `fourier_relay` in widefield → compiles

**Lines:** ~350 new/modified

---

### R8: BeliefState derives from mismatch ParamSpecs

**Problem.** `build_belief_from_graph` scans `graph_op.learnable_params` and creates
`ParameterSpec` with naive `±2 × |current_value|` bounds. This produces physically
invalid ranges (e.g., sigma=2.0 → bounds [-2, 6], but sigma must be positive).
The rich `ParameterSpec` already on `GraphNode.parameter_specs` (with proper bounds,
prior distribution, drift model, identifiability hint) is completely ignored.

**Changes.**

`mismatch/belief_state.py`:
- `build_belief_from_graph` rewritten:
  1. **Primary path:** read `GraphNode.parameter_specs` from `graph_op.spec.nodes`
     (the original OperatorGraphSpec, preserved on GraphOperator). If a node has
     `parameter_specs` populated, use those bounds/prior/drift/identifiability directly.
  2. **Fallback path:** for nodes with `learnable` list but no `parameter_specs`,
     use **primitive-aware defaults** instead of naive ±2x:
     - PSF sigma: lower=0.1, upper=20.0, prior="log_normal", units="px"
     - Gain/QE: lower=0.01, upper=100.0, prior="log_uniform"
     - Displacement (dx, dy): lower=-50.0, upper=50.0, prior="uniform", units="px"
     - Angle (theta): lower=-π, upper=π, prior="uniform", units="rad"
     - Generic: lower=val/10, upper=val*10 (if val > 0), prior="log_uniform"
  3. `ParameterSpec.parameterization` honored in `to_theta_space()`:
     - `"log"`: ThetaSpace operates on log(param), bounds = log(lower), log(upper)
     - `"logit"`: ThetaSpace operates on logit(param), bounds mapped accordingly
     - `"identity"`: pass through (current behavior)
  4. `ParameterSpec.drift_model` stored on BeliefState; `update()` can use drift
     prediction for temporal extrapolation (initial: store only, predict later)
  5. `ParameterSpec.identifiability_hint` logged during calibration as diagnostic

- Remove duplicate `StrictBaseModel` copy; import from `ir_types`

`mismatch/parameterizations.py`:
- `graph_theta_space(adapter)` also updated to use primitive-aware defaults
  (same logic as belief_state, or delegates to `build_belief_from_graph`)
- Remove code duplication between `graph_theta_space` and `build_belief_from_graph`

`graph/graph_operator.py`:
- `GraphOperator` stores reference to original `OperatorGraphSpec` as `.spec`
  (so `build_belief_from_graph` can access `parameter_specs`)

`graph/compiler.py`:
- `compile()` passes original `spec` through to `GraphOperator(..., spec=spec)`

**Tests.** Update `tests/test_belief_state.py`:
- Node with `parameter_specs` → BeliefState uses exact bounds from spec
- Node with only `learnable` list → uses primitive-aware defaults (not ±2x)
- PSF sigma gets positive-only bounds
- `to_theta_space()` with `parameterization="log"` produces log-scale bounds
- `build_belief_from_graph` with drift_model → stored on BeliefState

**Lines:** ~250 new/modified

---

### R9: Prompt parser outputs typed ExperimentSpec

**Problem.** `parse_prompt()` returns `ParsedPrompt` (a plain class with untyped fields),
not the Pydantic `ExperimentSpec` that the rest of the API expects. The caller must
manually construct `ExperimentSpec` from the parsed fields, duplicating logic and
losing validation.

**Changes.**

`api/prompt_parser.py`:
- Rename `ParsedPrompt` → `_ParsedFields` (internal)
- New `parse_prompt(prompt: str) -> ExperimentSpec`:
  1. Extract fields via `_parse_fields(prompt) -> _ParsedFields` (existing keyword logic)
  2. Map modality → graph template_id: `f"{modality}_graph_v2"`
  3. Map mode → `TaskKind`:
     - `simulate` → `simulate_recon_analyze`
     - `invert` → `reconstruct_only`
     - `calibrate` → `calibrate_and_reconstruct`
  4. Map budget → `BudgetState(photon_budget={"max_photons": value})`
  5. Map solvers → `ReconSpec(portfolio=ReconPortfolio(solvers=[SolverSpec(id=s) for s in ids]))`
  6. Construct full `ExperimentSpec`:
     ```python
     ExperimentSpec(
         id=f"prompt_{uuid4().hex[:8]}",
         input=ExperimentInput(
             mode=InputMode.simulate if mode==simulate else InputMode.measured,
             operator=OperatorInput(
                 kind=OperatorKind.parametric,
                 parametric=OperatorParametric(operator_id=template_id),
             ),
         ),
         states=ExperimentStates(
             physics=PhysicsState(modality=modality),
             budget=budget_state,
             task=TaskState(kind=task_kind),
         ),
         recon=recon_spec,
     )
     ```
  7. Validate via Pydantic before returning

- Keep backward-compat `parse_prompt_raw(prompt) -> _ParsedFields` for callers
  that only need the raw extraction

`api/types.py`:
- Add `graph_template_id: Optional[str] = None` to `PhysicsState` — links modality
  to a specific graph template
- Use `StrictBaseModel` for `ExperimentSpec` and all sub-models (migrate from plain
  `BaseModel`) — enforce extra="forbid" + NaN/Inf rejection
- Add `SourceSpec` and `NoiseModelSpec` (from R3) as optional fields on
  `ExperimentStates` (future-proofing wiring of source/noise models into the spec)

**Tests.** `tests/test_prompt_parser.py`:
- `parse_prompt("simulate widefield low-dose")` returns valid `ExperimentSpec`
- `spec.states.physics.modality == "widefield"`
- `spec.states.task.kind == TaskKind.simulate_recon_analyze`
- `spec.states.budget.photon_budget == {"max_photons": 1000.0}`
- `spec.recon.portfolio.solvers` is non-empty when solver detected
- Round-trip: `spec.model_dump()` → `ExperimentSpec.model_validate(...)` succeeds
- Invalid modality returns spec with `modality=None`, logs warning
- `ExperimentSpec` with extra fields raises ValidationError (strict mode)

**Lines:** ~200 new/modified

---

## Dependency graph

```
Wave 0:  R10 (this document — doc split, no code)

Wave 1 (independent, run in parallel):
  R1  Multi-input graph nodes
  R3  NoiseModel / Objective separation
  R5  Multi-channel sensors
  R7  TierPolicy + FourierRelay + Maxwell stub
  R8  BeliefState from ParamSpecs

Wave 2 (depend on Wave 1):
  R2  physics_subrole          ← needs R1 (interaction nodes may be multi-input)
  R4  OperatorCorrectionNode   ← needs R1 (correction can be multi-input: y + correction_data)
  R9  Prompt parser → ExperimentSpec  ← needs R3 (NoiseModelSpec type)

Wave 3 (depends on Waves 1+2):
  R6  Fix CT/PA/NeRF/3DGS      ← needs R1 (multi-input), R2 (subroles), R3 (noise models), R7 (tier stubs)
```

```
R1 ──→ R2 ──→ R6
R1 ──→ R4       ↑
R3 ──→ R9     R7┘
R5 (independent)
R7 ──→ R6
R8 (independent)
```

---

## File inventory (estimated)

### New files

| File | Est. lines | Work item |
|------|-----------|-----------|
| `objectives/noise_model.py` | 200 | R3 |
| `graph/tier_policy.py` | 150 | R7 |
| `tests/test_multi_input.py` | 80 | R1 |
| `tests/test_subrole.py` | 80 | R2 |
| `tests/test_noise_model.py` | 100 | R3 |
| `tests/test_correction_node.py` | 100 | R4 |
| `tests/test_multi_channel.py` | 80 | R5 |
| `tests/test_tier_policy.py` | 120 | R7 |
| `tests/test_prompt_parser.py` | 80 | R9 |
| `docs/v3_implementation_status.md` | 70 | R10 |

### Modified files

| File | Work items |
|------|-----------|
| `graph/ir_types.py` | R1 (PortSpec), R2 (PhysicsSubrole), R4 (CorrectionKind) |
| `graph/primitives.py` | R1 (multi-input), R2 (subroles), R4 (correction nodes), R5 (multi-channel), R6 (new physics), R7 (relay, Maxwell) |
| `graph/graph_spec.py` | R2 (physics_subrole field) |
| `graph/compiler.py` | R1 (multi-input validation), R2 (subrole derivation), R7 (tier derivation), R8 (spec passthrough) |
| `graph/executor.py` | R1 (DAG execution), R3 (NoiseModel), R4 (correction), R5 (multi-channel noise) |
| `graph/canonical.py` | R2 (carrier_transition enforcement), R4 (correction node placement) |
| `graph/graph_operator.py` | R8 (store original spec) |
| `mismatch/belief_state.py` | R8 (ParamSpec-aware) |
| `mismatch/parameterizations.py` | R8 (deduplicate bounds logic) |
| `api/prompt_parser.py` | R9 (→ ExperimentSpec) |
| `api/types.py` | R9 (StrictBaseModel, graph_template_id) |
| `contrib/graph_templates.yaml` | R5 (MRI multi-coil), R6 (CT/PA/NeRF/3DGS rewrites) |
| `objectives/base.py` | R3 (integrate with NoiseModel) |
| `tests/test_belief_state.py` | R8 |
| `tests/test_canonical_chain.py` | R2, R6 |
| `tests/test_graph_executor.py` | R1, R3, R4 |

**Total estimated:** ~2,500 lines new/modified across ~26 files

---

## Acceptance criteria

| # | Criterion | Verified by |
|---|-----------|-------------|
| R1 | 2-input interference node compiles + executes | `test_multi_input.py` |
| R1 | All existing single-input graphs unchanged | Existing 950 tests pass |
| R2 | Photoacoustic graph rejected without interaction node (when carrier_transitions set) | `test_subrole.py` |
| R2 | All 26 v2 templates compile (backward compat) | `test_canonical_chain.py` |
| R3 | User can override objective: Huber on Poisson-Gaussian graph | `test_noise_model.py` |
| R3 | NoiseModel.sample() matches noise primitive statistics | `test_noise_model.py` |
| R4 | Affine correction node in graph → forward/adjoint correct | `test_correction_node.py` |
| R4 | `provided_operator` still works (deprecated, with warning) | `test_graph_executor.py` |
| R5 | CoilSensor n_coils=8 → output shape (8, H, W) | `test_multi_channel.py` |
| R5 | Single-channel sensors produce (H, W) — no regression | `test_multi_channel.py` |
| R6 | CT v2 has BeerLambert node, exp-attenuation forward | `test_canonical_chain.py` |
| R6 | Photoacoustic v2 has OpticalAbsorption + AcousticPropagation | `test_canonical_chain.py` |
| R6 | NeRF/3DGS v2 use stub primitives with correct tier3 tag | `test_canonical_chain.py` |
| R7 | `select_tier` returns correct tier for budget thresholds | `test_tier_policy.py` |
| R7 | FourierRelay adjoint passes dot-product test | `test_tier_policy.py` |
| R7 | MaxwellInterface raises NotImplementedError with message | `test_tier_policy.py` |
| R7 | Tier switch: conv2d → fourier_relay compiles | `test_tier_policy.py` |
| R8 | Node with parameter_specs → BeliefState uses exact bounds | `test_belief_state.py` |
| R8 | PSF sigma gets positive-only bounds (not ±2x) | `test_belief_state.py` |
| R8 | `to_theta_space()` with log parameterization | `test_belief_state.py` |
| R9 | `parse_prompt("simulate widefield")` returns valid ExperimentSpec | `test_prompt_parser.py` |
| R9 | ExperimentSpec round-trip serialization succeeds | `test_prompt_parser.py` |
| — | Full suite: 950+ tests pass, 0 regressions | `pytest tests/ packages/pwm_core/tests/` |
