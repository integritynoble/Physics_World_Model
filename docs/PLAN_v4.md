# PWM v4 Plan: Multi-Modality "Works Well"

**Baseline:** v3.1 — see `docs/STATUS_v3_1.md` for verified inventory
**Goal:** 32 modalities, each running Mode S/I/C end-to-end with correct physics, staged acceptance, and operator-correction from measured data.

---

## Gaps (future work only)

Every gap below is NEW work. Nothing listed here is "done."

| # | Gap | Severity | What to build | Owner file | Verified-by test |
|---|-----|----------|---------------|------------|-----------------|
| G1 | No electron source/sensor primitives | High | `ElectronBeamSource`, `ElectronDetectorSensor` | `graph/primitives.py` | `test_electron_primitives.py` |
| G2 | No X-ray radiography, Ultrasound, PET/SPECT operators | High | 5 new operators + templates | `physics/{radiography,ultrasound,nuclear}/` | `test_casepack_{xray,us,pet,spect}.py` |
| G3 | Correction nodes are element-wise; don't wrap A | High | `CorrectedOperator` abstraction + PrePost/LowRank families | `graph/corrected_operator.py` | `test_corrected_operator.py` |
| G4 | No correlated noise; incorrect likelihood would break Mode C | Medium | `CorrelatedNoiseSensor` (simulation-only until whitening) | `graph/primitives.py` | `test_correlated_noise.py` |
| G5 | No SinglePixelSensor, XRayDetectorSensor, AcousticReceiveSensor, ElectronDetectorSensor | Medium | 4 new sensor primitives | `graph/primitives.py` | `test_specialized_sensors.py` |
| G6 | TensorSpec not enforced at compile time | Medium | Shape compatibility checker in compiler | `graph/compiler.py` | `test_shape_validation.py` |
| G7 | No CasePack acceptance tests | High | CasePack runner framework + staged gates | `core/casepack_runner.py` | `test_all_modalities_acceptance.py` |
| G8 | No modality registry Python loader | Low | `ModalityInfo` model + loader | `core/modality_registry.py` | `test_modality_registry.py` |
| G9 | Source-to-x wiring not validated (D2 incomplete) | Medium | Pattern A/B + `requires_x_interaction` | `graph/canonical.py` | `test_canonical_chain.py` (extended) |
| G10 | Ultrasound state semantics undefined | High | Define canonical state chain (RF → channels → beamformed) | `physics/ultrasound/` | `test_casepack_ultrasound.py` |
| G11 | Fourier optics / Maxwell IO convention not standardized | Medium | `PhotonFieldSpec` + discretization contract | `graph/optics_convention.py` | `test_optics_convention.py` |
| G12 | No first-class ingestion for provided A (dense/sparse/callback) | High | `ExplicitLinearOperator` primitive + hashing for RunBundle | `graph/primitives.py` | `test_explicit_operator.py` |
| G13 | PSNR-only acceptance misleading for PET/US/SEM/TEM | Medium | `MetricRegistry` + per-modality default metrics | `core/metric_registry.py` | `test_metric_registry.py` |

---

## Success Points

### S0: Framework-Level (must pass for every modality)

| ID | Criterion | Verified by |
|----|-----------|-------------|
| S0.1 | Canonical chain: Source → Element(s) → Sensor → Noise enforced at compile time | `test_canonical_chain.py` |
| S0.2 | Mode S/I/C on same graph; not separate pipelines | `test_graph_executor.py` |
| S0.3 | Noise model drives NLL; can override; correlated noise blocked from Mode C unless whitened | `test_noise_model.py` + `test_correlated_noise.py` |
| S0.4 | Operator-correction wraps A explicitly: `A' = CorrectedOperator(A, params)`; Mode C fitting improves NLL(y \| A'x), not a residual | `test_corrected_operator.py` |
| S0.5 | Source-to-x wiring enforced (Pattern A or B); `requires_x_interaction` validated per modality | `test_canonical_chain.py` |
| S0.6 | TensorSpec validated at compile time; shape mismatch caught before runtime | `test_shape_validation.py` |
| S0.7 | Tier ladder: TierPolicy selects tier; higher tier has same IO contract (swappable) | `test_tier_policy.py` + `test_optics_convention.py` |
| S0.8 | RunBundle exports: graph hash, param hash, seeds, solver version, outputs | `test_sharepack.py` |

### S1: Modality-Level (per CasePack)

| ID | Criterion | Test pattern |
|----|-----------|-------------|
| S1.1 | Forward sanity: physically plausible outputs (nonneg, energy-correct, correct shape) | `test_casepack_forward_{modality}.py` |
| S1.2 | Recon baseline: Mode I reaches threshold on **primary metric** (per-modality: PSNR, CRC, CNR, FRC, SpectralAngle) | `test_casepack_recon_{modality}.py` |
| S1.3 | Calibration improves: Mode C lowers NLL **and** improves metric (not just residual) | `test_casepack_calib_{modality}.py` |
| S1.4 | Identifiability guardrails: refuses underdetermined params | `test_casepack_ident_{modality}.py` |

---

## Design Decisions

### D1: Element nodes typed by subrole

**Status: COMPLETE in v3.1.** `PhysicsSubrole` enum (`propagation`, `modulation`, `sampling`, `interaction`, `transduction`, `encoding`, `relay`). All 52 primitives tagged. Carrier-transition enforcement in `canonical.py`.

**v4 extension:** New element primitives (ThinObjectPhase, CTFTransfer, YieldModel, BeamformDelay, EmissionProjection, ScatterModel) must each declare a subrole.

### D2: Source must connect to x — compile-time invariant

**Status: NOT ENFORCED in v3.1.** The canonical validator checks chain topology but does not verify that `x` (the unknown/sample) actually influences the forward model.

**v4 requirement — two valid patterns:**

**Pattern A:** `SourceNode(x)` — the source node explicitly consumes `x` via an input port edge. Used when `x` IS the source (e.g., fluorescence imaging where `x` is fluorophore concentration and the source illumination is a parameter).

```
x ──edge──> SourceNode ──> Element(s) ──> Sensor ──> Noise
```

**Pattern B:** `SourceNode()` produces an incident field/flux, and at least one `interaction` or `transduction` subrole node consumes both the incident field AND `x`:

```
SourceNode() ──> InteractionNode(incident, x) ──> ... ──> Sensor ──> Noise
                                      ^
                                      |
                                x (pipeline input)
```

**Modality-level enforcement:**

```python
class ModalityInfo(StrictBaseModel):
    ...
    requires_x_interaction: bool  # True for CT, MRI, SEM/TEM, PA, etc.
```

When `requires_x_interaction = True`, the canonical validator checks that at least one node with `_physics_subrole in ("interaction", "transduction")` exists and has `x` as an input (either via multi-input DAG edge or as the pipeline root input).

**Reject:** Graphs where the source produces output, elements transform it, sensor reads it, noise corrupts it — but `x` (the sample) never enters the chain. These are "pretty graphs that don't model the sample."

**Implementation:**
- `graph/canonical.py`: add `_validate_x_wiring(spec, modality_info)` after existing chain checks
- `contrib/modalities.yaml`: add `requires_x_interaction: true/false` per modality
- `core/modality_registry.py`: expose `ModalityInfo.requires_x_interaction`

### D3: Operator-correction wraps A, not signals

**Status: v3.1 has 3 element-wise correction primitives** (Affine, Residual, FieldMap). These modify signals flowing through the graph but do NOT wrap the forward operator `A` itself. The LowRankCorrectionNode in the old plan added something to `x`, not to `Ax`.

**v4 requirement — `CorrectedOperator` abstraction:**

The correction must modify the **operator** so that Mode C optimizes `NLL(y | A'(x))` where `A'` is the corrected operator.

```python
class CorrectedOperator:
    """Wraps a base operator with learnable correction parameters."""

    def __init__(self, base_op: PhysicsOperator, correction: OperatorCorrection):
        self.base_op = base_op
        self.correction = correction

    def forward(self, x):
        return self.correction.apply_forward(self.base_op, x)

    def adjoint(self, y):
        return self.correction.apply_adjoint(self.base_op, y)
```

**Two correction families (implement first):**

**Family 1 — Pre/Post:**
```
A'(x) = P(alpha) . A( Q(beta)(x) )
```
- `Q(beta)` = pre-correction: shift, warp, blur applied to `x` before `A`
- `P(alpha)` = post-correction: gain, offset, blur applied to `A(x)` after `A`
- Learnable: `alpha`, `beta` (bounded in BeliefState)
- Use case: geometry mismatch (center-of-rotation in CT), relay blur (CASSI defocus)

**Family 2 — Low-Rank Update:**
```
A'(x) = A(x) + U . diag(alpha) . V^T . x
```
- `U` (M x r), `V` (N x r), `alpha` (r,) = low-rank perturbation to the system matrix
- When `r` is small, this is efficient and learnable
- `U`, `V` can be initialized from SVD of `(A_measured - A_nominal)` on calibration data
- Learnable: `alpha` (possibly `U`, `V` with regularization)
- Use case: systematic model error, unmodeled physics

**Success criterion (testable):**
In Mode C, fitting correction parameters must improve `NLL(y | A'(x))` on measured data. The test:
1. Generate ground truth: `y = A_true(x) + noise`
2. Start with mismatched `A_0 != A_true`
3. Run Mode C: fit correction params to get `A' = CorrectedOperator(A_0, theta_hat)`
4. Assert: `NLL(y | A'(x)) < NLL(y | A_0(x))`
5. Assert: `PSNR(A'^{-1}(y), x) > PSNR(A_0^{-1}(y), x)`

**v3.1 correction primitives (Affine, Residual, FieldMap) remain** as signal-level corrections for simple cases. `CorrectedOperator` is for operator-level correction.

**Implementation:**
- NEW `graph/corrected_operator.py`: `CorrectedOperator`, `OperatorCorrection` ABC, `PrePostCorrection`, `LowRankCorrection`
- Modify `graph/executor.py`: Mode C uses `CorrectedOperator` when correction config is present
- NEW `tests/test_corrected_operator.py`: NLL improvement test for both families

### D4: One integration style — graph templates are the canonical path

**Rule:** Every modality's forward model is defined as a **graph template** in `contrib/graph_templates.yaml`. The template is the source of truth for the physics chain.

**`physics/*/` files** (e.g., `physics/ultrasound/ultrasound_helpers.py`) are **helper modules** used by graph primitives — they implement the math that a primitive's `forward()`/`adjoint()` calls into. They are NOT standalone operators that bypass the graph.

**Concretely:**
- `physics/ultrasound/ultrasound_helpers.py` provides `delay_and_sum()`, `propagate_rf()` etc. as pure functions
- `graph/primitives.py::BeamformDelay.forward()` calls `delay_and_sum()` from the helper
- The graph template `ultrasound_graph_v2` wires `BeamformDelay` into the canonical chain
- `core/runner.py` always builds from the template via `GraphCompiler`

**What this prevents:**
- Dual-path bugs where `physics/foo/foo_operator.py` and `graph_templates.yaml` diverge
- "Operator files" that bypass canonical validation, noise modeling, or correction

**Exception:** `ExplicitLinearOperator` (G12) accepts a provided matrix/callback that is NOT a graph template. This is intentional — it's for measured/external A matrices that have no graph decomposition.

**Test:** No standalone operator file should implement `PhysicsOperator.forward()` directly. Operator files export helper functions only. Enforced by convention (no formal test needed — graph template compilation tests cover correctness).

---

## Phase 0: IR Hardening + Validation + Registries

**Goal:** Lock down foundations so every subsequent phase builds on validated IR.

**~650 new lines, 7 new files**

### P0.1: TensorSpec compile-time validation (G6)

**File:** `graph/compiler.py` — add `_validate_shapes(spec, node_map)`

For each edge `(src -> dst)`:
- If both nodes declare `TensorSpec` on output/input ports, verify shape compatibility
- `-1` in shape = dynamic axis (skip)
- Mismatch raises `GraphCompilationError`
- Called after `_validate_primitive_ids()`, before topological sort

**File:** `graph/primitives.py` — add `_output_shape_hint(input_shape) -> Tuple[int, ...]` to `BasePrimitive`
- Default: returns `input_shape`
- Override for shape-changers: CTRadon (N,N) -> (angles, N), MRIKspace, FrameIntegration, etc.

**Test:** `tests/test_shape_validation.py` (~80 lines)
- Compatible shapes pass
- Incompatible shapes raise `GraphCompilationError`
- Dynamic axes tolerated
- All 26 existing v2 templates still compile

### P0.2: Source-to-x wiring validation (G9, D2)

**File:** `graph/canonical.py` — add `_validate_x_wiring(spec, modality_info)`

- Pattern A check: source node has incoming edge from pipeline input
- Pattern B check: at least one `interaction`/`transduction` node exists when `requires_x_interaction=True`
- Reject graphs where `x` is disconnected from physics chain

**File:** `contrib/modalities.yaml` — add `requires_x_interaction` field per modality:
```yaml
ct:
  requires_x_interaction: true   # x = attenuation map, enters via Beer-Lambert
mri:
  requires_x_interaction: true   # x = spin density, enters via encoding
cassi:
  requires_x_interaction: true   # x = spectral cube, modulated by mask
widefield:
  requires_x_interaction: false  # x IS the fluorophore (source consumes x directly)
```

**Test:** add 4 tests to `tests/test_canonical_chain.py`:
- Pattern A graph passes
- Pattern B graph passes
- Disconnected-x graph rejected when `requires_x_interaction=True`
- `requires_x_interaction=False` allows source-only chain

### P0.3: Modality registry loader (G8)

**File:** `core/modality_registry.py` (~80 lines):
```python
class ModalityInfo(StrictBaseModel):
    display_name: str
    category: str
    keywords: List[str]
    default_solver: str
    default_template_id: str
    requires_x_interaction: bool
    acceptance_tier: Literal["A", "B", "C"]  # for staged gate
```
- `load_modalities() -> Dict[str, ModalityInfo]`
- `get_modality(key) -> ModalityInfo` (KeyError on unknown)

**Test:** `tests/test_modality_registry.py` (~40 lines)

### P0.4: CasePack acceptance framework (G7)

**File:** `core/casepack_runner.py` (~140 lines):
```python
class CasePackResult(StrictBaseModel):
    modality: str
    forward_ok: bool
    recon_psnr: Optional[float]              # always computed (universal sanity)
    primary_metric: Optional[float]          # modality-appropriate metric
    primary_metric_name: Optional[str]       # e.g. "crc", "cnr", "frc"
    secondary_metric: Optional[float]        # optional second metric
    secondary_metric_name: Optional[str]
    calib_nll_improvement: Optional[float]
    ident_flags: Dict[str, bool]

def run_casepack(modality: str, quick: bool = True) -> CasePackResult
```
- `quick=True`: tiny data (16x16 or 32x32), CPU, <30s per modality
- `quick=False`: realistic sizes, GPU if available, nightly CI

**Test:** `tests/test_casepack_framework.py` (~60 lines)

### P0.5: MetricRegistry — per-modality quality metrics (G13)

**Problem:** PSNR-only acceptance will pass synthetic tests but doesn't match "works well" for modalities where PSNR is misleading (PET: activity recovery matters more; Ultrasound: contrast-to-noise; TEM: frequency content fidelity).

**File:** `core/metric_registry.py` (~120 lines):

```python
class Metric(ABC):
    """Base class for reconstruction quality metrics."""
    name: str
    higher_is_better: bool
    @abstractmethod
    def __call__(self, x_recon: ndarray, x_true: ndarray, **kwargs) -> float: ...

class PSNR(Metric):          # Universal baseline (all modalities)
class SSIM(Metric):           # Structural (all modalities)
class NLL_Metric(Metric):     # Negative log-likelihood on held-out data
class CRC(Metric):            # Contrast recovery coefficient (PET/SPECT)
class CNR(Metric):            # Contrast-to-noise ratio (Ultrasound)
class FRC(Metric):            # Fourier ring correlation proxy (TEM/SEM)
class SpectralAngle(Metric):  # Spectral angle mapper (CASSI/hyperspectral)

METRIC_REGISTRY: Dict[str, Type[Metric]] = {...}

def build_metric(name: str) -> Metric: ...
```

**Per-modality default metrics (defined in acceptance_thresholds.yaml):**

| Modality | Primary metric | Secondary metric | Why not PSNR alone |
|----------|---------------|-----------------|-------------------|
| CT | PSNR | SSIM | PSNR is reasonable for CT |
| MRI | PSNR | SSIM | PSNR is reasonable for MRI |
| PET | CRC | NLL | Activity quantification matters more than pixel MSE |
| SPECT | CRC | NLL | Same as PET |
| Ultrasound | CNR | PSNR | Speckle makes PSNR misleading; contrast matters |
| SEM | FRC | PSNR | Resolution/frequency content > pixel-wise error |
| TEM | FRC | PSNR | CTF information transfer drives image quality |
| CASSI | SpectralAngle | PSNR | Spectral fidelity > spatial PSNR |
| SPC | PSNR | SSIM | Standard for single-image compressive |
| CACTI | PSNR | SSIM | Standard for video compressive |

**Integration:** `CasePackResult` gains `primary_metric: float` + `primary_metric_name: str` alongside `recon_psnr`. Acceptance tests check the primary metric against modality-specific thresholds.

**Test:** `tests/test_metric_registry.py` (~60 lines):
- Each metric computes correct value on known inputs
- `build_metric()` round-trips through registry
- Perfect reconstruction → PSNR = inf / SSIM = 1.0 / CRC = 1.0

### P0.6: Fourier optics IO convention (G11)

**File:** `graph/optics_convention.py` (~100 lines):

```python
class PhotonFieldSpec(StrictBaseModel):
    """Standardized discretization for all optics nodes."""
    wavelength_m: float                   # central wavelength
    wavelength_bins: Optional[List[float]] # for polychromatic
    grid_shape: Tuple[int, int]           # (Ny, Nx)
    pixel_pitch_m: float                  # real-space sampling
    # Derived:
    # freq_pitch = 1 / (grid_shape * pixel_pitch_m)
    # max_freq = 1 / (2 * pixel_pitch_m)  [Nyquist]

    def freq_grid(self) -> Tuple[ndarray, ndarray]: ...
    def real_grid(self) -> Tuple[ndarray, ndarray]: ...
```

**Contract:** Every optics primitive (FresnelProp, AngularSpectrum, FourierRelay, future Maxwell stub) must:
1. Accept `PhotonFieldSpec` in params (or inherit from graph metadata)
2. Produce output on the same grid (same `grid_shape` and `pixel_pitch_m`)
3. Higher-tier replacements (Tier 2 = full Maxwell) must preserve the same `PhotonFieldSpec`, making tiers **swappable**

**4f relay and Fourier optics share one convention:**
- FresnelProp: propagation in real+freq domain, uses `pixel_pitch_m` for correct scaling
- AngularSpectrum: full freq-domain, uses `freq_grid()` from same spec
- FourierRelay: 4f system = FresnelProp + lens phase + FresnelProp, shares grid
- Maxwell stub (Tier 2): FDTD/BPM on same grid; input/output = `PhotonFieldSpec`

**Test:** `tests/test_optics_convention.py` (~60 lines):
- FresnelProp and AngularSpectrum produce same output shape for same `PhotonFieldSpec`
- FourierRelay chain preserves grid
- Maxwell stub IO matches `PhotonFieldSpec`

---

## Phase 1: Extended Node Families

**Goal:** Fill primitive gaps for all 32 modalities.

**~1200 new lines, 8 new test files**

### P1.1: CorrectedOperator abstraction (G3, D3)

**NEW File:** `graph/corrected_operator.py` (~200 lines):

```python
class OperatorCorrection(ABC):
    """Base class for operator-level corrections."""
    @abstractmethod
    def apply_forward(self, base_op, x): ...
    @abstractmethod
    def apply_adjoint(self, base_op, y): ...
    @abstractmethod
    def learnable_params(self) -> Dict[str, ParameterSpec]: ...

class PrePostCorrection(OperatorCorrection):
    """A'(x) = P(alpha) . A( Q(beta)(x) )"""
    def __init__(self, pre_fn, post_fn, params): ...
    def apply_forward(self, base_op, x):
        return self.post_fn(base_op.forward(self.pre_fn(x)))

class LowRankCorrection(OperatorCorrection):
    """A'(x) = A(x) + U @ diag(alpha) @ V^T @ x"""
    def __init__(self, U, V, alphas): ...
    def apply_forward(self, base_op, x):
        Ax = base_op.forward(x)
        delta = self.U @ (self.alphas[:, None] * (self.V.T @ x.ravel()))
        return Ax + delta.reshape(Ax.shape)

class CorrectedOperator:
    """Wraps base_op with correction. Satisfies PhysicsOperator protocol."""
    def __init__(self, base_op, correction): ...
    def forward(self, x): return self.correction.apply_forward(self.base_op, x)
    def adjoint(self, y): return self.correction.apply_adjoint(self.base_op, y)
```

**Modify:** `graph/executor.py` — Mode C creates `CorrectedOperator` when `calibration_config.correction_type` is set; optimizes correction params to minimize `NLL(y | A'(x))`.

**Test:** `tests/test_corrected_operator.py` (~150 lines):
- PrePost: inject known shift, Mode C recovers it, NLL improves
- LowRank: inject known perturbation, Mode C recovers alpha, NLL improves
- Both: `PSNR(A'^{-1}(y), x) > PSNR(A_0^{-1}(y), x)`

### P1.2: ExplicitLinearOperator primitive (G12)

**File:** `graph/primitives.py` — add:

- **ExplicitLinearOperator** (`explicit_linear_operator`, role: transport, `_is_linear = True`):
  - Accepts a provided forward operator `A` as dense ndarray, scipy sparse, torch sparse, or a `(forward_fn, adjoint_fn)` callback pair
  - params: `matrix` (ndarray/sparse), `forward_fn` (callable), `adjoint_fn` (callable) — exactly one of `matrix` or `(forward_fn, adjoint_fn)` must be set
  - `forward(x)`: `A @ x.ravel()` reshaped (matrix mode) or `forward_fn(x)` (callback mode)
  - `adjoint(y)`: `A.T @ y.ravel()` reshaped (matrix mode) or `adjoint_fn(y)` (callback mode)

**Hashing/serialization for RunBundle:**
```python
def _compute_hash(self) -> str:
    """SHA256 of operator for reproducibility tracking."""
    if self._matrix is not None:
        # For sparse: convert to CSR, hash data+indices+indptr
        data = self._matrix.toarray() if issparse(self._matrix) else self._matrix
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]
    else:
        # Callback mode: hash is None (not reproducible from hash alone)
        return "callback_" + hex(id(self._forward_fn))[-8:]
```
- RunBundle stores: operator hash, shape `(M, N)`, nnz (for sparse), dtype
- For callback mode: RunBundle logs a warning that exact A is not serialized

**Integration with CorrectedOperator:**
This is the primary use case for "measured y + matrix A" correction:
```python
A_provided = ExplicitLinearOperator(matrix=A_measured_sparse)
corrected = CorrectedOperator(A_provided, PrePostCorrection(...))
# Mode C fits correction params on measured y
```

**Test:** `tests/test_explicit_operator.py` (~80 lines):
- Dense matrix: forward/adjoint match numpy matmul
- Sparse matrix: forward/adjoint match dense equivalent
- Callback mode: custom forward/adjoint
- Hash stability: same matrix → same hash
- CorrectedOperator wrapping: NLL improvement on synthetic data

### P1.3: Electron source + sensor primitives (G1, D2-compliant)

**File:** `graph/primitives.py` — add:

- **ElectronBeamSource** (`electron_beam_source`, role: source, carrier: electron):
  - params: `accelerating_voltage_kv`, `beam_current_na`, `coherence` (incoherent/partial/coherent)
  - `forward()`: produces incident beam state (probe/plane wave), **does NOT consume x**
  - Output: `ElectronState(voltage_kv, current_na, coherence)` — incident illumination
  - This is Pattern B: source emits incident field, interaction nodes consume `(incident, x)`

- **ElectronDetectorSensor** (`electron_detector_sensor`, role: sensor):
  - params: `detector_type` (SE/BSE/BF/ADF/HAADF), `collection_efficiency`, `gain`
  - `forward(x)`: `collection_efficiency * gain * x`

Register in `PRIMITIVE_REGISTRY`.

**Test:** `tests/test_electron_primitives.py` (~80 lines):
- ElectronBeamSource.forward() returns incident state (no x input)
- ElectronDetectorSensor linear scaling
- Multi-input wiring test: (incident, x) → interaction node

### P1.4: Specialized sensor primitives (G5)

**File:** `graph/primitives.py` — add:

- **SinglePixelSensor** (`single_pixel_sensor`, role: sensor):
  - params: `n_patterns`, `integration_time`
  - `forward(x)`: integrate against stored patterns → 1D vector
  - For SPC modality

- **XRayDetectorSensor** (`xray_detector_sensor`, role: sensor):
  - params: `scintillator_efficiency`, `pixel_pitch_um`, `gain`, `offset`
  - `forward(x)`: `gain * scintillator_efficiency * x + offset`
  - Default noise: **PoissonGaussian** (not Poisson-only)
  - For X-ray radiography and CT

- **AcousticReceiveSensor** (`acoustic_receive_sensor`, role: sensor):
  - params: `n_elements`, `impulse_response`, `sensitivity`
  - `forward(x)`: convolve with impulse response, scale
  - For ultrasound B-mode

Register in `PRIMITIVE_REGISTRY`.

**Test:** `tests/test_specialized_sensors.py` (~80 lines)

### P1.5: Correlated noise — simulation-only (G4)

**File:** `graph/primitives.py` — add:

- **CorrelatedNoiseSensor** (`correlated_noise_sensor`, role: noise, stochastic):
  - params: `base_sigma`, `correlation_type` (none/spatial/temporal/1_over_f), `correlation_length`
  - `forward(x)`: generates correlated noise and adds to x
  - `likelihood(y, y_clean)`: **raises `NotImplementedError("Correlated noise likelihood requires whitening. Use simulation-only mode.")`**
  - `_simulation_only: bool = True` — class attribute

**File:** `objectives/noise_model.py` — add `CorrelatedNoiseModel`:
- `sample()`: generates correlated noise
- `default_nll()`: **raises `RuntimeError("CorrelatedNoiseModel does not support NLL. Mark as simulation-only or implement whitening.")`**

**File:** `graph/executor.py` — Mode C check:
```python
if noise_primitive._simulation_only:
    raise ValueError(
        f"Noise primitive {noise_primitive.primitive_id} is simulation-only. "
        "Cannot use Mode C (calibrate) without a correct likelihood. "
        "Either implement whitening or switch to an i.i.d. noise model."
    )
```

**Future:** When whitening is implemented (`Sigma^{-1/2}` applied to residuals), set `_simulation_only = False` and implement `likelihood()`.

**Test:** `tests/test_correlated_noise.py` (~60 lines):
- Forward produces correlated samples (check autocorrelation)
- `likelihood()` raises `NotImplementedError`
- Mode C with correlated noise raises `ValueError`
- Mode S works fine (simulation-only path)

### P1.6: Element primitives for new modalities

**File:** `graph/primitives.py` — add:

- **ThinObjectPhase** (subrole: interaction, tier: tier1_approx, **multi-input**):
  - TEM: `forward(incident, x)` = `incident * exp(i * sigma * V * x)` (complex transmission)
  - Input ports: `incident` (from ElectronBeamSource), `x` (sample projected potential)
  - Output: transmitted wave
  - params: `sigma` (interaction parameter), `thickness_nm`
  - `physics_validity_regime: "thin_sample, weak_phase_approx"`
  - **D2 compliance:** explicitly consumes x via multi-input port

- **CTFTransfer** (subrole: propagation, tier: tier1_approx):
  - TEM: FFT → multiply by CTF → IFFT
  - params: `defocus_nm`, `Cs_mm`, `wavelength_pm`
  - `physics_validity_regime: "isoplanatic, no_spatial_incoherence"`

- **YieldModel** (subrole: interaction, tier: tier0_geometry, **multi-input**):
  - SEM: `forward(incident, x)` = `yield_coeff * incident_current * x` (SE/BSE yield proxy)
  - Input ports: `incident` (from ElectronBeamSource), `x` (sample material map)
  - Output: emitted secondary/backscattered signal
  - params: `yield_coeff`, `detector_type`
  - `physics_validity_regime: "homogeneous_material, normal_incidence"`
  - **D2 compliance:** explicitly consumes x via multi-input port

- **BeamformDelay** (subrole: propagation, tier: tier1_approx):
  - Ultrasound: delay-and-sum beamforming
  - params: `n_elements`, `speed_of_sound`, `focus_depth`, `element_pitch`

- **EmissionProjection** (subrole: sampling, tier: tier1_approx):
  - PET/SPECT: line-of-response system matrix projection
  - params: `n_detectors`, `n_angles`, `attenuation_map`

- **ScatterModel** (subrole: interaction, tier: tier1_approx):
  - X-ray/CT/PET: additive scatter estimation
  - params: `scatter_fraction`, `kernel_sigma`

All registered in `PRIMITIVE_REGISTRY`.

**Test:** `tests/test_new_element_primitives.py` (~120 lines)

---

## Phase 2: Compressive Imaging Trio — Full Mismatch

**Goal:** SPC, CASSI, CACTI with correct mismatch knobs and CasePack acceptance.

**~600 new lines, 3 test files**

### P2.1: SPC mismatch + CasePack

Template update `spc_graph_v2`:
```
photon_source -> coded_mask(DMD) -> single_pixel_sensor -> poisson_gaussian_sensor
```
Mismatch: `pattern_shift_x/y`, `timing_jitter`, `collection_efficiency`, `gain_drift`

**Test:** `tests/test_casepack_spc.py` (~80 lines): S1.1-S1.4, PSNR > 20 dB

### P2.2: CASSI mismatch + CasePack

Template update `cassi_graph_v2`:
```
photon_source -> coded_mask -> spectral_dispersion -> frame_integration -> photon_sensor -> poisson_gaussian_sensor
```
Mismatch: `mask_dx/dy`, `disp_slope`, `defocus_sigma`, `gain_drift`

**Test:** `tests/test_casepack_cassi.py` (~80 lines): S1.1-S1.4, PSNR > 25 dB

### P2.3: CACTI mismatch + CasePack

Template update `cacti_graph_v2`:
```
photon_source -> coded_mask -> temporal_mask -> frame_integration -> photon_sensor -> poisson_gaussian_sensor
```
Mismatch: `timing_jitter`, `rolling_shutter_skew`, `mask_motion_dx`

**Test:** `tests/test_casepack_cacti.py` (~80 lines): S1.1-S1.4, PSNR > 22 dB

---

## Phase 3: Medical Modalities

**Goal:** X-ray radiography, CT (enhanced), MRI (enhanced), Ultrasound, PET/SPECT.

**~1400 new lines, 6 test files**

### P3.1: X-ray Radiography (new modality)

**Helper module:** `physics/radiography/xray_radiography_helpers.py` (~60 lines)
- Pure functions: `planar_beer_lambert(x, I_0, mu)`, `scatter_estimate(y, fraction, sigma)`
- Used by `BeerLambert` and `ScatterModel` primitives in the graph template

**Template:** `xray_radiography_graph_v2`:
```
xray_source -> beer_lambert -> scatter_model(optional) -> xray_detector_sensor -> poisson_gaussian_sensor
```

**Default noise: PoissonGaussian** (realistic: scintillator + read electronics). Poisson-only available as explicit override:
```yaml
# In template metadata:
noise_default: poisson_gaussian_sensor
noise_alternatives: [poisson_only_sensor]  # for ideal-detector studies
```

Mismatch: gain/offset, detector blur, scatter scale.

**Test:** `tests/test_casepack_xray.py` (~70 lines)

### P3.2: CT enhanced (operator-correction mode)

CT already has `ct_graph_v2` with `beer_lambert`. Add:
- `CorrectedOperator` with `PrePostCorrection` for geometry correction: center-of-rotation, angular offset
- CasePack test: given measured y + baseline A0, Mode C estimates geometry params, improves NLL and recon

**Default noise: PoissonGaussian** (matching X-ray detector reality). Poisson-only available as alternative.

**Test:** `tests/test_casepack_ct.py` (~80 lines):
- S1.1-S1.4; FBP init + iterative PSNR > 30 dB
- Operator-correction: inject center-of-rotation error, Mode C recovers it

### P3.3: MRI enhanced (multi-coil + trajectory errors)

MRI already has `mri_graph_v2` with `CoilSensor(n_coils)`. Add:
- Mismatch: `b0_inhomogeneity`, `trajectory_error`, `coil_sensitivity_error`
- `FieldMapCorrectionNode` wired for B0 correction
- `CorrectedOperator` with `PrePostCorrection` for trajectory deviations

**Test:** `tests/test_casepack_mri.py` (~80 lines):
- S1.1-S1.4; SENSE recon, PSNR > 28 dB
- Mode C calibrates coil sensitivity drift

### P3.4: Ultrasound B-mode (new modality)

**Canonical state chain (mandatory):**
```
AcousticPressure (tissue domain) → ReceiveChannels (per-element RF) → BeamformedImage (B-mode)
```

**State semantics:**
| Stage | State type | Shape | Units |
|-------|-----------|-------|-------|
| Tissue | `AcousticPressure` | (Nz, Nx) | reflectivity (a.u.) |
| Transmit | `AcousticPressure` | (Nz, Nx) | Pa |
| RF channels | `ChannelData` | (n_elements, n_samples) | voltage (V) |
| Beamformed | `BeamformedImage` | (Nz, Nx) | amplitude (a.u.) |

**Helper module:** `physics/ultrasound/ultrasound_helpers.py` (~100 lines):
- Pure functions: `propagate_rf(x, speed, n_elements)`, `delay_and_sum(rf, speed, focus)`, `apply_impulse_response(rf, ir)`
- Used by `AcousticPropagation`, `BeamformDelay`, `AcousticReceiveSensor` primitives

**Template:** `ultrasound_graph_v2`:
```
acoustic_source -> acoustic_propagation -> acoustic_receive_sensor -> beamform_delay -> gaussian_sensor_noise
```

**Ultrasound PortSpec table (exact shapes and units):**

| Node | Port | Direction | TensorSpec | Unit | Notes |
|------|------|-----------|------------|------|-------|
| `acoustic_source` | `out` | output | `(1,)` | Pa | Transmit pulse amplitude scalar |
| `acoustic_propagation` | `in:incident` | input | `(1,)` | Pa | From source |
| `acoustic_propagation` | `in:x` | input | `(Nz, Nx)` | a.u. | Tissue reflectivity map (the unknown) |
| `acoustic_propagation` | `out` | output | `(n_elements, n_samples)` | V | RF channel data per transducer element |
| `acoustic_receive_sensor` | `in` | input | `(n_elements, n_samples)` | V | Raw RF channels |
| `acoustic_receive_sensor` | `out` | output | `(n_elements, n_samples)` | V | Filtered RF channels (impulse response applied) |
| `beamform_delay` | `in` | input | `(n_elements, n_samples)` | V | Filtered RF channels |
| `beamform_delay` | `out` | output | `(Nz, Nx)` | a.u. | Beamformed B-mode image |
| `gaussian_sensor_noise` | `in` | input | `(Nz, Nx)` | a.u. | Clean beamformed image |
| `gaussian_sensor_noise` | `out` | output | `(Nz, Nx)` | a.u. | Noisy beamformed image (measurement y) |

**Default quick sizes:** `Nz=64, Nx=64, n_elements=32, n_samples=512`
**Default full sizes:** `Nz=256, Nx=256, n_elements=128, n_samples=2048`

Mismatch: `speed_of_sound`, `attenuation_coeff`, `impulse_response_error`

**Test:** `tests/test_casepack_ultrasound.py` (~80 lines):
- State chain validated: correct shapes at each stage
- S1.1-S1.4; DAS beamforming baseline

### P3.5: PET (new modality)

**Helper module:** `physics/nuclear/pet_helpers.py` (~80 lines)
- Pure functions: `system_matrix_projection(x, angles, detectors)`, `attenuation_correction(sinogram, mu_map)`
- Used by `EmissionProjection` and `ScatterModel` primitives

**Template:** `pet_graph_v2`:
```
generic_source(emission) -> emission_projection -> scatter_model -> photon_sensor -> poisson_only_sensor
```

Mismatch: TOF timing offset, normalization factors, attenuation map mismatch.

**Test:** `tests/test_casepack_pet.py` (~60 lines)

### P3.6: SPECT (new modality)

**Helper module:** `physics/nuclear/spect_helpers.py` (~60 lines)
- Pure functions: `collimator_projection(x, geometry)`, `attenuation_correction(sinogram, mu_map)`
- Used by `EmissionProjection` primitive with collimator-specific params

**Template:** `spect_graph_v2`.

**Test:** `tests/test_casepack_spect.py` (~50 lines)

---

## Phase 4: Electron Microscopy Modalities

**Goal:** SEM, TEM/STEM, Electron Tomography at labeled Tier 0/1.

**~900 new lines, 3 test files**

### Physics fidelity labeling

Every electron modality declares its physics fidelity explicitly in CasePack metadata:

```yaml
# In CasePack JSON / template metadata:
physics_fidelity:
  tier: "tier0_geometry"           # or "tier1_approx"
  validity_regime: "thin_sample, weak_phase_approximation"
  known_limitations:
    - "No dynamical scattering"
    - "No spatial incoherence envelope"
  upgrade_path: "tier2_full: multislice simulation"
```

This keeps PWM honest about what it can and cannot model, and provides a clean upgrade path.

### P4.1: SEM (new modality)

**Tier: tier0_geometry** — yield proxy + scan + detector

**Helper module:** `physics/electron/sem_helpers.py` (~60 lines)
- Pure functions: `se_yield(material_map, voltage_kv)`, `bse_yield(material_map, voltage_kv, angle)`
- Used by `YieldModel` primitive

**Template:** `sem_graph_v2` (Pattern B: source + interaction(incident, x)):
```
electron_beam_source ──> yield_model(incident, x) ──> electron_detector_sensor ──> gaussian_sensor_noise
                                    ^
                                    |
                              x (sample material)
```
Graph edges: `source->yield_model:incident`, `x_input->yield_model:x`, `yield_model->sensor`, `sensor->noise`

**Fidelity:**
```yaml
physics_fidelity:
  tier: tier0_geometry
  validity_regime: "homogeneous_material, normal_incidence, no_charging"
  known_limitations: ["No Monte Carlo scattering", "No topographic contrast model"]
  upgrade_path: "tier1_approx: angular-dependent yield; tier2_full: MC electron transport"
```

Mismatch: beam landing drift, detector gain, charging proxy.

**Diagnostic test:** scan drift sensitivity sanity — inject systematic drift, verify it's detectable in output.

**Test:** `tests/test_casepack_sem.py` (~70 lines)

### P4.2: TEM/STEM (new modality)

**Tier: tier1_approx** — phase object + CTF

**Helper module:** `physics/electron/tem_helpers.py` (~80 lines)
- Pure functions: `compute_ctf(freqs, defocus, Cs, wavelength)`, `phase_object_transmission(x, sigma, V)`
- Used by `ThinObjectPhase` and `CTFTransfer` primitives

**Template:** `tem_graph_v2` (Pattern B: source + interaction(incident, x)):
```
electron_beam_source ──> thin_object_phase(incident, x) ──> ctf_transfer ──> electron_detector_sensor ──> gaussian_sensor_noise
                                        ^
                                        |
                                  x (sample potential)
```
Graph edges: `source->thin_object_phase:incident`, `x_input->thin_object_phase:x`, `thin_object_phase->ctf_transfer`, `ctf_transfer->sensor`, `sensor->noise`

**Fidelity:**
```yaml
physics_fidelity:
  tier: tier1_approx
  validity_regime: "thin_sample, weak_phase, isoplanatic_CTF"
  known_limitations: ["No multislice (dynamical)", "No partial coherence envelope", "No aberrations beyond Cs"]
  upgrade_path: "tier2_full: multislice + partial coherence"
```

STEM: similar but with scanning + ADF integration.

Mismatch: defocus, astigmatism (Cs), drift, stigmator.

**Diagnostic test:** CTF zeros behavior sanity — verify CTF has correct zero crossings for given defocus/Cs.

**Test:** `tests/test_casepack_tem.py` (~80 lines):
- CTF zeros test: compare zero positions to analytical formula
- Forward sanity + recon baseline

### P4.3: Electron Tomography (new modality)

**Tier: tier1_approx** — TEM model + tilt series

**Helper module:** `physics/electron/et_helpers.py` (~80 lines)
- Pure functions: `tilt_project(volume, angle, tem_params)`, `alignment_shift(proj, dx, dy)`
- Used by `ThinObjectPhase` + `CTFTransfer` primitives chained per tilt angle

**Template:** `electron_tomography_graph_v2`.

Mismatch: tilt axis error, alignment shifts per tilt, drift.

**Test:** `tests/test_casepack_et.py` (~60 lines)

---

## Phase 5: Prompt-Driven UX + Staged Acceptance Gate

**Goal:** One command runs any modality. Staged acceptance per tier. Operator-correction verified.

**~700 new lines, 3 test files**

### P5.1: Prompt → full pipeline

**File:** `api/prompt_parser.py` — enhance:
- Support all 32 modalities
- Detect operator-correction: "calibrate CT with measured data" → Mode C + CorrectedOperator
- Auto-select tier from prompt budget hints

### P5.2: Modality registry integration

**File:** `core/runner.py` — use `modality_registry.load_modalities()`:
- Validate modality exists before building operator
- Auto-select default solver and template

### P5.3: Staged acceptance gate

**Three tiers instead of "all 32 pass":**

#### Tier A (must pass early — 8 modalities):
SPC, CACTI, CASSI, CT, MRI, Ultrasound, SEM, TEM

Requirements:
- S1.1: Forward sanity (nonneg, finite, correct shape)
- S1.2: Recon threshold on **primary metric** (per-modality: PSNR, CRC, CNR, FRC, SpectralAngle)
- S1.3: At least 1 calibratable mismatch param, NLL improves after Mode C
- S1.4: Identifiability check

#### Tier B (must pass before release — 4 modalities):
PET, SPECT, Electron Tomography, STEM

Requirements:
- S1.1: Forward sanity
- S1.2: Recon baseline on **primary metric** (> lower threshold)
- S1.3: Calibration optional at first; must be implemented before v4 final

#### Tier C (smoke test — 20 long-tail modalities):
Widefield, Confocal, SIM, Lensless, Light-sheet, Ptychography, Holography, NeRF, 3DGS, Matrix, Panorama, Light Field, Integral, Phase Retrieval, FLIM, Photoacoustic, OCT, FPM, DOT, Widefield Low-Dose

Requirements:
- S1.1: Forward sanity
- Smoke test: no crash on Mode S + Mode I with default solver
- Full CasePack acceptance deferred until modality-specific CasePacks exist

**Two test run modes:**

```python
# Quick mode (CI, every PR): tiny data, CPU, <30s per modality
pytest tests/test_all_modalities_acceptance.py -m quick

# Full mode (nightly): realistic sizes, GPU, comprehensive
pytest tests/test_all_modalities_acceptance.py -m full
```

Implementation:
```python
@pytest.fixture(params=["quick", "full"])
def run_mode(request):
    return request.param

QUICK_SIZES = {"ct": (32, 32), "mri": (32, 32), "cassi": (16, 16, 8), ...}
FULL_SIZES = {"ct": (256, 256), "mri": (128, 128), "cassi": (64, 64, 31), ...}

@pytest.mark.quick
def test_tier_a_acceptance():
    for mod in TIER_A_MODALITIES:
        result = run_casepack(mod, quick=True)
        assert result.forward_ok
        # Primary metric (modality-appropriate: PSNR, CRC, CNR, FRC, etc.)
        assert result.primary_metric > THRESHOLDS[mod]["quick_threshold"], \
            f"{mod}: {result.primary_metric_name}={result.primary_metric} < {THRESHOLDS[mod]['quick_threshold']}"
        assert result.calib_nll_improvement > 0

@pytest.mark.full
def test_tier_a_acceptance_full():
    for mod in TIER_A_MODALITIES:
        result = run_casepack(mod, quick=False)
        assert result.primary_metric > THRESHOLDS[mod]["full_threshold"]
        # Also check secondary metric if defined
        if "secondary_metric" in THRESHOLDS[mod]:
            assert result.secondary_metric > THRESHOLDS[mod]["secondary_quick"]
```

**File:** `contrib/acceptance_thresholds.yaml` (~120 lines):
```yaml
# Tier A — primary_metric is the modality-appropriate quality measure
ct:
  tier: A
  primary_metric: psnr
  quick_threshold: 25    # dB
  full_threshold: 30
  secondary_metric: ssim
  calib_required: true
mri:
  tier: A
  primary_metric: psnr
  quick_threshold: 22
  full_threshold: 28
  secondary_metric: ssim
  calib_required: true
cassi:
  tier: A
  primary_metric: spectral_angle
  quick_threshold: 0.85  # cosine similarity (higher = better)
  full_threshold: 0.92
  secondary_metric: psnr
  secondary_quick: 20    # dB (sanity check)
  calib_required: true
spc:
  tier: A
  primary_metric: psnr
  quick_threshold: 18
  full_threshold: 22
  calib_required: true
cacti:
  tier: A
  primary_metric: psnr
  quick_threshold: 18
  full_threshold: 22
  calib_required: true
ultrasound:
  tier: A
  primary_metric: cnr
  quick_threshold: 1.5   # contrast-to-noise ratio (higher = better)
  full_threshold: 3.0
  secondary_metric: psnr
  secondary_quick: 12
  calib_required: true
sem:
  tier: A
  primary_metric: frc
  quick_threshold: 0.5   # FRC resolution proxy (higher = better)
  full_threshold: 0.7
  secondary_metric: psnr
  secondary_quick: 12
  calib_required: true
tem:
  tier: A
  primary_metric: frc
  quick_threshold: 0.5
  full_threshold: 0.7
  secondary_metric: psnr
  secondary_quick: 12
  calib_required: true

# Tier B — primary metric + PSNR sanity
pet:
  tier: B
  primary_metric: crc
  quick_threshold: 0.3   # contrast recovery coefficient
  full_threshold: 0.6
  secondary_metric: psnr
  secondary_quick: 10
  calib_required: false
spect:
  tier: B
  primary_metric: crc
  quick_threshold: 0.3
  full_threshold: 0.6
  calib_required: false
electron_tomography:
  tier: B
  primary_metric: psnr
  quick_threshold: 12
  full_threshold: 18
  calib_required: false
stem:
  tier: B
  primary_metric: frc
  quick_threshold: 0.4
  full_threshold: 0.6
  calib_required: false

# Tier C (long-tail): forward_ok + smoke only, PSNR as sanity
widefield: {tier: C, primary_metric: psnr, quick_threshold: 20, calib_required: false}
# ...etc for remaining 19 modalities...
```

**Test:** `tests/test_all_modalities_acceptance.py` (~250 lines)

### P5.4: Operator-correction E2E verification

**Test:** `tests/test_operator_correction_e2e.py` (~120 lines):
- CT: measured y + A0 → Mode C → PrePostCorrection (center-of-rotation) → NLL + PSNR improve
- CASSI: measured y + A0 → Mode C → PrePostCorrection (mask shift) → NLL improves
- SEM: measured y + A0 → Mode C → PrePostCorrection (drift) → NLL improves

### P5.5: CasePacks + solver registry for new modalities

**CasePacks:** JSON files for xray_radiography, ultrasound, pet, spect, sem, tem, stem, electron_tomography

**Solver registry additions:**
| Modality | Solvers |
|----------|---------|
| X-ray | direct inversion, iterative |
| Ultrasound | DAS beamforming, adaptive beamforming |
| PET | MLEM, OSEM, PnP-OSEM |
| SPECT | OSEM, MAP-EM |
| SEM | Wiener filter, TV denoising |
| TEM | CTF correction, Wiener in Fourier domain |
| ET | SIRT, ART, ADMM-TV |

---

## Solver Plan

Layers (each works for ALL supporting modalities):

| Layer | Solver | Status | Modalities |
|-------|--------|--------|-----------|
| 0 | Tikhonov / L2 | **v3.1 done** | All linear |
| 1 | TV-FISTA / ADMM-TV | **v3.1 done** | All linear |
| 2 | FBP | **v3.1 done** | CT, PET, SPECT, ET |
| 3 | SENSE / GRAPPA | **NEW** | MRI |
| 4 | DAS beamforming | **NEW** | Ultrasound |
| 5 | MLEM / OSEM | **NEW** | PET, SPECT |
| 6 | PnP-ADMM wrapper | **NEW** | All (plug-in denoiser) |
| 7 | Modality-specific DL | Deferred | Per modality |

---

## Dependency Graph

```
Phase 0 (IR hardening + validation + registries)
  |
  +---> Phase 1 (extended node families + CorrectedOperator)
  |       |
  |       +---> Phase 2 (compressive trio CasePacks)
  |       |
  |       +---> Phase 3 (medical modalities)
  |       |
  |       +---> Phase 4 (electron modalities)
  |
  +-------+-------+-------+
                          |
                    Phase 5 (acceptance gate + prompt UX)
                    requires: Phase 2, 3, 4 all complete
```

Phases 2, 3, 4 can run in parallel after Phase 1.

---

## File Inventory

### New files (~45)

| # | File | Est. lines | Phase | Gap |
|---|------|-----------|-------|-----|
| 1 | `core/modality_registry.py` | 80 | P0 | G8 |
| 2 | `core/casepack_runner.py` | 140 | P0 | G7 |
| 3 | `core/metric_registry.py` | 120 | P0 | G13 |
| 4 | `graph/corrected_operator.py` | 200 | P1 | G3 |
| 5 | `graph/optics_convention.py` | 100 | P0 | G11 |
| 6 | `physics/radiography/xray_radiography_helpers.py` | 60 | P3 | G2 |
| 7 | `physics/ultrasound/ultrasound_helpers.py` | 100 | P3 | G2, G10 |
| 8 | `physics/nuclear/pet_helpers.py` | 80 | P3 | G2 |
| 9 | `physics/nuclear/spect_helpers.py` | 60 | P3 | G2 |
| 10 | `physics/electron/sem_helpers.py` | 60 | P4 | G1 |
| 11 | `physics/electron/tem_helpers.py` | 80 | P4 | G1 |
| 12 | `physics/electron/et_helpers.py` | 80 | P4 | G1 |

> **D4 note:** All `physics/*/` files are **helpers** (pure functions), NOT standalone operators. They never implement `PhysicsOperator.forward()` directly.
| 13 | `contrib/acceptance_thresholds.yaml` | 120 | P5 | G7 |
| 14-21 | `contrib/casepacks/{new}.json` | 8x40 | P5 | — |
| 22 | `tests/test_shape_validation.py` | 80 | P0 | G6 |
| 23 | `tests/test_modality_registry.py` | 40 | P0 | G8 |
| 24 | `tests/test_casepack_framework.py` | 60 | P0 | G7 |
| 25 | `tests/test_optics_convention.py` | 60 | P0 | G11 |
| 26 | `tests/test_metric_registry.py` | 60 | P0 | G13 |
| 27 | `tests/test_corrected_operator.py` | 150 | P1 | G3 |
| 28 | `tests/test_explicit_operator.py` | 80 | P1 | G12 |
| 29 | `tests/test_electron_primitives.py` | 80 | P1 | G1 |
| 30 | `tests/test_specialized_sensors.py` | 80 | P1 | G5 |
| 31 | `tests/test_correlated_noise.py` | 60 | P1 | G4 |
| 32 | `tests/test_new_element_primitives.py` | 120 | P1 | — |
| 33-35 | `tests/test_casepack_{spc,cassi,cacti}.py` | 3x80 | P2 | — |
| 36-41 | `tests/test_casepack_{xray,ct,mri,us,pet,spect}.py` | 6x70 | P3 | — |
| 42-44 | `tests/test_casepack_{sem,tem,et}.py` | 3x70 | P4 | — |
| 45 | `tests/test_all_modalities_acceptance.py` | 250 | P5 | G7 |
| 46 | `tests/test_operator_correction_e2e.py` | 120 | P5 | G3 |

### Modified files

| File | Phase(s) | Changes |
|------|----------|---------|
| `graph/primitives.py` | P1 | +~600: electron (D2-compliant multi-input), sensors, ExplicitLinearOperator, correlated noise, elements |
| `graph/compiler.py` | P0 | +~50: shape validation |
| `graph/canonical.py` | P0 | +~60: x-wiring validation |
| `graph/executor.py` | P1, P5 | +~80: CorrectedOperator integration, correlated-noise guard |
| `objectives/noise_model.py` | P1 | +~30: CorrelatedNoiseModel (simulation-only) |
| `contrib/graph_templates.yaml` | P2-P4 | +~400: new templates |
| `contrib/modalities.yaml` | P0, P3-P4 | +~200: requires_x_interaction, new modalities |
| `contrib/solver_registry.yaml` | P3-P5 | +~150: new solver entries |
| `api/prompt_parser.py` | P5 | +~60: new modalities |
| `core/runner.py` | P5 | +~40: registry integration |

**Total estimated:** ~5,500 new lines across ~48 files

---

## Acceptance Criteria (mapped to phases)

| # | Criterion | Phase | Test |
|---|-----------|-------|------|
| 1 | TensorSpec validated at compile time | P0 | `test_shape_validation.py` |
| 2 | Source-to-x wiring enforced (Pattern A/B); electron modalities use Pattern B multi-input | P0 | `test_canonical_chain.py` (extended) |
| 3 | `requires_x_interaction` validated per modality | P0 | `test_canonical_chain.py` (extended) |
| 4 | MetricRegistry with per-modality metrics (CRC, CNR, FRC, SpectralAngle beyond PSNR) | P0 | `test_metric_registry.py` |
| 5 | CorrectedOperator wraps A; Pre/Post and LowRank pass NLL test | P1 | `test_corrected_operator.py` |
| 6 | ExplicitLinearOperator accepts dense/sparse/callback A with hashing for RunBundle | P1 | `test_explicit_operator.py` |
| 7 | Correlated noise blocked from Mode C unless whitened | P1 | `test_correlated_noise.py` |
| 8 | X-ray/CT default PoissonGaussian noise | P3 | `test_casepack_xray.py`, `test_casepack_ct.py` |
| 9 | Ultrasound: PortSpec enforced (exact shapes/units at each node) | P3 | `test_casepack_ultrasound.py` |
| 10 | Electron primitives: D2-compliant (interaction(incident,x)); labeled Tier 0/1 with `physics_validity_regime` | P4 | `test_casepack_sem.py`, `test_casepack_tem.py` |
| 11 | Fourier optics + Maxwell share `PhotonFieldSpec`; tiers swappable | P0 | `test_optics_convention.py` |
| 12 | Tier A (8 modalities) pass S1.1-S1.4 on primary metric in quick mode | P5 | `test_all_modalities_acceptance.py` |
| 13 | Tier B (4 modalities) pass S1.1-S1.2 on primary metric in quick mode | P5 | `test_all_modalities_acceptance.py` |
| 14 | Tier C (20 modalities) pass forward sanity + smoke | P5 | `test_all_modalities_acceptance.py` |
| 15 | Operator-correction E2E: CT + CASSI + SEM | P5 | `test_operator_correction_e2e.py` |
| 16 | Graph templates are the canonical path; `physics/*/` files are helpers only (D4) | All | Convention (no standalone operator files) |
| 17 | Full test suite: 1500+ tests, 0 failures | P5 | CI |
| 18 | No existing test regresses at any phase boundary | All | CI (1128 baseline) |
