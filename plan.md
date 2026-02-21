# PWM Reconstruction Structure Reorganization Plan

## Problem Statement

The flagship paper and the Finite Primitive Theorem paper define a clean theoretical framework:
- **10 canonical primitives**: P (Propagate), M (Modulate), Π (Project), F (Encode), C (Convolve), Σ (Accumulate), D (Detect), S (Sample), W (Disperse), R (Scatter)
- **4 physics-stage families**: Propagation → {P, C}; Interaction → {M, R}; Encoding-Projection → {Π, F}; Detection-Readout → {Σ, S, W, C, D}
- **5 Detect response families**: linear-intensity, logarithmic, sigmoid, Poisson-rate, coherent-field
- **Canonical DAG decompositions**: e.g. CASSI = M → W → Σ → D, MRI = M → F → S → D, CT = Π → D

The current code (`graph/primitives.py`) has **~90 implementation-level primitives** (fresnel_prop, angular_spectrum, coded_mask, dmd_pattern, spectral_dispersion, ct_radon, mri_kspace, etc.) with no explicit mapping to the paper's 10 canonical types. The `ir_types.py` `PhysicsSubrole` enum (propagation, modulation, sampling, interaction, transduction, encoding, relay) doesn't match the paper's 4-family classification either.

This means:
1. The code cannot produce or validate the canonical DAG decompositions claimed in the paper
2. The basis-growth saturation analysis has no code backing
3. The extension protocol is not implemented
4. The ε_tier2 fidelity metric is not computed
5. A reviewer or collaborator reading the paper cannot trace claims to code

---

## Phase 1: Canonical Primitive Enum & Mapping Layer

**Goal**: Create an authoritative mapping between the paper's 10 canonical primitives and the code's ~90 implementation primitives.

### 1.1 Add `CanonicalPrimitive` enum to `ir_types.py`

```python
class CanonicalPrimitive(str, Enum):
    """The 10 canonical primitives from the Finite Primitive Basis Theorem."""
    P = "propagate"       # Free-space wave propagation
    M = "modulate"        # Element-wise multiplication (mask, coil, absorption)
    Pi = "project"        # Radon line-integral projection
    F = "encode"          # Fourier-domain encoding (k-space)
    C = "convolve"        # Spatial convolution (PSF)
    Sigma = "accumulate"  # Summation over spectral/temporal axis
    D = "detect"          # Detector response (5 canonical families)
    S = "sample"          # Sub-sampling on index set
    W = "disperse"        # Wavelength-dependent spatial shift
    R = "scatter"         # Direction change and/or energy shift
```

### 1.2 Add `PhysicsStageFamily` enum to `ir_types.py`

```python
class PhysicsStageFamily(str, Enum):
    """The 4 physics-stage families from the FPB Theorem proof."""
    propagation = "propagation"                      # → {P, C}
    interaction = "interaction"                      # → {M, R}  (elastic + scattering)
    encoding_projection = "encoding_projection"      # → {Π, F}
    detection_readout = "detection_readout"           # → {Σ, S, W, C, D}
```

### 1.3 Add `DetectFamily` enum to `ir_types.py`

```python
class DetectFamily(str, Enum):
    """The 5 canonical Detect response families."""
    linear_intensity = "linear_intensity"   # η(x) = g|x|²
    logarithmic = "logarithmic"             # η(x) = g·log(1 + |x|²/x₀)
    sigmoid = "sigmoid"                     # η(x) = g·σ(|x|² - x₀)
    poisson_rate = "poisson_rate"           # η(x) = g|x|² (returns Poisson rate)
    coherent_field = "coherent_field"       # η(x) = g·Re[x·e^(iφ)]
```

### 1.4 Add `canonical_id` attribute to `BasePrimitive`

Every existing primitive class gets a `_canonical_id: Optional[CanonicalPrimitive]` class attribute. The mapping:

| Implementation primitive(s) | canonical_id | Physics stage |
|---|---|---|
| `fresnel_prop`, `angular_spectrum`, `acoustic_propagation` | `P` | propagation |
| `coded_mask`, `dmd_pattern`, `sim_pattern`, `thin_object_phase`, `beer_lambert`, `optical_absorption` | `M` | interaction |
| `ct_radon`, `emission_projection`, `sar_backprojection` | `Pi` | encoding_projection |
| `mri_kspace`, `reciprocal_space_geometry` | `F` | encoding_projection |
| `conv2d`, `conv3d`, `depth_optics`, `diffraction_camera` | `C` | propagation (or detection_readout when detector PSF) |
| `sum_axis`, `frame_integration`, `shutter_integration`, `bucket_integration`, `fluoro_temporal_integrator` | `Sigma` | detection_readout |
| `magnitude_sq`, `saturation`, `log_compress`, `photon_sensor`, `coil_sensor`, `transducer_sensor`, `generic_sensor`, `single_pixel_sensor`, `xray_detector_sensor`, `spad_tof_sensor`, `energy_resolving_detector` | `D` | detection_readout |
| `random_mask`, `temporal_mask`, `scan_trajectory`, `tof_gate`, `collimator_model` | `S` | detection_readout |
| `spectral_dispersion`, `chromatic_warp` | `W` | detection_readout |
| `scatter_model`, `multiple_scattering` | `R` | interaction |
| Sources (`photon_source`, `xray_source`, `acoustic_source`, `spin_source`, etc.) | `None` | — (not a physics primitive per FPB theorem; source nodes) |
| Noise (`poisson`, `gaussian`, `poisson_gaussian`, `fpn`, etc.) | `None` | — (additive noise, not part of forward operator H) |
| Corrections (`affine_correction`, `residual_correction`, `field_map_correction`) | `None` | — (correction layer, not in B_lib) |
| Utility (`identity`, `quantize`, `adc_clip`) | `None` | — (readout post-processing, not canonical primitives) |
| Rendering stubs (`volume_rendering_stub`, `gaussian_splatting_stub`) | `None` | — (Tier-3+, outside C_Tier2) |
| Optics-specific (`objective_lens`, `relay_lens`, `imbalanced_response`) | `C` or `M` | case-dependent |

Also add `_physics_stage: Optional[PhysicsStageFamily]` and `_detect_family: Optional[DetectFamily]` where applicable.

### 1.5 Build `CANONICAL_REGISTRY` in `primitives.py`

A reverse lookup: `Dict[CanonicalPrimitive, List[str]]` that maps each canonical ID to the list of implementation primitive_ids that realize it. Populated automatically from class attributes at module load.

**Files modified**: `ir_types.py`, `primitives.py`

---

## Phase 2: Canonical DAG Decomposition Registry

**Goal**: Implement the 31-modality decomposition table from the FPT paper (Table 1) as a code-level registry.

### 2.1 Create `graph/canonical_decompositions.py`

A new module containing the canonical DAG decomposition for each modality, expressed in the 10-primitive notation:

```python
@dataclass
class CanonicalDecomposition:
    modality: str
    dag: str                   # e.g. "M → W → Σ → D"
    primitives: List[CanonicalPrimitive]
    nodes: int
    depth: int
    carrier: str               # photon, electron, spin, acoustic, particle
    e_tier2: str               # e.g. "<1e-4"
    validation_level: str      # "full", "held_out", "exotic", "template"

CANONICAL_DECOMPOSITIONS: Dict[str, CanonicalDecomposition] = {
    # Full-validation modalities (Table 1, rows 1-7)
    "cassi":        CanonicalDecomposition("cassi",        "M → W → Σ → D", [M, W, Sigma, D], 4, 4, "photon",   "<1e-4",   "full"),
    "cacti":        CanonicalDecomposition("cacti",        "M → Σ → D",     [M, Sigma, D],     3, 3, "photon",   "<1e-4",   "full"),
    "spc":          CanonicalDecomposition("spc",          "M → Σ → D",     [M, Sigma, D],     3, 3, "photon",   "<1e-4",   "full"),
    "lensless":     CanonicalDecomposition("lensless",     "C → D",          [C, D],            2, 2, "photon",   "<1e-5",   "full"),
    "ptychography": CanonicalDecomposition("ptychography", "M → P → D",     [M, P, D],         3, 3, "photon",   "4.2e-4",  "full"),
    "mri":          CanonicalDecomposition("mri",          "M → F → S → D", [M, F, S, D],      4, 4, "spin",     "<1e-6",   "full"),
    "ct":           CanonicalDecomposition("ct",           "Π → D",          [Pi, D],           2, 2, "xray",     "<1e-5",   "full"),
    # Held-out modalities (Table 1, rows 8-12)
    "oct":              CanonicalDecomposition("oct",              "P+P → Σ → D",       [P, P, Sigma, D], 4, 3, "photon",   "3.8e-4",  "held_out"),
    "photoacoustic":    CanonicalDecomposition("photoacoustic",    "M → P → D",         [M, P, D],        3, 3, "acoustic",  "7.1e-4",  "held_out"),
    "sim":              CanonicalDecomposition("sim",              "M → C → D",         [M, C, D],        3, 3, "photon",   "2.5e-4",  "held_out"),
    "phase_contrast":   CanonicalDecomposition("phase_contrast",   "Π → P → M → D",    [Pi, P, M, D],    4, 4, "xray",     "1.2e-3",  "held_out"),
    "electron_ptycho":  CanonicalDecomposition("electron_ptycho",  "M → P → D",         [M, P, D],        3, 3, "electron", "5.6e-4",  "held_out"),
    # Exotic modalities (Table 1, rows 13-19)
    "ghost_imaging":  CanonicalDecomposition("ghost_imaging",  "M → Σ → D",             [M, Sigma, D],    3, 3, "photon", "1.9e-4",  "exotic"),
    "thz_tds":        CanonicalDecomposition("thz_tds",        "C → D_coh",             [C, D],           2, 2, "photon", "8.3e-4",  "exotic"),
    "compton":        CanonicalDecomposition("compton",        "M → R → D",             [M, R, D],        3, 3, "xray",   "6.7e-3",  "exotic"),
    "raman":          CanonicalDecomposition("raman",          "M → R → D",             [M, R, D],        3, 3, "photon", "4.1e-3",  "exotic"),
    "fluorescence":   CanonicalDecomposition("fluorescence",   "M → R → D",             [M, R, D],        3, 3, "photon", "3.8e-3",  "exotic"),
    "dot":            CanonicalDecomposition("dot",            "M → R∘P∘R → D",         [M, R, P, R, D],  5, 5, "photon", "8.9e-3",  "exotic"),
    "brillouin":      CanonicalDecomposition("brillouin",      "M → R → D",             [M, R, D],        3, 3, "photon", "5.2e-3",  "exotic"),
}
```

### 2.2 Add `to_canonical()` method to `GraphOperator`

Given a compiled graph, produce its canonical decomposition by mapping each node's `primitive_id` → `canonical_id` and collapsing the DAG to the 10-primitive notation. This enables runtime validation that an arbitrary graph matches the expected canonical form.

### 2.3 Add `validate_decomposition()` function

Given a modality key and a `GraphOperator`, verify:
- The canonical form matches `CANONICAL_DECOMPOSITIONS[modality]`
- Node count and depth are within bounds (N_max=20, D_max=10)
- All nodes map to known canonical primitives

**Files created**: `graph/canonical_decompositions.py`
**Files modified**: `graph/graph_operator.py`

---

## Phase 3: ε_tier2 Fidelity Metric

**Goal**: Implement the formal fidelity metric from the FPT paper to validate that DAG decompositions satisfy ε < 0.01.

### 3.1 Create `graph/fidelity.py`

Implement both metrics from the paper:

```python
def operator_norm_fidelity(H_true, H_dag, n_trials=20, seed=42) -> float:
    """Operator-norm relative error: ||H - H_dag|| / ||H|| ≤ ε
    Estimated via power iteration on random test vectors."""
    ...

def pointwise_fidelity(H_true, H_dag, X_test, delta=1e-8) -> float:
    """Pointwise metric (Eq. S1 in FPT supplementary):
    e_tier2 = sup_x ||H(x) - H_dag(x)||₂ / (||H(x)||₂ + δ)"""
    ...

def mean_fidelity(H_true, H_dag, X_test, delta=1e-8) -> float:
    """Mean pointwise metric (Eq. S2 in FPT supplementary):
    ē_tier2 = (1/|X_test|) Σ_x ||H(x) - H_dag(x)||₂ / (||H(x)||₂ + δ)"""
    ...
```

### 3.2 Add adjoint consistency check at canonical level

Verify that the composed DAG adjoint satisfies ⟨Ax, y⟩ = ⟨x, A†y⟩ using the randomized dot-product test described in Methods (Eq. 5 in methods.tex). This already exists for individual primitives; extend it to validate the canonical-level composition.

**Files created**: `graph/fidelity.py`

---

## Phase 4: Extension Protocol Implementation

**Goal**: Implement the 5-step formal extension protocol from Section 7 of the FPT paper.

### 4.1 Create `graph/extension_protocol.py`

```python
@dataclass
class ExtensionProposal:
    """A proposal to add a new canonical primitive to B_lib."""
    primitive_id: str
    canonical_name: str
    forward_adjoint_validated: bool       # Step 1: forward()/adjoint() pass dot-product test
    representation_gap: float             # Step 2: min_G e_tier2 > ε without new primitive
    error_with_primitive: float           # Step 3: e_tier2 < ε with new primitive
    modalities_requiring: List[str]       # Step 4: ≥2 modalities need it
    closure_test_passed: bool             # Step 5: backward-compatible, all existing decomps still valid

def validate_extension(proposal: ExtensionProposal, epsilon: float = 0.01) -> Tuple[bool, List[str]]:
    """All 5 criteria must pass. Returns (passed, list_of_failures)."""
    failures = []
    if not proposal.forward_adjoint_validated:
        failures.append("Step 1: forward/adjoint not validated")
    if proposal.representation_gap <= epsilon:
        failures.append(f"Step 2: representation gap {proposal.representation_gap:.4f} <= eps={epsilon}")
    if proposal.error_with_primitive > epsilon:
        failures.append(f"Step 3: error with primitive {proposal.error_with_primitive:.4f} > eps={epsilon}")
    if len(proposal.modalities_requiring) < 2:
        failures.append(f"Step 4: only {len(proposal.modalities_requiring)} modalities need it (need >=2)")
    if not proposal.closure_test_passed:
        failures.append("Step 5: closure test failed (backward compatibility broken)")
    return (len(failures) == 0, failures)
```

### 4.2 Add basis-growth tracking

A function that, given the modality registration order and their canonical decompositions, computes the K-vs-N basis-growth curve (Fig. 1 of FPT paper / Fig. 9 of flagship) and checks for saturation.

```python
def basis_growth_curve(registration_order: List[str]) -> List[Tuple[int, int]]:
    """Returns [(N, K)] where N = number of modalities registered so far,
    K = number of distinct canonical primitives needed."""
    ...

def is_saturated(curve: List[Tuple[int, int]], window: int = 5) -> bool:
    """Returns True if K has not increased in the last `window` modalities."""
    ...
```

**Files created**: `graph/extension_protocol.py`

---

## Phase 5: Update `NodeTags` and `GraphNode` Schemas

**Goal**: Thread the new canonical types through the graph IR so they're available at compile time and in serialized specs.

### 5.1 Extend `NodeTags` in `ir_types.py`

Add fields:
```python
canonical_id: Optional[CanonicalPrimitive] = None
physics_stage: Optional[PhysicsStageFamily] = None
detect_family: Optional[DetectFamily] = None
```

### 5.2 Update `GraphCompiler` to populate canonical tags

During the Bind step (Step 3 of `compile()`), auto-populate `canonical_id`, `physics_stage`, and `detect_family` from primitive class attributes into `NodeTags`. This is analogous to the existing code that populates `physics_tier` and `physics_subrole`.

### 5.3 Update `GraphNode` in `graph_spec.py`

Add optional `canonical_id` field so that YAML templates can declare canonical type explicitly (for validation against auto-inferred type).

**Files modified**: `ir_types.py`, `graph_spec.py`, `compiler.py`

---

## Phase 6: Update YAML Graph Templates & Add Tests

**Goal**: Annotate existing graph templates with canonical primitive types so they can be validated.

### 6.1 Update `graph_templates.yaml`

For each modality template, add:
- `canonical_decomposition`: the 10-primitive DAG string (e.g., "M → W → Σ → D")
- Per-node `canonical_id` annotations

### 6.2 Add a CI test

A test that loads every graph template, compiles it, extracts its canonical form via `to_canonical()`, and asserts it matches the declared `canonical_decomposition`.

**Files modified**: `contrib/graph_templates.yaml` (or wherever templates live)
**Files created**: `tests/test_canonical_decompositions.py`

---

## Summary of Changes

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| 1 | — | `ir_types.py`, `primitives.py` |
| 2 | `graph/canonical_decompositions.py` | `graph/graph_operator.py` |
| 3 | `graph/fidelity.py` | — |
| 4 | `graph/extension_protocol.py` | — |
| 5 | — | `ir_types.py`, `graph_spec.py`, `compiler.py` |
| 6 | `tests/test_canonical_decompositions.py` | `contrib/graph_templates.yaml` |

**Total**: 4 new files, 6 modified files. No existing functionality is broken; all changes are additive.

## Execution Order

Phases 1 → 5 → 2 → 3 → 4 → 6

Phase 1 (enums + mapping) and Phase 5 (schema updates) come first because all other phases depend on the canonical types being defined. Phase 2 (decomposition registry) depends on Phase 1. Phase 3 (fidelity) and Phase 4 (extension protocol) are independent. Phase 6 (templates + tests) comes last as integration validation.

## Non-Goals

- **No changes to reconstruction solvers** (`recon/` module). The reorganization is purely in the forward-model representation layer.
- **No changes to physics operators** (`physics/` module). The existing modality-specific operators continue to work; this adds a canonical abstraction on top.
- **No changes to the agent system** (`agents/` module). The MismatchAgent already operates on the graph; canonical types give it richer metadata but don't change its logic.
- **No removal of existing primitives**. The ~90 implementation primitives remain; they gain a `canonical_id` attribute that maps them to the paper's 10 types.
