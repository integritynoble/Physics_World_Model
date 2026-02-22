# Targeting System Implementation Plan

> The harness is the rail. Solvers are trains. The OperatorGraph is the track gauge.
>
> *"Whoever defines the metrics defines the economy."* -- `docs/pwm_lockin_strategy.md`

---

## Context: Where This Fits

### The 18-Month Lock-In Window

Per `docs/pwm_lockin_strategy.md`, standards for computational imaging evaluation are **not yet set**. The first evaluation protocol adopted by 5+ labs becomes permanent (the QWERTY Moment). PWM must establish its position within this window. The targeting system is the centerpiece.

### Maturity: L1 -> L2 -> L3

Per `rails/maturity_levels.md`, PWM is transitioning from **L1 (Measurable)** to **L2 (Repeatable)**. The targeting system implementation serves this transition:

| Transition | What the targeting system must do |
|-----------|-----------------------------------|
| L1 -> L2 | Validate the 4-scenario protocol. Make calibration workflows repeatable via `pwm evaluate`. |
| L2 -> L3 | Automate evaluation across 20+ modalities. Open harness to external submissions. Achieve rho >= 0.80 on 10+ modalities. |
| L3 -> L4 | Multiple competing agents on the open harness. CISP operational. |

### LIP-Arena Phases

Per `docs/targeting_system.md` (440-line frozen spec), LIP-Arena matures in 4 phases:

| Phase | Timeline | What Gets Built |
|-------|----------|-----------------|
| **A (Internal)** | 0-6 months | `pwm evaluate` runs locally. Sealed-simulator only. 3 modalities (CASSI, SPC, CACTI). Red Team = self-testing. First 3 counterfactual packs. |
| B (Pilot External) | 6-12 months | First live-lab partners. 10+ modalities. Independent Red Team. Third-party submissions accepted. |
| C (Full Operation) | 12-24 months | 5+ partner labs. 20+ modalities. Quarterly rounds. Prospective dominance scoring. |
| D (Utility) | 24+ months | Rolling submissions. Hardware-in-the-loop. Protocol extractable as standalone standard. |

**This implementation plan builds Phase A.** Everything for Phase A must work before external submissions are possible.

### How All 10 Gears Connect to the Targeting System

Per `rails/README.md` and `rails/industrial_stack.md`:

| Gear | Relationship to Targeting System |
|------|----------------------------------|
| **1. Targeting System** | **IS** the harness -- this plan |
| 2. Outcome Contracts | Harness *verifies* the outcomes that contracts pay for (rho, oracle_gap, RoIC) |
| 3. Compute Escrow | Harness *enforces* compute budgets (2x disqualification, RoIC tracking) |
| 4. Action Networks | Harness *scores* the outcomes of software actuation (16 validated modalities) |
| 5. Data Trusts | Harness *uses* synthetic-first data; real-data via dataset registry |
| 6. Decision Logs | Every submission *produces* a RunBundle with DR-IS records |
| 7. Two-Source Rule | Multi-solver portfolio *tested* on the same harness |
| 8. Compute + Energy | RoIC metric *makes* compute measurable |
| 9. Fairness Targets | Tail-risk weighting + anti-Goodhart *built into* scoring |
| 10. Literacy | `pwm evaluate --help` + contribution guides teach everyone to use the harness |

---

## The Rail / Train Distinction

Per `docs/pwm_lockin_strategy.md` Part 1:

> *"PWM does NOT develop new reconstruction methods -- it evaluates them. PWM does NOT pick winners -- the targeting system picks winners based on outcomes."*

### What Is Rail (PWM builds, freezes, maintains)

| Component | Why it's rail | Key files |
|-----------|---------------|-----------|
| **OperatorGraph IR** | Universal representation for any imaging forward model | `graph/graph_spec.py`, `graph/compiler.py` |
| **Physics Fidelity Ladder** (Tier 0-3) | 4-tier hierarchy: geometry → approx → full transport → learned surrogate | `graph/ir_types.py`, `graph/tier_policy.py` |
| **Primitive registry** | ~30 atomic operators (FresnelProp, Radon, CodedMask...) | `graph/primitives.py` |
| **Graph compiler** | OperatorGraphSpec → executable GraphOperator | `graph/compiler.py` |
| **Targeting harness** (LIP-Arena) | 4-scenario engine + Commit-Measure-Score + Red Team | `docs/targeting_system.md` (frozen spec) |
| **Scoring formulas** | rho, oracle gap, RoIC, anti-Goodhart weights | `docs/targeting_system.md` S3, S5 |
| **RunBundle v0.3.0** | Audit trail + SHA-256 hashing | `docs/contracts/runbundle_schema.md` |
| **LinearLikeOperator protocol** | `forward()`, `adjoint()`, `x_shape`, `y_shape` | `recon/protocols.py` |
| **YAML registries** | 6 registries: modalities, mismatch, photon, compression, metrics, solver | `contrib/*.yaml` |
| **Triad Law** | Recoverability, Carrier Budget, Operator Mismatch diagnosis | `docs/purpose.md` |

### What Is Train (community contributes, competes, replaces)

| Component | Why it's train | Contribution difficulty |
|-----------|---------------|----------------------|
| **Reconstruction solvers** | `run_solver(y, physics, cfg) -> (x_hat, info)`. Stateless. Replaceable. | **Easy** |
| **Calibrators** | `calibrate(y, H_nom, budget) -> (H_hat, info)`. Estimate the corrected operator. | **Medium** |
| **Graph templates** | YAML specs defining a new modality's OperatorGraph topology. | **Medium** |
| **New primitives** | New atomic physics operators (e.g., a Tier 2 scattering kernel). RFC process. | **Hard** |

### The Boundary Rule

> A contributed solver never touches the OperatorGraph, the compiler, or the harness. It only calls `physics.forward()` and `physics.adjoint()` through the `LinearLikeOperator` protocol.

---

## How Others Contribute (4 Levels)

### Level 1: New Solver (easiest)

**Who**: Any ML researcher, PhD student, or imaging lab.

**What they do**: Write one function:

```python
def run_my_solver(y, physics, cfg):
    """
    Args:
        y: measurements (numpy array)
        physics: LinearLikeOperator (.forward(), .adjoint(), .x_shape, .y_shape)
        cfg: dict of solver-specific parameters
    Returns:
        (x_hat, info_dict)
    """
    x_hat = my_algorithm(y, physics.forward, physics.adjoint, **cfg)
    return x_hat, {"solver": "my_solver", "iters": cfg.get("iters", 100)}
```

**The solver never knows what modality it's solving.** Write once, compete on all 64+ modalities.

**Path**: Fork -> implement -> add to `solver_registry.yaml` -> `pwm evaluate --solver my_solver --modality cassi` -> submit RunBundle -> leaderboard.

**Publishable**: "Our method achieves rho=0.85 across 20 modalities on LIP-Arena" is a paper.

### Level 2: New Calibrator (medium)

**Who**: Self-calibration, blind deconvolution, operator learning researchers.

**What they do**: Write a calibration function that uses `get_theta()`/`set_theta()` on the GraphOperator:

```python
def calibrate_my_method(y, H_nom, budget):
    theta_hat = estimate_parameters(y, H_nom, budget)
    H_hat = H_nom.with_updated_params(theta_hat)
    return H_hat, {"method": "my_calibrator", "params_found": theta_hat}
```

**Publishable**: "Our blind calibrator reduces oracle gap from 12 dB to 2 dB on compound mismatch" is a paper.

### Level 3: New Graph Template (medium-hard)

**Who**: Domain experts with a modality PWM doesn't cover (e.g., 4D-STEM, terahertz).

**What they do**: Write a YAML graph template composing existing primitives + add entries to `modalities.yaml`, `mismatch_db.yaml`, `photon_db.yaml`, `metrics_db.yaml`, `solver_registry.yaml`.

**Publishable**: "We formalize 4D-STEM as an OperatorGraph and benchmark 10 solvers" is a paper.

### Level 4: New Primitive (hardest -- RFC process)

**Who**: Physics experts willing to implement a new atomic operator at a specific fidelity tier.

**What they do**: Implement the `PrimitiveOp` protocol with `_physics_tier` attribute.

**This IS touching the rail**, but in a controlled way: they add a node type, not modify the compiler or harness. We review and curate.

**Publishable**: "Our full-wave primitive improves calibration fidelity by 3 dB across 5 modalities" is a paper.

---

## What the Harness Does (the engine)

The harness is **modality-agnostic**. It doesn't know CASSI from MRI. It reads from existing YAML registries and the OperatorGraph compiler:

```
pwm evaluate --modality cassi --solver mst_l --track correct --budget 600

┌─────────────────────────────────────────────────────────────────┐
│                    THE ONE HARNESS (frozen)                      │
│                                                                  │
│  Inputs (from existing PWM registries):                          │
│    modality  → graph_templates.yaml → compile → GraphOperator   │
│    mismatch  → mismatch_db.yaml → parameter sampler             │
│    solver    → solver_registry.yaml → load solver function       │
│    noise     → photon_db.yaml → noise model                     │
│    metrics   → metrics_db.yaml → which metrics to compute       │
│                                                                  │
│  Protocol (from docs/targeting_system.md):                       │
│    1. Compile GraphOperator for modality (H_true)               │
│    2. Sample mismatch → create nominal operator (H_nom)         │
│    3. Generate synthetic scene (from modality x_shape)           │
│    4. Generate measurement: y = H_true(x) + noise               │
│    5. Run 4 scenarios (I: ideal, II: assumed, III: corrected,   │
│       IV: oracle mask)                                           │
│    6. Score: rho, oracle_gap, RoIC                              │
│    7. Apply anti-Goodhart penalties (S5 of targeting spec)       │
│    8. Check safety brakes (S6.4 of targeting spec)              │
│    9. Emit RunBundle v0.3.0 with SHA-256 hashes + DR-IS         │
│   10. Output leaderboard-ready JSON                              │
│                                                                  │
│  The harness NEVER imports modality-specific code.               │
│  Everything comes through OperatorGraph + registries.            │
└─────────────────────────────────────────────────────────────────┘
```

### 4 Evaluation Tracks (from `docs/targeting_system.md` S3)

| Track | Goal | Key Scoring | Weight |
|-------|------|-------------|--------|
| **1: Correct** | Infer H_hat, correct mismatch, reconstruct | rho (0.30), param recovery (0.20), uncertainty (0.15), tail-risk (0.15), cross-modality (0.10), RoIC (0.10) | 0.35 |
| **2: Diagnose** | Attribute failure to Triad gate | Attribution accuracy (0.35), evidence quality (0.25), action relevance (0.20), shift robustness (0.20) | 0.20 |
| **3: No-GT** | Correct without ground truth | Self-consistency (0.30), physical invariants (0.25), held-out channels (0.20), trap survival (0.15), RoIC (0.10) | 0.25 |
| **4: Design** | Specify robust OperatorGraph | Constraint satisfaction (0.25), Pareto efficiency (0.20), robustness margin (0.25), calibration cost (0.20), prediction accuracy (0.10) | 0.20 |

Composite: `S_total = 0.35*S_correct + 0.20*S_diagnose + 0.25*S_no_gt + 0.20*S_design`

### Anti-Goodhart Scoring (from `docs/targeting_system.md` S5)

**Prospective dominance**: `S_rank = 0.3 * S_retro + 0.7 * S_prospective`

**Gaming penalties** (mechanically enforced):

| Check | Penalty |
|-------|---------|
| Wrong Triad attribution | -15% of track score |
| Overconfident uncertainty (< 75% coverage at 90% CI) | -10% |
| Identifiability inconsistency | -10% |
| Compute dishonesty (declared < 0.5x actual) | Disqualification |
| Missing TriadReport or operator estimate | Not scored |

### Safety Brakes (from `docs/targeting_system.md` S6.4)

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Recovery ratio regression | rho < 0.30 | Block deployment |
| Uncertainty miscalibration | Coverage deviates > 15% | Flag as uncalibrated |
| Out-of-family miss | Wrong gate diagnosis | Mandatory retraining |
| Budget exceeded | > 2x declared | Disqualification |
| Consistency violation | Re-projection error > 3x median | Quarantine |

### Red Team (from `docs/targeting_system.md` S4)

7 injection categories per round: novel mismatch, compound mismatch, out-of-family physics, distribution shift, compute traps, gate-flip, misleading consistency.

Escalation: Rounds 1-2 mild → 3-4 moderate → 5-6 severe → 7+ catastrophic.

---

## New Additions (Community Adoption + Lock-In Hardening)

### Addition 1: Rail Constitution (`docs/RAIL_CONSTITUTION.md`)

A top-level governance document codifying what is frozen vs evolvable. Prevents benchmark drift and protects lock-in. See `docs/RAIL_CONSTITUTION.md` (created alongside this plan).

**Key articles**:
- **Article 1 (Frozen)**: OperatorGraph compiler, LinearLikeOperator protocol, 4-scenario protocol, scoring formulas, safety brakes, RunBundle v0.3.0, solver/calibrator signatures
- **Article 2 (Evolvable)**: New primitives (RFC), new metrics (additive), new modalities, new solvers/calibrators, new tracks, new safety brakes (tighten only)
- **Article 3 (Governance)**: Who decides, change process for frozen components (90-day comment, unanimous vote, major version bump), experimental → candidate → stable promotion
- **Article 4 (Anti-Drift)**: Regression tests on frozen formulas, SHA-256 hash anchoring of spec docs, solver isolation enforcement via CI

### Addition 2: `pwm scaffold` Command (10-Minute Onboarding)

Level-1 solver contribution is the growth engine. Onboarding must be < 10 minutes.

```bash
# Scaffold a new solver
pwm scaffold solver my_solver

# Generates:
# contrib/solvers/my_solver/
#   ├── solver.py          # Correct signature, inline docs, physics.forward/adjoint examples
#   ├── config.yaml        # Default parameters + solver_registry.yaml entry template
#   ├── test_local.py      # Self-test against a toy 64x64 operator (runs in <5s)
#   └── README.md          # "What to do next" checklist

# Scaffold a new modality
pwm scaffold modality 4dstem

# Generates:
# contrib/modalities/4dstem/
#   ├── graph.yaml         # Graph template skeleton with annotated fields
#   ├── mismatch.yaml      # Mismatch parameter template
#   ├── photon.yaml        # Noise model template
#   ├── metrics.yaml       # Metric selection template
#   ├── solver_entry.yaml  # solver_registry.yaml entry
#   └── README.md          # "How to fill in the physics" guide
```

**Implementation**: `targeting/scaffold.py` -- template generator using string formatting (no heavy deps). Wired into CLI as `pwm scaffold`.

**Why**: Removes the "where do I start?" barrier. A PhD student types one command and has a working skeleton in their editor.

### Addition 3: Sandbox Mode (`pwm evaluate --sandbox`)

Full harness is expensive and intimidating. Sandbox mode is the "hello world" for contributors.

```bash
pwm evaluate --sandbox --solver my_solver --modality cassi
```

Sandbox mode:
- **Tiny operators**: 32x32 or 64x64 (not 256x256). Compiles in <1s.
- **1 scene only**: No multi-scene averaging
- **No Red Team**: Skip adversarial injection
- **No escrow**: Budget tracking informational, never disqualifies
- **No prospective scoring**: Retrospective only (no sealed data)
- **Fast**: Completes in < 60 seconds on CPU
- **Full output**: Still emits RunBundle, still computes rho/oracle_gap/RoIC -- just on toy data

**Purpose**: Contributors can verify their solver works, see the scoring pipeline, understand the output format -- before committing to a full evaluation run.

**Implementation**: `Harness.__init__` accepts `sandbox=True` which overrides scene dimensions, scene count, and disables Red Team / escrow / prospective splits.

### Addition 4: Physics Adapter Layer (Tier-2/3 Integration)

The biggest risk: Tier-2/3 physics (full Maxwell, Monte Carlo, quantum corrections) is hard for outsiders to contribute. Solution: adapter wrappers that handle the complexity.

```
graph/adapters/
  ├── __init__.py
  ├── tier_adapter.py       # Base adapter with validation, adjoint checking, error handling
  ├── tier0_adapter.py      # Geometry: coordinate transforms, projection, ray tracing
  ├── tier1_adapter.py      # Wave approx: Fourier optics, paraxial, Born approximation
  ├── tier2_adapter.py      # Full transport: Maxwell, scattering, Monte Carlo wrapping
  └── tier3_adapter.py      # Learned surrogates: uncertainty calibration, error bar enforcement
```

Each adapter provides:

```python
class Tier2Adapter:
    """Wraps a raw physics kernel into a valid PrimitiveOp.

    Handles:
    - Adjoint correctness validation (automatic dot-product test)
    - Shape checking and dtype normalization
    - Physics tier tag injection (_physics_tier = "tier2_full")
    - Serialization boilerplate
    - Error handling with clear messages for common mistakes

    Contributors implement only:
    - forward_kernel(x, params) -> y
    - adjoint_kernel(y, params) -> x  (or auto-derived for linear ops)
    """

    def wrap(self, forward_kernel, adjoint_kernel=None, params=None) -> PrimitiveOp:
        ...
```

**Why**: A physics PhD can contribute a scattering kernel as a raw function without learning the PrimitiveOp protocol, BasePrimitive, serialization, or graph integration. The adapter handles all of that.

### Addition 5: Contributor Reputation Tracking

People contribute for visibility, not just science. Make contributions legible.

```
community/reputation.yaml
```

Schema:

```yaml
contributors:
  - github: "@alice_optics"
    name: "Alice Chen"
    affiliation: "MIT"
    contributions:
      solvers: ["pnp_alice_v1"]
      calibrators: []
      modalities: ["4dstem"]
      primitives: ["multislice_v2"]
    score: 4               # count of accepted contributions
    first_contribution: "2026-03-15"
    leaderboard_positions:  # best rho achieved on each modality
      cassi: 0.82
      spc: 0.91
```

- Updated automatically on PR merge via CI
- Displayed on leaderboard alongside scores
- Enables "Top Contributors" recognition at CISP events
- Links to RunBundles proving each contribution

**Why**: Social capital drives open-source contribution. A PhD student's PWM reputation is visible to hiring committees and conference reviewers.

### Addition 6: Stable vs Experimental Rail

Protect scoring integrity while allowing innovation.

| Namespace | Location | Used for official scoring? | Promotion path |
|-----------|----------|---------------------------|----------------|
| **Stable** | `targeting/` | Yes | -- |
| **Experimental** | `targeting/experimental/` | No -- informational only | 2 rounds of validation → governance vote → promote to stable |

**Rules**:
- Official leaderboard rankings use only stable-rail scoring
- Experimental metrics/tracks are computed and displayed but labeled `[EXPERIMENTAL]`
- Contributors can propose experimental metrics via PR
- Experimental metrics that prove valuable get promoted to candidate → stable (per Rail Constitution Article 3.3)

**Implementation**: `scoring.py` loads scoring weights from a config that distinguishes stable vs experimental. Experimental metrics are computed but excluded from `final_score`.

### Addition 7: Interface Isolation Tests (Anti-Cheating)

Solvers that peek at internals undermine the entire evaluation.

```python
# tests/test_solver_isolation.py

def test_solver_does_not_import_compiler():
    """Solver must not import graph compiler internals."""
    solver_module = importlib.import_module("my_solver")
    source = inspect.getsource(solver_module)
    assert "graph.compiler" not in source
    assert "graph.primitives" not in source
    assert "targeting" not in source

def test_solver_does_not_access_ground_truth():
    """Solver receives only y and physics (H_nom), never H_true or x_gt."""
    # Run solver through harness with a mock that tracks attribute access
    spy_physics = SpyLinearLikeOperator(H_nom)
    x_hat, info = solver(y, spy_physics, cfg)
    assert "H_true" not in spy_physics.accessed_attributes
    assert "x_gt" not in spy_physics.accessed_attributes

def test_solver_signature_compliance():
    """Solver must accept exactly (y, physics, cfg) and return (x_hat, info)."""
    sig = inspect.signature(solver_fn)
    params = list(sig.parameters.keys())
    assert params == ["y", "physics", "cfg"]
```

**Enforcement**: These tests run automatically on every solver PR. Failure = automatic rejection.

### Addition 8: Frozen Baseline Pack

To become CASP-like, you need official reference baselines that anchor the field.

```
baselines/
  ├── README.md                    # "What baselines are and how to use them"
  ├── gap_tv/
  │   ├── config.yaml             # Frozen parameters
  │   ├── runbundles/             # Reference RunBundles for CASSI, SPC, CACTI
  │   │   ├── cassi_gap_tv_v1.zip
  │   │   ├── spc_gap_tv_v1.zip
  │   │   └── cacti_gap_tv_v1.zip
  │   └── scores.json             # Official baseline scores (rho, oracle_gap, RoIC)
  ├── fista_tv/
  │   ├── config.yaml
  │   ├── runbundles/
  │   └── scores.json
  ├── mst_l/
  │   ├── config.yaml
  │   ├── runbundles/
  │   └── scores.json
  └── pnp_admm/
      ├── config.yaml
      ├── runbundles/
      └── scores.json
```

**Rules**:
- Baselines are frozen: same parameters, same seeds, same results forever
- Every new submission is compared against baselines in the leaderboard
- Baselines are re-run on new modalities as they are added (but parameters don't change)
- Baselines span solver families: classical (GAP-TV, FISTA-TV), PnP (PnP-ADMM), deep (MST-L)
- Baselines provide the "floor" -- if you can't beat GAP-TV, something is wrong

**Known baseline scores** (from `rails/benchmarks.md`):

| Baseline | CASSI rho | SPC rho | Status |
|----------|-----------|---------|--------|
| GAP-TV | 0.60 | -- | Classical reference |
| FISTA-TV | -- | 0.81 | Classical reference |
| MST-L | 0.47 | -- | Deep learning reference |
| HATNet | -- | 0.90 | Deep learning reference |

### Addition 9: CISP Annual Challenge Framework (Start Now)

Per `docs/pwm_lockin_strategy.md` Part 2, CISP must launch within months 7-12. The infrastructure should exist from day one, even if small.

```
community/challenges/
  ├── template/                    # Existing template (already built)
  ├── 2026-W10/ .. 2026-W13/     # Existing weekly challenges (already built)
  ├── 2026_CISP/                  # Annual challenge framework
  │   ├── README.md               # Rules, timeline, tracks, prizes
  │   ├── tracks/
  │   │   ├── track1_correct.md   # Spectral imaging (CASSI) under mismatch
  │   │   ├── track2_temporal.md  # Video compressive imaging (CACTI) under motion blur
  │   │   ├── track3_medical.md   # MRI/CT under acquisition artifacts
  │   │   └── track4_cross.md    # Cross-modal transfer performance
  │   ├── datasets/
  │   │   ├── generate_cisp_data.py  # Sealed-simulator data generator
  │   │   └── README.md              # Data format, access rules
  │   ├── scoring/
  │   │   ├── cisp_scorer.py      # Wraps targeting/scoring.py with CISP-specific weighting
  │   │   └── cisp_leaderboard.py # Extends community/leaderboard.py
  │   ├── submissions/            # Received RunBundles
  │   ├── results/                # Published round results
  │   └── governance/
  │       ├── stewards.md         # Independent steward board (3-5 neutral parties)
  │       └── rules.md            # Anti-gaming rules, DQ conditions, appeal process
  ```

**Even if CISP 2026 has only 5 submissions**, the infrastructure exists. This signals seriousness and creates the Schelling point for the community.

### Addition 10: Contributor Profiles (Persona-Based Onboarding)

Instead of just levels, map **personas** to concrete first tasks.

```
docs/contributors/profiles.md
```

| Profile | Background | First Task | Time to First Result | Starter Issue Label |
|---------|-----------|------------|---------------------|-------------------|
| **ML Student** | PyTorch, optimization | Add solver via `pwm scaffold solver` | 1 day | `good-first-solver` |
| **Imaging PhD** | Optics, MRI, spectral | Add modality via `pwm scaffold modality` | 2-3 days | `good-first-modality` |
| **Physicist** | Maxwell, scattering, MC | Add Tier-2/3 primitive via adapter | 1 week | `physics-tier-help` |
| **Industry Engineer** | Product, deployment | Run benchmark, report results | 1 day | `benchmark-extension` |

Each profile includes:
- **1-day starter path**: step-by-step from clone to first `pwm evaluate` output
- **Common pitfalls**: "Don't import from `graph.compiler`", "Your solver must be stateless"
- **Example paper**: what a publication using PWM looks like for this persona
- **Mentorship pointer**: link to community Discord/forum channel for this persona

**Why**: People self-identify as personas, not "Level 1 contributors". This converts docs-readers into contributors.

### Addition 11: `pwm contrib check` (One-Command PR Validation)

Contributors shouldn't need to guess if their PR will pass. One command validates everything locally.

```bash
pwm contrib check my_solver
```

Runs in sequence:
1. **Signature test**: `(y, physics, cfg) -> (x_hat, info)` compliance
2. **Isolation test**: no forbidden imports (`graph.compiler`, `graph.primitives`, `targeting.*`)
3. **Ground truth test**: solver never accesses `H_true` or `x_gt`
4. **Sandbox eval**: `pwm evaluate --sandbox --solver my_solver` on 3 modalities
5. **Registry validation**: solver entry valid YAML, all required fields present
6. **Format check**: PEP 8, type annotations on public API, docstring on main function

Output:
```
✓ Signature compliance
✓ Isolation (no forbidden imports)
✓ Ground truth isolation
✓ Sandbox eval: CASSI rho=0.45, SPC rho=0.62, CACTI rho=0.38
✓ Registry entry valid
✓ Format check passed

Ready for PR. Run: gh pr create --title "Add solver: my_solver"
```

**Implementation**: `targeting/contrib_check.py` -- orchestrates existing tests + sandbox harness. Wired into CLI as `pwm contrib check`.

**Why**: Eliminates human review back-and-forth. If `pwm contrib check` passes, the PR is mechanically correct. Reviewers only evaluate scientific merit.

### Addition 12: Reference Implementations (Copy, Don't Read)

Templates show structure. Reference implementations show **working code**.

```
examples/
  ├── level1_solver/
  │   ├── solver.py              # Working FISTA solver (~40 lines)
  │   ├── config.yaml            # Real config that produces rho=0.55 on CASSI
  │   ├── run_example.sh         # "pwm evaluate --sandbox --solver fista_example"
  │   └── expected_output.json   # What the output should look like
  ├── level2_calibrator/
  │   ├── calibrator.py          # Working grid-search calibrator (~60 lines)
  │   ├── config.yaml
  │   ├── run_example.sh
  │   └── expected_output.json
  ├── level3_modality/
  │   ├── graph.yaml             # Working graph template for a simple modality
  │   ├── registry_entries.yaml  # All 5 registry entries filled in
  │   ├── run_example.sh
  │   └── expected_output.json
  └── level4_primitive/
      ├── primitive.py           # Working Tier-1 primitive (~80 lines)
      ├── adjoint_test.py        # Dot-product adjoint correctness proof
      ├── run_example.sh
      └── expected_output.json
```

**Key difference from templates**: These are **real submissions** with **real scores**. A contributor can:
1. Copy `examples/level1_solver/solver.py`
2. Replace the algorithm body
3. Run `pwm contrib check my_solver`
4. Submit

**Why**: People copy working code, not documentation. Every major framework (PyTorch, HuggingFace, FastAPI) grows through examples.

### Addition 13: External Plugin Loading (No Fork Required)

Forking is a barrier. Support installing external solvers directly.

```bash
# Install a solver from a git repo
pwm install https://github.com/alice/my_solver

# Or from a local directory
pwm install ./my_solver_dir

# What it does:
# 1. Validates solver signature and isolation
# 2. Copies solver to contrib/solvers/my_solver/
# 3. Adds entry to solver_registry.yaml
# 4. Runs pwm contrib check automatically
# 5. Solver is now usable: pwm evaluate --solver my_solver
```

**Plugin contract**:
```
my_solver/
  ├── solver.py         # Must export run_my_solver(y, physics, cfg) -> (x_hat, info)
  ├── config.yaml       # Must contain: name, version, family, supported_modalities
  └── requirements.txt  # Optional: extra dependencies (numpy-only preferred)
```

**Implementation**: `targeting/plugin_loader.py` -- validates, copies, registers. Wired into CLI as `pwm install`.

**Rules**:
- Installed plugins are local-only by default (not pushed to main repo)
- To share: contributor submits PR with the validated plugin
- Plugin isolation tests run at install time (not just at PR time)
- Plugins that fail isolation are rejected with a clear error

**Why**: Lowers contribution from "fork repo + learn git workflow + PR" to "install and run". HuggingFace Hub, pip install, and npm proved this scales.

### Addition 14: Contributor Credits and Authorship Policy

Academic incentive is the primary driver for computational imaging contributions.

```
docs/contributors/CREDITS.md
```

**Policy**:

| Contribution | Recognition |
|-------------|------------|
| Accepted solver/calibrator | Listed on `pwm.ai/contributors` with affiliation and link to RunBundle |
| Top-3 on any modality leaderboard | Highlighted on leaderboard with badge |
| Accepted modality (Level 3) | Co-author on next PWM benchmark paper |
| Accepted primitive (Level 4) | Co-author on next PWM benchmark paper + named in RAIL_CONSTITUTION.md |
| CISP top-3 per track | Named in CISP proceedings, invited to CISP workshop talk |
| Community steward | Listed in governance, acknowledgment in all papers |

**Automatic credit generation**:
- Every RunBundle records `contributor_id` in provenance
- `community/leaderboard.py` links scores to contributor profiles
- CISP proceedings auto-generated from RunBundle metadata

**Why**: In academia, credit = career advancement. Making credit automatic and visible makes PWM contribution a rational career choice for PhD students and postdocs.

### Addition 15: Tier-2/3 Wrapper Templates (Physics Kernel Only)

Complement the adapter layer (Addition 4) with concrete templates that physicists fill in.

```python
# contrib/templates/tier2_wrapper.py

"""
Tier-2 Physics Kernel Template
===============================
You implement: forward_kernel() and adjoint_kernel()
We handle: validation, shape checking, serialization, graph integration

Example: Mie scattering kernel for microscopy
"""

import numpy as np
from graph.adapters import Tier2Adapter

# === YOU IMPLEMENT THESE ===

def forward_kernel(x: np.ndarray, params: dict) -> np.ndarray:
    """Your physics forward model.

    Args:
        x: input field (shape defined by your primitive)
        params: physical parameters (wavelength, refractive_index, etc.)
    Returns:
        y: output field after physics
    """
    # YOUR CODE HERE
    # Example: y = mie_scatter(x, params["wavelength"], params["n_sphere"])
    raise NotImplementedError

def adjoint_kernel(y: np.ndarray, params: dict) -> np.ndarray:
    """Your physics adjoint (transpose of Jacobian).

    If your forward is linear, this is the matrix transpose.
    If nonlinear, this is the Jacobian-vector product.

    Args:
        y: output-space vector
        params: same physical parameters
    Returns:
        x: input-space vector (adjoint application)
    """
    # YOUR CODE HERE
    raise NotImplementedError

# === WE HANDLE THE REST ===

adapter = Tier2Adapter()
my_primitive = adapter.wrap(
    forward_kernel=forward_kernel,
    adjoint_kernel=adjoint_kernel,
    params={"wavelength": 532e-9, "n_sphere": 1.5},
    input_shape=(256, 256),
    output_shape=(256, 256),
)

# Auto-runs:
# 1. Adjoint correctness test (dot-product test, tolerance 1e-6)
# 2. Shape validation
# 3. Energy conservation check (for linear ops)
# 4. Serialization test
# 5. Registers as PrimitiveOp with _physics_tier = "tier2_full"
```

**Also provides**:
- `contrib/templates/tier2_test.py` -- standalone adjoint correctness test
- `contrib/templates/tier3_wrapper.py` -- learned surrogate template with uncertainty calibration

**Why**: Physicists know their forward model and its adjoint. They should not need to learn PrimitiveOp, BasePrimitive, serialization, graph IR, or any software engineering abstractions. Fill in two functions → done.

### Addition 16: Three-Speed Governance Model

Prevent bottlenecks by matching review speed to risk level.

| Lane | What | Review | Merge Condition |
|------|------|--------|----------------|
| **Fast Lane** (auto-merge) | Solvers, calibrators, config tweaks | CI only | `pwm contrib check` passes |
| **Review Lane** (1-2 maintainers) | Modalities, metrics, track tweaks | 1-2 PWM maintainer reviews | CI passes + maintainer approval |
| **Governance Lane** (RFC) | Primitives, scoring changes, protocol | RFC + 90-day comment + core team | Per Rail Constitution Article 3 |

**Implementation**: GitHub Actions workflow with labels:
- `fast-lane`: auto-merge on CI pass (solver/calibrator PRs)
- `review-lane`: request 1-2 reviewers (modality/metric PRs)
- `governance`: block merge, open RFC discussion (primitive/scoring PRs)

**PR auto-labeling** (in CI):
```yaml
# .github/workflows/pr_classify.yml
# If PR only touches contrib/solvers/ → fast-lane
# If PR touches contrib/modalities/ or contrib/metrics_db.yaml → review-lane
# If PR touches graph/primitives.py or targeting/scoring.py → governance
```

**Why**: CASP succeeded because submitting results was frictionless (fast lane) while changing the evaluation was hard (governance lane). Most open-source projects die because easy PRs get stuck behind hard PRs in a single review queue.

### Addition 17: Community Steward Board

External legitimacy requires independent oversight.

```yaml
# community/stewards.yaml
stewards:
  - name: "TBD - Spectral Imaging Expert"
    affiliation: "TBD"
    role: "Review Tier-2/3 primitives for spectral modalities"
    term: "2026-2028"

  - name: "TBD - Medical Imaging Expert"
    affiliation: "TBD"
    role: "Review medical modalities and fairness targets"
    term: "2026-2028"

  - name: "TBD - Optimization/ML Expert"
    affiliation: "TBD"
    role: "Review solver scientific merit, audit anti-Goodhart"
    term: "2026-2028"

governance:
  min_stewards: 3
  max_stewards: 5
  term_years: 2
  responsibilities:
    - Review governance-lane PRs
    - Vote on Rail Constitution changes (per Article 3)
    - Validate CISP challenge design and results
    - Annual public report on rail health
```

**Recruitment**: Target stewards from labs already publishing in spectral/medical/computational imaging. Offer co-authorship on PWM benchmark papers as incentive.

**Why**: Self-governance lacks credibility. CASP has independent assessors. ImageNet has an advisory board. PWM needs 3-5 respected external voices to be taken seriously by the field.

### Addition 18: Formal Governance Document (`docs/GOVERNANCE.md`)

The three-speed model (Addition 16) needs **hard rules with deadlines**, not just labels.

See `docs/GOVERNANCE.md` (created alongside this plan).

**Key rules**:

| Rule | Lane | Enforcement |
|------|------|-------------|
| **No human veto on solvers** | Fast Lane | If `pwm contrib check` passes → auto-merge in 48h. No maintainer can block. |
| **7-day review deadline** | Review Lane | 2 independent reviewers required. If no review in 7 days → auto-escalate to steward. No silent blocking. |
| **90-day RFC minimum** | Governance Lane | Per Rail Constitution Article 3. Steward vote required. |
| **No self-scoring** | All lanes | PWM team solvers compete on the same harness as external solvers. No shortcuts. |
| **Rotation** | Stewards | 2-year terms. No perpetual gatekeepers. |

**Why**: Without hard deadlines and no-veto rules, PWM becomes "your lab's benchmark". With them, it becomes "the field's benchmark". CASP succeeded because nobody controlled outcomes personally.

### Addition 19: External Submission Channel (`pwm submit`)

Many labs will compete but never submit PRs. CASP allowed submission without joining the dev team.

```bash
# Submit a RunBundle without touching the repo
pwm submit runbundle.zip

# What it does:
# 1. Validates RunBundle integrity (SHA-256 hashes, manifest, artifacts)
# 2. Verifies solver isolation (no H_true in artifacts, no forbidden imports in logs)
# 3. Scores the RunBundle using targeting/scoring.py
# 4. Uploads to leaderboard
# 5. Returns: score, rank, permalink

# Flow:
# run locally → pwm submit → score → leaderboard
# No fork. No PR. No code merge.
```

**Submission tiers**:

| Tier | Requirement | Leaderboard Visibility |
|------|------------|----------------------|
| **Anonymous** | Valid RunBundle only | Score shown, contributor anonymous |
| **Identified** | RunBundle + contributor profile | Score + name + affiliation shown |
| **Reproducible** | RunBundle + source code | Score + name + "reproducible" badge |

**Only reproducible submissions are eligible for CISP prizes.** But anonymous submissions still appear on the leaderboard, encouraging early/experimental participation.

**Winners who want their solver in the official registry** submit a PR after competing -- but competition does not require it.

**Implementation**: `targeting/submit.py` -- validates RunBundle, scores, updates leaderboard. Wired into CLI as `pwm submit`.

**Why**: Requiring PRs for competition kills adoption. Many industry labs and foreign institutions won't contribute code but will compete. This is exactly how CASP works.

### Addition 20: Modality Pack Specification (`docs/modality_pack_spec.md`)

Formalize what a self-contained modality looks like so external consortia can add modalities without politics.

See `docs/modality_pack_spec.md` (created alongside this plan).

```
my_modality_pack/
  ├── graph.yaml           # OperatorGraphSpec (composing existing primitives)
  ├── mismatch.yaml        # Mismatch parameters, ranges, distributions
  ├── photon.yaml          # Noise model (type, SNR range)
  ├── metrics.yaml         # Which metrics to compute
  ├── meta.yaml            # Name, version, author, description, domain, license
  ├── README.md            # Physics description, reference papers
  └── LICENSE              # MIT/BSD/Apache required
```

**Validation**:
```bash
pwm install-modality ./my_modality_pack

# Validates:
# 1. All 5 YAML files present and well-formed
# 2. graph.yaml references only existing primitives in PRIMITIVE_REGISTRY
# 3. mismatch.yaml parameters match graph.yaml's tunable parameters
# 4. Compiles successfully via graph/compiler.py
# 5. Runs in sandbox mode with at least 1 baseline solver
# 6. Passes test_registry_integrity.py after insertion
```

**Rules**:
- Must be self-contained (no external dependencies beyond existing primitives)
- Must run in sandbox
- Must pass validation
- Must include LICENSE (MIT/BSD/Apache only for registry inclusion)
- Installed modality packs are local by default; PR required for official registry

**Why**: Industry, new domains (terahertz, quantum imaging, astronomical), and external consortia need a formal spec to contribute modalities without navigating the full codebase.

### Addition 21: Plugin Tier Separation

Clear distinction between local experiments and official standards.

| Tier | Location | Leaderboard? | Baselines? | How to get there |
|------|----------|-------------|-----------|-----------------|
| **Local** | `~/.pwm/plugins/` | No -- local only | No | `pwm install` or `pwm install-modality` |
| **Community** | `contrib/solvers/`, `contrib/modalities/` | Yes | No | PR → fast-lane or review-lane merge |
| **Official** | `baselines/` | Yes + **BASELINE** badge | Yes (frozen) | PWM team curates; never changes |

**Rules**:
- Local plugins are invisible to leaderboard and other users
- Community contributions appear on leaderboard after merge
- Official baselines are frozen reference points (per Addition 8)
- Promotion: local → community requires PR; community → official requires governance vote

**Implementation**: `targeting/plugin_loader.py` tracks plugin tier. `community/leaderboard.py` filters by tier.

**Why**: Without tier separation, experiments get confused with standards. A local experimental solver should never appear as an "official PWM result".

### Addition 22: IP and Licensing Policy (`docs/IP_POLICY.md`)

Industry won't contribute without legal clarity.

See `docs/IP_POLICY.md` (created alongside this plan).

**Key rules**:

| Component | Required License | Rationale |
|-----------|-----------------|-----------|
| Solvers | MIT or Apache-2.0 | Must be reproducible by anyone |
| Calibrators | MIT or Apache-2.0 | Same |
| Modality packs | MIT or Apache-2.0 | Must be usable by competing labs |
| Primitives | Apache-2.0 (with patent grant) | Rail components need patent protection |
| Datasets (synthetic) | CC-BY-4.0 | Academic reuse |
| Datasets (real) | Per data trust agreement | See `rails/gear05_data_trusts.md` |
| RunBundles | CC-BY-4.0 | Results must be citable and reusable |
| PWM core (rail) | Apache-2.0 | Per `community/OPEN_CORE_BOUNDARY.md` |

**CLA (Contributor License Agreement)**:
- Required for all contributions that touch rail (primitives, harness, scoring)
- Not required for train contributions (solvers, calibrators, modality packs)
- CLA grants PWM Foundation the right to relicense rail code (not solver code)

**Patent policy**:
- Contributing a primitive grants a royalty-free patent license to all PWM users
- Solvers retain contributor's patent rights (no grant required)

**Why**: Without explicit IP policy, industry legal teams will block contributions. Every successful open-source project (Linux, Apache, CNCF) has a clear IP framework. This is table stakes for institutional adoption.

### Addition 23: Rail Charter (`docs/RAIL_CHARTER.md`)

The trust document. 5 commitments that make PWM credible as a field standard.

See `docs/RAIL_CHARTER.md` (created alongside this plan).

**The 5 Commitments**:

1. **PWM exists to evaluate, not promote.** PWM does not develop reconstruction methods. It evaluates them. No PWM-authored solver receives privileged treatment on the harness.

2. **All scoring is reproducible.** Every score on the leaderboard can be reproduced from the RunBundle. Every RunBundle is integrity-verified. Every scoring formula is published and frozen.

3. **All baselines are frozen.** Reference baselines (GAP-TV, FISTA-TV, MST-L, PnP-ADMM) use the same parameters, same seeds, same results forever. They are the field's fixed reference points.

4. **Governance rotates.** No individual or lab permanently controls the rail. Stewards serve 2-year terms. The Rail Constitution requires external votes for changes to frozen components.

5. **The community owns the outcomes.** RunBundles, leaderboard results, and CISP proceedings belong to the community (CC-BY-4.0). PWM provides the infrastructure; the community provides the science.

**Why**: This is the "constitution preamble" -- the document that reviewers, program committees, and funding agencies read first. It must be short, principled, and unambiguous. CASP's credibility rests on similar public commitments.

### Addition 24: Updated Architecture Diagram

With all additions integrated:

```
         US (PWM team)                          THEM (community)
    ┌──────────────────────────┐          ┌───────────────────────────────┐
    │   RAIL (frozen)          │          │      TRAINS (compete)         │
    │   [Charter + Constitution│          │                               │
    │    + Governance]         │          │  Profiles: ML / Imaging /     │
    │                          │          │    Physics / Industry         │
    │ OperatorGraph IR         │          │                               │
    │ Graph compiler           │          │  Level 1: New solvers          │
    │ Primitive registry       │          │    pwm scaffold solver →       │
    │ Targeting harness        │          │    pwm contrib check →         │
    │ Scoring (stable)         │          │    pwm evaluate --sandbox →    │
    │ Scoring (exp.)           │◄────────►│    pwm evaluate --full →       │
    │ RunBundle schema         │          │    leaderboard + reputation    │
    │ Anti-Goodhart            │          │                               │
    │ Safety brakes            │          │  Level 2: New calibrators     │
    │ Isolation tests          │          │  Level 3: New modalities      │
    │ Baselines (frozen)       │          │    Modality Pack Spec →        │
    │                          │          │    pwm install-modality →      │
    │ The track gauge:         │          │  Level 4: New primitives      │
    │ LinearLikeOperator       │          │    Tier wrappers → RFC →       │
    │                          │          │    steward review              │
    │ Governance:              │          │                               │
    │   Fast lane (auto 48h)   │          │  Submit without PR:           │
    │   Review lane (7d max)   │          │    pwm submit runbundle.zip    │
    │   Governance lane (RFC)  │          │    Anonymous / Identified /    │
    │   No human veto on       │          │    Reproducible tiers          │
    │   solvers                │          │                               │
    │                          │          │  Plugins: Local / Community /  │
    │ Steward board (rotating) │          │    Official tiers              │
    │ CISP infra               │          │                               │
    │ IP Policy (Apache/MIT)   │          │  Credits:                     │
    │ Ref. implementations     │          │    Papers / website / CISP    │
    └──────────────────────────┘          └───────────────────────────────┘
```

---

## Community Contribution Pipeline (End-to-End Flow)

Two paths: **PR path** (code contributions) and **Submit path** (competition-only).

### Path A: PR Path (code enters registry)

```
Step 1: Discover                    Step 2: Scaffold
  Read profiles.md                    pwm scaffold solver my_solver
  Pick persona + starter issue        (or: pwm scaffold modality, calibrator)
  Clone repo                          Working skeleton in <1 minute
       │                                    │
       ▼                                    ▼
Step 3: Develop                     Step 4: Validate
  Implement algorithm                 pwm contrib check my_solver
  Copy from examples/level1/          Signature ✓ Isolation ✓ Sandbox ✓
  Iterate locally                     Registry ✓ Format ✓
       │                                    │
       ▼                                    ▼
Step 5: Evaluate                    Step 6: Submit PR
  pwm evaluate --sandbox (fast)       gh pr create
  pwm evaluate --full (official)      Auto-labeled: fast-lane
  See rho, oracle_gap, RoIC           CI runs pwm contrib check
       │                                    │
       ▼                                    ▼
Step 7: Merge + Register            Step 8: Recognition
  Fast lane: auto-merge in 48h       Leaderboard updated
  Solver in solver_registry.yaml      Contributor profile updated
  Globally usable by all users        Credits on pwm.ai/contributors
       │                                    │
       ▼                                    ▼
Step 9: Compete                     Step 10: Publish
  CISP annual challenge               "rho=0.85 on LIP-Arena" → paper
  Quarterly leaderboard rounds        CISP top-3 → invited talk
  Cross-modality rankings             Co-author on benchmark paper
```

### Path B: Submit Path (competition without code merge)

```
Step 1: Download                    Step 2: Run Locally
  pip install pwm                     pwm evaluate --solver my_solver
  Read profiles.md                    --modality cassi --output ./results
       │                                    │
       ▼                                    ▼
Step 3: Submit                      Step 4: Compete
  pwm submit ./results/runbundle.zip  Score appears on leaderboard
  Choose: anonymous / identified      Rank updated in real-time
       │                                    │
       ▼                                    ▼
Step 5: Win                         Step 6: (Optional) Merge
  CISP prizes for top-3               Submit PR to make solver official
  Reproducible badge if code shared   Moves from leaderboard → registry
```

**Key insight**: Path B lets labs compete **without touching the repo**. This is how CASP scaled to 100+ groups. Path A is for contributors who want their code in the ecosystem permanently.

---

## Implementation Plan

### Phase 1: Harness Core Engine

**Goal**: Build the modality-agnostic evaluation engine that reads from existing registries and implements the protocol defined in `docs/targeting_system.md`.

**All files under `packages/pwm_core/pwm_core/targeting/`**:

#### 1. `__init__.py`

Package init. Version = "1.0.0". Exports: `Harness`, `ScoredResult`, `BudgetGuard`.

#### 2. `harness.py` -- The One Harness

The central engine. Modality-agnostic.

```python
class Harness:
    def __init__(self, modality: str, solver: str, track: str, budget_s: int):
        """
        Loads everything from existing YAML registries:
        - graph_templates.yaml → OperatorGraphSpec → compile → GraphOperator (H_true)
        - mismatch_db.yaml → mismatch parameter names, ranges, distributions
        - photon_db.yaml → noise model (type, parameters)
        - solver_registry.yaml → solver function (module, function name)
        - metrics_db.yaml → which metrics to compute for this modality
        """

    def run(self, n_scenes: int, seed: int, severity: str) -> HarnessResult:
        """
        For each scene:
        1. Generate synthetic scene x from modality x_shape
        2. Compile H_true from graph template
        3. Sample mismatch params → create H_nom
        4. Generate measurement: y = H_true(x) + noise
        5. Run 4 scenarios via scenarios.py
        6. Score via scoring.py
        7. Check safety brakes
        8. Emit RunBundle via runbundle_emitter.py
        Returns HarnessResult with all scores, RunBundle path.
        """
```

`HarnessResult` dataclass: modality, solver, track, per_scene_results, aggregate_scores, runbundle_path, timing, budget_report.

#### 3. `scenarios.py` -- 4-Scenario Protocol

Implements the 4-scenario protocol from `docs/targeting_system.md`:

```python
@dataclass
class ScenarioResult:
    scenario_id: str  # "I", "II", "III", "IV"
    psnr: float
    ssim: float
    x_hat: np.ndarray
    runtime_s: float
    budget_used_s: float

def run_scenario_I(scene, H_true, solver_fn, cfg, budget) -> ScenarioResult:
    """Ideal: measure with H_true, reconstruct with H_true."""

def run_scenario_II(scene, H_true, H_nom, solver_fn, cfg, budget) -> ScenarioResult:
    """Assumed: measure with H_true, reconstruct with H_nom."""

def run_scenario_III(scene, H_true, H_nom, calibrator_fn, solver_fn, cfg, budget) -> ScenarioResult:
    """Corrected: calibrate H_nom → H_hat, then reconstruct with H_hat."""

def run_scenario_IV(scene, H_true, H_nom, solver_fn, cfg, budget) -> ScenarioResult:
    """Oracle Mask: reconstruct with partial oracle (e.g., true H with nominal dispersion)."""
```

#### 4. `scoring.py` -- Metrics + Anti-Goodhart

Implements scoring from `docs/targeting_system.md` S3 and S5:

```python
def compute_recovery_ratio(psnr_i, psnr_ii, psnr_iii) -> float:
    """rho = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II)"""

def compute_oracle_gap(psnr_i, psnr_iii) -> float:
    """PSNR_I - PSNR_III"""

def compute_roic(psnr_gain_db, gpu_seconds) -> float:
    """dB recovered per GPU-hour"""

def compute_track_score(results, track, weights) -> float:
    """Weighted combination per track (S3 of targeting spec)"""

def apply_anti_goodhart(raw_score, diagnostics) -> float:
    """S_rank = 0.3 * S_retro + 0.7 * S_prospective, minus gaming penalties"""

def check_safety_brakes(results) -> list[SafetyBrakeViolation]:
    """Check all 5 pre-committed thresholds from S6.4"""
```

`ScoredResult` dataclass: rho, oracle_gap, roic, track_scores, penalties, safety_violations, final_score, per_scene_breakdown.

Also computes composite metrics from `docs/pwm_benchmarks_solveeverything.md`:
- **OFS** (Operator Fidelity Score): PSNR_III / PSNR_I ratio
- **MSI** (Mask-Sensitivity Index): degradation per unit mismatch (dB/px)
- **Calibration Efficiency**: dB recovered per GPU-second

#### 5. `budget.py` -- Compute Budget Enforcement

Per `rails/gear03_compute_escrow.md`:

```python
class BudgetGuard:
    """Context manager enforcing wall-clock and GPU-hour limits.

    - Tracks actual consumption during solver execution
    - Raises BudgetExceeded if > 2x declared budget (disqualification)
    - Detects sandbagging if declared < 0.5x actual (dishonesty penalty)
    - Returns BudgetReport: seconds_used, gpu_hours, fraction_of_declared
    """
```

#### 6. `mismatch_sampler.py` -- Sample Mismatch from Registry

Reads `contrib/mismatch_db.yaml`:

```python
def sample_mismatch(modality: str, severity: str, rng) -> dict[str, float]:
    """
    Sample mismatch parameters for a modality.
    Severity: mild (0.25x range), moderate (0.5x), severe (1.0x), catastrophic (2.0x+)
    Maps to Red Team escalation schedule from targeting_system.md S4.
    """

def inject_mismatch(H_true: GraphOperator, theta_mismatch: dict) -> GraphOperator:
    """Create H_nom by applying mismatch to H_true's parameters via set_theta()."""
```

### Phase 2: RunBundle Emission + CLI

#### 7. `runbundle_emitter.py` -- Produce RunBundle v0.3.0

Per `docs/contracts/runbundle_schema.md`:

```python
def emit_runbundle(result: HarnessResult, output_dir: Path) -> Path:
    """
    Produces a RunBundle directory:
    - runbundle_manifest.json (v0.3.0 schema)
    - artifacts: x_gt.npy, y.npy, x_hat.npy per scenario
    - metrics.json: all computed metrics
    - dr_is_records.json: DR-IS decision chain (per gear06)
    - SHA-256 hashes for every artifact
    - provenance: git_hash, seeds, platform, pwm_version
    """
```

Integrates with existing `community/validate.py` (326 lines) for integrity verification and `community/leaderboard.py` (352 lines) for ranking.

#### 8. `cli.py` -- `pwm evaluate` Subcommand

Per the README.md CLI spec:

```bash
# Core usage (already documented in README.md)
pwm evaluate --method my_solver --modality cassi --track correct
pwm evaluate --method my_solver --modality cassi --scenarios I,II,III,IV
pwm evaluate --method my_solver --method mst_l --modality cassi

# Extended flags for harness
pwm evaluate --modality cassi --solver mst_l --track correct --budget 600
pwm evaluate --modality spc --solver my_solver --scenes 10 --severity moderate
pwm evaluate --modality cassi --solver gap_tv --red-team --output ./results/
pwm evaluate --modality cassi --solver my_solver --dry-run  # validate only
```

Flags:
- `--modality`: looks up graph template + mismatch + noise + metrics from YAML registries
- `--solver` / `--method`: looks up in `solver_registry.yaml`
- `--track`: correct | diagnose | no_gt | design
- `--budget`: compute budget in seconds (default from modality config)
- `--scenes`: number of synthetic scenes (default 5)
- `--severity`: mild | moderate | severe | catastrophic
- `--red-team`: include adversarial scenarios (7 categories from S4)
- `--output`: RunBundle output directory
- `--dry-run`: validate setup without executing (contribution testing)

Console output: table of scenario PSNRs, rho, oracle_gap, RoIC, OFS, safety brake status, pass/fail.

### Phase 3: Contribution Interfaces

**Goal**: Make it trivial for external researchers to contribute trains.

#### 9. `contrib/templates/contrib_solver_template.py`

Documented skeleton showing:
- The `run_<solver>(y, physics, cfg) -> (x_hat, info)` signature
- How to use `physics.forward()`, `physics.adjoint()`, `physics.x_shape`, `physics.y_shape`
- Self-test function against a toy operator
- How to add entry to `solver_registry.yaml`

Extends existing `contrib/templates/new_operator_template.py` and `contrib/templates/new_calibrator_template.py`.

#### 10. `contrib/templates/contrib_calibrator_template.py`

Documented skeleton showing:
- The `calibrate_<method>(y, H_nom, budget) -> (H_hat, info)` signature
- How to use `H_nom.get_theta()`, `H_nom.set_theta()`, `H_nom.forward()`
- Self-test against known mismatch (Scenario II → III gain verification)

#### 11. `contrib/templates/contrib_graph_template_example.yaml`

Annotated YAML showing:
- How to compose a new modality from existing primitives
- Which fields are required in `modalities.yaml`, `mismatch_db.yaml`, `photon_db.yaml`, `metrics_db.yaml`, `solver_registry.yaml` (all 5 registries -- per the registry integrity tests)
- How to set `_physics_tier` on each node
- How to run `pwm evaluate` to validate the new modality

#### 12. Update `community/CONTRIBUTING_CHALLENGE.md`

Add 4-level contribution guide:
- "How to contribute a solver" (Level 1 -- easiest, one function)
- "How to contribute a calibrator" (Level 2)
- "How to contribute a new modality" (Level 3 -- add to all 6 registries + graph template)
- "How to propose a new primitive" (Level 4 -- RFC issue -> discussion -> implementation -> adjoint tests)

References existing `community/OPEN_CORE_BOUNDARY.md` for licensing.

### Phase 4: Validation + Integration Tests

#### 13. `tests/test_harness.py`

End-to-end tests:
- Run harness on CASSI with GAP-TV: 4 scenarios, check rho > 0, check rho matches known benchmarks from `rails/benchmarks.md` (rho=0.60 for GAP-TV)
- Run harness on SPC with FISTA-TV: check rho matches known benchmarks (rho=0.81)
- Budget enforcement: solver exceeding 2x budget gets disqualified
- Anti-Goodhart: synthetic gaming attempt gets penalized
- RunBundle emission: valid v0.3.0 schema, correct SHA-256 hashes, all artifacts present
- Safety brakes: rho < 0.30 triggers block
- Integration with existing `community/validate.py` and `community/leaderboard.py`

#### 14. `tests/test_contribution_path.py`

Contribution path validation:
- A minimal solver function (from template) works through full harness
- A new graph template compiles and evaluates correctly
- Dry-run mode validates without executing
- Invalid solver signature rejected with clear error
- All 6 registries remain consistent after adding a test entry (runs `test_registry_integrity.py`)

---

## Existing Code to Integrate (NOT rewrite)

The harness should integrate with, not duplicate, existing infrastructure:

| Existing Component | Location | How Harness Uses It |
|-------------------|----------|-------------------|
| Graph compiler | `graph/compiler.py` | Compile modality → GraphOperator |
| Graph operator | `graph/graph_operator.py` | H_true and H_nom for scenarios |
| Tier policy | `graph/tier_policy.py` | Select physics fidelity per node |
| Primitives | `graph/primitives.py` | ~30 atomic operators |
| Solver registry | `contrib/solver_registry.yaml` | Look up solver function by name |
| Modality registry | `contrib/modalities.yaml` | Look up modality dimensions and config |
| Mismatch DB | `contrib/mismatch_db.yaml` | Sample mismatch parameters |
| Photon DB | `contrib/photon_db.yaml` | Noise model for measurements |
| Metrics DB | `contrib/metrics_db.yaml` | Which metrics per modality |
| Validation | `community/validate.py` | Verify emitted RunBundles |
| Leaderboard | `community/leaderboard.py` | Score and rank submissions |
| Challenge infra | `community/challenges/` | Weekly challenge format |
| Physics operators | `physics/*/` | Legacy operators (for equivalence testing) |
| Recon solvers | `recon/` | 43+ solver implementations |
| Analysis metrics | `analysis/metrics.py` | PSNR, SSIM, SAM computation |
| BudgetState | `world/budget.py` | Existing compute budget model |
| RunBundle format | `docs/contracts/runbundle_schema.md` | v0.3.0 schema (frozen) |

---

## The Frozen Contract (v1.0)

Once shipped, these interfaces are **additive-only** (new optional fields ok, breaking changes never):

| Interface | Frozen | Extensible |
|-----------|--------|------------|
| `LinearLikeOperator` protocol | `forward()`, `adjoint()`, `x_shape`, `y_shape` | New optional methods |
| Solver signature | `(y, physics, cfg) -> (x_hat, info)` | New optional keys in cfg/info |
| Calibrator signature | `(y, H_nom, budget) -> (H_hat, info)` | New optional keys in info |
| 4-scenario definitions | I/II/III/IV semantics per `docs/targeting_system.md` | New scenarios additive |
| Scoring formulas | rho, oracle_gap, RoIC, anti-Goodhart weights per S5 | New metrics additive |
| Track weights | 0.35/0.20/0.25/0.20 per S5.3 | New tracks additive |
| RunBundle v0.3.0 | Required manifest fields per `runbundle_schema.md` | New optional fields |
| `PrimitiveOp` protocol | `forward()`, `adjoint()`, `serialize()`, `_physics_tier` | New optional methods |
| Safety brake thresholds | 5 conditions per S6.4 | New brakes additive |

---

## File Summary

### Phase 1-4: Core Harness + CLI + Contribution + Tests

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `targeting/__init__.py` | 1 | Package init, version 1.0.0 |
| 2 | `targeting/harness.py` | 1 | **The one harness** -- modality-agnostic engine |
| 3 | `targeting/scenarios.py` | 1 | 4-scenario protocol (I/II/III/IV) |
| 4 | `targeting/scoring.py` | 1 | rho, oracle_gap, RoIC, OFS, MSI, anti-Goodhart, safety brakes |
| 5 | `targeting/budget.py` | 1 | Compute budget enforcement (2x = DQ) |
| 6 | `targeting/mismatch_sampler.py` | 1 | Sample mismatch from mismatch_db.yaml |
| 7 | `targeting/runbundle_emitter.py` | 2 | RunBundle v0.3.0 output + SHA-256 |
| 8 | `targeting/cli.py` | 2 | `pwm evaluate` CLI subcommand |
| 9 | `contrib/templates/contrib_solver_template.py` | 3 | Solver contribution template |
| 10 | `contrib/templates/contrib_calibrator_template.py` | 3 | Calibrator contribution template |
| 11 | `contrib/templates/contrib_graph_template_example.yaml` | 3 | Graph template contribution example |
| 12 | `community/CONTRIBUTING_CHALLENGE.md` (update) | 3 | 4-level contribution guide |
| 13 | `tests/test_harness.py` | 4 | End-to-end harness tests |
| 14 | `tests/test_contribution_path.py` | 4 | Contribution path validation |

### Phase 5: Community Adoption Infrastructure

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 15 | `docs/contributors/profiles.md` | 5 | Persona-based onboarding (ML / Imaging / Physics / Industry) |
| 16 | `targeting/contrib_check.py` | 5 | `pwm contrib check` one-command validation |
| 17 | `targeting/plugin_loader.py` | 5 | `pwm install` external plugin loading |
| 18 | `targeting/scaffold.py` (update) | 5 | Add calibrator scaffold support |
| 19 | `examples/level1_solver/` | 5 | Working FISTA solver reference implementation |
| 20 | `examples/level2_calibrator/` | 5 | Working grid-search calibrator reference implementation |
| 21 | `examples/level3_modality/` | 5 | Working modality graph template reference implementation |
| 22 | `examples/level4_primitive/` | 5 | Working Tier-1 primitive reference implementation |
| 23 | `contrib/templates/tier2_wrapper.py` | 5 | Tier-2 physics kernel template |
| 24 | `contrib/templates/tier3_wrapper.py` | 5 | Tier-3 learned surrogate template |
| 25 | `docs/contributors/CREDITS.md` | 5 | Authorship and recognition policy |
| 26 | `community/stewards.yaml` | 5 | External steward board |
| 27 | `.github/workflows/pr_classify.yml` | 5 | Three-speed PR auto-labeling |
| 28 | `tests/test_solver_isolation.py` | 5 | Interface isolation tests (anti-cheating) |

### Phase 6: Governance + Trust + Legal Infrastructure

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 29 | `docs/GOVERNANCE.md` | 6 | Formal merge authority with hard deadlines and no-veto rules |
| 30 | `docs/RAIL_CHARTER.md` | 6 | 5 trust commitments (evaluate-not-promote, reproducible, frozen baselines, rotation, community ownership) |
| 31 | `docs/IP_POLICY.md` | 6 | Licensing requirements per component type, CLA, patent policy |
| 32 | `docs/modality_pack_spec.md` | 6 | Formal modality pack specification (self-contained, validated, licensed) |
| 33 | `targeting/submit.py` | 6 | `pwm submit` RunBundle submission without PR |
| 34 | `targeting/plugin_loader.py` (update) | 6 | Plugin tier separation (local / community / official) |

---

## What We Build vs. What They Build

```
         US (PWM team)                    THEM (community)
    ┌────────────────────┐          ┌──────────────────────────┐
    │   RAIL (frozen)    │          │    TRAINS (compete)      │
    │                    │          │                          │
    │ OperatorGraph IR   │          │ Level 1: New solvers     │
    │ Graph compiler     │          │ Level 2: New calibrators │
    │ Primitive registry │◄────────►│ Level 3: New modalities  │
    │ Targeting harness  │          │ Level 4: New primitives  │
    │ Scoring engine     │          │   (RFC → we review)      │
    │ RunBundle schema   │          │                          │
    │ Anti-Goodhart      │          │ All via:                 │
    │ Safety brakes      │          │   physics.forward()      │
    │ Triad Law          │          │   physics.adjoint()      │
    │                    │          │   get_theta/set_theta    │
    │ The track gauge:   │          │   solver_registry.yaml   │
    │ LinearLikeOperator │          │   mismatch_db.yaml       │
    └────────────────────┘          └──────────────────────────┘
```

---

## Why This Draws the Computational Imaging Community

### The Academic Value Proposition

| Who | What they contribute | What they get | Publishable outcome |
|-----|---------------------|---------------|-------------------|
| **PhD student** (ML/optimization) | A new solver (Level 1) | Instant benchmarking on 64 modalities; citable leaderboard | "rho=0.85 across 20 modalities on LIP-Arena" |
| **PhD student** (optics/imaging) | A new graph template (Level 3) | 43+ solvers automatically benchmarked on their modality | "We formalize 4D-STEM as an OperatorGraph" |
| **Professor / lab** | A calibrator or solver | Transparent, adversarial comparison | "Our blind calibrator reduces oracle gap from 12 dB to 2 dB" |
| **Physics researcher** | A Tier 2 primitive (Level 4) | Improves every modality that uses it | "Our full-wave primitive improves fidelity by 3 dB across 5 modalities" |
| **Industry user** | Real-world measurement data | Validated calibration + audit trail | Calibration-as-a-service integration |

### The Lock-In Logic (per `docs/pwm_lockin_strategy.md`)

1. **Phase 1 (months 1-6)**: `pwm evaluate` becomes usable. First 3-5 external labs validate methods on our protocol. The 4-scenario protocol, rho, oracle_gap become the field's vocabulary.
2. **Phase 2 (months 7-12)**: CISP (Critical Assessment of Spectral/Spatial Imaging Prediction) launches. 20+ submissions. Leaderboard cited at conferences.
3. **Phase 3 (months 13-18)**: PWM's `LinearLikeOperator` becomes the API surface all imaging methods must implement.

**The QWERTY Moment**: The first evaluation protocol adopted by 5+ labs becomes permanent. This targeting system is how we get there.

### The Abundance Flywheel (per `docs/pwm_lockin_strategy.md` Part 3)

```
Commitment (labs escrow GPU hours)
    → Focus (R&D targets CISP leaderboard)
        → Collapse (calibration automated, rho >= 0.80)
            → Surplus (imaging cost drops to GPU-minutes)
                → Reinvestment (surplus funds next modality)
                    → Back to Commitment, spanning more modalities
```

---

## Ordering

Phase 1 (harness core) → Phase 2 (CLI + RunBundle) → Phase 3 (contribution templates) → Phase 4 (tests) → Phase 5 (community adoption) → Phase 6 (governance + trust)

- **Phase 1-2**: Build the rail. ~8 files of engine code. This is Phase A of LIP-Arena.
- **Phase 3**: Open the on-ramp. Templates + docs. Enables Level 1-4 contributions.
- **Phase 4**: Prove it works. End-to-end tests against known benchmarks.
- **Phase 5**: Scale the community. Reference implementations, plugin loading, `pwm contrib check`, three-speed governance, steward board, credits policy. This is the "PyTorch moment" -- making PWM feel like a platform people want to build on, not a repo they have to understand.
- **Phase 6**: Earn trust. Formal governance with hard deadlines, Rail Charter, IP policy, modality pack spec, `pwm submit` without PRs, plugin tier separation. This is the "CASP moment" -- making PWM a credible, neutral, field-owned standard that no single lab controls.

After Phase 4, the targeting system is ready for internal use. After Phase 5, it's ready for community-scale adoption. After Phase 6, it's ready to be **the field's standard**. The 18-month lock-in clock starts at Phase 4 completion.
