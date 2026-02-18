# Rail Constitution

> The rules that govern the railroad. Changed only by governance, never by convenience.

**Version**: 1.0.0
**Status**: ACTIVE
**Rationale**: Per `docs/pwm_lockin_strategy.md`, the first evaluation protocol adopted by 5+ labs becomes permanent. Benchmark drift destroys lock-in. This document codifies what is frozen, what can evolve, and who decides.

---

## Article 1: Frozen Components (require governance vote to modify)

These components are the load-bearing walls of the rail. Changing them breaks backward compatibility, invalidates existing RunBundles, and undermines trust.

### 1.1 OperatorGraph Compiler

- **File**: `graph/compiler.py`
- **Frozen**: The 5-step compilation pipeline (Validate → Bind → Plan Forward → Plan Adjoint → Export)
- **Frozen**: Input type `OperatorGraphSpec`, output type `GraphOperator`
- **Frozen**: DAG acyclicity requirement, primitive_id lookup in `PRIMITIVE_REGISTRY`

### 1.2 LinearLikeOperator Protocol

- **File**: `recon/protocols.py`
- **Frozen**: `forward(x) -> y`, `adjoint(y) -> x`, `x_shape`, `y_shape`, `all_linear`
- **Rationale**: Every solver and calibrator in the ecosystem depends on this contract

### 1.3 4-Scenario Protocol

- **Source**: `docs/targeting_system.md` S3
- **Frozen**: Scenario I (ideal), II (assumed), III (corrected), IV (oracle mask) -- their definitions and semantics
- **Frozen**: Recovery ratio formula: rho = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II)
- **Frozen**: Oracle gap formula: PSNR_I - PSNR_III
- **Frozen**: RoIC formula: dB recovered per GPU-hour

### 1.4 Scoring Formulas

- **Source**: `docs/targeting_system.md` S5
- **Frozen**: Anti-Goodhart prospective dominance: S_rank = 0.3 * S_retro + 0.7 * S_prospective
- **Frozen**: Track weights: 0.35 (Correct) + 0.20 (Diagnose) + 0.25 (No-GT) + 0.20 (Design)
- **Frozen**: Gaming penalties: -15% (wrong Triad), -10% (overconfident), -10% (identifiability), DQ (compute dishonesty)

### 1.5 Safety Brakes

- **Source**: `docs/targeting_system.md` S6.4
- **Frozen**: 5 pre-committed thresholds (rho < 0.30, uncertainty > 15% deviation, wrong gate, > 2x budget, > 3x re-projection error)
- **Frozen**: Consequences (block, flag, DQ, quarantine)

### 1.6 RunBundle v0.3.0 Schema

- **Source**: `docs/contracts/runbundle_schema.md`
- **Frozen**: Required manifest fields (version, spec_id, timestamp, provenance, metrics, artifacts, hashes)
- **Frozen**: SHA-256 as the hash algorithm
- **Frozen**: Artifact integrity verification protocol

### 1.7 Solver Function Signature

- **Frozen**: `run_<solver>(y, physics, cfg) -> (x_hat, info)`
- **Frozen**: `physics` must satisfy `LinearLikeOperator` protocol
- **Rationale**: Every contributed solver depends on this

### 1.8 Calibrator Function Signature

- **Frozen**: `calibrate_<method>(y, H_nom, budget) -> (H_hat, info)`
- **Frozen**: `H_nom` exposes `get_theta()`, `set_theta()`, `forward()`, `adjoint()`

---

## Article 2: Evolvable Components (community can extend, additive only)

These components grow the ecosystem without breaking existing code.

### 2.1 New Primitives (via RFC)

- **Process**: Open RFC issue → physics discussion → tier placement agreed → implement `PrimitiveOp` → adjoint correctness tests → merge into `PRIMITIVE_REGISTRY`
- **Rule**: New primitives are additive. Existing primitive IDs and their `forward()`/`adjoint()` behavior never change.
- **Governance**: PWM core team reviews and approves. RFC must include: physics justification, tier classification, adjoint proof, at least 2 modalities that benefit.

### 2.2 New Metrics (additive only)

- **Rule**: New metrics (e.g., SAM, CNR, diagnostic accuracy) can be added to `metrics_db.yaml` and to scoring
- **Rule**: Existing metric formulas (rho, oracle_gap, RoIC) never change
- **Rule**: New metrics start with weight 0.0 in scoring (informational) until promoted by governance vote

### 2.3 New Modalities

- **Process**: Add entries to all 6 YAML registries + graph template → pass `test_registry_integrity.py` → merge
- **Rule**: Adding a modality never modifies existing modality entries
- **Rule**: New modalities start in `experimental` status until validated on the harness

### 2.4 New Solvers and Calibrators

- **Process**: Implement function with correct signature → add to `solver_registry.yaml` → pass harness evaluation → merge
- **Rule**: Adding a solver never modifies existing solver entries
- **Rule**: A solver becomes default for a modality only if it beats the current default on the harness

### 2.5 New Evaluation Tracks

- **Rule**: New tracks (Track 5, 6, ...) can be added alongside existing ones
- **Rule**: Existing track definitions (1-4) and their weights never change
- **Rule**: New tracks start with weight 0.0 in composite score until promoted

### 2.6 New Safety Brakes

- **Rule**: New brake conditions can be added (tightening safety)
- **Rule**: Existing brake thresholds can only be tightened (made stricter), never loosened
- **Rule**: Removing a safety brake requires governance vote

### 2.7 New Red Team Categories

- **Rule**: New adversarial injection categories can be added
- **Rule**: Existing categories never removed
- **Rule**: Escalation schedule can be adjusted by Red Team stewards

---

## Article 3: Governance Process

### 3.1 Who Decides

| Decision | Authority | Quorum |
|----------|-----------|--------|
| Freeze a new component (Article 1) | PWM core team + 2 external stewards | Unanimous |
| Add evolvable component (Article 2) | PWM core team | Majority |
| Modify a frozen component | PWM core team + 3 external stewards | Unanimous + 90-day comment period |
| Emergency safety brake addition | PWM core team | Any single member (ratified within 7 days) |

### 3.2 Change Process for Frozen Components

1. **RFC**: Publish detailed rationale, impact analysis, and migration path
2. **Comment period**: 90 days for community feedback
3. **Impact assessment**: Document all RunBundles, solvers, and calibrators affected
4. **Version bump**: Frozen component changes require a major version bump (e.g., v1.0 → v2.0)
5. **Migration tooling**: Provide automated migration for all affected submissions
6. **Vote**: Unanimous approval from core team + all external stewards

### 3.3 Experimental Rail

New ideas that might eventually become frozen go through an experimental phase:

| Stage | Namespace | Used for scoring? | Duration |
|-------|-----------|-------------------|----------|
| **Experimental** | `targeting/experimental/` | No -- informational only | Minimum 2 rounds |
| **Candidate** | `targeting/candidate/` | Optional track, weight 0.0 | Minimum 2 rounds |
| **Stable** | `targeting/` | Yes -- full scoring weight | Permanent |

Promotion from experimental → candidate → stable requires governance vote at each stage.

---

## Article 4: Anti-Drift Protections

### 4.1 Regression Tests

Every frozen component has a corresponding regression test in `tests/test_rail_constitution.py`:
- Scoring formula outputs are checked against known reference values
- Safety brake thresholds are asserted as constants
- Protocol signatures are asserted via type checking
- RunBundle schema validation is tested against reference bundles

### 4.2 Hash Anchoring

The frozen spec documents are SHA-256 hashed and recorded:
- `docs/targeting_system.md` hash recorded at v1.0 release
- `docs/contracts/runbundle_schema.md` hash recorded at v1.0 release
- Any change to these files triggers CI failure until governance process completes

### 4.3 Solver Isolation Enforcement

Contributed solvers are tested for isolation:
- A solver must not import from `graph.compiler`, `graph.primitives`, or `targeting.*`
- A solver must not access `H_true` (ground truth operator) during Scenario III
- A solver must not read files outside its declared inputs
- Violations are automatically rejected by CI

---

## Article 5: Version History

| Version | Date | Change | Authority |
|---------|------|--------|-----------|
| 1.0.0 | 2026-02-18 | Initial constitution | PWM core team |

---

## References

- `docs/targeting_system.md` -- LIP-Arena frozen specification (440 lines)
- `docs/contracts/runbundle_schema.md` -- RunBundle v0.3.0 schema
- `docs/pwm_lockin_strategy.md` -- Lock-in strategy and foundry window
- `docs/purpose.md` -- ISA purpose statement and Industrial Intelligence Stack
- `rails/gear01_targeting_system.md` -- Targeting system gear overview
