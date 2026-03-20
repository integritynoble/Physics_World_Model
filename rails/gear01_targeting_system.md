# Gear 1: Targeting System -- The Foundation

> A public, honest, continuous target that makes problems legible.

**Status: BUILT**

---

## The Principle

A targeting system is a public, adversarial evaluation protocol that makes truth cheap to verify and progress cheap to measure. It must test on future events that did not exist at training time, so nobody can memorize the answers. It must be funded and structured to try to break the system.

---

## PWM Implementation

PWM's targeting system is **LIP-Arena** (Live Imaging Physics Arena) -- a built-in evaluation harness that ships with PWM itself. It is not a separate benchmark or a static dataset. Users run `pwm evaluate` locally to score any method against the same protocol used for official benchmarks.

### Core Protocol: Commit-Measure-Score

1. **Commit**: Teams submit containerized pipelines + declared compute budgets before the measurement deadline.
2. **Measure**: New measurement sets generated *after* the commit deadline (sealed-simulator + live-lab).
3. **Execute**: All submissions run in a sealed environment with no ground truth access.
4. **Score**: Fully automated scoring; all RunBundles and methodologies published.

### 4-Scenario Evaluation Protocol

| Scenario | Measurement | Reconstruction Operator | Purpose |
|----------|-------------|------------------------|---------|
| I (Ideal) | True H | True H | Oracle upper bound |
| II (Assumed) | True H | Nominal H_nom | Mismatch impact baseline |
| III (Corrected) | True H | Calibrated H_hat | Calibration benefit |
| IV (Oracle Mask) | True H | Partial oracle | Partial upper bound |

### 4 Evaluation Tracks

| Track | Goal | Key Metric |
|-------|------|------------|
| Track 1: Correct | Infer and correct operator mismatch | Recovery ratio rho |
| Track 2: Diagnose | Attribute failure to Triad gate | Gate attribution accuracy |
| Track 3: No-GT | Correct without ground truth | Self-consistency + invariants |
| Track 4: Design | Specify robust imaging systems | Constraint satisfaction + robustness |

### Red Team Module

Dedicated adversarial layer injecting novel mismatch types, compound failures, out-of-family physics, gate-flip scenarios, and compute traps every round.

### Anti-Goodhart Scoring

Prospective score dominates (70% weight). Gaming penalized: wrong diagnosis, overconfident uncertainty, or missing artifacts result in rank loss.

---

## Key Files

| File | Description |
|------|-------------|
| `docs/targeting_system.md` | Full LIP-Arena specification (440 lines) |
| `docs/purpose.md` Layer 4 | Targeting system in Industrial Intelligence Stack context |
| `community/leaderboard.py` | Leaderboard computation and ranking |
| `community/validate.py` | Submission validation and RunBundle integrity checks |
| `packages/pwm_core/contrib/solver_registry.yaml` | 43+ registered solvers competing on the harness |

---

## What's Built

- **LIP-Arena specification**: Complete Commit-Measure-Score protocol with 4 phases
- **4-scenario protocol**: Validated on CASSI (10 KAIST scenes), SPC, CACTI
- **`pwm evaluate` CLI**: Score any method against any modality on any track
- **Anti-Goodhart scoring**: Prospective dominance (0.7 weight), gaming penalties, tail-risk weighting
- **Red Team injection categories**: 7 categories with escalation schedule
- **Safety brakes**: 5 pre-committed thresholds (rho < 0.30, uncertainty miscalibration, compute excess, etc.)
- **Sealed-simulator prospective sets**: Operational for 3 modalities

---

## What's Next

- **Live-lab prospective sets** (Phase B): Partner labs capturing real hardware measurements post-deadline
- **Curated counterfactual packs**: First 3 packs (CASSI, SPC, CACTI) to be published
- **Independent Red Team**: Dedicated budget and mandate separate from development team
- **External submissions**: Open the harness to third-party method submissions (Phase B, 6-12 months)
- **Quarterly rounds**: Rolling schedule with escalating difficulty

---

## Connections

- **Gear 2 (Outcome Contracts)**: The harness *verifies* the outcomes that contracts pay for
- **Gear 3 (Compute Escrow)**: The harness *enforces* compute budgets (2x disqualification)
- **Gear 6 (Decision Logs)**: Every submission produces a RunBundle with DR-IS records
- **Gear 7 (Two-Source Rule)**: Multi-solver portfolio tested on the same harness
- **Gear 9 (Fairness Targets)**: Tail-risk weighting and cross-modality transfer built into scoring
