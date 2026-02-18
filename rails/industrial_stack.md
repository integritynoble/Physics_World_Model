# The Industrial Intelligence Stack: 9 Layers

PWM is designed as a complete Industrial Intelligence Stack -- not just a model, but the full infrastructure required to industrialize imaging. Each layer must be built, and the targeting system (Layer 4) only functions when the layers beneath it are solid.

---

## The 9 Layers

| # | Layer | PWM Status | Description |
|---|-------|-----------|-------------|
| 1 | Purpose and Payoff | **Defined** | Recovery ratio >= 0.80, oracle gap <= 2 dB, RoIC tracked |
| 2 | Task Taxonomy | **Built** | OperatorGraph IR, 64 modalities, 89 templates, atomic task decomposition |
| 3 | Observability | **Partial** | RunBundle exists, DR-IS specified, drift monitor planned |
| 4 | Targeting System | **Built** | LIP-Arena ships with PWM; 4-scenario protocol; `pwm evaluate` |
| 5 | Model Layer | **Active** | Current best methods shipped: Alg 1+2 calibration, 5 reconstruction solvers, Triad diagnosis |
| 6 | Actuation | **Software only** | Corrected operator feeds reconstruction; hardware API planned |
| 7 | Verification | **Strong** | 2900+ tests, Red Team module, cross-validation on 10 KAIST scenes |
| 8 | Governance | **Foundation** | Outcome-based evaluation defined; compute escrow and allocation not yet implemented |
| 9 | Distribution | **Foundation** | Standardized OperatorGraph IR, multi-solver redundancy, rate-based operations |

---

## Layer Details

### Layer 1: Purpose and Payoff

Quantified, falsifiable targets -- not vague aspirations.

| Metric | Definition | Target |
|--------|-----------|--------|
| Recovery ratio rho | (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II) | >= 0.80 across 20+ modalities |
| Oracle gap | PSNR_I - PSNR_III | <= 2 dB under bounded compute |
| RoIC | dB recovered per GPU-hour | Tracked per modality, improving monotonically |

### Layer 2: Task Taxonomy

Every ISA task decomposes into OperatorGraph operations: **Compile** (parse spec, instantiate graph) -> **Diagnose** (evaluate Triad gates) -> **Correct** (estimate and apply minimal intervention) -> **Verify** (re-project, check invariants, issue RunBundle).

### Layer 3: Observability

- **RunBundle**: Permanent record of every run (inputs, operator state, Triad diagnosis, correction trajectory, outputs, uncertainty, compute)
- **DR-IS**: Cryptographically signed decision records
- **Drift Monitor**: Continuous operator fidelity tracking (planned)

### Layer 4: Targeting System (LIP-Arena)

The engine that makes truth cheap to verify. Ships with PWM, runs via `pwm evaluate`. Commit-Measure-Score protocol with quarterly prospective rounds, Red Team injection, and anti-Goodhart scoring.

### Layer 5: Model Layer (Current Best Methods)

The methods that currently win on the harness: OperatorGraph compiler, Triad diagnostics, Alg 1 (grid search), Alg 2 (gradient refinement), and solvers (GAP-TV, MST-L, HDNet, EfficientSCI, etc.). When someone beats them on the harness, they become the new default.

### Layer 6: Actuation

Software actuation: corrected operator fed into reconstruction pipeline. Hardware actuation: calibration commands to instrument controllers (future). All actions logged in DR-IS.

### Layer 7: Verification and Red Teaming

- Adversarial Red Team with escalating difficulty
- DR-IS audit trail for every decision
- 2900+ regression tests
- Cross-validation on held-out datasets

### Layer 8: Governance and Incentives

- Outcome-based evaluation (recovery ratio, not publication count)
- Compute escrow (GPU budgets tracked, RoIC as first-class metric)
- Open harness: third-party methods compete on the same protocol

### Layer 9: Distribution and Maintenance

- OperatorGraph IR as universal protocol
- Multi-solver redundancy (no single-point-of-failure)
- Continuous monitoring with re-calibration triggers
- Rate-based operations (modalities/week, dB/GPU-hour, scenarios/quarter)

---

## How the 10 Gears Map to the 9 Layers

| Gear | Primary Layer(s) | Relationship |
|------|-----------------|--------------|
| 1. Targeting System | Layer 4 | *Is* the targeting system |
| 2. Outcome Contracts | Layer 1, 8 | Defines payoff (L1) + enforces incentives (L8) |
| 3. Compute Escrow | Layer 8 | Governance mechanism for compute allocation |
| 4. Action Networks | Layer 6 | *Is* the actuation layer |
| 5. Data Trusts | Layer 2, 3 | Feeds task taxonomy (L2) + enables observability (L3) |
| 6. Decision Logs | Layer 3 | *Is* the observability layer |
| 7. Two-Source Rule | Layer 7 | *Is* the verification layer |
| 8. Compute + Energy | Layer 9 | Infrastructure for distribution |
| 9. Fairness Targets | Layer 1, 4 | Embedded in purpose (L1) and targeting (L4) |
| 10. Literacy | Layer 2 | Enables understanding of task taxonomy |

---

## Implementation Status Summary

| Layer | Gears Involved | Status |
|-------|---------------|--------|
| 1. Purpose | Gear 2, 9 | Metrics defined, fairness targets partial |
| 2. Tasks | Gear 5, 10 | 64 modalities, 89 templates, 26 working-process docs |
| 3. Observability | Gear 6 | RunBundle built, DR-IS specified, drift monitor planned |
| 4. Targeting | Gear 1 | LIP-Arena operational (sealed-simulator) |
| 5. Models | -- | 43+ solvers, 5 calibration algorithms |
| 6. Actuation | Gear 4 | Software actuation for 16 modalities |
| 7. Verification | Gear 7 | Multi-solver, 2900+ tests, safety brakes |
| 8. Governance | Gear 2, 3 | Outcome metrics defined, escrow not implemented |
| 9. Distribution | Gear 8 | OperatorGraph IR, multi-solver, RoIC tracking |

---

## Reference

Full Industrial Intelligence Stack definition: `docs/purpose.md` S"The Industrial Intelligence Stack for Imaging"
