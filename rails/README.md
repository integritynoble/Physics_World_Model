# The Rails: PWM as the SolveEverything Trail

PWM is the first repository that implements all 10 gears of the [SolveEverything.org](https://solveeverything.org/) abundance engine as a concrete, runnable reference. This directory maps each gear to PWM's implementation -- what's built, what's partial, and what's planned.

---

## Thesis

The SolveEverything framework describes 10 interlocking gears that, together, form an abundance engine for any domain. PWM instantiates this engine for **computational imaging**: 64 modalities, 89 graph templates, 43+ reconstruction solvers, a built-in adversarial evaluation harness (LIP-Arena), and a complete audit trail (RunBundle + DR-IS). The rails/ directory is the guided tour.

---

## The 10 Gears: PWM Status

| # | Gear | Status | PWM Implementation | Key Doc |
|---|------|--------|--------------------|---------|
| 1 | [Targeting System](gear01_targeting_system.md) | **BUILT** | LIP-Arena, 4-scenario protocol, `pwm evaluate` | `docs/targeting_system.md` |
| 2 | [Outcome Contracts](gear02_outcome_contracts.md) | **PARTIAL** | Recovery ratio, oracle gap, RoIC metrics defined | `docs/purpose.md` Layer 1 |
| 3 | [Compute Escrow](gear03_compute_escrow.md) | **PARTIAL** | BudgetState, calibration budgets, 2x enforcement | `packages/pwm_core/pwm_core/world/budget.py` |
| 4 | [Action Networks](gear04_action_networks.md) | **PLANNED** | Software actuation (corrected operators), hardware API roadmap | `docs/purpose.md` Layer 6 |
| 5 | [Data Trusts](gear05_data_trusts.md) | **FOUNDATION** | Dataset registry, MIT license, synthetic-first policy | `packages/pwm_core/contrib/` |
| 6 | [Decision Logs](gear06_decision_logs.md) | **BUILT** | DR-IS schema, RunBundle v0.3.0, SHA-256 hashing | `docs/contracts/runbundle_schema.md` |
| 7 | [Two-Source Rule](gear07_two_source_rule.md) | **PARTIAL** | Multi-solver portfolio, safety brakes | `packages/pwm_core/contrib/solver_registry.yaml` |
| 8 | [Compute + Energy](gear08_compute_energy.md) | **OUT OF SCOPE** | RoIC metric makes compute measurable | `docs/purpose.md` Layer 9 |
| 9 | [Fairness Targets](gear09_fairness_targets.md) | **PARTIAL** | Tail-risk scoring, prospective dominance, anti-Goodhart | `docs/targeting_system.md` S5 |
| 10 | [Literacy](gear10_literacy.md) | **PARTIAL** | 26 working-process docs, quickstart guides | `docs/quickstart/` |

---

## Supporting References

| Document | What It Covers |
|----------|---------------|
| [Maturity Levels (L0-L5)](maturity_levels.md) | PWM's progression from L0 (muddle) to L5 (commoditized) |
| [Industrial Stack (9 Layers)](industrial_stack.md) | The 9-layer Industrial Intelligence Stack and how gears map to layers |
| `docs/purpose.md` | Full ISA purpose statement, stack definition, maturity levels, roadmap |
| `docs/targeting_system.md` | LIP-Arena specification (Commit-Measure-Score, Red Team, anti-Goodhart) |

---

## Current Position: L1 -> L2

PWM is transitioning from **L1 (Measurable)** to **L2 (Repeatable)**:

- **L1 evidence**: Clear metrics (recovery ratio, oracle gap, RoIC), comparable results via 4-scenario protocol, 3 modalities validated on Track B (Correct).
- **L2 gaps**: Calibration SOPs not yet documented for all modalities; procedures still require expert parameter tuning for new modalities.

See [maturity_levels.md](maturity_levels.md) for the full L0-L5 framework.

---

## Document Convention

Each gear document follows this structure:

1. **Title** -- Gear number + name + SolveEverything one-liner
2. **Status** -- BUILT / PARTIAL / PLANNED / OUT OF SCOPE
3. **The Principle** -- What this gear does in the abstract
4. **PWM Implementation** -- How PWM implements it concretely
5. **Key Files** -- File paths with brief descriptions
6. **What's Built** -- Current implementation evidence
7. **What's Next** -- Roadmap items
8. **Connections** -- How this gear meshes with others
