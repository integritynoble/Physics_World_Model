# Gear 7: Two-Source Rule -- The Safety Brake

> Critical decisions confirmed by two independent systems.

**Status: PARTIAL**

---

## The Principle

No critical decision should depend on a single system's output. The two-source rule requires that important conclusions -- "this operator is correct," "this reconstruction is reliable," "Gate 3 is the dominant bottleneck" -- be confirmed by at least two independent methods before action is taken. When the two sources disagree, the system escalates to human review rather than guessing.

---

## PWM Implementation

PWM implements the two-source principle through its **multi-solver portfolio** and **safety brake** system. Every modality has multiple independent reconstruction solvers, and critical thresholds trigger automatic flags when results are inconsistent.

### Multi-Solver Portfolio

Each modality ships with 2-4 solvers spanning different algorithmic families:

| Family | Examples | Strength |
|--------|----------|----------|
| Classical (TV/iterative) | GAP-TV, FISTA-TV, ADMM | Robust to operator mismatch, interpretable |
| Plug-and-Play | PnP-ADMM, PnP-FFDNet, PnP-FISTA | Better denoising, moderate robustness |
| Deep unrolling | MST-L, HDNet, ELP-Unfolding | Highest quality under correct operator |
| End-to-end learned | EfficientSCI, FlatNet, CARE | Fast inference, task-specific |

**Key insight**: Classical solvers degrade gracefully under mismatch (GAP-TV drops ~2 dB); learned solvers degrade catastrophically (MST-L drops ~14.5 dB). Disagreement between solver families is a strong signal that operator mismatch is present.

### Solver Agreement as Diagnostic Signal

| Agreement Pattern | Diagnosis | Action |
|-------------------|-----------|--------|
| All solvers agree (within 2 dB) | Operator likely correct | Report results with high confidence |
| Classical high, learned low | Operator mismatch likely (Gate 3) | Trigger calibration; report both results |
| All solvers low | Sampling or noise issue (Gate 1/2) | Investigate measurement quality |
| Learned high, classical low | Solver-specific artifact possible | Flag for review; report both |

### Safety Brakes

Pre-committed thresholds that trigger automatic flags:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Recovery ratio regression | rho < 0.30 on any validated modality | Block deployment, root-cause analysis required |
| Uncertainty miscalibration | Coverage deviates > 15% from declared CI | Flag all outputs as "uncalibrated" |
| Out-of-family miss | System confidently diagnoses wrong gate | Mandatory retraining on expanded family |
| Compute budget exceeded | > 2x declared GPU-hours | Submission disqualified for that scenario |
| Consistency violation | Re-projection error > 3x median | Output quarantined pending review |

### Layer 7: Verification and Red Teaming

The broader verification layer includes:
- **Adversarial Red Team**: Dedicated effort to break capabilities before deployment
- **Regression suite**: 2900+ automated tests; any code change must pass the full suite
- **Cross-validation**: Results on one dataset not trusted until replicated on a different instrument or lab

---

## Key Files

| File | Description |
|------|-------------|
| `packages/pwm_core/contrib/solver_registry.yaml` | 43+ solvers with tier classification and supported modalities |
| `docs/purpose.md` Layer 7 | Verification and Red Teaming layer |
| `docs/purpose.md` Layer 9 | Multi-solver redundancy |
| `docs/targeting_system.md` S6.4 | Safety brakes specification |

---

## What's Built

- **43+ registered solvers**: Spanning 4 algorithmic families across 64 modalities
- **Multi-solver evaluation**: `pwm evaluate` scores multiple solvers on the same scenario
- **Safety brakes**: 5 pre-committed thresholds with automatic flagging
- **Regression suite**: 2900+ tests preventing regressions
- **Solver disagreement observed empirically**: GAP-TV vs MST-L disagreement under mismatch is a validated diagnostic signal (14.5 dB divergence under 1px shift)

---

## What's Next

- **Formal two-source protocol**: Codify when solver agreement is required (all Track 1 submissions? all production deployments?)
- **Divergence thresholds**: Per-modality thresholds for "acceptable disagreement" vs "escalate to human review"
- **Independent verification solvers**: Designate specific solvers as "verification-only" (never used for production, only for cross-checking)
- **Escalation protocol**: Define the human review process when solvers disagree beyond threshold

---

## Connections

- **Gear 1 (Targeting System)**: The harness enforces safety brakes and scores multi-solver submissions
- **Gear 6 (Decision Logs)**: Solver disagreement events are logged as DR-IS records with full evidence
- **Gear 9 (Fairness Targets)**: Two-source verification is especially important for tail-risk scenarios where a single solver might silently fail
