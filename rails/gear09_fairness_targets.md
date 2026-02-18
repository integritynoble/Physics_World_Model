# Gear 9: Fairness Targets -- The Steering

> Fairness goals built directly into the targets we pay for.

**Status: PARTIAL**

---

## The Principle

Fairness targets ensure that the abundance engine doesn't optimize only for easy cases or well-resourced teams. If the targeting system only rewards average-case performance, methods will neglect hard scenarios, rare modalities, and under-resourced labs. Fairness must be baked into the scoring function, not bolted on as an afterthought.

---

## PWM Implementation

PWM builds fairness into the evaluation protocol through three mechanisms: **tail-risk scoring**, **prospective dominance**, and **anti-Goodhart protections**.

### Tail-Risk Scoring

The harness weights worst-case performance heavily:
- **Bottom-10% emphasis**: Performance on the hardest 10% of scenarios carries disproportionate weight in the composite score
- **Tail-risk clause in contracts**: Minimum performance on every scene, not just average (e.g., rho >= 0.50 on every scene, not just rho >= 0.80 on average)
- **Gate-flip scenarios**: The Red Team injects cases where the dominant Triad gate is different from historical priors, penalizing systems that memorize "CASSI = Gate 3"

### Prospective Dominance

The anti-memorization mechanism:

```
S_rank = 0.3 * S_retro + 0.7 * S_prospective
```

A system that scores 95% retrospective but 60% prospective ranks below one that scores 80% on both. This prevents overfitting to public datasets and rewards genuine generalization.

### Anti-Goodhart Scoring

Gaming penalties ensure methods are right *for the right reasons*:

| Check | Penalty |
|-------|---------|
| Wrong Triad attribution | -15% of track score |
| Overconfident uncertainty (< 75% coverage at 90% CI) | -10% of track score |
| Identifiability inconsistency | -10% of track score |
| Compute dishonesty (declared < 0.5x actual) | Disqualification |
| Missing TriadReport or operator estimate | Not scored |

### Cross-Modality Transfer Scoring

10% of Track 1 score comes from cross-modality transfer: performance on modalities *not declared* in the submission. This prevents hyper-specialization on a single modality and rewards general-purpose methods.

---

## Key Files

| File | Description |
|------|-------------|
| `docs/targeting_system.md` S5 | Anti-Goodhart scoring (prospective dominance, gaming penalties) |
| `docs/targeting_system.md` S4 | Red Team module (gate-flip scenarios, escalation schedule) |
| `docs/purpose.md` Layer 4 | Targeting system with tail-risk emphasis |

---

## What's Built

- **Prospective dominance**: 70% weight on post-deadline measurements in composite score
- **Gaming penalties**: 5 penalty conditions mechanically enforced
- **Tail-risk weighting**: Bottom-10% scenarios weighted heavily; catastrophic failures penalized
- **Cross-modality transfer**: 10% of Track 1 score from unseen modalities
- **Red Team escalation**: Difficulty increases predictably across rounds (mild -> catastrophic)
- **Gate-flip injection**: Scenarios where the dominant gate differs from historical priors

---

## What's Next

- **Explicit fairness targets across modality tiers**: Ensure coverage across all 10 modality tiers, not just Tier 1-3 (currently most validation focuses on CASSI/SPC/CACTI)
- **Geographic diversity**: Fairness in lab partnerships -- not just top-tier university labs, but community clinics, developing-world facilities, industrial sites
- **Compute accessibility**: Fairness targets for resource-constrained teams (e.g., dedicated "low-compute" track where methods must achieve rho >= 0.60 in under 1 GPU-hour)
- **Modality equity scoring**: Track whether new method contributions improve all tiers proportionally, not just the popular modalities
- **Accessibility audit**: Ensure documentation, tools, and harness are usable by teams without computational imaging expertise

---

## Connections

- **Gear 1 (Targeting System)**: Fairness targets are embedded in the harness scoring function
- **Gear 2 (Outcome Contracts)**: Outcome metrics include tail-risk and cross-modality, not just average PSNR
- **Gear 10 (Literacy)**: Fairness requires that everyone can read a target and understand the scoring -- literacy is a prerequisite for equitable participation
