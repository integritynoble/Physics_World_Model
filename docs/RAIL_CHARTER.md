# Rail Charter

> The 5 commitments that make PWM a credible field standard.

**Version**: 1.0.0
**Status**: ACTIVE
**Rationale**: Trust is the foundation of any community benchmark. CASP, ImageNet, and MLPerf succeeded because the community believed the evaluation was fair, reproducible, and not controlled by a single group. This charter codifies what PWM promises to its community.

---

## The 5 Commitments

### 1. PWM exists to evaluate, not promote.

PWM does not develop reconstruction methods. It evaluates them.

- No PWM-authored solver receives privileged treatment on the harness.
- PWM team solvers compete on the same harness, under the same rules, with the same deadlines as external solvers.
- The harness does not know who authored a solver. Scoring is identity-blind.
- PWM will never reject a solver because it outperforms a PWM-authored solver.

**Test**: Can an external lab beat every PWM solver and see it reflected on the leaderboard? If yes, this commitment holds.

### 2. All scoring is reproducible.

Every score on the leaderboard can be independently verified.

- Every score is derived from a RunBundle (v0.3.0) with SHA-256 integrity verification.
- Every scoring formula is published in `docs/targeting_system.md` and frozen per `docs/RAIL_CONSTITUTION.md`.
- Any researcher can download a RunBundle and recompute the score using `pwm score runbundle.zip`.
- Scoring code is open-source (Apache-2.0). No hidden weights, no private adjustments.

**Test**: Can a researcher who has never used PWM reproduce any leaderboard score from its RunBundle? If yes, this commitment holds.

### 3. All baselines are frozen.

Reference baselines are the field's fixed reference points.

- Baseline solvers (GAP-TV, FISTA-TV, MST-L, PnP-ADMM) use frozen parameters, frozen seeds, frozen configurations.
- Baseline scores never change after publication. If a baseline is found to have a bug, the corrected score is published alongside the original, not replacing it.
- New baselines can be added (additive only). Existing baselines are never removed.
- Baselines are re-evaluated on new modalities using the same frozen parameters.

**Test**: Are baseline scores from 2026 still valid reference points in 2030? If yes, this commitment holds.

### 4. Governance rotates.

No individual or lab permanently controls the rail.

- External stewards serve 2-year terms with a 2-term limit (per `docs/GOVERNANCE.md`).
- Stewards must come from at least 2 different institutions and 2 different countries.
- Changes to frozen components require unanimous approval from core team + all external stewards (per `docs/RAIL_CONSTITUTION.md` Article 3).
- The PWM core team cannot unilaterally change scoring, protocol, or safety brakes.

**Test**: If the PWM founding team disappeared, could the community continue running LIP-Arena and CISP under the existing rules? If yes, this commitment holds.

### 5. The community owns the outcomes.

PWM provides infrastructure. The community provides science.

- RunBundles are licensed CC-BY-4.0. Anyone can cite, analyze, and build on published results.
- Leaderboard data is publicly accessible. No paywalls, no registration walls for viewing results.
- CISP proceedings are open-access.
- Contributed solvers remain the intellectual property of their authors (per `docs/IP_POLICY.md`).
- The harness, scoring, and infrastructure are open-source (Apache-2.0).

**Test**: Can a researcher at any institution, in any country, access every PWM result, reproduce every score, and compete on equal footing? If yes, this commitment holds.

---

## What This Charter Is Not

- **Not a technical specification.** Technical details live in `docs/targeting_system.md` and `docs/RAIL_CONSTITUTION.md`.
- **Not a governance procedure.** Procedures live in `docs/GOVERNANCE.md`.
- **Not a license.** Licensing lives in `docs/IP_POLICY.md`.
- **Not binding on solver authors.** Solver authors choose their own licenses and retain their IP.

This charter is a **public promise** from the PWM project to the computational imaging community.

---

## Amending This Charter

This charter can only be amended by:

1. RFC with detailed rationale
2. 90-day public comment period
3. Unanimous approval from PWM core team + all external stewards
4. Published record of the change and its rationale

Minor clarifications (typos, formatting) do not require this process.

---

## Version History

| Version | Date | Change | Authority |
|---------|------|--------|-----------|
| 1.0.0 | 2026-02-18 | Initial charter | PWM core team |

---

## References

- `docs/RAIL_CONSTITUTION.md` -- What is frozen, what can evolve
- `docs/GOVERNANCE.md` -- Who decides, how fast, with what authority
- `docs/IP_POLICY.md` -- Licensing and patent policy
- `docs/targeting_system.md` -- LIP-Arena frozen specification
