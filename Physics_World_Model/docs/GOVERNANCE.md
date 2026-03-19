# Governance

> Hard rules with deadlines. No gatekeeping on trains. No shortcuts on rail.

**Version**: 1.0.0
**Status**: ACTIVE
**Rationale**: Per `docs/RAIL_CHARTER.md`, governance must be explicit, time-bound, and rotation-based. This document codifies merge authority, review deadlines, and escalation paths.

---

## 1. Three-Speed Merge Authority

### 1.1 Fast Lane (Solvers, Calibrators, Config Tweaks)

| Rule | Detail |
|------|--------|
| **Scope** | PRs that only touch `contrib/solvers/`, `contrib/calibrators/`, or solver/calibrator config files |
| **Validation** | `pwm contrib check` must pass in CI |
| **Merge** | Auto-merge within 48 hours of CI pass |
| **Human veto** | **Not allowed.** No maintainer can block a fast-lane PR that passes CI. |
| **Rationale** | Solvers are trains. They compete on the harness. Blocking a solver is blocking science. |

**Exception**: If a solver PR introduces a security vulnerability (e.g., arbitrary code execution, network calls), any maintainer can flag it for security review. Security flags require documented evidence.

### 1.2 Review Lane (Modalities, Metrics, Track Tweaks)

| Rule | Detail |
|------|--------|
| **Scope** | PRs that touch `contrib/modalities/`, `contrib/metrics_db.yaml`, modality packs, or non-scoring targeting config |
| **Reviewers** | 2 independent reviewers (1 PWM maintainer + 1 domain expert or steward) |
| **Deadline** | 7 calendar days from PR submission |
| **Escalation** | If no review in 7 days → auto-escalate to steward board. Steward must respond within 3 days. |
| **Silent blocking** | **Not allowed.** A reviewer must provide written rationale for rejection. |

### 1.3 Governance Lane (Rail Changes)

| Rule | Detail |
|------|--------|
| **Scope** | PRs that touch `graph/primitives.py`, `targeting/scoring.py`, `targeting/harness.py`, `targeting/scenarios.py`, frozen protocol definitions, or any file listed in Rail Constitution Article 1 |
| **Process** | Per Rail Constitution Article 3.2: RFC → 90-day comment period → impact assessment → migration tooling → unanimous vote |
| **Authority** | PWM core team + all external stewards |
| **Quorum** | Unanimous approval required |
| **Version bump** | Major version bump required (e.g., v1.0 → v2.0) |

---

## 2. Self-Scoring Prohibition

| Rule | Detail |
|------|--------|
| **No privileged treatment** | PWM team solvers compete on the same harness as external solvers. No internal shortcuts, no early access to sealed data, no scoring exceptions. |
| **Leaderboard equality** | PWM-authored solvers appear on the leaderboard with the same formatting as external solvers. No "official" badge on PWM solvers. |
| **Audit** | Any community member can request the RunBundle of any PWM-authored solver result. Refusal = result removed from leaderboard. |

---

## 3. Steward Board

### 3.1 Composition

| Field | Requirement |
|-------|-------------|
| **Size** | 3-5 external stewards |
| **Independence** | No steward may be employed by or funded by the PWM core team's institution |
| **Diversity** | At least 2 different institutions, at least 2 different countries |
| **Expertise** | Each steward must have published in computational imaging, inverse problems, or a related field |

### 3.2 Terms and Rotation

| Rule | Detail |
|------|--------|
| **Term length** | 2 years |
| **Term limit** | Maximum 2 consecutive terms (4 years), then mandatory 2-year gap |
| **Staggered rotation** | Terms staggered so no more than 2 stewards rotate in the same year |
| **Vacancy** | If a steward resigns, replacement appointed within 60 days by remaining stewards + core team majority vote |

### 3.3 Responsibilities

- Review all governance-lane PRs
- Vote on Rail Constitution changes (per Article 3)
- Validate CISP challenge design, sealed data integrity, and results
- Publish annual public report on rail health (scoring drift, safety brake triggers, community growth)
- Mediate disputes between contributors and maintainers

### 3.4 Compensation

- Co-authorship on PWM benchmark papers
- Named acknowledgment in all PWM publications
- Travel support for CISP events (when funded)
- No monetary compensation (to preserve independence)

---

## 4. Dispute Resolution

### 4.1 Contributor Disputes

| Stage | Action | Timeline |
|-------|--------|----------|
| **1. Direct** | Contributor contacts PR reviewer with specific objection | Immediate |
| **2. Escalation** | Contributor opens issue tagged `dispute` with evidence | Within 7 days of rejection |
| **3. Steward review** | Steward board reviews evidence and makes binding decision | Within 14 days |
| **4. Public record** | Decision and rationale published in `community/decisions/` | Permanent |

### 4.2 Scoring Disputes

| Stage | Action | Timeline |
|-------|--------|----------|
| **1. Reproduce** | Challenger provides RunBundle + expected vs actual score | Immediate |
| **2. Verify** | PWM team re-runs scoring on challenger's RunBundle | Within 48 hours |
| **3. Resolve** | If discrepancy confirmed → scoring bug fixed + all affected scores recalculated | Within 7 days |
| **4. Post-mortem** | Root cause analysis published | Within 14 days |

---

## 5. Emergency Procedures

| Situation | Authority | Action |
|-----------|-----------|--------|
| **Security vulnerability in harness** | Any core team member | Immediate patch, ratified by team within 24 hours |
| **Safety brake triggered on production submission** | Automatic (harness) | Block deployment, notify contributor + stewards |
| **Scoring formula bug discovered** | Core team | Freeze leaderboard, fix, recalculate all affected scores, publish post-mortem |
| **Steward misconduct** | Remaining stewards | Majority vote to remove, replacement within 60 days |

---

## 6. Version History

| Version | Date | Change | Authority |
|---------|------|--------|-----------|
| 1.0.0 | 2026-02-18 | Initial governance document | PWM core team |

---

## References

- `docs/RAIL_CONSTITUTION.md` -- Frozen vs evolvable components
- `docs/RAIL_CHARTER.md` -- 5 trust commitments
- `docs/IP_POLICY.md` -- Licensing and patent policy
- `docs/targeting_system.md` -- LIP-Arena frozen specification
- `community/stewards.yaml` -- Steward board roster
