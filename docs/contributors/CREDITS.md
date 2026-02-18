# Authorship and Recognition Policy

> Clear rules for credit. Every contribution is tracked. Every contributor is recognized.

**Version**: 1.0.0
**Status**: ACTIVE
**Rationale**: Open-source projects fail when contributors feel invisible. Academic contributors need authorship clarity before their advisors approve time spent. Industry contributors need their company's legal team to see a clear IP and credit framework. This document removes ambiguity.

---

## 1. Recognition by Contribution Level

| Contribution Level | Type | Recognition |
|--------------------|------|-------------|
| **Level 1** -- Solver | `run_<name>(y, physics, cfg)` | Listed on contributors page with solver name and affiliation |
| **Level 2** -- Calibrator | `calibrate_<name>(y, H_nom, budget)` | Listed on contributors page with calibrator name and affiliation |
| **Level 1/2** -- Top-3 on any modality | Solver or calibrator ranked top-3 on LIP-Arena leaderboard | Leaderboard badge (gold/silver/bronze) + highlighted on project homepage |
| **Level 3** -- Modality | Full modality pack (graph + mismatch + photon + metrics + meta YAML) | Co-author on PWM benchmark paper |
| **Level 4** -- Primitive | New `PrimitiveOp` in `PRIMITIVE_REGISTRY` | Co-author on benchmark paper + named in `docs/RAIL_CONSTITUTION.md` Article 2.1 acknowledgments |
| **CISP Top-3** | Top-3 in any CISP challenge round | Listed in CISP proceedings + invited talk at CISP workshop |
| **Steward** | External steward board member | Listed in `docs/GOVERNANCE.md` + `community/stewards.yaml` + co-author on benchmark papers + named acknowledgment in all PWM publications |

---

## 2. Detailed Recognition Matrix

### 2.1 Contributors Page

All contributors at any level are listed on the project's contributors page.

| Field | Source | Required |
|-------|--------|----------|
| Name | `contributor_id` in RunBundle or PR author | Yes |
| Affiliation | Self-declared in PR or RunBundle | Yes |
| Contribution type | Solver / Calibrator / Modality / Primitive | Yes |
| Contribution ID | Registry ID (e.g., `cassi_fista_tv_v1`) | Yes |
| Date | Merge date or RunBundle timestamp | Automatic |
| ORCID | Self-declared | Optional but encouraged |

### 2.2 Leaderboard Badges

| Badge | Criteria | Display |
|-------|----------|---------|
| Gold | Rank 1 on any modality's LIP-Arena leaderboard | Gold badge next to solver name on leaderboard |
| Silver | Rank 2 on any modality | Silver badge |
| Bronze | Rank 3 on any modality | Bronze badge |
| Multi-modal | Top-3 on 5+ modalities simultaneously | Special "universal" badge |

Badges are recalculated when new submissions are scored. A badge may be lost if a better submission displaces yours from the top-3.

### 2.3 Benchmark Paper Co-authorship

| Eligibility | Criteria | Authorship Position |
|-------------|----------|---------------------|
| Level 3 -- Modality contributor | Merged modality pack used in the benchmark | Middle author (alphabetical within modality contributors) |
| Level 4 -- Primitive contributor | Merged primitive used by 2+ modalities in the benchmark | Middle author (alphabetical within primitive contributors) |
| Steward | Active steward during the benchmark period | Senior author acknowledgment section |
| PWM core team | Core maintainers | First/last author positions |

**Authorship is opt-in.** Contributors may decline authorship. Contributors who decline are listed in the acknowledgments section instead.

**Authorship requires response.** When a benchmark paper is drafted, all eligible contributors are notified via email (from their `contributor_id`). Contributors have 30 days to confirm or decline. Non-response after 30 days = listed in acknowledgments (not as co-author).

### 2.4 RAIL_CONSTITUTION Recognition

Level 4 (Primitive) contributors are named in `docs/RAIL_CONSTITUTION.md` Article 2.1 in the following format:

```
Primitive: <primitive_id>
  Author: <name> (<affiliation>)
  RFC: #<issue_number>
  Merged: <date>
```

This entry is permanent. It survives even if the primitive is later deprecated (deprecated primitives are marked but never removed).

### 2.5 CISP Recognition

| Placement | Recognition |
|-----------|-------------|
| Top-3 overall | Listed in CISP proceedings (open-access publication) |
| Top-3 overall | Invited talk at CISP workshop (travel support when funded) |
| Top-3 per track | Listed in CISP proceedings for that track |
| All participants | Listed in CISP proceedings participant appendix |
| Challenge designer | Co-author on CISP proceedings if challenge is used |

### 2.6 Steward Recognition

| Recognition | Detail |
|-------------|--------|
| Governance listing | Named in `docs/GOVERNANCE.md` and `community/stewards.yaml` |
| Benchmark paper | Co-author on all benchmark papers published during term |
| Publications | Named acknowledgment in all PWM publications during term |
| CISP | Co-author on CISP proceedings during term |
| Travel | Travel support for CISP events (when funded) |

---

## 3. Automatic Credit Generation via RunBundle

### 3.1 How It Works

Every RunBundle (v0.3.0) contains a `contributor_id` field in the manifest's `provenance` section:

```json
{
  "version": "0.3.0",
  "provenance": {
    "contributor_id": "jane_doe_mit_2026",
    "solver_id": "cassi_fista_tv_v1",
    "affiliation": "MIT",
    "orcid": "0000-0001-2345-6789",
    "timestamp": "2026-03-15T14:30:00Z"
  }
}
```

### 3.2 Credit Flow

```
Contributor submits PR or RunBundle
    |
    v
RunBundle manifest includes contributor_id
    |
    v
Harness scores the submission
    |
    v
Leaderboard updates automatically
    |
    v
Contributors page regenerated from all RunBundle manifests
    |
    v
If top-3: badge assigned automatically
    |
    v
If modality/primitive: flagged for benchmark paper inclusion
```

### 3.3 contributor_id Requirements

| Field | Format | Example |
|-------|--------|---------|
| `contributor_id` | `<name>_<affiliation>_<year>` | `jane_doe_mit_2026` |
| `solver_id` | Per registry conventions: `<domain>_<name>_v<N>` | `cassi_fista_tv_v1` |
| `affiliation` | Free text | `MIT`, `Google DeepMind`, `ETH Zurich` |
| `orcid` | ORCID format (optional) | `0000-0001-2345-6789` |

### 3.4 Dispute Resolution

If a `contributor_id` is missing or incorrect:

1. Open an issue tagged `credit-dispute`.
2. Provide evidence (PR link, RunBundle hash, email from submission).
3. Core team resolves within 7 days.
4. Corrected credit is retroactive.

---

## 4. What Does NOT Earn Credit

| Activity | Why Not |
|----------|---------|
| Opening an issue | Appreciated but not a tracked contribution |
| Reviewing a PR (unless steward) | Appreciated but handled by maintainer acknowledgments |
| Forking the repo | No contribution to the ecosystem |
| Running PWM on your data (without submitting) | No public contribution |
| Submitting a solver that fails CI | Must pass `pwm contrib check` to be listed |

---

## 5. Credit Timeline

| Event | Credit Action |
|-------|---------------|
| PR merged (solver/calibrator) | Added to contributors page within 24 hours |
| PR merged (modality) | Added to contributors page + flagged for benchmark paper |
| PR merged (primitive) | Added to contributors page + added to RAIL_CONSTITUTION |
| RunBundle scored on leaderboard | Leaderboard badge assigned if top-3 |
| CISP round completed | Proceedings updated within 30 days |
| Benchmark paper drafted | All eligible contributors notified within 14 days |

---

## 6. Version History

| Version | Date | Change | Authority |
|---------|------|--------|-----------|
| 1.0.0 | 2026-02-18 | Initial authorship and recognition policy | PWM core team |

---

## References

- `docs/GOVERNANCE.md` -- Steward board composition and responsibilities
- `docs/RAIL_CONSTITUTION.md` -- Frozen components and primitive acknowledgments
- `docs/RAIL_CHARTER.md` -- Community ownership commitment
- `docs/IP_POLICY.md` -- Licensing (contributors retain IP)
- `docs/contracts/runbundle_schema.md` -- RunBundle v0.3.0 manifest schema
- `docs/contributors/profiles.md` -- Persona-based onboarding guide
- `community/stewards.yaml` -- Steward board roster
