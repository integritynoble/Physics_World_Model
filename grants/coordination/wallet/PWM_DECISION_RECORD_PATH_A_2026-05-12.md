# PWM Decision Record: Path A — Director Holds All 5 Founder Slots Through DAO Migration

**Date:** 2026-05-12
**Author:** Director
**Status:** ✅ **ACCEPTED**
**Type:** Architecture Decision Record (ADR)
**Supersedes:** None
**Superseded by:** None
**Related docs:**
- `PWM_FOUNDER_HANDOFF_2026-05-12.md` (concentration vs handoff analysis)
- `PWM_V2_AUTO_RETIREMENT_DESIGN_2026-05-12.md` (v2-lite auto-retirement)
- `PWM_DAO_BUILD_AND_AUTHORITY_TRANSFER_2026-05-12.md` (Path A vs Path B analysis)
- `PWM_FOUNDER_SUNSET_2026-05-12.md` (retirement timeline)

---

## Decision

**Director will hold all 5 founder slots through the completion of both v2-lite migration AND DAO migration. Trustee distribution begins only after these migrations are complete (D9 + 186 days target).**

This is **Path A** as analyzed in `PWM_DAO_BUILD_AND_AUTHORITY_TRANSFER_2026-05-12.md`.

---

## Context

PWM v1 governance is a 5-founder 3-of-5 multisig with 48-hour timelock. The current contract:
- Has `activateDAO()` (one-way switch retiring multisig) but no time-based enforcement
- Has no auto-retirement guarantee — cooperation-dependent
- Has no DAO contract deployed for post-retirement governance
- Will deploy to Base mainnet with Director initially holding all 5 founder slots

Two key post-launch upgrades were identified:
1. **v2-lite migration**: Add `publicActivateDAO()` callable by anyone after 36-month hard sunset (contract-enforced retirement guarantee)
2. **DAO migration**: Deploy DAO contract + transfer parameter authority to it (post-retirement governance)

Both upgrades require **3-of-5 + 48h approval** via the existing PWMGovernance proposal flow. The key strategic question: do these migrations happen while Director holds all 5 keys (unilateral) or after trustees are distributed (cooperative)?

Three paths were analyzed in `PWM_DAO_BUILD_AND_AUTHORITY_TRANSFER_2026-05-12.md`:

- **Path A**: Director holds all 5 through both migrations (~6 months); trustee onboarding begins after
- **Path B**: Distribute trustees early (D9 + 60-90); DAO migration requires cooperation later
- **Hybrid**: v2-lite unilateral first; trustees onboard D9 + 90; DAO migration cooperative later

Director has chosen **Path A**.

---

## Rationale

### Why Path A wins

| Factor | Path A | Path B | Hybrid |
|---|---|---|---|
| DAO migration guaranteed | ✅ Yes (unilateral) | ⚠️ Cooperation-dependent | ⚠️ Cooperation-dependent for DAO |
| v2-lite migration guaranteed | ✅ Yes (unilateral) | ✅ Yes (also early) | ✅ Yes (early) |
| Trustee obstruction risk for migrations | ❌ Zero | ✅ Yes | ✅ Yes for DAO |
| Trustees inherit complete system | ✅ Yes (both upgrades in place) | ❌ No | ⚠️ Partial |
| Trustee onboarding speed | Slower (D9 + 186) | Fast (D9 + 60-90) | Medium (D9 + 90) |
| Key management risk window | ~6 months | ~60-90 days | ~90 days |
| Public perception during bootstrap | "Central for 6 months" | "Decentralizing early" | "Mostly decentralizing" |

The decisive factor: **Path A is the only path that guarantees both critical infrastructure upgrades complete without any cooperation risk.**

### What Path A trades

Path A accepts the following costs:

| Trade-off | Magnitude | Director's assessment |
|---|---|---|
| Longer key-management window (6 mo vs 3 mo) | Manageable with cold-storage discipline | ✅ Acceptable |
| Public perception risk (centralized for 6 months) | Mitigated by clear narrative as "commitment device" | ✅ Acceptable |
| Regulatory profile during bootstrap | Director is clearly the "control person" for 6 months | ✅ Acceptable with legal counsel |
| Trustee onboarding delayed | Trustees recruited under "you'll inherit a complete system" framing | ✅ Acceptable; potentially advantageous |
| Bus-factor risk during 6-month all-keys window | Mitigated by Shamir-split emergency recovery | ✅ Acceptable with planning |
| DAO development must complete on time | Hard deadline; aggressive milestone tracking required | ✅ Acceptable; Director will manage |

### What Path A gains

| Gain | Why it matters |
|---|---|
| Contract-enforced retirement guarantee (v2-lite) | Even if all 5 trustees later refuse to retire, protocol auto-retires at D9 + 36 mo |
| DAO governance in place before trustees | Trustees can never block DAO migration because they inherit a system where it's already done |
| Reduced cooperative-decision dependency | Director needs no trustee cooperation for the 2 most critical upgrades |
| Trustees can be recruited "to operate" rather than "to migrate" | Cleaner trustee responsibilities; easier recruitment |
| Single Director-controlled phase rather than mixed phase | Easier to manage operationally |
| Clean public narrative: "constitutional commitment, then distribution" | Better story than "early decentralization with retirement-coordination risk" |

---

## Consequences

### Timeline (committed)

```
D9 (mainnet launch)              Director holds all 5 founder slots
                                  v1 PWMGovernance active

D9 + 0 to D9 + 90 days           v2-lite migration
                                  ├─ Develop PWMGovernance v2 with publicActivateDAO()
                                  ├─ HARD_SUNSET_TIMESTAMP = deploy + 36 months
                                  ├─ Audit (~2-3 weeks delta)
                                  ├─ Deploy to mainnet
                                  ├─ Director proposes 5 migration tx via v1
                                  ├─ Director signs 3× each (Director-holds-all)
                                  ├─ Wait 48h timelock
                                  └─ Execute → admin contracts now use v2

D9 + 0 to D9 + 180 days          DAO contract development (parallel)
                                  ├─ Engage Solidity team (~$80-150k from Reserve)
                                  ├─ Months 1-3: Implementation
                                  ├─ Months 4-5: Audit
                                  ├─ Month 6: Deployment prep
                                  └─ End of month 6: DAO live on mainnet

D9 + 180 to D9 + 186 days        DAO authority migration
                                  ├─ Director submits migration proposals
                                  ├─ 5 admin contracts → DAO governance
                                  │  (or 1 atomic proposal via DAOAdoptionHelper)
                                  ├─ Director signs all 3 approvals
                                  ├─ Wait 48h timelock
                                  └─ Execute → DAO is now parameter authority

D9 + 186 days+                   Trustee onboarding begins
                                  ├─ Slot 2 → Trustee #1 (Trezor #1)
                                  ├─ Slot 3 → Trustee #2
                                  ├─ Slot 4 → Trustee #3
                                  ├─ Slot 5 → Trustee #4
                                  └─ Reach 3-of-5 distributed by D9 + 12 mo

D9 + 360 to D9 + 540 days        activateDAO() called
                                  ├─ 7-criterion graduation gate green
                                  ├─ 3-of-5 founders propose
                                  ├─ 48h timelock
                                  └─ Multisig retires permanently

D9 + 36 months (D9 + 1080 days)  HARD sunset backstop
                                  └─ If activateDAO hasn't fired,
                                     anyone can call publicActivateDAO()
```

### Director's commitments under Path A

| Commitment | Detail |
|---|---|
| Hold all 5 HW devices for 6 months | Ledger + 4 Trezors, with cold-storage discipline |
| Complete v2-lite migration by D9 + 90 | Auto-retirement guarantee in place |
| Complete DAO development by D9 + 180 | Hard deadline; aggressive PM |
| Complete DAO authority migration by D9 + 186 | Full post-retirement governance in place |
| Begin trustee onboarding by D9 + 186 | First external trustee receives Slot 2 |
| Maintain operational security throughout | Cold storage, Shamir backup, key verification |
| Public transparency on progress | Quarterly status reports |
| Engage legal counsel | Risk management for control-person period |

### Risks accepted

| Risk | Severity | Mitigation strategy |
|---|---|---|
| **Key compromise during 6-month window** | High | Cold storage of 4 of 5 HW; only Slot 1 active for daily ops; quarterly key verification; phishing hygiene |
| **DAO development slips past 6 months** | Medium | Modular DAO (Phase 1 MVP); aggressive milestone tracking; audit slot booked at D9 + 90 |
| **Director incapacitated during 6 months** | Low but catastrophic | Shamir-split (3-of-5) seed phrase deposited with trusted parties; documented succession plan |
| **Public perception of centralization** | Medium | Pre-published CONSTITUTION.md commitment; transparency reports; framing as "commitment device, then distribution" |
| **Regulatory exposure as control person** | Medium | Legal counsel engaged; structured trustee agreements pre-signed for D9 + 186 activation; jurisdiction analysis |
| **DAO contract bug post-deployment** | Low | Independent audit; bug bounty; pre-activateDAO reversion path (multisig still active until D9 + 12 mo) |
| **Reserve grant funding shortfall** | Low | Reserve allocation budgeted ($200k of 4% Reserve pool); foundation grant as backup |

### Mitigations committed (Director will execute)

| Mitigation | Status | Deadline |
|---|---|---|
| Cold storage for 4 of 5 HW devices | To set up | Pre-mainnet (D9 - 7 days) |
| Shamir-split (3-of-5) emergency recovery seed | To document and deposit | D9 + 7 days |
| Public commitment in CONSTITUTION.md | To draft | Pre-mainnet (D9 - 3 days) |
| Engage DAO development firm | To engage | D9 + 0 (launch day) |
| Book DAO audit slot | To book | D9 + 0-30 |
| Engage legal counsel | To engage | D9 + 0-30 |
| Trustee agreement template (pre-signed) | To draft | D9 + 90-180 |
| Quarterly status reports | To publish | D9 + 90, +180, +270, +360, ... |

---

## Alternatives considered

### Alternative 1: Path B (early trustee distribution)

**Description**: Distribute trustees at D9 + 60-90; DAO migration requires real 3-of-5 cooperation later.

**Why rejected**: DAO migration becomes subject to trustee obstruction risk. A 3-trustee coalition could block migration indefinitely, leaving protocol permanently without DAO governance. This contradicts the entire purpose of building the DAO.

### Alternative 2: Hybrid (v2-lite unilateral, DAO migration cooperative)

**Description**: v2-lite migration during Director-holds-all (D9 + 0-90); trustee onboarding D9 + 90+; DAO migration cooperatively at D9 + 180+.

**Why rejected**: Still has cooperation risk for the more important migration (DAO). Saves only ~90 days on trustee onboarding, at the cost of cooperation dependency for the critical post-retirement governance infrastructure.

### Alternative 3: Defer DAO migration indefinitely

**Description**: Do v2-lite only; never migrate to DAO; accept frozen parameters post-activateDAO.

**Why rejected**: Removes PWM's ability to evolve parameters over decades. Even if content layer remains permissionless, locked parameters would gradually become misaligned with reality (PWM/USD prices change, threat models evolve, etc.). Frozen-parameter PWM would be less viable than DAO-governed PWM.

### Alternative 4: Build DAO with Director as initial governor

**Description**: Deploy DAO with Director as initial admin; Director progressively cedes control to DAO membership.

**Why rejected**: This is effectively the same as having a 6th centralized authority. PWM's DAO design uses voting-weight from contribution history; "Director admin override" would corrupt this. Better to have the DAO migration happen cleanly via PWMGovernance.

### Alternative 5: Skip the DAO entirely; multisig forever

**Description**: Never call activateDAO; multisig remains the governance authority indefinitely.

**Why rejected**: Contradicts PWM's stated mission of decentralized scientific protocol; creates permanent regulatory exposure; creates permanent founder-key risk. Already rejected in `PWM_FOUNDER_SUNSET_2026-05-12.md`.

---

## Validation criteria

This decision is "successful" if all of the following are true by D9 + 12 months:

| Criterion | Target | Status |
|---|---|---|
| v2-lite migration complete | By D9 + 90 | Pending |
| DAO contract deployed | By D9 + 180 | Pending |
| DAO authority migration complete | By D9 + 186 | Pending |
| First external trustee onboarded | By D9 + 210 | Pending |
| 3-of-5 trustees external | By D9 + 12 months | Pending |
| All 5 HW devices secure throughout | Continuous | Pending |
| No protocol incidents during Director-holds-all phase | Continuous | Pending |
| Public commitment honored (transparency reports) | Quarterly | Pending |

If any of these fail, Director will publish an update explaining the deviation and revised plan.

---

## Reversal conditions

This decision should be revisited if:

| Condition | Action |
|---|---|
| DAO development slips past D9 + 9 months | Reassess: extend Path A window or pivot to Hybrid |
| Critical security incident with Director's keys | Emergency rotation; accelerate trustee onboarding |
| Director's health concerns | Trigger Shamir recovery; accelerate succession |
| Major regulatory ruling against control-person model | Legal counsel; possible accelerated distribution |
| Path A is found to violate constitutional principles | Re-examine in light of new information |

None of these are expected. If they occur, this ADR will be marked superseded and a new decision record will be created.

---

## Acceptance signature

**Decision accepted by:** Director
**Date:** 2026-05-12
**Effective:** D9 (mainnet launch date)
**Review date:** D9 + 12 months (when activateDAO is called or hard sunset approaches)

---

## Appendix: the complete thirteen-doc founder governance + evolution set

| # | Doc | Question answered |
|---|---|---|
| 1 | `PWM_FOUNDER_ROTATION_2026-05-10.md` | WHY rotation exists |
| 2 | `PWM_FOUNDER_ROTATION_MECHANISM_2026-05-12.md` | HOW to rotate a single founder |
| 3 | `PWM_FOUNDER_GOVERNANCE_POLICY_2026-05-12.md` | POLICY: composition, identity |
| 4 | `PWM_FOUNDER_ROLE_AND_AUTHORITY_2026-05-12.md` | WHAT founders can/cannot do |
| 5 | `PWM_FOUNDER_SUNSET_2026-05-12.md` | WHEN to retire |
| 6 | `PWM_FOUNDER_HANDOFF_2026-05-12.md` | WHO holds keys during bootstrap |
| 7 | `PWM_FOUNDER_RETIREMENT_GUARANTEE_2026-05-12.md` | CAN retirement be ensured |
| 8 | `PWM_V2_AUTO_RETIREMENT_DESIGN_2026-05-12.md` | v2 auto-retirement spec |
| 9 | `PWM_EVOLUTION_MECHANISMS_2026-05-12.md` | HOW PWM evolves (3-layer overview) |
| 10 | `PWM_DISTRIBUTION_AND_GATES_EVOLUTION_2026-05-12.md` | Distribution + S1-S4 specifics |
| 11 | `PWM_POST_FOUNDER_MECHANISM_AND_CONSTITUTION_2026-05-12.md` | DAO primer + post-founder mechanism + constitution |
| 12 | `PWM_DAO_BUILD_AND_AUTHORITY_TRANSFER_2026-05-12.md` | WHO builds DAO + WHEN it needs founder approval |
| 13 | **`PWM_DECISION_RECORD_PATH_A_2026-05-12.md` (this doc)** | **LOCKED DECISION: Path A — Director holds all 5 through DAO migration** |

These thirteen docs comprehensively cover PWM's founder lifecycle from design to execution decision.
