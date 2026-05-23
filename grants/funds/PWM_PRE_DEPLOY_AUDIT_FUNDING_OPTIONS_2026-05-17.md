# Pre-Deploy Audit Funding Options — 2026-05-17

**Date:** 2026-05-17
**Audience:** Director + future Claude sessions
**Purpose:** Capture the realistic funding options for pre-deploy smart-contract audit, given Director's actual budget constraints, and the decision framework for choosing among them.
**Scope:** Narrow — this doc is specifically about *audit funding before the Phase 5 mainnet deploy*. Post-mainnet operating funding is covered separately in `PWM_FUNDING_STRATEGY_AND_MAINNET_SEQUENCING_2026-05-02.md` and the `funds/` track docs.

---

## UPDATE 2026-05-18 evening — accelerated path adopted; key context shifts

This doc was written 2026-05-17 assuming the original Jul 24 mainnet timeline. On 2026-05-18 Director adopted the **accelerated 36-48 hour dispatch path** per [`../deploy/PWM_MULTI_SERVER_DISPATCH_2026-05-18.md`](../deploy/PWM_MULTI_SERVER_DISPATCH_2026-05-18.md). Three things have changed since the original write:

| What changed | Old assumption | New reality |
|---|---|---|
| **Mainnet target date** | 2026-07-24 (post-audit) | **2026-05-19 evening or 2026-05-20 morning** (~36-48 hr from now) |
| **Audit sequencing** | Pre-deploy paid audit ($25-50K, 5-8 weeks) → THEN mainnet | **Multi-agent review (✅ COMPLETE 2026-05-18; commit `37fd967c`) + soft-launch caps → mainnet → paid audit at ~D9+30 when grants land** |
| **$0 safety stack** | "do this regardless of funding outcome" | ✅ **ALREADY DONE** — Slither + multi-pass Claude review across 8 of 10 agents; 2 CRITICAL + 4 HIGH + 5 MEDIUM all FIXED; 188/188 tests GREEN |
| **Soft-launch cap values** | STAKING_TVL_CAP_USD = $1,000 | Actual on-chain: **maxTotalStakeWei = 100 PWM ≈ $130**, maxBenchmarkPoolWei = 100 PWM, mintingPaused=true, transfersPaused=true, submissionPermissionless=false |
| **What audit funding is for** | Pre-deploy audit to enable deploy | **Post-deploy audit (~D9+30 to D9+90) to enable cap-raise** from 100 PWM → $10K+ |
| **Application urgency** | High — audit blocks deploy | Moderate — audit blocks cap-raise (D9+30+), not deploy itself. Still apply soon to bound the soft-launch window. |

What's still valid:
- Tier 1 funding sources (Base Builder Grants, EF ESP) — same fit, same amounts, same application effort
- The "Why PWM's audit need is smaller than DeFi pricing implies" argument
- The MAINNET_FIRST_PLAN policy-narrowing rationale
- Director's eligibility constraints (Research Associate, not PI; pure 501(c)(3) path)
- The $25-50K total audit budget envelope

Sections below should be read through this accelerated-path lens. The Tier 1/Tier 2/Tier 3 funding source tables remain authoritative; the "Recommended action plan" week-by-week schedule is fully refreshed below.

---

## TL;DR

Director's safety-first / earliest-possible / personal-budget-constrained situation has a workable path:

1. **This week:** apply to two pre-mainnet-eligible grants in parallel — **Base Builder Grants** ($5-25K) and **Ethereum Foundation ESP** ($10-50K). Both fund smart-contract audits explicitly for nonprofit infrastructure. Combined application time ~20-25 hours. Realistic 4-8 week turnaround.
2. **In parallel:** run the **$0 safety stack** (Slither + Mythril + Claude multi-pass review) so when funding lands, audit firms see a Slither-clean codebase → faster, cheaper engagement.
3. **Optional fallback:** soft-launch with conservative parameter caps (Staking TVL $1K, minting paused, treasury paused) so the at-risk surface is bounded to ~$1K until a full audit completes post-deploy.
4. **Policy decision needed:** narrow the existing `MAINNET_FIRST_PLAN_2026-04-27` "no grants until mainnet + 30 days stable" rule to allow pre-deploy *audit-only* grant applications. Rationale + draft decision-log entry at the end of this doc.

**~~Realistic mainnet date if grants land normally: 2026-07 to 2026-08~~** — SUPERSEDED 2026-05-18: accelerated path now puts mainnet at **2026-05-19/20** (~36-48h from this doc's write date). Paid audit is now post-deploy (~D9+30 when grants land); the four points above remain otherwise correct, but applications are sized for post-deploy audit funding rather than pre-deploy gating.

---

## Director's actual constraints (2026-05-17)

| Constraint | Reality |
|---|---|
| Personal cash available | ~$30-50 + ~$200-500 borrowable |
| Time per week for grant writing | TBD (Director to estimate) |
| Institutional position | Research Associate at UTSW; **NOT PI-eligible** for NIH R-series; Co-I status requires PI mentor (Track K) |
| Spousal entity option | Ruled out (pure 501(c)(3) path locked) |
| Foundation status | Pre-formation; NumFOCUS Round 4 closes 2026-10-15 (after current 2026-07-24 deploy target) |
| Reserve token liquidity | Illiquid pre-launch; cannot pay vendors in PWM Reserve until secondary market exists |
| Existing operating constraint | `MAINNET_FIRST_PLAN_2026-04-27`: no grant applications until mainnet + 30 days stable |

These constraints are what determine which funding sources are realistically reachable in the 1-3 month pre-deploy window.

---

## Why PWM's audit need is smaller than typical DeFi audit market pricing implies

Standard smart-contract audit pricing ($20-50K) is calibrated for typical DeFi protocols holding $10M+ of user funds — AMMs, lending protocols, oracle-integrated derivatives. **PWM has none of those characteristics.** Specifically:

- **No user fund custody.** No deposit/withdraw contract. The 17.22M PWM token supply is Foundation-controlled (Reserve + emissions), not user-deposited.
- **No oracles, no flash loans.** No external price feeds; no composability primitives.
- **Conservative architecture.** PWMRegistry is write-only (no upgrade, no deletion). PWMGovernance has 3-of-5 multisig + 48 h timelock — the timelock IS a kill-switch.
- **Per-principle treasury isolation.** A bug in one principle's treasury doesn't drain the whole system.
- **Fixed staking floors (10/2/1 PWM)** — no oracle-based pricing, no slashing of user funds.

A typical $30K audit budget covers protocols that could lose $10M+ to a bug. PWM's realistic worst-case loss from a bug is the Reserve allocation cap on the affected contract — and with soft-launch caps (below), that's bounded at ~$1K USD-equivalent for the first 30 days.

This isn't an argument to skip audit — it's an argument that PWM is in the **$15-25K audit budget tier**, not the $30-60K rapid-audit tier. Smaller scope means realistic grant amounts are achievable.

---

## Funding sources — ranked by fit, timeline, and reachability

### Tier 1 — apply this week (best fit for audit funding; pre- or post-deploy applicable)

| Source | URL | Amount | Cycle | Audit-eligible? | Application effort |
|---|---|---|---|---|---|
| **Base Builder Grants** | https://base.org/grants | $5K-$25K | Rolling | ✅ Yes — they have funded audits for Base-native projects | ~8-10 hr (5-page proposal + budget) |
| **Ethereum Foundation ESP** | https://esp.ethereum.foundation | $10K-$50K typical for audits | Rolling | ✅ Yes — ESP has explicitly funded audits for Sismo, Bedrock contributors, others | ~12-15 hr (detailed proposal + milestones + team) |
| **Code4rena** | https://code4rena.com | Sponsor posts $30K-$100K bounty pool; pays only on valid findings | Rolling private + monthly public | ✅ Yes — competitive audit marketplace; ~30-100 independent auditors | ~2 days outreach + scope spec |
| **Spearbit / Cantina** | https://cantina.xyz | $25K-$80K sponsor-paid; private competition | Rolling | ✅ Yes — Spearbit-vetted senior auditors; competitive format | ~3-5 days outreach + scope |
| **Sherlock** | https://sherlock.xyz | $30K-$100K + insurance backstop | Rolling | ✅ Yes — audit + insurance; co-financing possible | ~3-5 days outreach |
| **Trail of Bits — discounted/free audits for high-impact OSS** | https://www.trailofbits.com | $0-$30K (full audit usually $50-150K; OSS discount programs exist for eligible protocols) | Application | ✅ Yes — case-by-case for academic/nonprofit | ~10-15 hr proposal |
| **OpenZeppelin** | https://www.openzeppelin.com/security-audits | Sometimes discount/credit programs for OSS | Direct outreach | ✅ Yes | ~5-8 hr outreach |
| **Coinbase Cloud free tier + Base RPC credits** | https://www.coinbase.com/cloud | $0 cash; ~$50-100/mo value | Self-serve | N/A — infrastructure not audit | ~30 min signup |
| **Coinbase Developer Platform credits** | https://www.coinbase.com/developer-platform | $0 cash + RPC/api credits | Self-serve | N/A | ~30 min |

**Best fit for PWM (in priority order):**
1. **Base Builder Grants** — fastest application, on-network alignment, Base team has explicitly funded audits
2. **Code4rena / Spearbit / Sherlock competitive marketplace** — pay-for-findings model often cheaper than fixed-fee; covers post-deploy audit cleanly; multiple parallel auditors catch issues solo audit misses
3. **EF ESP** — larger amount, more proposal effort, longer turnaround (4-8 weeks)
4. **Direct firm engagement (Trail of Bits / OpenZeppelin)** — full audit but expensive; case-by-case OSS discount; engage only after Base/EF/competitive funding is clearer

Competitive audit programs (Code4rena, Spearbit, Sherlock, Cantina) are particularly well-suited for PWM's accelerated post-deploy audit path: they accept code that's already deployed (with soft-launch caps protecting users), they pay only for valid findings (often cheaper than $30K fixed fee for the same coverage), and the audit can begin as early as D9+7.

### Tier 2 — apply within 30 days, 3-6 month turnaround

| Source | URL | Amount | Cycle | PWM fit |
|---|---|---|---|---|
| **Chan Zuckerberg Initiative — Essential Open Source Software for Science** | https://chanzuckerberg.com/eoss/ | $200K-$1M | Annual (RFA typically Spring) | Strong — PWM = scientific OSS infrastructure |
| **Sloan Foundation OSS** | https://sloan.org/programs/research/digital-information-technology | $200K-$1M | Semiannual | Strong — same framing as CZI |
| **Mozilla Open Source Support (MOSS)** | https://www.mozilla.org/en-US/moss/ | $10K-$250K | 2-3 month cycle | Possible if framed as research infrastructure tooling |

These won't help D9 timing but should be applied to for ongoing operating budget once Foundation status is clearer.

### Tier 3 — gated on Track K (UTSW PI mentor recruitment)

| Source | URL | Amount | Why blocked | If Track K lands |
|---|---|---|---|---|
| **NSF POSE** (Pathways to Open-Source Ecosystems) | https://new.nsf.gov/funding/opportunities/pose | $300K-$1.5M | Director not R-series PI-eligible | Excellent — Co-I role with new PI |
| **NIH R03 / R21 small grants** | https://grants.nih.gov | $50K-$275K | Same PI restriction | Possible — see `PWM_RESEARCH_ASSOCIATE_AND_MENTOR_CONSTRAINTS_2026-05-13.md` |

These remain Tier 3 until Track K progresses. They should be reactivated when a supportive PI mentor is confirmed.

### Tier 4 — defer to post-mainnet per the original policy

| Source | URL | Why deferred |
|---|---|---|
| **NumFOCUS Round 4** | https://numfocus.org | Round 4 closes 2026-10-15 (after target D9); fiscal sponsorship is enabling-not-grant — helps subsequent applications more than this one |
| **Optimism Retroactive Public Goods Funding** | https://app.optimism.io/retropgf | Retroactive only; no upfront money |
| **Gitcoin Grants** | https://gitcoin.co | Quadratic matching requires community-in-place; pre-mainnet has no community |

### Tier 5 — bootstrap-from-personal (already at the limit)

| Source | Amount | Notes |
|---|---|---|
| Director personal cash | ~$30-50 + ~$200-500 borrowable | Already at constraint; insufficient for any formal audit |
| Friends-and-family round | $0-$5K typical | Relationship risk; possible for 501(c)(3) framing (donation, not equity); not yet pursued |
| Personal credit card cash advance | $5-10K possible | High interest; personal liability; not recommended for nonprofit audit |
| Crowdfunding (Kickstarter / Open Collective) | Unpredictable | Long timeline; needs polished outreach; better post-launch |

These remain available as last resort but should not be the primary path.

---

## The $0 safety stack — ✅ COMPLETED 2026-05-18 (multi-agent review)

Original section recommended running these tools whether or not grants land. As of **2026-05-18 evening this stack is DONE** — see `../deploy/findings/SECURITY_REVIEW_2026-05-18.md` (final aggregator from agent A10) and `../deploy/findings/STATUS_2026-05-18_final.md` for the full record. Coverage map:

| Tool | Status | Findings |
|---|---|---|
| **Slither** | ✅ DONE — A4 triage `cf752fa7` | 58 raw findings → 0 deploy-blocking after triage |
| **Mythril** | Deferred per dispatch playbook (overnight scan) | Not blocking; scheduled for post-deploy |
| **Echidna** fuzz tests | A7 deferred per dispatch playbook §2 (~12-24 hr unattended); A5 economic-attacks reviewed in `cec0d988` | Property fuzz not blocking; scheduled for pre-audit prep |
| **Multi-pass Claude review** (A1/A2/A3 + re-passes) | ✅ DONE — `203df847`, `fe3ba529`, `41efadb8`, `114df918`, `6d5ec7a6` | 2 CRITICAL + 4 HIGH + 5 MEDIUM all FIXED |
| **Cross-contract review (A6)** | ✅ DONE — `9dd3e80a` | 1 HIGH cross-validates A8; 4 MED; 2 LOW |
| **Deploy-script audit (A8)** | ✅ DONE — `92e946e5` | 1 HIGH + 3 MED + 3 LOW + 2 INFO |
| **Spec-vs-code consistency (A9)** | ✅ DONE — `e699ec10` | 11 MATCH, 8 NEW state vars need doc amendment |
| **Aggregator (A10)** | ✅ DONE — `6cb56537` then `480e7522` | Final SECURITY_REVIEW with all findings and resolutions |
| **Test suite** | ✅ 188/188 GREEN | Was 165 pre-patch; +23 new tests across 5 patch commits |

**Deliverable status:** `pwm-team/deploy/findings/SECURITY_REVIEW_2026-05-18.md` committed. All deploy-blocking findings closed. Director already approved the multi-agent review as audit-equivalent for the 30-day soft-launch window.

**For grant applications**: the `SECURITY_REVIEW_2026-05-18.md` aggregator is the artifact to reference. It demonstrates rigorous pre-deploy security work and identifies remaining gaps (symbolic execution, formal verification, MEV analysis) that a paid audit would close. Application text: "PWM completed a multi-agent Claude-based security review prior to mainnet deploy on 2026-05-18; 188/188 tests passing; 2 CRITICAL + 4 HIGH + 5 MEDIUM findings all addressed. Grant funding requested for a paid follow-up audit to close residual symbolic-execution / formal-verification gaps before raising soft-launch caps from $130 to $10K+."

---

## Soft-launch parameter caps — IMPLEMENTED (values lower than original projection)

The deploy script (`deploy/erc20.js`) now bakes in conservative parameter values at construction time per SECURITY_REVIEW M-8. Governance proposes to relax them after the post-deploy paid audit completes. **Actual on-chain values are LOWER than the 2026-05-17 projection** — verified by `post_deploy_verify.js` section [E]:

| Parameter | Soft-launch value (actual on-chain) | Production target | Lever it pulls |
|---|---|---|---|
| `maxTotalStakeWei` | **100 PWM** (~$130 at $1.30) | $10,000+ equivalent | Total stake across protocol bounded; max exploit ceiling for staking bugs ≈ $130 |
| `maxBenchmarkPoolWei` | **100 PWM** (~$130) | TBD | Per-benchmark reward pool bounded |
| `mintingPaused` | **true** | false (governance unpause post-audit) | Reserve cannot emit new PWM; bug in `PWMMinting` can't drain Reserve |
| `transfersPaused` | **true** | false | Per-principle treasuries exist but can't be withdrawn from; protects T_k accumulation |
| `submissionPermissionless` | **false** (new from C-1 fix) | true | PWMCertificate.submit gated to governance-approved submitters; prevents self-deal attack |
| Stake floors (10/2/1 PWM per layer) | unchanged | unchanged | No change needed |
| Reserve token transfers | **30-day cliff (auto-clears D9+30); then governance-controlled** | governance-controlled | No transfers out of Reserve until cliff OR governance vote |

After 30 days of incident-free operation + paid audit complete → governance proposes `setMaxTotalStakeWei($10K equiv)`, `setMintingPaused(false)`, `setTransfersPaused(false)`, `setSubmissionPermissionless(true)` via `executeCall` → 48 h timelock each → execute. Same contract addresses become production PWM (no v2 rebrand).

**With these caps, max blast radius from any contract bug during the first 30 days is ~$130-260 USD-equivalent** (one pool, both contracts). That's a safety budget Director can afford to lose if a worst-case bug surfaces, and it's a credible launch posture for grant reviewers. Note: the dispatch playbook's "$1K cap" framing is rounded up; actual is lower.

---

## Recommended action plan — ACCELERATED PATH (revised 2026-05-18 evening)

The original week-by-week plan (pre-deploy audit gating deploy) has been superseded by the accelerated dispatch path. Sequencing is now: **multi-agent review (DONE) → mainnet deploy with soft-launch caps (D9 = 2026-05-19/20) → engage paid audit during D9+0 to D9+7 → audit completes ~D9+30 → governance raises caps**.

### This week — 2026-05-18 (today) → 2026-05-24

| Day | Action | Effort | Output |
|---|---|---|---|
| **TODAY (2026-05-18 evening)** | Approve MAINNET_FIRST_PLAN narrowing for audit-grants only (see "Policy question" section below) | 5 min | Decision recorded; pre-deploy audit-grant applications unblocked |
| TODAY-2026-05-19 | Phase 3 Step 9 fires (2026-05-18 21:42:56 local); Step 9.5 + Phase 4 preflight | ~1 hr active | Sepolia complete; ready for Phase 5 |
| TODAY-2026-05-20 | **Phase 5 deploy** with 5.4a/b/c sub-sequence per `REGISTRY_HANDOFF_DECISION_2026-05-18.md` | ~3-5 hr active | Mainnet live with soft-launch caps + 1,597 genesis Principles on-chain |
| 2026-05-20 → 2026-05-23 | **Draft Base Builder Grant application** (post-deploy framing: "PWM live on Base mainnet; soft-launch caps; need follow-up audit to raise caps") | 8-10 hr | Submitted to https://base.org/grants |
| 2026-05-21 → 2026-05-24 | **Draft Ethereum Foundation ESP application** (similar framing; larger ask $25-50K) | 12-15 hr | Submitted to https://esp.ethereum.foundation |
| In parallel | **Engage Code4rena / Spearbit / Sherlock** for competitive audit scoping | 2-3 days outreach + scope spec | Audit slot scheduled for ~D9+7 to D9+14 |

### Week 2-4 (D9+7 to D9+28) — audit engagement

| Action | Effort | Output |
|---|---|---|
| Competitive audit contest runs (Code4rena 1-2 week public contest, or Spearbit/Cantina 2-3 week competitive review) | Director: ~5 hr/wk monitoring + question answering | Audit findings report |
| Triage findings; fix HIGH/MEDIUM | 1-3 days engineering | Patched contracts |
| Re-run multi-agent + tests on patched code | ~2 hr | 188+ tests still GREEN |
| Continue monitoring soft-launch deploy: daily watchdog, weekly indexer health, no P0/P1 incidents | ~5 min/day | Stable mainnet |

### Week 4-8 (D9+28 to D9+56) — paid-audit completion + cap-raise

| Grant outcome | Action |
|---|---|
| **Base + EF + Competitive (combined $40-100K)** | Engage Trail of Bits / OpenZeppelin / Spearbit for formal-verification + symbolic-execution audit (closes the residual gap multi-agent + competitive don't cover). 4-week fieldwork. |
| **Single source ($15-25K)** | Engage smaller-scope audit (Cantina senior review at $15-25K). 2-week turnaround. |
| **No grants in 8 weeks** | Use Reserve PWM (~$30K equivalent) to fund the audit directly per vision §6.2 "self-funding war chest." Governance proposal → 48h timelock → execute → audit firm gets paid in PWM-via-USDC-swap. |

### Week 8+ (D9+56 to D9+90) — cap-raise governance

| Action | Effort | Output |
|---|---|---|
| Governance proposes `setMaxTotalStakeWei($10K equiv)` via executeCall | 1 multisig tx + 48h timelock | After execute: cap = $10K |
| Governance proposes `setMintingPaused(false)` | 1 multisig tx + 48h timelock | Minting unpaused; PWM emission begins |
| Governance proposes `setTransfersPaused(false)` | 1 multisig tx + 48h timelock | T_k withdrawals enabled |
| Governance proposes `setSubmissionPermissionless(true)` | 1 multisig tx + 48h timelock | Permissionless cert submission enabled |
| Announce "formal launch" (same contract addresses; just cap-raise; not a v2) | 1 hr | Public announcement |

**By ~D9+90 (early August 2026), PWM is operating at production parameters with the same contract addresses deployed on 2026-05-19/20.** The deploy at D9 IS the formal protocol — soft-launch is a posture, not a v0.

---

## The MAINNET_FIRST_PLAN policy question

The current policy (`pwm-team/funds/MAINNET_FIRST_PLAN.md`, locked 2026-04-27) says:

> No grant applications until mainnet + 30 days stable. First application 2026-08-25.

The policy was made to avoid the "vaporware fundraising" perception that hurts open-source projects when they fundraise on promises rather than working software. The policy is sound in spirit.

**Audit-specific grant applications don't trigger the vaporware critique** because:

1. The deliverable (audit) is concrete, third-party-verified, and time-bounded
2. The grant is for *infrastructure that enables mainnet*, not for "fund our operations on faith"
3. PWM already has working software (116/116 tests, Phase 3 Sepolia round-trip complete, contracts deployed to testnet) — this is not pre-product fundraising

**Recommended policy narrowing:**

```
2026-05-17 | MAINNET_FIRST_PLAN policy NARROWED, not revoked. Pre-deploy
audit-funding applications via Base Builder Grants and Ethereum Foundation
ESP are APPROVED. Scope: audit fees only ($25-50K target). General
operating-budget grants (NumFOCUS Round 4, CZI, Sloan, NSF) remain
deferred to mainnet + 30 days stable per original policy. Rationale:
audit grants fund a specific deliverable that is itself a mainnet-
enabling step; they don't carry the vaporware risk that motivated the
original policy. Decision authority: Director sole executive (Path A
bootstrap). Cross-reference: PWM_PRE_DEPLOY_AUDIT_FUNDING_OPTIONS_2026-05-17.md.
```

This is a clean, defensible narrowing. The original spirit is preserved; the practical blocker is removed.

---

## Open questions / blockers Director must decide

1. **Approve the MAINNET_FIRST_PLAN policy narrowing?** Yes / No / Modify. **(Still open as of 2026-05-18 — required to unblock Base Builder Grants + EF ESP applications.)**
2. **Time commitment for grant writing this week (post-deploy)?** ~10 hr (Base only), ~25 hr (Base + EF), or more (Base + EF + competitive engagement)? **(Plan now assumes deploy first, applications immediately after; Director's bandwidth in the D9+0 to D9+7 window is the limiting factor.)**
3. **Track K (UTSW PI mentor) status?** Has progressed since 2026-05-13 doc? If yes, Tier 3 grants (NSF POSE) become realistic on a 6-month horizon. **(Still open; doesn't affect pre-deploy audit funding.)**
4. ~~Soft-launch fallback acceptable?~~ **RESOLVED 2026-05-18: YES — Director chose soft-launch as the active path on 2026-05-18 when adopting the accelerated dispatch playbook. The "fallback" framing in the original doc is moot; soft-launch IS the path.**
5. **Any specific audit firm preference?** Director's research advisor, university contacts, or crypto-twitter referrals could shortcut firm selection. **(Still open. Competitive audit programs — Code4rena / Spearbit / Sherlock — may be a better fit than direct firm engagement given budget; consider engaging post-deploy.)**
6. **(NEW 2026-05-18)** Reserve PWM as audit-funding fallback acceptable? If no external grants land in 8 weeks, Director can use ~$30K equivalent of Reserve PWM (~2% of available Reserve) to fund the paid audit. Vision §6.2 explicitly contemplates this. **Recommended default: YES.**

---

## Cross-references

### Original cross-refs

- `pwm-team/funds/MAINNET_FIRST_PLAN.md` — current policy (to be narrowed per this doc)
- `pwm-team/funds/PWM_FUNDING_STRATEGY_AND_MAINNET_SEQUENCING_2026-05-02.md` — broader funding strategy
- `pwm-team/funds/PWM_INDEPENDENT_FUNDING_PATHS_2026-05-13.md` — alternate funding paths
- `pwm-team/funds/PWM_RESEARCH_ASSOCIATE_AND_MENTOR_CONSTRAINTS_2026-05-13.md` — Director's eligibility constraints
- `pwm-team/funds/PWM_PI_TRANSITION_STRATEGY_2026-05-13.md` — Track K (UTSW PI mentor) strategy
- `pwm-team/coordination/wallet/PWM_PHASE_4_FINAL_PRE_DEPLOY_2026-05-09.md` — Phase 4 prep canonical doc
- `pwm-team/coordination/wallet/PWM_PHASE_4_PROGRESS_2026-05-17.md` — Phase 4 prep current state
- `pwm-team/coordination/wallet/PWM_PHASE_5_DEPLOY_DAY_2026-05-09.md` — D9 deploy procedure
- `pwm-team/coordination/DIRECTOR_RUNBOOK_D1_TO_D10_2026-05-01.md` — original D1-D10 runbook (the source of "D9")
- `pwm-team/coordination/MAINNET_BLOCKERS_2026-04-30.md` — original 12-step deploy blocker list

### Added 2026-05-18 (accelerated path)

- **`pwm-team/deploy/PWM_MULTI_SERVER_DISPATCH_2026-05-18.md`** — the accelerated dispatch playbook (commit `ea0bf2c7`); source of the 36-48 hour path
- **`pwm-team/deploy/findings/SECURITY_REVIEW_2026-05-18.md`** — final A10 aggregator from multi-agent review; the audit-equivalent evidence for grant applications
- **`pwm-team/deploy/findings/STATUS_2026-05-18_final.md`** — all 12 deploy-relevant issues CLOSED; 188/188 tests
- **`pwm-team/deploy/findings/REGISTRY_HANDOFF_DECISION_2026-05-18.md`** — defers registry handoff to Phase 5.6 so 1,597 genesis Principles register at D9
- **`pwm-team/deploy/findings/ABORT_DECISION_2026-05-18.md`** — patch-vs-abort decision record for A3 CRITICAL
- **`pwm-team/coordination/wallet/PWM_PHASE_5_PROGRESS_2026-05-17.md`** — updated 2026-05-18 with 5.4a/b/c sub-sequence (commit `15ca88e5`)

### External grant sources

- https://base.org/grants — Base Builder Grants (Tier 1)
- https://esp.ethereum.foundation — Ethereum Foundation ESP (Tier 1)
- https://code4rena.com — Code4rena competitive audit marketplace (Tier 1)
- https://cantina.xyz — Spearbit / Cantina competitive audit (Tier 1)
- https://sherlock.xyz — Sherlock competitive audit + insurance (Tier 1)
- https://www.trailofbits.com — Trail of Bits direct (Tier 1; OSS discount)
- https://www.openzeppelin.com/security-audits — OpenZeppelin (Tier 1; OSS programs)
- https://www.coinbase.com/cloud — Coinbase Cloud free tier
- https://www.coinbase.com/developer-platform — Coinbase Developer Platform credits
- https://chanzuckerberg.com/eoss/ — CZI EOSS (Tier 2; post-mainnet)
- https://sloan.org/programs/research/digital-information-technology — Sloan OSS (Tier 2; post-mainnet)
- https://www.mozilla.org/en-US/moss/ — Mozilla MOSS (Tier 2; post-mainnet)
