# PWM User Acquisition Strategy — Product-First Two-Track Launch

**Date:** 2026-05-22
**Owner:** Director
**Status:** Canonical reference for the demand-side strategy through D9+18 months
**Audience:** Director + Heyang (intern) + future co-founders + grant reviewers asking "who actually uses PWM"
**Supersedes:** the implicit "Track 9 alone is the demand wedge" framing in prior planning conversations
**Companion to:** `PWM_API_VS_WEBSITE_2026-05-20.md` (website-first phasing), `plan/track_9/PWM_TRACK_9_LOW_DOSE_CT_2026-05-16.md` (medical flagship), `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md` (intern execution capacity)

---

## TL;DR

**PWM must solve user acquisition before token mining can matter.** Per the comparable-tokens analysis in `coordination/prevent_copy/PWM_TOKEN_VALUE_DEFENSE_2026-05-20.md` §5.3, three of six closest analog tokens (Filecoin, Helium, UMA) failed at this step — supply-side mining inflated indefinitely while demand-side revenue never materialized; ~90% token-price collapse followed.

**Two-track user acquisition:**

- **Track A — Fast academic/research wedge** (launches Months 4-6 post-D9). Researchers/PhD students submit methods to a public-data benchmark for leaderboard credibility + citation value. Goal: 10-30 external submitters within 6-9 months. No Track K mentor required.
- **Track B — Medical imaging flagship** (Track 9; full RSNA/ISBI 2028 launch unchanged). Track 9 stays the long-term medical credibility play. Builds on Track A's proven mechanism. Requires Track K mentor + IRB + clinical partners.

**Strategic formula (canonical):**

> Benchmark first. Users second. Verified jobs third. Infrastructure fourth. Token value last.

---

## 1. The reframe

| Old framing (retire) | New framing (canonical) |
|---|---|
| "PWM is a token mining protocol for scientific computing" | "PWM is a verified benchmark platform for physics-grounded AI, starting with computational imaging and medical imaging" |
| Token-first; abstract; speculative-sounding | Product-first; concrete; aligns with what researchers + grant reviewers already understand |
| Single-vertical demand bet on Track 9 (medical) | Two-track: fast research wedge (Track A) + medical flagship (Track B = Track 9) |
| First user = someone with money | First user = PhD student / postdoc / AI imaging researcher who wants leaderboard credibility |
| Demand proof = "user pays PWM to verify" | Demand proof = "10+ external submitters care about leaderboard rank" |

**Why this matters for messaging:** when explaining PWM to a grant reviewer, NumFOCUS evaluator, prospective co-founder, or AI lab evaluator, the new framing answers "what does it do" in concrete benchmark terms. The token mechanics are a downstream implementation detail of the benchmark platform, not the headline.

---

## 2. The two-track structure

### Track A — Fast academic/research wedge (launch FIRST)

**Target user:** PhD student, postdoc, or AI imaging researcher who:
- Wants a credible leaderboard result for their method
- Wants citations + visibility + comparison against baselines
- Already understands benchmarks, reproducibility, scoring
- Does NOT primarily care about token value at this stage

**Why this user:** they already submit to Kaggle, MIDL, ML4H, SPIE Medical Imaging, NeurIPS reproducibility tracks. PWM offering "verified leaderboard with on-chain reproducibility proofs + small PWM prize" is a recognizable + low-friction value proposition. They don't need to be convinced that benchmarks matter — they already build their careers on them.

**Candidate first benchmark (pick ONE for the first launch):**

| Option | Domain | Director fit | External market |
|---|---|---|---|
| **CASSI calibration / reconstruction** | Compressive spectral imaging | ✅ Strong (Director's published work) | Moderate (specialty community) |
| **Public low-dose CT reconstruction (LIDC-IDRI + AAPM 2016)** | Medical imaging without clinical scope | Moderate (overlaps with Track 9 baseline work) | Strong (active ML4H community) |
| **General inverse-problem reconstruction** | Broad ML | Moderate (Director's InverseNet paper) | Strong (broad ML researcher market) but less concrete |

**Director's stated preference:** CASSI or public low-dose CT, not too broad. **Decision needed (target: post-D9+14 days): pick one and commit.**

**Doctrine: do NOT make this clinical at first.** Public datasets only. No IRB. No claims about clinical adoption. The first goal is to prove that the on-chain mechanism (submit → score → leaderboard → cert → small reward) works at small scale with real external participants.

### Track B — Medical imaging flagship (build AFTER Track A proves mechanism)

**This is Track 9 (per `plan/track_9/PWM_TRACK_9_LOW_DOSE_CT_2026-05-16.md`), unchanged timeline.**

**External-facing storytelling for Track B (the headline message PWM eventually deploys for grant reviewers, hospitals, investors):**

> "PWM verifies whether AI reconstruction methods truly reduce CT dose while preserving image quality."

That sentence is comprehensible to a radiology department chair, an NIH program officer, a journalist, or a Schmidt Futures evaluator. It is the long-game framing — but it requires clinical credibility, IRB-approved data acquisition, and a Track K-class attending mentor at UTSW (NOT Dr. Zaman, per the COI gate in `plan/track_9`).

**Track B sequencing (unchanged):**

| Phase | When | Gating dependency |
|---|---|---|
| Reference method prototype + paper-quality figures | D9+30 to D9+90 | Heyang execution capacity |
| Clinical-data mini-competition | D9+6 to D9+12 months | Track K mentor + IRB |
| Full RSNA/ISBI 2028 launch | 2028 | Above + ~$420-625K Reserve commitment |

Track A unblocks Track B by proving the mechanism works at low risk before larger clinical commitments are made.

---

## 3. PWM-CI-1 — the first benchmark launch

**Codename:** PWM Computational Imaging Benchmark #1 (PWM-CI-1)
**Launch target:** Months 4-6 post-D9 (~2026-09 to 2026-11 calendar)
**Owner:** Director (strategic) + Heyang (execution, weeks 10-16 of intern workplan)
**Prize pool:** 10,000 PWM from Reserve (governance-decided disbursement; small enough for fast multisig approval per Director Decision §4.2 of `coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md`)

### "One of each" launch checklist

The first launch ships the minimum complete product surface — not the perfect one:

| # | Component | Purpose | Effort |
|---|---|---|---|
| 1 | One landing page (subroute on `physicsworldmodel.org`) | Submitter onboarding | 1-2 days |
| 2 | One GitHub repo (`integritynoble/pwm-ci-1`) | Code + data scripts + leaderboard | 2-3 days |
| 3 | One dataset or data-loading script | Reproducible input | 1-2 days (depends on dataset choice) |
| 4 | One baseline method | Floor for submitters to beat | 3-5 days |
| 5 | One evaluation script | Deterministic scoring (PSNR/SSIM/MSE per choice) | 2-3 days |
| 6 | One leaderboard | Public ranking | 1 day (table-driven) |
| 7 | One small prize (10K PWM) | Demand-side signal | governance proposal + 48h timelock |
| 8 | One technical report | Citable artifact | 5-10 days |
| 9 | One community channel (Discord OR WeChat OR Slack — Director picks) | Submitter Q&A | 1 day |
| 10 | One "submit your method" guide | Reduce friction | 1-2 days |

**Total effort: ~3-5 weeks of focused work.** Fits Heyang's weeks 10-16 with Director reviewing.

### What is NOT in the first launch

- No clinical claims
- No payment requirement to submit
- No multi-benchmark portfolio (just ONE benchmark first)
- No mining client infrastructure scaling (mining infrastructure scales only AFTER demand is proven, per the strategic formula)
- No partnerships with hospitals or institutional users
- No press release (silent launch; let participants find it organically + via Director's existing network)

---

## 4. KPI Ladder — the only metric that matters

| Stage | Target external submitter count | Implication |
|---|---|---|
| **Internal demo** | 3-5 (Director + Heyang + 1-3 trusted readers) | Mechanism works end-to-end on-chain |
| **Soft launch** | 5-10 external | Proof of concept; mechanism survives outside-the-team usage |
| **First benchmark report published** | 10-30 total submissions | Genuine community interest; report becomes citable artifact |
| **Strong traction** | 50+ submissions OR 5+ labs involved | Demand-side validation; second benchmark (PWM-CI-2) becomes justified |
| **Real protocol demand** | Users pay or stake to run verified evaluation | Token economics activates; mining infrastructure scaling becomes appropriate |

**Stop condition: if PWM-CI-1 cannot reach 10 serious external submitters within 6 months of launch, do NOT scale mining infrastructure or launch additional benchmarks.** Re-evaluate the wedge — either re-pick the benchmark domain, re-pick the user demographic, or accept that demand-side adoption is going to be longer than 12 months and conserve Reserve.

---

## 5. Payment sequence — do NOT extract too early

**Critical doctrine:** the protocol cannot ask users to pay PWM before they trust the platform. Forcing payment in the first 6-12 months reduces adoption + signals desperation.

**Phased monetization (years, not months):**

| Phase | When | Submitter pays? | Submitter rewarded? |
|---|---|---|---|
| **Phase 1: Free + sponsored** | PWM-CI-1 launch + first ~6 months | NO — free or sponsored runs | YES — small PWM rewards for ranked submissions (10K pool) |
| **Phase 2: PWM-rewarded contributors** | D9+6 to D9+12 months | Still no | YES — broader reward structure as 2-3 benchmarks live |
| **Phase 3: Advanced verified runs require payment or staking** | D9+12 to D9+24 months | YES for premium tier (large-scale verification, private benchmarks) | YES — protocol-level rewards from T_k per-Principle treasuries |
| **Phase 4: Infrastructure providers earn PWM from real jobs** | D9+18+ months | YES at scale | YES — full Zeno emission active |

**Rule: do not ask users to pay before they trust the platform. First create value. Then monetize verification.**

---

## 6. The strategic formula (canonical)

```
Benchmark first.
Users second.
Verified jobs third.
Infrastructure fourth.
Token value last.
```

**Interpretation:**

1. **Benchmark first** — ship PWM-CI-1 with public data, simple metrics, ONE concrete leaderboard
2. **Users second** — recruit 10-30 external researchers; meet them where they already are (Kaggle/MIDL/ML4H culture)
3. **Verified jobs third** — once submitters are present, the on-chain scoring/cert mechanism gets exercised at small scale
4. **Infrastructure fourth** — mining infrastructure (CP nodes, scoring validators, IPFS pinners) scales only AFTER real jobs exist to justify it
5. **Token value last** — PWM price discovery follows demand, not the other way around. Do not market the token; market the benchmark platform.

**Each layer is a precondition for the next.** Inverting the order (token first → infrastructure → jobs → users → benchmark) is the Helium/UMA pattern that fails.

---

## 7. The first PWM user — concrete options (Director must pick within 14 days post-D9)

A mini-competition with public data + PSNR scoring tests whether the **supply side** (miners/submitters) works. It does NOT by itself test the **demand side** (users with their own data + a problem to solve). The demand side needs deliberate seeding.

Three paths to seed the first real demand-side event, none requiring Track K:

### Option A — Director becomes User #1

Director's published arXiv papers (InverseNet, ct_qc_copilot, Proof-of-Solution, universal_simulation, system_design) all have reconstruction methods that benefit from verified-reproducibility runs.

**Mechanism:** Post-PWM-CI-1 launch (Months 4-6), Director submits a "verify my paper's reconstruction" job, pays a tiny PWM amount, gets a cert back. Cost: trivial. Output: first real demand-side on-chain event + citable evidence the payment mechanism works end-to-end.

### Option B — Cold-outreach 10 paper-author contacts

Director's network includes co-authors, reviewers, and people who've cited the published papers.

**Mechanism:** Months 3-6 post-D9, Director sends cold emails: *"Want me to verify your method's reconstruction on the [LIDC-IDRI / CASSI / etc.] test set? Free; you'd write a short blog post about the experience."* Target: 1-2 takers per month. Each is real demand-side user behavior even if no money changes hands at first.

### Option C — Recruit one PWM-CI-1 participant as long-term user

After PWM-CI-1 publishes its report (Months 6-9), identify the top 3 submitters; offer them ongoing free verified runs of their methods on new test sets in exchange for a quote / blog post / case study.

**Mechanism:** Convert submitter (supply-side) → ongoing user (demand-side). Low cost; uses existing relationship.

**Recommendation:** A + B in combination. Option C only if Option A + B don't produce enough demand by D9+9 months.

---

## 8. Dependencies + what's blocked / unblocked

| Workstream | Blocked by Track K? | Heyang capacity? | Can start when |
|---|---|---|---|
| PWM-CI-1 benchmark choice (CASSI / public CT / inverse problem) | ❌ No | n/a — Director decision | Within 14 days post-D9 |
| PWM-CI-1 design + scoping | ❌ No | ✅ Yes (weeks 10-12) | D9+30 |
| PWM-CI-1 launch (full "one of each" checklist) | ❌ No | ✅ Yes (weeks 10-16) | Months 4-6 post-D9 |
| First benchmark report | ❌ No | ✅ Yes (post-launch) | Months 6-9 post-D9 |
| Director-as-User #1 transaction | ❌ No | n/a | Months 2-3 post-D9 |
| Cold-outreach 10 paper contacts | ❌ No | n/a | Months 3-6 post-D9 |
| **Track B clinical mini-competition (Track 9 sub-launch)** | ✅ **YES — gated on Track K mentor + IRB** | n/a until unblocked | TBD (~D9+9-15 months when Track K lands) |
| **Full RSNA/ISBI 2028 launch** | ✅ **YES — gated on Track K + clinical partners** | n/a until unblocked | 2028 unchanged |

**Critical insight: Track K being pending does NOT block the demand-side proof.** Track A is fully runnable on public data + Heyang's existing workplan capacity.

---

## 9. What this DOES NOT change

This strategy is additive to existing canonical decisions; it does not retract any of them:

- ✅ **Mainnet deploy schedule unchanged** — Phase 5 still targets 2026-05-19/20 per accelerated path
- ✅ **Soft-launch caps unchanged** — `mintingPaused=true`, `transfersPaused=true`, `submissionPermissionless=false`, `maxTotalStakeWei=100 PWM`, `maxBenchmarkPoolWei=100 PWM`. PWM-CI-1 prize pool (10K PWM) operates within these caps because it's governance-disbursement, not user staking
- ✅ **Founder compensation structure unchanged** — Director retains 100% of 0.63M L1 allocation per the 2026-05-22 update to `PWM_DEVELOPER_COMPENSATION_2026-05-22.md`; co-founder economic stake via L3 mechanisms
- ✅ **Track 9 long-term timeline unchanged** — RSNA/ISBI 2028 stays the medical flagship target
- ✅ **Website-first phasing unchanged** — Year 1 website / Year 2 API / Year 3+ agents per `PWM_API_VS_WEBSITE_2026-05-20.md`; PWM-CI-1 IS what the website does in Year 1
- ✅ **Walled-garden trigger unchanged** — remains the only override condition for Year 3+ agent SDK acceleration
- ✅ **Bounty board structure unchanged** — 8 OPEN/SPEC + 2 RESERVED per `bounties/INDEX.md`; the bounty framing already supports "benchmark platform" reading without rewrite

---

## 10. Director decisions needed

Three concrete decisions, with default + suggested deadlines:

| # | Decision | Default | Deadline | Implication |
|---|---|---|---|---|
| 1 | **First benchmark domain:** CASSI / public low-dose CT / inverse problem? | **Public low-dose CT** (best storytelling for grant reviewers) OR **CASSI** (best Director expertise fit) | Post-D9+14 days | Locks Heyang's weeks 10-16 deliverable scope |
| 2 | **Be User #1 with own paper as test case?** | **YES** (low cost, high signal value) | Post-D9+30 days | First demand-side on-chain event happens Months 2-3 |
| 3 | **Commit to 10 cold-outreach emails Months 3-6?** | **YES** (Director's existing network is the cheapest user pipeline) | Post-D9+60 days | Generates 1-2 real users per month via existing relationships |

Items #2 and #3 are individually small; #1 is the strategic lock that determines the entire PWM-CI-1 scope.

---

## 11. Honest stop conditions

The strategy includes explicit failure paths:

- **If PWM-CI-1 ships but gets 0-2 external submitters in 3 months:** the wedge is wrong. Revisit domain choice; do not scale.
- **If PWM-CI-1 gets 3-9 external submitters in 6 months:** marginal; defer mining infrastructure scaling; revisit benchmark choice + outreach intensity.
- **If PWM-CI-1 gets 10+ external submitters and a published technical report cites 5+ external methods:** mechanism validated; proceed to PWM-CI-2 (second benchmark) AND begin Track B (medical flagship clinical-data scoping when Track K lands).
- **If after 12 months no consistent submitter community has formed:** the protocol is in zombie state. Conserve Reserve; the demand-side wedge needs fundamental rethink; do NOT continue scaling supply-side infrastructure.

**The KPI ladder in §4 has explicit stop-conditions to prevent the Helium-pattern failure mode (supply-side over-investment without demand-side proof).**

---

## 12. What gets updated downstream (proposed; awaiting Director confirmation)

If this strategy is approved as canonical, downstream docs to update:

| Doc | Change |
|---|---|
| `coordination/PWM_API_VS_WEBSITE_2026-05-20.md` | Add PWM-CI-1 as the concrete Year 1 product the website serves |
| `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md` | Add PWM-CI-1 design (weeks 10-12) + launch prep (weeks 13-16) as new deliverables |
| `plan/track_9/PWM_TRACK_9_LOW_DOSE_CT_2026-05-16.md` | Insert PWM-CI-1 as a Track 9 sub-phase ("Phase 0 fast wedge") preceding the clinical-data competition |
| `coordination/prevent_copy/PWM_COMPETITIVE_DEFENSE_2026-05-20.md` | Reframe product positioning per §1 of this doc |
| `coordination/prevent_copy/PWM_TOKEN_VALUE_DEFENSE_2026-05-20.md` | Add the strategic formula (§6) as an explicit defense against Helium-pattern collapse |
| `bounties/INDEX.md` | Light edit to reference the benchmark-platform framing in the intro |
| `pwm-team/plan/PLAN.md` | Add a new Track 10 or Track 3d entry for PWM-CI-1 execution |

**Not done in this commit.** Director needs to sign off on this strategy first; downstream cascade follows.

---

## 13. Cross-references

- `pwm-team/coordination/prevent_copy/PWM_TOKEN_VALUE_DEFENSE_2026-05-20.md` §5.3 — comparable tokens (Helium/UMA/Filecoin failure-pattern analysis)
- `pwm-team/coordination/PWM_API_VS_WEBSITE_2026-05-20.md` — website-first phasing decision (Year 1 website / Year 2 API / Year 3+ agents)
- `pwm-team/plan/track_9/PWM_TRACK_9_LOW_DOSE_CT_2026-05-16.md` — Track B (medical flagship) spec
- `pwm-team/papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md` — Heyang execution capacity + week-by-week breakdown
- `pwm-team/coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` §4 — Reserve discretionary mechanism (governance flow for the 10K PWM prize pool)
- `pwm-team/coordination/strategy/AUTH_AND_WALLET_STRATEGY.md` — auth doctrine for the PWM-CI-1 landing page (wallet connect at submit time only, no web2 auth)
- `pwm-team/bounties/INDEX.md` — infrastructure bounty board (Track A is built on top of this infrastructure)

---

## 14. The single most important sentence

**The product should lead. The token should follow.**

Every existing strategic doc assumes the protocol is what users want. This strategy is the corrective: PWM is not a product to most researchers; the **benchmark + leaderboard + reproducibility cert** is the product. PWM is the token that infrastructure providers earn for running the benchmark platform. Lead with the product. Discuss the token only when asked.

---

*This doc is the canonical reference for PWM user acquisition through D9+18 months. Update §3 (PWM-CI-1 scope) when Director picks the first benchmark domain. Update §10 (Director decisions) when each decision lands. Update §4 (KPI ladder) with actual submitter counts as they accrue. Keep cross-references current.*
