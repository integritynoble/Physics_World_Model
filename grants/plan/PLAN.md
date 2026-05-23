# PWM Master Plan — All Tracks A-N

**Date:** 2026-05-15 (last revised 2026-05-18 evening)
**Author:** Director
**Status:** Active execution master plan; consolidates all tracks from sub-plans
**Mainnet target:** **2026-05-19/20** (accelerated path per `pwm-team/deploy/PWM_MULTI_SERVER_DISPATCH_2026-05-18.md`; was 2026-07-24). Path A bootstrap — Director-holds-all-5 multisig at deploy. Soft-launch caps protect the 30-day post-deploy window; paid audit deferred to ~D9+30 via Track 7 grants.
**Year-1 funding target:** $190K-$700K (Tier 1 crypto-foundation grants Q4 2026 onwards)

This single plan integrates every active track across the PWM project. It supersedes nothing — each sub-plan remains the detailed source of truth — but provides the unified calendar and gate sequencing Director needs to execute week-by-week.

---

## Recent Progress (2026-05-15/16)

Changes since the plan was first drafted:

| What | Commits | Effect on plan |
|---|---|---|
| **Track C complete** — PWMToken + PWMVesting + PWMFaucet + ERC20 sibling contracts shipped, 165/165 tests GREEN | `d558afa0` | Critical path now blocks only on Track D (audit). |
| **Multi-agent security review COMPLETE (2026-05-18)** — 10 agent runs (A1/A2/A3 + re-passes, A4 Slither, A6 cross-contract, A8 deploy-script, A9 spec, A10 aggregator); **2 CRITICAL + 4 HIGH + 6 MEDIUM all FIXED**; test count **165 → 188** (+23 new tests across `203df847`, `fe3ba529`, `41efadb8`, `114df918`, `6d5ec7a6`); 5 soft-launch caps wired + verified by `post_deploy_verify.js` section [E] (25 checks total). | `6cb56537` (A10 aggregator), `37fd967c` (STATUS FINAL), `839705f8` (SECURITY_REVIEW) | Replaces paid audit on pre-deploy critical path. Paid audit deferred to ~D9+30 via Track 7. |
| **Accelerated-path adoption (2026-05-18)** — mainnet D9 shifts 2026-07-24 → 2026-05-19/20; soft-launch caps (100 PWM TVL, `mintingPaused`, `transfersPaused`, `submissionPermissionless=false`) cap blast radius until cap-raise via governance proposal | `ea0bf2c7` (dispatch playbook) | Track 1 path collapses from ~10 weeks to ~36-48 hours. |
| **Registry-handoff DEFER decision (2026-05-18)** — `deploy/erc20.js` does 5/6 sibling handoff at deploy; PWMRegistry stays deployer-owned through Phase 5.4a (`register_batch.py` for 6 founder-vetted Principles), then Phase 5.4b (`scripts/transfer_registry_to_governance.js`) hands off; verifier becomes phase-aware via `PWM_VERIFY_PHASE` env var (5 ↔ 5.6) | `22409cbc` (decision), `3404c96f` (revert + phase-aware verifier), `15ca88e5` (Phase 5 runbook 5.4a/b/c) | Preserves 1,597 genesis artifact registration capability at D9 without splitting genesis content registration. |
| **Track E COMPLETE** — All 1,597 genesis artifacts registered on Base Sepolia (was: 192/1,597) | `78382351`, `9cf9e64a`, `614ebbc7`, `5d8428dc`, `00b0ee38`, `3fd778f2`, `5ca0df31`, `b3362d40`, `b2d0488c` | Track E removed from critical-path-blocker list. Explorer now displays full 529 L1 / 531 L2 / 531 L3 catalog. |
| **Track F shipped** — SIWE + Privy + Web3Auth + ERC-1271 + EIP-6492 + 30-day JWT session, 19 pages + 5 API routes build green | `b274aaef`, `efd6d3f9`, `5375e552`, `e5d6e8c1` | Director's only remaining Track F action is Privy app ID + WalletConnect ID + DNS. |
| **Track G chunk 1+2** — Sidebar nav, `/templates` page, task-type filter chips, benchmark-first homepage, benchmark detail Example Data | `c456def7`, `379eb51c` | Track G no longer blocked on Track F; UI surface is live. Remaining: Principle/Spec "Understanding" prose, `/mine` 4-path tree, "Use This Solution" modal. |
| **Ops tooling** — `scripts/bridge_eth_sepolia_to_base_sepolia.py`, `scripts/validate_forward_models.py`, `scripts/register_batch.py` (replaces deprecated `register_genesis.js`) | `2ca08766`, `78382351`, `9cf9e64a`, `b3362d40` | Reusable infrastructure for future testnet ops + mainnet promotion. |
| **NumFOCUS application package** — 11 docs drafted | `f8346c9e` | Ready to submit Round 4 (Oct 15) once co-founders confirmed. |
| **Solo-founder funding plan** | `fd56e417` | Documents path forward without external co-founders for Tier 1 grants. |

**Net effect (refreshed 2026-05-18 evening):** Accelerated-path mainnet **2026-05-19/20** is now blocked solely on **Director operational checklist items** — Coinbase release for deployer ETH (24-72 hr typical), 4 `.env` vars (Reserve Safe + Liquidity + Team + Alchemy BASE_RPC_URL), Phase 3 Step 9 (auto-fires tonight 01:30 UTC), Phase 3 Step 9.5 kill-switch rehearsal (paste-ready walkthrough at `coordination/wallet/STEP_9_5_KILL_SWITCH_REHEARSAL_2026-05-19.md`), then Phase 4 preflight → Phase 5 deploy. **Multi-agent security review COMPLETE** replaces paid audit pre-deploy per 2026-05-18 dispatch; paid audit reslotted to D9+30 via Track 7 grants. Track E moved from "behind schedule" to "done"; Track G shipped earlier than expected because chunks 1-2 didn't actually require Track F to be live.

---

## Track Consolidation (2026-05-16, extended 2026-05-18)

Reduced from 14 tracks (A-N) to **6 unified tracks (1-6)** for cognitive simplicity. Two additional tracks (7-8) added 2026-05-18 to cover work that didn't fit existing tracks. The old labels remain valid in source plans; this PLAN.md uses the new numbering. Mapping:

| New track | Old labels | Why combined |
|---|---|---|
| **1. Smart Contracts & Governance** | B + B+ + C + D | One contract-engineering lifecycle: write (C) → audit (D) → multisig deploy (B) → post-mainnet rotations (B+) |
| **2. Genesis Content** | E + I | E = on-chain registration (done); I = domain red-team review (post-hoc / optional) |
| **3. Web Explorer** | F + G + H | All `pwm-team/infrastructure/agent-web/` work: wallet (F), redesign (G), verification routes port (H) |
| **4. Foundation & Funding** | A + J + L + M | One foundation-and-money workstream: NumFOCUS legal wrapper (A) ← co-founder recruitment (J) → unlocks Tier 2 grants (M); Tier 1 grants (L) need no wrapper |
| **5. Academic Career** | K | UTSW PI partnership transition — single-topic |
| **6. Community** | N | Discord, workshops, governance forums — single-topic |
| **7. Audit Funding (pre+post deploy)** | new (2026-05-18) | Pre-deploy audit grant applications (Base Builder + EF ESP). Originally blocked by MAINNET_FIRST_PLAN; narrowed 2026-05-18. Parallel to Phase 5 soft-launch — funds the post-deploy audit at D9+30 if grants land in time. |
| **8. Paper Implementation** | new (2026-05-18) | Empirical / engineering work supporting active PWM-authored papers (specific paper TBD per Director). |

---

## Status Snapshot (refreshed 2026-05-18 evening)

| Track | Topic | Owner | Status | Blocks mainnet? |
|---|---|---|---|---|
| **1** | Smart Contracts & Governance | agent-contracts + Director | ✅ 1a Code done (188/188 GREEN); ✅ 1b Multi-agent review COMPLETE (2C+4H+6M fixed; paid audit deferred to D9+30 via Track 7); ✅ 1c HW wallets + Sepolia rehearsals complete; ✅ **PHASE 5 MAINNET DEPLOY COMPLETE 2026-05-22T18:52:09Z** (33 tx, deploy commit `0c731a7c`, deploy log `f8ddbd5f`, post-deploy verifier 25/25 GREEN x2, independent verification 10/10 PASS); ⏳ 1d Rotations Months 1-6 post-mainnet (B+) | ✅ DONE — PWM live on Base mainnet (chainId 8453) |
| **2** | Genesis Content | Director + agent-contracts + reviewers | ✅ **Testnet: 1,597/1,597 on Base Sepolia (2026-05-15) — 1,591 stub-tier auto-generated metadata.** ✅ **Mainnet 2026-05-22: 6 v3-verified artifacts LIVE on Base** (CASSI L1-025 + CACTI L1-027 + L2/L3 children) via `scripts/batches_genesis_curated/batch_00_cassi_cacti.json` (commit `3dfcc4e7`); independent on-chain verification `exists()=true` for all 6 (2026-05-22 evening). Per Director's Option A decision 2026-05-22. Other 1,591 stubs stay on testnet until polished to v3 via Bounty 7. See `coordination/PWM_GENESIS_PRINCIPLES_REALITY_CHECK_2026-05-22.md`. | ✅ DONE (mainnet live) |
| **3** | Web Explorer | agent-web | ✅ Wallet code shipped (F); ✅ Sidebar+templates+task filter+hero shipped (G chunks 1-2); ⏳ Verification routes port (H), Use-This-Solution modal, /mine 4-path deferred | ❌ No |
| **4** | Foundation & Funding | Director + co-founders | ⏳ Co-founder outreach starts week 1 (J); ⏳ NumFOCUS Round 4 target Oct 15 (A); ⏳ Tier 1 grants from Aug 25 (L); ⏳ Tier 2 grants Q4+ pending NumFOCUS (M) | ❌ No |
| **5** | Academic Career | Director | ⏳ UTSW COI disclosure → new mentor → R21 in Q3 2027 (K) | ❌ No |
| **6** | Community | Director | ⏳ Discord live before NumFOCUS (N) | ❌ No |
| **7** | Audit Funding | Director | ⏳ Apply Base Builder + EF ESP this week; fund post-deploy audit at D9+30 | ❌ No (soft-launch path) |
| **8** | Paper Implementation | agent-contracts + future contributors | ⏳ Started: smart contracts (in Track 1); explorer (Track 3); pwm-node CLI + scoring engine deferred to post-mainnet bounties | ❌ No |

**Critical path to mainnet (accelerated path):** Phase 3 Step 9 (Sepolia executeProposal — auto-fires tonight 01:30 UTC) → Phase 3 Step 9.5 (kill-switch rehearsal AM) → Phase 4 preflight → 0 FAIL (gated on 4 `.env` vars + Coinbase ETH release) → Phase 5 deploy with 5.4a/b/c sub-sequence per `REGISTRY_HANDOFF_DECISION_2026-05-18.md` → Phase 6/7. Multi-agent security review COMPLETE closes the audit prerequisite; paid audit moved to Track 7 post-deploy. Tracks 2-8 do not block mainnet deploy; they run in parallel.

---

## Critical Path Calendar

```
                      [Pre-Mainnet, weeks 1-10]                         [Post-Mainnet, months 1-12]
Week:        1   2   3   4   5   6   7   8   9   10  ┊  M1  M2  M3  M6  M9  M12
T1 contracts:████████████████ (✅C shipped)            ┊                                Track 1 — Smart Contracts
T1 audit:             ████████████████████            ┊                                  audit delta wk 4-8 (D)
T1 multisig: ██                                       ┊  ████████████████████             setup (B) + rotations (B+)
T2 genesis:  ████ (✅ COMPLETE 2026-05-15)            ┊                                Track 2 — Genesis Content
T3 wallet:   ████████████ (✅ shipped)                ┊                                Track 3 — Web Explorer
T3 redesign: ██████ (chunks 1+2 shipped) ─ ─          ┊                                  G partial; modal needs F deploy
T3 H:                          ████████████           ┊                                  verification routes port
T4 funding:  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┊  ████████████████████████████  Track 4 — Foundation & Funding
                                                       ┊                                  (J recruit + A NumFOCUS + L Tier 1 + M Tier 2)
T5 career:                                            ┊      ████████████████████        Track 5 — UTSW PI transition (K)
T6 community:████ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─        ┊                                Track 6 — Discord, workshops (N)
                                                  ▲
                                                  2026-07-24 MAINNET DEPLOY
```

Legend: `████` = active; `─ ─ ─` = parallel/background; `┊` = mainnet milestone.

---

## Track 1 — Smart Contracts & Governance

**Goal:** Working PWM token contracts (audited) and PWMGovernance multisig owning admin keys at mainnet deploy day; co-founder rotations onboard real signers Months 1-6 post-mainnet.

**Old labels covered:** B (multisig setup) + B+ (founder rotations) + C (PWMToken + ERC20 migration) + D (audit delta)

**Source plans:** `PWM_ACCELERATED_MAINNET_PLAN_2026-05-15.md` §§ B, B+, C, D; `pwm-team/coordination/wallet/PWM_DECISION_RECORD_PATH_A_2026-05-12.md`; `PWM_FOUNDER_ROTATION_2026-05-10.md`

### 1a — PWMToken stack & ERC20 migration (old Track C) — ✅ CODE COMPLETE + ✅ MULTI-AGENT REVIEW CLOSED

All 7 new contracts written, **188/188 tests GREEN** (was 165 pre-review; +23 tests from `203df847`, `fe3ba529`, `41efadb8`, `114df918`, `6d5ec7a6` patches), end-to-end `deploy/erc20.js` smoke-tested on local hardhat (commit `d558afa0`), then hardened through 10-agent multi-agent security review (2 CRITICAL + 4 HIGH + 6 MEDIUM all FIXED; commit `6cb56537` A10 aggregator, `37fd967c` STATUS FINAL).

| Week | Action | Status |
|---|---|---|
| 1 | Write `PWMToken.sol` (10 tests), `PWMVesting.sol` (10 tests), `PWMFaucet.sol` (13 tests) | ✅ Done |
| 2 | ERC20 sibling versions: `PWMStakingERC20.sol`, `PWMRewardERC20.sol`, `PWMMintingERC20.sol`, `PWMTreasuryERC20.sol` | ✅ Done |
| 3 | Tests: 165/165 GREEN; `deploy/erc20.js` smoke test on local hardhat | ✅ Done |
| 4 | Multi-agent security review (10 agents); 188/188 tests; 5 soft-launch caps wired; 25-check `post_deploy_verify.js` | ✅ Done 2026-05-18 |
| D9 | Phase 4 preflight → Phase 5 deploy with 5.4a/b/c sub-sequence (deferred registry handoff per `REGISTRY_HANDOFF_DECISION_2026-05-18.md`) | Director (~3-5 hr active) |

**Director action remaining:** complete Phase 4 preflight (4 `.env` vars + deployer ETH from Coinbase release), then Phase 5 deploy. Sepolia full erc20 deploy is no longer a prerequisite (accelerated path goes direct to mainnet; genesis 1,597 artifacts already on testnet per Track 2).

### 1b — Multi-agent security review (replaces old Track D paid audit on pre-deploy critical path) — ✅ COMPLETE 2026-05-18

**Scope:** 10 agent runs covering full deploy-relevant attack surface; replaces the originally-planned paid audit delta on the pre-deploy critical path. Paid audit is **deferred to ~D9+30** via Track 7 grants (Base Builder + EF ESP). Soft-launch caps cap blast radius at ~$130 USD (100 PWM TVL cap) during the gap.

| Agent run | Scope | Status |
|---|---|---|
| A1 / A1-v2 / A1-v3 | High-impact path coverage | ✅ Closed (CRITICAL + HIGH findings remediated) |
| A2 / A2-v2 | Spec drift vs. behavior | ✅ Closed |
| A3 / A3-v2 | State-machine + access control | ✅ Closed |
| A4 | Slither static analysis | ✅ Clean (re-run post-fixes) |
| A6 | Cross-contract invariants | ✅ Closed |
| A8 | Deploy-script verification (`deploy/erc20.js` + `post_deploy_verify.js`) | ✅ Closed (M-8 fix added 5 cap checks → 25 total) |
| A9 | Spec / acceptance criteria | ✅ Closed |
| A10 | Aggregator across all runs — final gate | ✅ Closed (`6cb56537`) |

**Findings closed (12 deploy-relevant):** C-1, C-2 (CRITICAL — cert.submit access control + governance executeCall primitive); H-1, H-2, H-3, H-4 (HIGH — mintingPaused, transfersPaused, ExecProposal timelock, registry transferOwnership); M-1, M-2 (MEDIUM — threshold-reached timelock on Proposal + FounderChange); M-8, M-10 (MEDIUM — verifier soft-launch cap checks, hard mainnet guard against PWM_SKIP_GOVERNANCE_HANDOFF); CC-2 (MEDIUM — PWMCertificate.finalize soft-launch compatible).

**Audit-equivalent spend avoided:** ~$20-50K of paid-audit effort substituted by ~$100-200 of Claude API. Paid audit (Track 7d) becomes a **cap-raise gate at D9+30-60**, not a deploy gate.

| Track 7 phase | Action |
|---|---|
| Week 1 post-deploy | Submit Base Builder Grant ($5-25K) + EF ESP ($10-50K) applications |
| Week 1-8 | Applications under review; soft-launch caps held |
| Week 5-8+ | If ≥$15K lands: engage Code4rena / Spearbit / Cantina / Sherlock / Trail of Bits OSS / OpenZeppelin |
| D9+30-60 | Audit complete; governance proposes cap-raise via 3 executeCall proposals (48h timelock × 3) |

**Estimated cost (paid audit deferred):** $15-50K from Track 7 grants OR Reserve PWM fallback (~$30K equiv from ~2% of 1.09M Reserve via governance proposal; see `pwm-team/funds/PWM_PRE_DEPLOY_AUDIT_FUNDING_OPTIONS_2026-05-17.md` §Q6).

### 1c — Multisig setup, Path A bootstrap (old Track B) — ✅ COMPLETE except Phase 3 Step 9 auto-fire + Step 9.5 rehearsal

**Goal:** PWMGovernance multisig contract owns admin keys at mainnet deploy. Honest framing: interim self-custody (all 5 keys controlled by Director) until rotations (1d) onboard real co-founders Months 1-6 post-mainnet.

| Day | Action | Owner | Status |
|---|---|---|---|
| 1-2 | Verify all 5 HW wallets respond and addresses match expected EIP-55 checksums | Director | ✅ Phase 2 done (`94324738`) — 4 Trezor + 1 Ledger |
| 3-7 | Sepolia rehearsal Steps 1-7 (propose / approve × 3) | Director | ✅ Done (`30642f91`); known bytes32 input-corruption documented + root-caused |
| 8 | 48 h timelock | passive | ⏳ wraps **2026-05-19 01:30:10 UTC** |
| 9 | `executeProposal(0)` — auto-fires tonight | Director (Ledger) | ⏳ tonight after timelock; ~5 min |
| 9.5 | Kill-switch rehearsal per paste-ready walkthrough | Director | ⏳ tomorrow AM (`STEP_9_5_KILL_SWITCH_REHEARSAL_2026-05-19.md`, sub-GPU `7d177574`) — ~20-30 min |

**Honest interim disclosure:** The Path A bootstrap state is acceptable for Tier 1 crypto-foundation grants (Track 4 sub-track 4c) but NOT for Tier 2 major foundation grants via NumFOCUS — which is why rotations (1d) must complete before NumFOCUS approval lands.

### 1d — Path A founder rotations (old Track B+) — Months 1-6 post-mainnet

**Goal:** Replace Director's 5 keys one-at-a-time with co-founder keys via Path A founder rotation mechanism. Genuine 3-of-5 decentralized governance by Month 6. **Hard prerequisite:** Track 4 sub-track 4a (co-founder recruitment) — co-founders confirmed and committed.

**Dependency callout** ✅ APPROVED 2026-05-18 (see §1e Decision 2): If Track 4 sub-track 4a (co-founder recruitment) slips beyond Month 1 post-mainnet, the entire 1-6 month rotation cadence slips, which in turn blocks NumFOCUS Round 4 (Oct 15) and Tier 2 grants. Recruitment outreach starts Week 1 post-mainnet under Track 1c day 5 / Track 4a — the dependency chain is critical-path for Tier 2 funding but NOT for mainnet deploy itself.

| Month | Action |
|---|---|
| 1-2 | Rotation #1 (replace 1 of 5 Director-keys with co-founder #2's key) |
| 2-3 | Rotation #2 (co-founder #3's key replaces another Director slot) |
| 3-4 | Rotation #3 (co-founder #4's key) |
| 4-5 | Rotation #4 — Director controls only 1 of 5; genuine 3-of-5 multi-party governance achieved |
| 5-6 | Optional: Rotation #5 if Comprehensive FSA recruited a 5th independent signer |

Each rotation: 3-of-5 governance proposal + 48h timelock + execute. During interim, Director self-approves with his own keys; once first co-founder joins, Director + co-founder both required for next rotation.

### 1e — Director decision log (2026-05-18 evening)

Two Track 1 decisions outstanding from the morning Track 1 audit (`track_1/PWM_TRACK_1_AUDIT_2026-05-18.md` P1 section) approved this evening as part of the accelerated-path adoption + multi-agent review closure:

#### Decision 1 — D5 pre-deploy gate retirement ✅ APPROVED 2026-05-18

| Aspect | Value |
|---|---|
| **Original gate** | D5 / G4 from the 2026-04-30 plan: "≥20 non-founder L4 submissions on Base Sepolia before mainnet deploy" |
| **Source docs** | `pwm-team/coordination/DIRECTOR_RUNBOOK_D1_TO_D10_2026-05-01.md` row D5; `pwm-team/coordination/MAINNET_DEPLOY_PLAN_SYNTHESIS_2026-05-18.md` row D5 |
| **Decision** | **Formally retired.** Soft-launch caps deployed at D9 supersede this gate. |
| **Rationale** | The D5 gate's purpose was to bound blast radius from un-vetted external submissions before deploy. Soft-launch caps achieve the same risk control at deploy time via on-chain enforcement: `maxTotalStakeWei = 100 PWM` (~$130), `maxBenchmarkPoolWei = 100 PWM`, `mintingPaused = true`, `transfersPaused = true`, `submissionPermissionless = false`. All 5 caps are verified by `post_deploy_verify.js` section [E] (added in M-8 fix, commit `114df918`; verifier is now 25 checks total). Max blast radius during 30-day soft-launch: ~$130-260 USD-equivalent. Cap-raise happens via governance proposal + 48h timelock × 3 once paid audit completes at D9+30-60. |
| **Optional Route B remains** | Self-stress with 20 disposable wallets (~2 hours of activity) is a nice-to-have for **post-deploy monitoring confidence** — verifies indexer + Tenderly alerts fire on real L4 submission flow — but is no longer a deploy gate. Director may run it during D9+0 to D9+7 stabilization window. |
| **Doc-consistency follow-up** | The "Mainnet Deploy Day Gate Checklist" section of this PLAN.md (line ~580+) and `MAINNET_DEPLOY_PLAN_SYNTHESIS_2026-05-18.md` row D5 still reference the pre-retirement framing; a separate consistency pass should propagate this decision. Track 1 closeout treats this as deferred doc hygiene, not a re-open. |

#### Decision 2 — Track 1d co-founder dependency callout ✅ APPROVED 2026-05-18

| Aspect | Value |
|---|---|
| **Issue** | Pre-refresh §1d implied that Track 4a (co-founder recruitment) slip would also slip mainnet deploy. The dependency chain was implicit and ambiguous. |
| **Decision** | **Wording approved as written in §1d above.** The dependency callout explicitly clarifies that Track 4a slip blocks Tier 2 funding (NumFOCUS Round 4 + Tier 2 grants) but does **NOT** block mainnet deploy. Path A bootstrap (Director-holds-all-5) is the lawful interim state at D9. |
| **Rationale** | Honesty about the dependency chain matters for grant copy + reviewer expectations. Conflating "rotations needed for Tier 2 funding" with "rotations needed for deploy" overstates the deploy gate and risks reviewer pushback when they realize Director controls all 5 keys at D9. The Path A decision record (`pwm-team/coordination/wallet/PWM_DECISION_RECORD_PATH_A_2026-05-12.md`) already establishes interim self-custody as the lawful state; this §1d wording brings PLAN.md into alignment with that decision. |
| **Effect** | Tier 1 crypto-foundation grants (Track 4 sub-track 4c) remain eligible at D9 with Path A disclosure; Tier 2 (NumFOCUS-routed) waits for first rotation to land (~Month 1-2 post-mainnet). |

#### What's NOT a Track 1 decision

The following are explicitly out of scope for this Track 1 closeout — they belong to other tracks or to follow-up doc-hygiene passes:

- Track 7 audit-funding decisions (MAINNET_FIRST_PLAN narrowing, grant-writing time commitment, Reserve PWM fallback) — see `pwm-team/funds/PWM_PRE_DEPLOY_AUDIT_FUNDING_OPTIONS_2026-05-17.md` Q1/Q2/Q6 and §7a above which records MAINNET_FIRST_PLAN narrowing as APPROVED
- Mainnet Deploy Day Gate Checklist date + content refresh (line ~580)
- Critical Path Calendar Gantt diagram refresh (lines 67-83)
- `MAINNET_DEPLOY_PLAN_SYNTHESIS_2026-05-18.md` D5 + D10 row updates

These can be consolidated into a single doc-hygiene pass after D9 deploy or in a post-D9+7 closeout sweep.

---

## Track 2 — Genesis Content ✅ Registration COMPLETE

**Goal:** Register all 1,597 genesis artifacts (529 L1 + 531 L2 + 531 L3 + 6 founder-vetted) on `PWMRegistry` at Base Sepolia, and (optionally) recruit domain experts to red-team the AI-authored stubs post-registration.

**Old labels covered:** E (genesis batch registration) + I (domain expert red-team review)

**Source plan:** `PWM_WEBSITE_AND_GENESIS_TESTNET_PLAN_2026-05-15.md` §§ E, I
**Status:** **✅ 1,597 / 1,597 (100%) on-chain as of 2026-05-15 (commit `b2d0488c`).**

### 2a — Batch registration (old Track E) — ✅ COMPLETE

| Batch | Domain | Count | Status |
|---|---|---|---|
| 1 | CASSI + CACTI (founder-vetted) | 6 | ✅ Done 2026-04-19 |
| 2 | Compressive + depth imaging | 23 | ✅ Done 2026-05-15 (`614ebbc7`) |
| 3 | Medical imaging (a/b/c/d) | 192 | ✅ Done 2026-05-15 (`00b0ee38`, `3fd778f2`) |
| 4 | Fluid dynamics | 75 | ✅ Done 2026-05-15 (`5d8428dc`, `5ca0df31`) |
| 5 | Electromagnetics | 75 | ✅ Done 2026-05-15 |
| 6 | Structural mechanics | 77 | ✅ Done 2026-05-15 |
| 7 | Geophysics + acoustics + heat | 150 | ✅ Done 2026-05-15 |
| 8 | Quantum + plasma + nuclear | 111 | ✅ Done 2026-05-15 |
| 9 | Chemistry + materials | 138 | ✅ Done 2026-05-15 |
| 10 | Bio + astro + signal + applied + imaging-remainder (15 sub-batches) | 750 | ✅ Done 2026-05-15 (`b2d0488c`) |
| **Total** | | **1,597** | **✅ 100%** |

**Tooling shipped during execution** (reusable for mainnet promotion):
- `pwm-team/infrastructure/agent-contracts/scripts/register_batch.py` — generalized batched registration, handles content/ schema (canonical Python; deprecates `register_genesis.js`)
- `scripts/validate_forward_models.py` — structural validation across 1,597 artifacts (caught 2 missing `parent_l1` fields + 1 single-vertex DAG false-positive case in the validator itself)
- `scripts/bridge_eth_sepolia_to_base_sepolia.py` — L1→L2 testnet ETH bridge via OptimismPortal direct send (~3 min L2 credit time)

**3 testnet ETH bridges executed** (deployer `0x0c566f…`): 0.05 ETH each from L1 Sepolia → L2 Base Sepolia.

### 2b — Domain expert red-team review (old Track I) — ⏳ Optional

**Goal:** For high-stakes domains (medical imaging, fluid dynamics, EM, structural), recruit one expert reviewer per domain to validate the AI-authored L1/L2/L3 artifacts. Not blocking — artifacts are now on-chain; errors get caught via the 7-day challenge mechanism + governance Type 3a upgrades.

| Domain | Priority | Effort |
|---|---|---|
| Medical imaging | Critical (UTSW alignment) | 20-25 L1s × 1 reviewer |
| Fluid dynamics | High | 25 L1s × 1 reviewer |
| Electromagnetics | High | 50 L1s × 1 reviewer |
| Structural mechanics | High | 50 L1s × 1 reviewer |
| Geophysics + seismology | Medium | Optional |
| All others | Low | Skip; community review post-registration |

**Total reviewer effort:** ~110 hours total across priority domains.

---

## Track 3 — Web Explorer

**Goal:** Upgrade the existing public PWM explorer at **`explorer.physicsworldmodel.org`** + the public landing page at **`physicsworldmodel.org`** into the **verified AI4Science platform for physics-grounded problems** (per `coordination/PWM_VALUE_FRAMING_2026-05-22.md`). Three subtracks: wallet+login integration (Track 3a; SIWE via RainbowKit + Privy), benchmark-first redesign with AI4Science framing (Track 3b), and full verification-routes parity with the legacy FastAPI site (Track 3c). Phase 1a deliverable (must be LIVE by D9+30, when mining activates per `PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md`).

**Canonical URL stays `explorer.physicsworldmodel.org`.** This is the live, self-hosted (GCP `34.63.169.185`), Docker-bundled (`pwm-explorer:latest`) public read-only window into PWM's on-chain state — and the **Bounty #2 reference implementation** (80,000 PWM) per `pwm-team/coordination/strategy/EXPLORER_PURPOSE.md`. The Track G work lives in `pwm-team/infrastructure/agent-web/frontend/`, which is the same code that already serves `explorer.physicsworldmodel.org`. We are not building a new site — we are shipping a new container to the existing URL.

**Optional staging subdomain `next.physicsworldmodel.org`** can be used during the cutover for low-risk verification before swapping the production container. After verification, the canonical URL stays `explorer.physicsworldmodel.org`; the staging subdomain is deleted.

**Old labels covered:** F (wallet integration) + G (benchmark-first redesign) + H (verification routes port)

**Source plan:** `PWM_WEBSITE_AND_GENESIS_TESTNET_PLAN_2026-05-15.md` §§ F, G, H
**Status docs:** `pwm-team/plan/track_f/PWM_TRACK_F_STATUS_2026-05-15.md`, `pwm-team/plan/PWM_TRACK_G_STATUS_2026-05-15.md`
**Purpose + scope:** `pwm-team/coordination/strategy/EXPLORER_PURPOSE.md`
**Hosting + deploy:** `pwm-team/website/EXPLORER_HOSTING_OPTIONS.md`, `pwm-team/coordination/EXPLORER_REDEPLOY_RUNBOOK_2026-05-02.md`

**Out of scope (separate product):** the SpecLab chat product at `physicsworldmodel.org/speclab` lives in `pwm-team/pwm_product/platform/` (FastAPI + Jinja2). It has a different audience (clinicians, non-developers), a different identity model (web2 auth + file upload + billing), and is not part of Track 3. See `pwm-team/coordination/strategy/AUTH_AND_WALLET_STRATEGY.md`.

### 3a — Wallet integration (old Track F) — ✅ CODE SHIPPED

Wagmi + Privy + SIWE all live in code. All 8 SIWE end-to-end tests pass. `next build` PASS (19 pages + 5 API routes). Commit `b274aaef`.

| Item | Status |
|---|---|
| SIWE end-to-end (Wagmi + RainbowKit + siwe@3) | ✅ Live in code |
| ERC-1271 + EIP-6492 sig verify (smart wallets) | ✅ Live (commit `5d16e82`) |
| Privy `Sign in with Google / email` | ✅ Code live, gracefully hidden when env unset |
| 30-day httpOnly JWT session | ✅ Live |
| `/api/auth/me`, `/api/auth/logout` | ✅ Live |
| Nav LoginButton + WalletBalance | ✅ Live |
| `next build` passes (19 pages, 5 API routes) | ✅ Verified |
| Web3Auth integration | ✅ Per `PWM_WEB3AUTH_EXPLAINED_2026-05-15.md` |
| Coinbase Smart Wallet via ERC-1271 | ✅ Per `PWM_LOGIN_AUTH_STATUS_2026-05-15.md` |
| **Privy app ID** in `.env.local` | ❌ Director task (~30 min) |
| **WalletConnect Project ID** | ❌ Optional, mobile QR (~5 min) |
| **Public deployment target** for Next.js explorer | ❌ Director task (~1 hr; e.g., `next.physicsworldmodel.org`) |

**Director action remaining (3a):**
1. Sign up Privy at privy.io with platformaiyang@gmail.com; create app "PWM"; configure email + Google + Base; whitelist `physicsworldmodel.org`, `explorer.physicsworldmodel.org`, and `localhost:3000`
2. Add `NEXT_PUBLIC_PRIVY_APP_ID` and `PRIVY_APP_SECRET` to `.env.local`
3. Decide deploy strategy: **(a) direct cutover** — rebuild `pwm-explorer:latest` Docker image with Track F/G changes and replace the container at `explorer.physicsworldmodel.org` (1 hr, low risk because the existing site is already read-only and we can roll back to the prior image); OR **(b) staging-first** — deploy to `next.physicsworldmodel.org` first, verify, then swap DNS. Option (a) is simpler and recommended; `next.*` is only needed if you want a parallel verification environment during the transition.

### 3b — Benchmark-first redesign (old Track G) — ✅ Chunks 1+2 shipped

Sidebar nav + `/templates` + task-type filter + hero + benchmark detail Example Data shipped 2026-05-15 (`c456def7`, `379eb51c`). **Track F files untouched** during Track G work.

| Page / Feature | Status |
|---|---|
| Sidebar nav (Browse / Explore / Action / Verify / Meta tiers) | ✅ Shipped — every page, mobile overlay |
| `/` (hero + 3 featured benchmarks + search input + supporting stats) | ✅ Shipped (benchmark-first homepage) |
| `/benchmarks` (filterable grid + task-type chips + chain-status chips) | ✅ Shipped (multi-select task-type filter) |
| `/benchmarks/{hash}` (task-type chips + parent breadcrumbs + Example Data section) | ✅ Shipped (placeholder Example Data) |
| `/benchmarks/{hash}` "Use This Solution" modal + Wagmi-signed submit | ⏳ Deferred — requires Track F deployed in production |
| `/templates` (4 .md templates: L1/L2/L3/L4, download + copy + preview) | ✅ Shipped |
| `/principles` (browse all 529) | ✅ Existing; surfaces full catalog now that Track E populated the registry |
| `/principles/{hash}` (detail page + "Understanding This Principle" educational prose) | ⏳ Deferred — needs domain prose, better authored by `agent-imaging`/`agent-physics` |
| `/specs/{hash}` ("Understanding This Spec") | ⏳ Deferred — same content-authoring constraint |
| `/cert/{hash}` (L4 cert page + challenge button) | ✅ Existing; challenge button gates on Track F |
| `/leaderboard/{hash}` (with task chip + miner badges) | ✅ Existing |
| `/submit` (L4 cert submission) | ⏳ Skeleton; needs Wagmi wiring (Track F deploy) |
| `/mine` (4-path decision tree: solution / benchmark / spec / principle) | ⏳ Deferred — existing `/contribute` page is informal placeholder |
| `/profile/{address}` (wallet page) | ⏳ Deferred — needs Track F deployed |

### 3c — Verification routes port (old Track H) — ⏳ Pending

**Goal:** Move legacy FastAPI verification routes (`/gates`, `/reproduce`, `/redteam`, `/claims`, `/contributors`) from `pwm_platform/` (Jinja2) to the Next.js explorer.

All 5 routes need to work in Next.js with the same data as the legacy FastAPI versions. Can start now that 3a + 3b chunk 1-2 are shipped.

---

## Track 4 — Foundation & Funding

**Goal:** Operational 501(c)(3) status under NumFOCUS umbrella + a calendar-driven grant pipeline (Tier 1 crypto grants → Tier 2 major foundations & federal).

**Old labels covered:** A (NumFOCUS sponsorship) + J (co-founder recruitment) + L (Tier 1 grants) + M (Tier 2 grants)

**Source plans:** `PWM_ACCELERATED_MAINNET_PLAN_2026-05-15.md` § A; `PWM_FUND_EXECUTION_PLAN_2026-05-15.md`; `pwm-team/numfocus/application/CONTRIBUTORS.md` § Section 5

### 4a — Co-founder recruitment (old Track J)

**Hard prerequisite for** sub-tracks 4b (NumFOCUS) and 4d (Tier 2 grants), and for Track 1d (founder rotations).

| Slot | Profile | Where to find | Time estimate |
|---|---|---|---|
| 2 | Computational imaging researcher at non-UTSW university | Conference network; collaborator referrals | 2-4 weeks |
| 3 | Open-source contributor (no academic affiliation) | OSS communities, Discord servers | 4-6 weeks |
| 4 | Crypto/blockchain ecosystem contributor | Base Builder community, Ethereum researcher community | 4-6 weeks |
| 5 (optional) | Domain expert in benchmark methodology | MLcommons, OpenML, RCT communities | 6-8 weeks |

**Outreach template:** `numfocus/application/CONTRIBUTORS.md` § Section 5.
**Excluded:** Director's wife (spousal SFI); NextGen PlatformAI staff (common-affiliation issue per NumFOCUS rules).

### 4b — NumFOCUS sponsorship (old Track A)

**Goal:** Operational 501(c)(3) status under NumFOCUS umbrella; unlocks major foundation grants (Sloan, CZI, Schmidt, Moore).
**Application package:** `pwm-team/numfocus/application/` (11 docs, ready)

| Phase | When | Action |
|---|---|---|
| 1 | 2026-05-15 → 2026-07-01 | Co-founder outreach (covered by sub-track 4a above) |
| 2 | 2026-06-15 | Verify NumFOCUS portal reopened (apps closed until end Q2) |
| 3a | 2026-07-01 (decision point) | If 4+ co-founders confirmed → submit Round 3 (deadline Jul 15; notification Aug 31) |
| 3b | 2026-07-01 (decision point) | If <4 confirmed → defer to Round 4 (deadline Oct 15; notification Nov 30) — RECOMMENDED |
| 4 | At submission | Materials already drafted; copy from `numfocus/application/` |
| 5 | Notification + 30 days | Sign Comprehensive FSA; NumFOCUS opens project bank account |
| 6 | Year 2-3 | Optional graduation to independent PWM Foundation 501(c)(3) |

**Cost:** $0 upfront; 5-15% admin fee on grants only (typically 10%).

### 4c — Tier 1 grants (old Track L) — crypto foundations, solo-eligible

**Goal:** Apply for solo-eligible crypto-foundation grants. No 501(c)(3) required; no co-founders required; no UTSW PI required.
**Locked policy:** No applications until 2026-08-25 (mainnet + 30 days stable).

| Week (post-mainnet) | Funder | URL | Award |
|---|---|---|---|
| Aug 25-31 | Base Builder Grants / Builder Rewards | docs.base.org/get-started/get-funded | 1-5 ETH ($3-15K) |
| Sep 1-7 | Ethereum Foundation Ecosystem Support | esp.ethereum.foundation | $50K-500K |
| Sep 8-14 | 0xPARC | 0xparc.org/grants | $50K-200K |
| Sep 15-21 | Filecoin Foundation grants | fil.org/grants | $100K-1M |
| Sep 22-28 | Gitcoin Grants (next round) | explorer.gitcoin.co | $5-50K |
| Sep 29-Oct 5 | Mantle Foundation | mantle.xyz/grants | $5-100K |
| Oct 6-12 | Optimism RetroPGF (next round) | retrofunding.optimism.io | Variable |

**Realistic Tier 1 outcome by Oct 31:** $190K expected; $400-700K upside.

### 4d — Tier 2 grants (old Track M) — major foundations + federal

**Goal:** Apply for major foundation grants (Sloan, CZI, Schmidt, Moore) + first NIH R21 (with new UTSW PI as named PI).
**Hard prerequisite:** NumFOCUS approval (sub-track 4b) for major foundation grants; Track 5 complete for federal NIH grants.

| Quarter | Funder | Mechanism | Award range |
|---|---|---|---|
| 2026-Q4 | Sloan Reproducibility Initiative | LOI | $250K-1M |
| 2026-Q4 | Schmidt Sciences AI-for-Science | RFP | $500K-5M |
| 2027-Q1 | Moore Foundation imaging program | Invited proposal | $250K-5M |
| 2027-Q1 | Chan Zuckerberg Initiative Science | Application cycle | $250K-5M |
| 2027-Q2 | NIH R21 (with new UTSW PI as named PI; Director as Co-I) | Standard NIH cycle | $275K (Director's Co-I share ~$50-100K) |

**Realistic Tier 2 outcome by Q2 2027:** $300K expected; $700K-$1.5M upside.

---

## Track 5 — Academic Career (UTSW PI Transition)

**Goal:** Find a supportive UTSW PI mentor (NOT Dr. Zaman) so Director can pursue NIH R-series grants as Co-I in Q3 2027.

**Old labels covered:** K

**Source plans:** `pwm-team/funds/PWM_RESEARCH_ASSOCIATE_AND_MENTOR_CONSTRAINTS_2026-05-13.md`; `pwm-team/funds/PWM_PI_TRANSITION_STRATEGY_2026-05-13.md`
**Strategic constraint:** Director is Research Associate, NOT PI-eligible; current PI Dr. Zaman is unsupportive of PWM. Strategic sequence requires NOT disclosing PWM to Dr. Zaman until new mentor is in place.

| Phase | When | Action |
|---|---|---|
| 1 | 2026-08 → 2026-10 | UTSW Conflict of Interest Office disclosure (confidential from Dr. Zaman) |
| 2 | 2026-08 → 2026-10 | Identify 2-3 supportive UTSW faculty as candidate new mentors |
| 3 | 2026-11 → 2026-12 | Initial conversations with candidate PIs; gauge support |
| 4 | 2027-Q1 | Formalize new mentor relationship; PI change paperwork at UTSW |
| 5 | 2027-Q2 | Draft NIH R21 specific aims with new PI |
| 6 | 2027-Q3 | Submit R21 (Jun 16 or Oct 16 deadline) — first available cycle |
| 7 | 2027-Q4 / 2028-Q1 | NIH review cycle; potential funding mid-2028 |

---

## Track 6 — Community

**Goal:** Public discussion forum + Code of Conduct enforcement + quarterly workshops. Required for Track 4 sub-track 4b (NumFOCUS).

**Old labels covered:** N

**Source plans:** `pwm-team/community/`; `numfocus/application/CODE_OF_CONDUCT.md`

| Asset | Status |
|---|---|
| Discord server | ❌ Setup before Track 4 sub-track 4b submission |
| GitHub Discussions on `pwm-public` | ❌ Enable before Track 4 sub-track 4b submission |
| `conduct@physicsworldmodel.org` enforcement contact | ❌ Configure DNS + mail forwarding |
| First quarterly community workshop | Q4 2026 (post-mainnet) |
| Mailing list (low-volume; major announcements) | Optional |

---

## Track 7 — Audit Funding (pre + post deploy)

**Goal:** Apply for grant funding to cover the formal smart-contract audit. Originally blocked by `MAINNET_FIRST_PLAN` policy ("no applications until mainnet + 30 days stable"). **Policy narrowed 2026-05-18** to allow pre-deploy *audit-specific* grant applications. Current operational reality: PWM is on the **soft-launch path** (deploy 2026-05-19/20 with caps; audit at D9+30 when grants land), so Track 7 funds the **post-deploy audit** rather than the pre-deploy audit.

**Old labels covered:** none — new track 2026-05-18

**Source plans:** `pwm-team/funds/PWM_PRE_DEPLOY_AUDIT_FUNDING_OPTIONS_2026-05-17.md` (canonical); `pwm-team/funds/MAINNET_FIRST_PLAN.md` (narrowed)

### 7a — Policy narrowing decision ✅ APPROVED

| Day | Action | Owner | Status |
|---|---|---|---|
| 2026-05-18 | Decision-log entry: narrow `MAINNET_FIRST_PLAN` to permit audit-only grant applications pre-deploy. General operating grants remain deferred to mainnet + 30 days. | Director | ✅ Approved in this PLAN.md update |

### 7b — Base Builder Grant application

| Day | Action | Owner | Time |
|---|---|---|---|
| Week 1 (2026-05-19 → 23) | Draft 5-page proposal + budget; submit to https://base.org/grants | Director (+ Claude drafting support) | 8-10 hr |
| Week 1-5 | Application under review (typical Base turnaround) | Base team | — |
| Week 5+ | If approved: $5-25K disbursed; engage audit firm | Director + Track 1b |

**Best fit:** PWM is Base-native. Coinbase explicitly funds Base ecosystem nonprofits. SECURITY_REVIEW doc (199/199 tests + multi-agent audit + Slither + Mythril) is strong supporting evidence.

### 7c — Ethereum Foundation ESP application

| Day | Action | Owner | Time |
|---|---|---|---|
| Week 1 (2026-05-19 → 23) | Draft detailed proposal + milestones + team; submit to https://esp.ethereum.foundation | Director (+ Claude drafting support) | 12-15 hr |
| Week 1-8 | Application under review (typical ESP turnaround) | ESP committee | — |
| Week 8+ | If approved: $10-50K disbursed; can cover full audit | Director + Track 1b |

### 7d — Audit firm engagement (triggered by 7b OR 7c funding landing)

| Trigger | Action | Audit firm options | Cost target |
|---|---|---|---|
| ≥$15K landed | 2-week scoping engagement | Cantina, Code4rena private contest, Spearbit standard | $15-25K |
| ≥$25K landed | Standard 5-week fieldwork | Spearbit, Zellic, OpenZeppelin standard, Halborn | $25-40K |
| <$15K in 8 weeks | Fall back to bug bounty + soft-launch caps maintained | — | $0 |

### 7e — Post-audit cap raise

| Trigger | Action | Owner |
|---|---|---|
| Audit complete + all HIGH/MED resolved | Governance proposes `setMaxTotalStakeWei($10K equiv)`, `setMintingPaused(false)`, `setTransfersPaused(false)` via executeCall | PWMGovernance multisig |
| 48 h timelock × 3-of-5 approval × 3 proposals | Sequential execution; same addresses become production PWM | Governance |
| Announcement | "PWM audit complete; soft-launch ends; full protocol live" | Director + community |

**Realistic timeline (current path):** Mainnet 2026-05-20 → grants land Q3 2026 → audit Aug-Oct 2026 → cap raise Sep-Nov 2026.

---

## Track 8 — Paper Implementation

**Goal:** Track empirical and engineering work supporting active PWM-authored research papers. Each sub-track maps to one paper folder under `pwm-team/papers/` or `papers/` and tracks its venue + timing.

**Old labels covered:** none — new track 2026-05-18

**Source plans:** each paper folder has its own `plan.md` / `README.md`. This track is the master timeline view.

**Note:** Sub-tracks ordered by **estimated completion date**, earliest first. Status reflects most recent activity per folder.

### 8a — InverseNet ECCV paper (CASSI validation) — **2026-Q3** (most advanced)

**Folder:** `papers/inversenet/`
**Plan doc:** `papers/inversenet/plan.md` (InverseNet Paper Polish Plan — ECCV 2026)
**Target venue:** ECCV 2026 (resubmission) or NeurIPS 2026
**Status:** Implementation complete (`IMPLEMENTATION_STATUS.md` shows all production-ready as of 2026-02-15); 15+ deliverable docs; SPC + CACTI solvers shipped; `eccv_resubmission/` + `before_submission/` + `arxiv_submission/` folders indicate multi-cycle
**Remaining work:** Polish + resubmit. Visual reconstruction examples (G1 gap). Supplementary material (G7 gap).
**Estimated completion:** Submission **2026-Q3** (June-September); decision Q4 2026 / Q1 2027

### 8b — PWMI-CASSI ECCV paper (Differentiable Calibration) — **2026-Q3**

**Folder:** `papers/pwmi_cassi/`
**Plan doc:** `papers/pwmi_cassi/README.md`
**Target venue:** ECCV 2026 (resubmission) or follow-up venue
**Status:** Full pipeline ready (`scripts/run_all.sh` runs in ~4-6 hr on a single GPU). Validation, sensitivity sweep, ablation, figures, tables all scripted.
**Key innovation (paper claim):** Straight-Through Estimator (STE) for integer dispersion offsets enables gradient-based optimization through the CASSI forward model. Recovers -16 dB MST-L drop from operator mismatch.
**Remaining work:** Final results in paper text; figure quality pass; venue selection.
**Estimated completion:** Submission **2026-Q3**; decision Q4 2026

### 8c — Universal Simulation Framework (Judge Agent) — **2026-Q3 to Q4**

**Folder:** `papers/universal_simulation/`
**Plan doc:** `papers/universal_simulation/presubmission_enquiry.md` (Nature presubmission already sent)
**Target venue:** **Nature** (presubmission enquiry sent)
**Key claim:** Judge Agent validation reduces LLM-generated scientific simulation silent-failure rate from 42% → 1.5% across 12 scientific domains. 72 blinded prospective tasks from 12 independent scientists / 9 countries. CT reconstruction subtask reaches 99% expert quality.
**Status:** Pre-submission enquiry sent; awaiting Nature editor response.
**Remaining work:** If Nature accepts presubmission → 2-4 weeks to assemble full submission. Otherwise reframe for Science / PNAS / Nature Communications.
**Estimated completion:** Full submission **2026-Q3 to Q4**; decision Q4 2026 / Q1 2027

### 8d — PWM Flagship Nature paper (Eleven Primitives + Three Gates) — **2026-Q4 to 2027-Q1**

**Folder:** `papers/pwm_flagship/`
**Plan doc:** `papers/pwm_flagship/plan.md` (Restructuring Plan v3)
**Target venue:** **Nature**
**Status:** Draft + identified review weaknesses (`nature_review_weaknesses.md`) — major revisions needed:
  - Real hardware mask perturbation (currently software-simulated)
  - Comparison against state-of-the-art calibration (ESPIRiT, ptychography, etc.)
  - Independent peer-reviewed validation (Track 8a InverseNet must publish first OR be folded in)
  - Visual evidence figures (mismatch artifacts before/after)
  - Co-author breadth (currently 2 NextGen + 1 university)
**Hard prerequisite:** Track 8a (InverseNet) must reach peer-review status to be citable; otherwise self-citation is circular
**Estimated completion:** Submission **2026-Q4 to 2027-Q1**; decision 2027-Q2

### 8e — Finite Primitive Basis Theorem — **2026-Q4** (companion to 8d)

**Folder:** `papers/finite_primitive_theorem/`
**Plan doc:** none (only `main.tex` + supplementary)
**Target venue:** Companion to Track 8d Nature submission (Supplementary Note) OR standalone math venue
**Status:** Draft in `main.tex` + `supplementary.tex` + `preamble.tex`
**Key claim:** Every imaging forward model in a broad operator class can be decomposed into a DAG of exactly 10 canonical primitives.
**Estimated completion:** **2026-Q4** (alongside or as part of flagship submission)

### 8f — RAIL paper — **2026-Q4 to 2027-Q1**

**Folder:** `papers/rail/`
**Plan doc:** none visible; `main.tex` + bibliography exist
**Target venue:** TBD
**Status:** Draft in `main.tex`; compile log indicates active work
**Estimated completion:** TBD — Director to confirm

### 8g — System Design (Prospective Algorithm-Comparison Expert Study) — **2027-Q1**

**Folder:** `papers/system_design/`
**Plan doc:** `papers/system_design/USAGE_DEMO.md`
**Target venue:** TBD (likely a methods journal)
**Status:** Demo/usage docs; appears to be later-stage work
**Estimated completion:** **2027-Q1**

### 8h — CT QC Platform paper revision — **2027-Q1 to Q2** (gated on intern empirical work)

**Folder:** `papers/ct_qc_platform/`
**Plan doc:** main TeX file; intern workplan at `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md`
**Target venue:** Med Phys / Radiology AI / IEEE TMI
**Status:** Late-stage paper revision; intern Heyang Zhao's 16-week baseline reproductions (Summer/Fall 2026) feed the comparison evidence
**Hard prerequisite:** Intern's reproductions complete (Track 8h depends on intern work weeks 1-12)
**Estimated completion:** **2027-Q1 to Q2** (after intern delivers full baselines + Director writes comparison sections)

### 8i — Track 9 Low-Dose CT (PWM Grand Challenge) (= Track B in two-track strategy) — **2027-Q2 to Q3** (future)

**Folder:** `papers/track_9_low_dose_ct/` (not yet created)
**Plan doc:** `plan/track_9/PWM_TRACK_9_LOW_DOSE_CT_2026-05-16.md` (revised 2026-05-22 with two-track sequencing); intern workplan
**Target venue:** **Nature Scientific Data**
**Status:** Future — depends on:
  - Mainnet deploy + ≥6 months of protocol activity (live miners + submissions to benchmark)
  - **NEW (2026-05-22) sequencing gate: PWM-CI-1 (Track A) must hit 30+ external submissions by D9+180 before Track 9 (Track B) activates clinical infrastructure** (Track K mentor commitment, IRB filing, hospital partner recruitment)
  - Intern's baseline + grand-challenge dataset construction
  - Co-author recruitment (Track 4a)
**Estimated completion:** **2027-Q2 to Q3** (unchanged; RSNA / ISBI 2028 target also unchanged)
**Cross-reference:** `coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` (two-track strategy); `coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` (Phase 2 activation gate)

---

## Paper Submission Timeline (Track 8 summary)

```
Quarter:    2026-Q3   2026-Q4   2027-Q1   2027-Q2   2027-Q3
8a InverseNet:       ████►submit                                  (ECCV/NeurIPS)
8b PWMI-CASSI:       ████►submit                                  (ECCV)
8c UniversalSim:     ████►full submit after presub                (Nature)
8d Flagship:                 ████►submit (after 8a clears review) (Nature)
8e Finite Theorem:           ████►submit (companion to 8d)        (Nature SI / math)
8f RAIL:                     ████ TBD                              (TBD)
8g System Design:                  ████►submit                    (methods journal)
8h CT QC Platform:                 ████►submit (after intern)     (Med Phys/Rad AI)
8i Track 9 Grand Challenge:                ████►submit            (Nature Sci Data)
```

**3 papers in Q3 2026** (the "implementation-complete" cohort: 8a, 8b, 8c)
**3 papers in Q4 2026 / Q1 2027** (8d flagship + 8e companion + 8f RAIL)
**2 papers in Q1-Q2 2027** (8g system design + 8h CT QC after intern delivers)
**1 paper in Q2-Q3 2027** (8i grand challenge — depends on protocol live data)

---

## Mainnet Deploy Day Gate Checklist (2026-07-24)

All seven must be GREEN before deploy transaction is signed. From `PWM_ACCELERATED_MAINNET_PLAN_2026-05-15.md` (revised 2026-05-15):

| # | Gate | Verification |
|---|---|---|
| 1 | All 8 contracts compiled and verified-ready | `forge build` + verified ABIs in `contracts_abi/` |
| 2 | PWMGovernance multisig owns admin keys (NOT Director EOA) | `addresses.json` shows multisig as owner. **Path A bootstrap interim:** all 5 signing keys controlled by Director until rotations complete. Honest disclosure in announcement materials. |
| 3 | ≥7 founder mining transactions queued | Test transactions ready to fire in first hour post-deploy |
| 4 | ≥1 external (non-founder) wallet ready to interact | Coordinated with at least one external tester |
| 5 | Audit delta closed with zero open High/Medium findings | Auditor sign-off doc in `pwm-team/funds/AUDIT_DELTA_V1.1.0.md` |
| 6 | $10K TVL cap intact and contract-enforced | Verified in `PWMStaking.sol` test suite |
| 7 | ~~NumFOCUS application submitted~~ | **Removed from gate.** NumFOCUS targets Round 4 (Oct 15) per Track A revision; mainnet proceeds without NumFOCUS dependency. |

---

## Director's Action Checklist (rolling)

Total time: ~8-10 hours of Director attention. Tracks all run in parallel. Items shipped at the time this checklist was last revised are marked with ✅ for traceability.

### Already completed (do not redo)

| # | Action | Track | Evidence |
|---|---|---|---|
| ✅ | Genesis batch registration (Batches 1-10, 1,597/1,597) | 2a | commit `b2d0488c` |
| ✅ | Bridge testnet ETH from L1 Sepolia → L2 Base Sepolia (3 bridges, 0.15 ETH total) | 2a | `scripts/bridge_eth_sepolia_to_base_sepolia.py`, txs in batch_results/ |
| ✅ | Web Explorer chunks 1+2 (sidebar, /templates, task-type filter, homepage redesign) | 3b | commits `c456def7`, `379eb51c` |
| ✅ | Smart Contract code (PWMToken stack + ERC20 siblings) | 1a | commit `d558afa0` |
| ✅ | Wallet integration code (Wagmi + Privy + SIWE) | 3a | commit `b274aaef` |

### Open — top priority (this week)

| # | Action | Time | Track |
|---|---|---|---|
| 1 | Run `deploy/erc20.js --network baseSepolia` (with env vars `FOUNDER_ADDRESSES`, `PWM_RESERVE_MULTISIG`, `PWM_LIQUIDITY_ADDR`, `PWM_TEAM_BENEFICIARY` set) | 1 hr | 1a |
| 2 | Email audit firm about delta scope; request 4-week turnaround | 1 hr | 1b |
| 3 | Fund Base mainnet deployer `0xA5349f9E…0f325` with ~0.05 ETH (real ETH, not testnet) | 30 min | 1a/1b |
| 4 | Verify all 5 hardware wallets respond and addresses match expected EIP-55 checksums | 30 min | 1c |
| 5 | Sign up Privy at privy.io; create app "PWM"; configure email + Google + Base; whitelist `physicsworldmodel.org` + `localhost:3000` | 30 min | 3a |
| 6 | Add `NEXT_PUBLIC_PRIVY_APP_ID` + `PRIVY_APP_SECRET` to `.env.local` (see `pwm-team/plan/track_f/PWM_TRACK_F_ENV_FILES_2026-05-15.md`) | 10 min | 3a |
| 7 | Decide deploy strategy: **direct cutover** (replace the running container at `explorer.physicsworldmodel.org`) OR **staging first** (deploy to `next.physicsworldmodel.org`, verify, then DNS-swap). See `pwm-team/coordination/EXPLORER_REDEPLOY_RUNBOOK_2026-05-02.md`. | 1 hr | 3a |

### Open — parallel / no rush

| # | Action | Time | Track |
|---|---|---|---|
| 8 | Begin co-founder recruitment outreach (template in `numfocus/application/CONTRIBUTORS.md` § Section 5); send to 8-10 candidates | 2 hrs | 4a |
| 9 | Verify NumFOCUS application portal status (target reopen ~Jun 30) | 5 min | 4b |
| 10 | Set up funding-applications tracking spreadsheet | 30 min | 4c |
| 11 | Schedule UTSW Conflict of Interest Office disclosure meeting (confidential from Dr. Zaman) | 30 min | 5 |
| 12 | Identify 2-3 candidate new UTSW PI mentors | 1 hr | 5 |
| 13 | Set up Discord server + GitHub Discussions on `pwm-public` (preparation) | 1 hr | 6 |
| 14 | Pre-write Base Builder Grants + Ethereum Foundation ESP application drafts (don't submit yet — wait for mainnet + 30 days per locked policy) | 3 hrs | 4c |

---

## Decision Points by Date

| Date | Decision | Owner | Default if no decision |
|---|---|---|---|
| 2026-06-15 | Verify NumFOCUS portal reopened | Director | Defer to Round 4 (Oct 15) |
| 2026-07-01 | Round 3 vs Round 4 of NumFOCUS | Director | Round 4 (recommended) |
| 2026-07-15 | NumFOCUS Round 3 deadline (if applying) | Director | Skip; Round 4 |
| 2026-07-24 | Mainnet deploy go/no-go (all 6 gates GREEN) | Director | Slip 1-2 weeks |
| 2026-08-24 | First crypto-foundation grant application | Director | Per locked policy: do not apply earlier |
| 2026-08-31 | NumFOCUS Round 3 notification (if Round 3 was submitted) | NumFOCUS | Apply Round 4 |
| 2026-10-15 | NumFOCUS Round 4 deadline | Director | Apply Round 1 of 2027 |
| 2026-11-30 | NumFOCUS Round 4 notification | NumFOCUS | Pursue alternate sponsor |
| 2027-Q1 | Begin DAO contract development (~$200K from Reserve) | Director + multisig | Defer activateDAO |
| 2027-Q2 | First NIH R21 submission with new PI (Track K complete) | Director + new PI | Defer NIH path |

---

## Cumulative Funding Projection (Tracks L + M)

| Date | Cumulative expected | Cumulative upside |
|---|---|---|
| 2026-10-31 (end Track L sprint) | ~$190K | $400-700K |
| 2027-04-30 (end Track M Phase) | ~$490K | $1.1-2.2M |
| 2027-12-31 (Track M ramping) | ~$700K | $1.5-3M |

Easily clears NumFOCUS graduation threshold ($50K accumulated revenue) by end Q4 2026 → enables independent 501(c)(3) PWM Foundation planning Q1 2027 (per `funds/PWM_INSTITUTIONAL_LIFECYCLE_2026-05-13.md` Phase 1 → Phase 2).

---

## Cross-References to Source Plans

This master plan integrates the following authoritative sub-plans. When a discrepancy arises, the sub-plan is authoritative for its track:

| New track | Source plan(s) |
|---|---|
| **1** Smart Contracts & Governance (1a, 1b, 1c, 1d) | `pwm-team/plan/PWM_ACCELERATED_MAINNET_PLAN_2026-05-15.md` (sections B, B+, C, D) |
| **2** Genesis Content (2a, 2b) | `pwm-team/plan/PWM_WEBSITE_AND_GENESIS_TESTNET_PLAN_2026-05-15.md` (sections E, I) |
| **2** ops tooling | `scripts/bridge_eth_sepolia_to_base_sepolia.py`, `scripts/validate_forward_models.py`, `pwm-team/infrastructure/agent-contracts/scripts/register_batch.py`, `pwm-team/infrastructure/agent-contracts/scripts/batches/` (manifests for all 34 sub-batches) |
| **3** Web Explorer (3a, 3b, 3c) | `pwm-team/plan/PWM_WEBSITE_AND_GENESIS_TESTNET_PLAN_2026-05-15.md` (sections F, G, H); `pwm-team/plan/track_f/PWM_TRACK_F_STATUS_2026-05-15.md` + `PWM_TRACK_F_DIRECTOR_GUIDE_2026-05-15.md` + `PWM_TRACK_F_ENV_FILES_2026-05-15.md`; `pwm-team/plan/PWM_TRACK_G_STATUS_2026-05-15.md` |
| **4** Foundation & Funding (4a-4d) | `PWM_ACCELERATED_MAINNET_PLAN_2026-05-15.md` § A; `pwm-team/plan/PWM_FUND_EXECUTION_PLAN_2026-05-15.md`; `pwm-team/numfocus/application/CONTRIBUTORS.md` § Section 5 |
| **5** Academic Career | `pwm-team/funds/PWM_RESEARCH_ASSOCIATE_AND_MENTOR_CONSTRAINTS_2026-05-13.md`; `pwm-team/funds/PWM_PI_TRANSITION_STRATEGY_2026-05-13.md` |
| **6** Community | `pwm-team/community/`; `pwm-team/numfocus/application/CODE_OF_CONDUCT.md` |
| Path A decision | `pwm-team/coordination/wallet/PWM_DECISION_RECORD_PATH_A_2026-05-12.md` |
| Founder rotation mechanism | `pwm-team/coordination/wallet/PWM_FOUNDER_ROTATION_2026-05-10.md` |
| Phase 5 deploy day procedure | `pwm-team/coordination/wallet/PWM_PHASE_5_DEPLOY_DAY_2026-05-09.md` |
| Phase 7 routine ops | `pwm-team/coordination/wallet/PWM_PHASE_7_ROUTINE_OPERATIONS_2026-05-09.md` |
| Mainnet-first locked policy | `pwm-team/funds/MAINNET_FIRST_PLAN.md` |
| Master strategy framework | `pwm-team/funds/MASTER_PLAN.md` |
| Pure non-profit recommended path | `pwm-team/funds/PWM_PURE_NONPROFIT_TENURE_STRATEGY_2026-05-13.md` |
| Institutional lifecycle | `pwm-team/numfocus/PWM_INSTITUTIONAL_LIFECYCLE_2026-05-13.md` |
| NumFOCUS application package | `pwm-team/numfocus/application/` (11 files) |
| Long-term vision | `pwm-team/long-term-vision/` (12 files) |
| Solo-founder funding plan | `pwm-team/numfocus/application/SOLO_FOUNDER_FUNDING_PLAN.md` |

---

## Risk-Adjusted Summary

| Scenario | Probability | Mainnet date | Year-1 funding |
|---|---|---|---|
| Best case (all tracks on schedule, 3+ Tier 1 grants land) | 25% | 2026-07-24 | $400-700K |
| Realistic case (audit slips 1-2 wks, 2 Tier 1 grants land) | 50% | 2026-08-07 → 2026-08-21 | $190-300K |
| Worst case (1 High audit finding requires rewrite, 1 Tier 1 grant lands) | 20% | 2026-09-04 | $50-150K |
| Disaster case (NumFOCUS rejects + audit fails + no grants Year 1) | 5% | Q4 2026 fallback | $0-50K (donations only) |

**All four scenarios cover Year-1 operating costs** (hosting, audits, conference travel, occasional bounties — Director's UTSW salary covers personal income). Even disaster case enables protocol operation; only major foundation grants (Track M) and NumFOCUS-dependent activities are gated on better outcomes.

---

## What This Plan Does NOT Cover

| Topic | Where it lives |
|---|---|
| Smart contract source code + tests | `pwm-team/infrastructure/agent-contracts/` |
| Web frontend code | `pwm-team/infrastructure/agent-web/` |
| Reference miner | `pwm-team/infrastructure/agent-miner/` |
| Indexer / API | `pwm-team/infrastructure/agent-coord/` (interfaces, indexer) |
| Genesis benchmark data | `pwm-team/pwm_product/genesis/` |
| Customer-facing guide content | `pwm-team/customer_guide/`, `pwm-team/content/` |
| Bounty descriptions | `pwm-team/bounties/` |
| Per-phase runbooks (deploy day, etc.) | `pwm-team/coordination/wallet/` |
| 21 funding strategy docs | `pwm-team/funds/` |
| Token economy detail | `pwm-team/community/` |
| 12 long-term vision docs | `pwm-team/long-term-vision/` |
| Website design specs | `pwm-team/website/` |

This plan is the **calendar layer** — the scheduling and gate-sequencing of work. The **technical layer** lives in the `infrastructure/` folder; the **strategy layer** lives in `funds/` and `long-term-vision/`. Each layer is independently authoritative for its scope.

---

## Post-Mainnet Strategic Integration (added 2026-05-22)

Five new strategic docs in `coordination/` define the post-mainnet sequence. They integrate the existing Track 1 (Smart Contracts) + Track 3 (Web Explorer) deliverables with the user-acquisition / value-framing / phased-mining strategy. Director questions 2026-05-22 ("how to attract users", "what's PWM's value", "when does mining activate", "is login part of wallet") are answered by these docs.

### Strategic doc index

| Doc | Purpose |
|---|---|
| [`coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md`](../coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md) | Two-track demand-side strategy (Track A computational imaging fast wedge + Track B medical imaging flagship). First user = miner submitting AI4Science methods. Per Director: "Benchmark first. Users second. Verified jobs third. Infrastructure fourth. Token value last." |
| [`coordination/PWM_VALUE_FRAMING_2026-05-22.md`](../coordination/PWM_VALUE_FRAMING_2026-05-22.md) | PWM's value framing: "verified AI4Science platform for physics-grounded problems." AI4Science is the WHAT (product); verification is the HOW (moat). Lead with AI4Science. |
| [`coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md`](../coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md) | Phased deployment: Phase 1a (audit, no mining) → Phase 1b (mining ACTIVE D9+30) → Phase 2 (LP, multi-benchmark) → Phase 3 (mine-to-use; users pay) → Phase 4 (industry scale). Login = wallet integration in PWM (SIWE; not separate). |
| [`coordination/PWM_TOKEN_UTILITY_AND_VALUE_2026-05-22.md`](../coordination/PWM_TOKEN_UTILITY_AND_VALUE_2026-05-22.md) | 6 value drivers + 8 use cases. Genesis miners NEVER pay; users pay starting Phase 3 for advanced verified runs only. Avoids Helium failure pattern. |
| [`coordination/PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md`](../coordination/PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md) | Implementation-ready landing page copy for D9. Sub-GPU implements in `pwm_product/platform/pwm_platform/`. 14 sections including hero, PWM-CI-1 spec, FAQ, audience-specific copy variations. |

### Phase-deliverable mapping (cross-references existing tracks + new deliverable)

| Phase | Window | Deliverables | Track / Owner |
|---|---|---|---|
| **Phase 1a** | D9 to D9+30 (audit window) | Token system deployed (all 9 contracts live); soft-launch caps active (`MINTING_PAUSED=true`); wallet+login integration LIVE (SIWE via RainbowKit + Privy); landing page at physicsworldmodel.org LIVE; PWM-CI-1 spec published; whitelisted-only submissions | Track 1 (contracts ✅) + Track 3 (website, sub-GPU + Bounty 2 reference) |
| **Phase 1b** | D9+30 to D9+180 (mining active) | Mining ACTIVATES (`MINTING_PAUSED=false`); full ranked-draw rewards from `PWMMintingERC20` + 10K Reserve sponsor bonus for PWM-CI-1 top-3; 5-30 external AI4Science submissions; arXiv companion paper published; Bounty 9 (MCP) + Bounty 10 (Mobile) ship by D9+60 | NEW: PWM-CI-1 launch deliverable (Director + Heyang + sub-GPU); Track 3 (mobile + MCP integration via bounties) |
| **Phase 2** | Months 6-12 | LP seeded (Uniswap v3 PWM/USDC, 1.05M PWM + $50K); PWM-CI-2 (compressed sensing) + PWM-CI-3 (spectral imaging) + PWM-MED-1 (mini low-dose CT, public data) launched; T_k royalty distributions; Bounty 5 + Bounty 6 open | NEW: multi-benchmark expansion; Track 3 + Track 4 |
| **Phase 3** | Months 12-24 | Mine-to-use activates (users pay PWM for advanced verified runs: large-scale verification, private benchmarks, clinical-grade); RSNA / ISBI 2028 pre-launch; Foundation 501(c)(3) operational | NEW: Phase 3 demand-side activation (Director + Foundation board) |
| **Phase 4** | Year 2-3+ | Full Zeno emission; 50+ benchmarks across CI + MED + other physics domains; industry-standard verification venue; AI/data partnerships generate API traffic | NEW: scale phase |

### Key revisions to existing track plans (per the 2026-05-22 strategic docs)

Status: 3 of 7 revisions IMPLEMENTED in source docs as of commit `90c97a6b` (2026-05-22). Others noted for sub-GPU / Director execution.

1. **Track 3 (Web Explorer) hero framing** ✅ **DONE 2026-05-22** (commit `90c97a6b`) — Track 3 goal statement in §Track 3 below updated from "Verified benchmarks for physics-grounded AI" → "Verified AI4Science platform for physics-grounded problems." Phase 1a deliverable flag added (LIVE by D9+30). Cross-references `PWM_VALUE_FRAMING` + `PWM_PHASED_ARCHITECTURE_DEPLOYMENT`.

2. **Track 1 (Smart Contracts) Phase 5 deploy** ✅ NO CHANGE NEEDED — soft-launch caps already protect Phase 1a window per `PHASES_1_7_FAST_TRACK_2026-05-16.md` Step 9.5 kill-switch rehearsal. Existing Track 1 plan unchanged.

3. **Mining activation timing** ✅ DOCUMENTED in `PWM_PHASED_ARCHITECTURE_DEPLOYMENT` §4.0 — Originally framed as Phase 2 activation; revised to Phase 1b (D9+30) per terminology correction (submitter = miner; mining IS the production of AI4Science). Director's directive 2026-05-22. Operational impact: at D9+30, sub-GPU + Director unpause `MINTING_PAUSED` so miners earn full ranked-draw rewards from `PWMMintingERC20` (not just the 10K Reserve sponsor bonus).

4. **Track 9 (Low-Dose CT) = Track B sequencing** ✅ **DONE 2026-05-22** (commit `90c97a6b`) — `plan/track_9/PWM_TRACK_9_LOW_DOSE_CT_2026-05-16.md` title updated to "Track 9 (= Track B in two-track strategy)"; status changed from "Months 1-24 post-mainnet" to "Months 6-24 post-mainnet"; new "Two-track sequencing" section added with the gating rule (Track 9 holds clinical infrastructure investment UNTIL PWM-CI-1 hits 30+ submissions at D9+180). RSNA / ISBI 2028 grand challenge scope UNCHANGED.

5. **New deliverable: PWM-CI-1 launch** ⏳ PROPOSED (not yet in PLAN.md as a numbered track) — CASSI reconstruction benchmark, Phase 1b. Owner: Director + Heyang (weeks 10-16 of intern workplan, optionally re-scoped per `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md` Scenario Y); sub-GPU for website infrastructure. Spec PDF + arXiv companion paper drafting begins D9+0. **Future PLAN.md revision should add this as a numbered track (e.g., "Track 9b: PWM-CI-1 launch" or "Track 10: AI4Science Launch") when Director approves the user acquisition strategy decisions in `PWM_USER_ACQUISITION_STRATEGY` §10.**

6. **Heyang workplan optional re-scope** ✅ **DONE 2026-05-22** (commit `90c97a6b`) — `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md` "Optional weeks 10-16 re-scope" section added; Director chooses Scenario X (unchanged) vs Scenario Y (re-scope to PWM-CI-1 launch support) by D9+45 mid-internship checkpoint.

7. **Login system clarification** ✅ DOCUMENTED in `PWM_PHASED_ARCHITECTURE_DEPLOYMENT` §4.0 — Login = wallet integration in PWM (SIWE; not a separate system). Confirms Director's 2026-05-21 decision to reject a separate Login bounty. No PLAN.md change needed since Track 3a already covers Wagmi + Privy + SIWE wallet integration.

8. **User payment timing** ✅ DOCUMENTED in `PWM_PHASED_ARCHITECTURE_DEPLOYMENT` §4.0 + `PWM_TOKEN_UTILITY_AND_VALUE` §3.0 — Phase 1-2 users pay nothing. Phase 3+ users pay PWM only for advanced verified runs (large-scale verification, private benchmarks, clinical-grade). Basic access (browse, download MIT methods, cite cert hashes) always FREE.

### Cross-references back to PLAN.md tracks

- Wallet integration (Track 3a) provides the login + wallet infrastructure for Phase 1a
- Web Explorer redesign (Track 3b) provides the landing page + leaderboard infrastructure for Phase 1a
- Verification routes port (Track 3c) provides the verification UI for Phase 1b mining flow
- Smart Contracts (Track 1) provides the on-chain token + minting + reward distribution infrastructure
- Bounty 9 (MCP server) extends the AI4Science platform to AI-assistant queries (Phase 1b)
- Bounty 10 (Mobile UX) extends the AI4Science platform to mobile users (Phase 1b)
- Track 4 (Foundation & Funding) provides the institutional credibility for Group B + C user acquisition

---

## TL;DR

| Question | Answer |
|---|---|
| When does mainnet deploy? | ✅ **DEPLOYED 2026-05-22T18:52:09Z** on Base mainnet (chainId 8453). Path-A soft-launch caps active until D9+30 (~2026-06-21). See `deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md`. |
| What's the critical path? | ✅ ALL DONE 2026-05-22. Track 1a (188/188) → 1b multi-agent review → 1c Sepolia rehearsals → Phase 4 preflight → Phase 5 deploy (5.1-5.4c all GREEN) → 5.7 deploy log committed → independent verification 10/10 PASS. Next: 5.5 explorer cutover + 5.6 announcements (post-launch housekeeping). Paid audit at D9+30 via Track 7. |
| What's already complete? | Tracks 1a (contract code), 2a (genesis registration), 3a (wallet code), 3b chunks 1-2 (web redesign) — see "Recent Progress" section above |
| What does Director need to do this week (~8-10 hours)? | See § Director's Action Checklist above (top-priority items: ERC20 mainnet deploy script, audit firm engagement, Privy + DNS for Track F) |
| Why no co-founders by mainnet? | Path A bootstrap: Director holds all 5 keys at deploy; rotations onboard real co-founders Months 1-6 post-mainnet |
| When can grants start being applied for? | 2026-08-25 (mainnet + 30 days stable per locked policy) |
| When does NumFOCUS approval land? | Round 3 → Aug 31 (if 4 co-founders by Jul 1); Round 4 → Nov 30 (recommended) |
| Realistic Year-1 funding outcome? | $190-300K (50% probability); $400-700K upside |
| When does Director need new UTSW PI? | Q1 2027 to enable Q3 2027 NIH R21 submission |
| When does PWM graduate from NumFOCUS to own Foundation? | Year 2-3 (when accumulated revenue >$50K, target Q1 2027) |
| Are the genesis artifacts live? | **Testnet: Yes — 1,591 stub-tier + 6 founder-vetted on Base Sepolia.** **Mainnet: ✅ YES (as of 2026-05-22T18:52:09Z) — 6 v3-verified artifacts on Base mainnet (CASSI L1-025 + CACTI L1-027 + L2/L3 children) via `scripts/batches_genesis_curated/batch_00_cassi_cacti.json`. On-chain `exists()=true` verified.** Other 1,591 stubs stay on testnet until polished to v3 (Bounty 7). See `coordination/PWM_GENESIS_PRINCIPLES_REALITY_CHECK_2026-05-22.md` + `deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md`. |
| Single source of truth for current sequencing? | This document. Sub-plans in same folder are authoritative for their tracks. |

---

*Last revised: 2026-05-16. This master plan is updated whenever a sub-plan changes. Review monthly; revise dates as actuals diverge from estimates.*
*Revision 2026-05-16 (consolidation): Reduced 14 tracks (A-N) to 6 unified tracks (1-6). New numbering — Track 1 Smart Contracts & Governance (B+B++C+D), Track 2 Genesis Content (E+I), Track 3 Web Explorer (F+G+H), Track 4 Foundation & Funding (A+J+L+M), Track 5 Academic Career (K), Track 6 Community (N). Old labels remain valid in source plans; sub-track suffixes (1a/1b/1c/1d, etc.) preserve the granularity.*
*Revision 2026-05-16 (earlier): Marked old Track E COMPLETE (1,597/1,597); marked old Track G chunks 1-2 SHIPPED; added "Recent Progress" section; refreshed Director's Action Checklist (now distinguishes Already-Complete from Open items); added cross-refs for new ops tooling (`scripts/bridge_eth_sepolia_to_base_sepolia.py`, `scripts/validate_forward_models.py`, `register_batch.py`).*
