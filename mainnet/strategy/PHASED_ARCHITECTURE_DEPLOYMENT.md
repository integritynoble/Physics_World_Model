# PWM Phased Architecture Deployment — Canonical Reference

**Date:** 2026-05-22
**Audience:** Director + sub-GPU + bounty winners + co-founders + grant reviewers
**Status:** Canonical reference for what ships at each phase of PWM mainnet deployment
**Purpose:** Resolves Director's 2026-05-22 strategic question: *"Is PWM token mining necessary at launch? Should users mine first to use solutions? Can we launch with Solution + Mismatch only (no Imaging Design / Sci-Simu)?"*

This doc complements:
- `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` — WHO uses PWM (two-track strategy)
- `PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` — HOW PWM is marketed at launch
- `PWM_DEVELOPER_COMPENSATION_2026-05-22.md` — WHO gets paid (Layer 1/2/3 framework)

This doc answers: **WHAT architectural surface ships at each phase, and WHY this sequencing avoids the Helium pattern.**

---

## TL;DR

1. **Token mining is NOT required at launch.** The token system is already built (`PWMMintingERC20`, `PWMRewardERC20`, `PWMStakingERC20`, `PWMTreasuryERC20` exist in `infrastructure/agent-contracts/contracts/`). The question is what to **FEATURE in the launch narrative**, not what to build.

2. **Director's "mine-first-to-use" mechanism is good design but wrong timing.** It's a Phase 3 mechanism (Year 2+), not a Phase 1 launch mechanism. Forcing mining-before-use at launch inverts the trust direction and recreates the Helium failure pattern.

3. **Director's Phase 1 simplification is correct: Solution + Mismatch only.** Imaging Design (L3-001) + Sci-Simu (L3-002) layers are deferred to Phase 2/3. The v3 anchors already include hardware + simulation methodology embedded in the principle definition; researchers DON'T need to submit new imaging designs.

   **CORRECTION 2026-05-22 (Option A):** Earlier framing in this doc references "30 v3 anchors at launch." Reality check (`coordination/PWM_GENESIS_PRINCIPLES_REALITY_CHECK_2026-05-22.md`) confirms only **2 v3-polished anchors exist as committed files: CASSI (L1-003) + CACTI (L1-004)**. Director's Option A decision 2026-05-22: deploy only these 2 at mainnet D9. References to "30 v3 anchors" elsewhere in this doc reflect aspirational long-term scope, NOT current launch state.

4. **"AI4Science" is a useful rename for the Solution layer in MARKETING ONLY.** Internal contracts + code still call it "Solution" (L3-003). External materials call it "AI4Science" — recognizable to the academic ML community, less generic than "Solution."

5. **Three-phase deployment sequence:**
   - **Phase 1 (Months 1-6):** Audit window (1a: D9 to D9+30, no mining); then mining ACTIVE (1b: D9+30 to D9+180, full ranked-draw rewards + Reserve sponsor bonus for PWM-CI-1 top-3). AI4Science + Mismatch only; no user payment.
   - **Phase 2 (Months 6-12):** VERIFY to EARN — add mining for reproduction; LP seeded; token has price discovery; small entry stake (1-10 PWM, refundable)
   - **Phase 3 (Year 2+):** MINE to USE — Director's proposed mechanism; users pay PWM to run inference on own data; full mine-to-use economics

6. **Sequencing rationale:** Each phase de-risks the next. Skipping phases = Helium pattern (~90% collapse). Following phases = Chainlink / Arweave pattern (sustained growth).

---

## 1. The Strategic Question (Director, 2026-05-22)

Director asked four interrelated questions:

> "In order to attract users and build community, is it necessary to build PWM token mining?
>
> When users want to use some solutions, users need to mine first, which can increase benchmark and solutions?
>
> If so, we also need to build the PWM token system and AI4science.
>
> Solution is AI4science. But, I think we can first only the solution and mismatch without imaging design and sci-simu."

Parsed:

1. Is PWM token mining NECESSARY to attract users / build community?
2. Should users be required to mine first (earn PWM) before they can use solutions (spend PWM)? Does this circular economy help bootstrap?
3. If yes → do we need to fully build the token system AND AI4Science layer at launch?
4. Director's tentative simplification: launch with only Solution (= AI4Science) + Mismatch layers, deferring Imaging Design + Sci-Simu.

---

## 2. Direct Answers

| Question | Short answer | Reasoning |
|---|---|---|
| Is token mining NECESSARY at launch? | **NO** — not for Phase 1 | Token system is already built; question is what to FEATURE in launch narrative |
| Should users mine first to use solutions? | **NOT at launch** | Creates wrong-direction friction; first users want to SUBMIT not USE; Phase 3 mechanism, not Phase 1 |
| Solution + Mismatch only (skip Imaging Design + Sci-Simu)? | **YES** — correct simplification | Minimum viable architectural surface for Phase 1 |
| AI4Science = the Solution layer? | **YES** — useful marketing rename | "Solution" is generic; "AI4Science" is recognizable to academic ML community |

---

## 3. Why "Mine-First-to-Use" Is the Wrong Friction for Launch

Director's proposed mechanism: users mine first → earn PWM → spend PWM to use solutions → creates demand+supply loop.

**The mechanism is good design.** It's how mature PWM should work. **But the timing is wrong for Phase 1.** Four specific reasons:

### 3.1 First users want to SUBMIT, not USE

Per `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` §2.2 + §6, the first PWM user is:

> A PhD student, postdoc, or AI imaging researcher who wants a credible leaderboard result for their method.

Their motivation:
- ✅ Want to publish a paper using PWM-CI-1 as comparison benchmark
- ✅ Want a citable score against an established baseline
- ✅ Want visibility (their name on a public leaderboard)
- ✅ Want comparison against other methods in their subfield

**These users want to SUBMIT their methods, not USE other methods.** Forcing them to mine someone else's method first asks them to do work they don't care about. Wrong direction.

### 3.2 Mining requires technical expertise

Reproducing a method (cloning repo, running on test set, getting it to verify) takes 2-10 hours minimum. First users won't spend this BEFORE seeing if PWM is worth their time.

**The friction at launch must be LOW.** Mining-first adds 2-10 hours of work before the user sees any value.

### 3.3 Mine-first inverts the trust direction

First users need PWM to prove value to THEM. Mine-first asks users to give first (mine, expend compute, generate verifications) before they receive value (recognition for their AI4Science solution within the verified PWM catalog). The value users seek is **AI4Science recognition**, not abstract verification — see `PWM_VALUE_FRAMING_2026-05-22.md` for canonical framing.

**Trust must be earned, not demanded.** The launch sequence should be:
1. PWM proves value (verified leaderboard, citable cert hash)
2. User invests minimal effort (submit method)
3. User receives high value (rank, citation, optional PWM prize)
4. User trusts PWM and engages more deeply

Mine-first breaks this sequence — user invests heavily BEFORE proof of value.

### 3.4 The Helium pattern risk

Per `coordination/prevent_copy/PWM_TOKEN_VALUE_DEFENSE_2026-05-20.md` §5.3, three of six comparable tokens (Filecoin, Helium, UMA) failed at the demand-acquisition step:

- Supply-side mining inflated indefinitely with rewards
- Demand-side revenue never materialized
- Token collapsed ~90% from peak
- Protocol became a zombie

**Mine-first-to-use at launch recreates this pattern.** PWM would have:
- Lots of supply-side activity (users mining to earn PWM)
- No verified demand for using solutions (because there are no published solutions yet)
- Token inflates with no real demand
- Collapse

The mine-to-use loop ONLY works once there are real solutions worth paying to use. At Phase 3 (Year 2+) this exists. At Phase 1 (D9 launch) it does not.

---

## 4. The Three-Phase Deployment Sequence

### 4.0 Canonical PWM role terminology (CORRECTED 2026-05-22)

**CRITICAL TERMINOLOGY CORRECTION** (Director, 2026-05-22):

> "There is no submitter. Here submitter is like the miner. The original plan, the miner in Genesis will never pay."

In PWM canonical vocabulary:
- **Submitter = Miner.** Same entity. They submit certificates (L4) and earn from ranked-draw rewards.
- **Mining is certificate submission.** Whether the algorithm is novel (your own method) OR a reproduction, generating a verified L4 certificate IS mining.
- **Users are DIFFERENT from miners.** Users consume verified outputs; miners produce them.
- **Genesis miners NEVER pay.** Only USERS pay (Phase 3+).

#### The two PWM roles

| Role | What they do | Earns? | Pays? | Phase active |
|---|---|---|---|---|
| **MINER** (= "Submitter" in casual usage) — has sub-roles: | | | | |
|   • **SP / AC** (Solution Provider / Algorithm Contributor) | Wrote the algorithm; registers on-chain | ✅ `p × 55%` per ranked cert | ❌ NO (Genesis never pays) | Phase 1+ |
|   • **CP** (Compute Provider) | Runs the algorithm on benchmark data | ✅ `(1-p) × 55%` per ranked cert | ❌ NO (Genesis never pays) | Phase 1+ |
| **USER** (researcher / clinician / AI agent) | Queries leaderboard for solutions; consumes verified outputs | ❌ NO (not a token-earner) | ❌ Phase 1-2 (free); ✅ Phase 3+ (premium runs) | All phases (free until Phase 3) |

#### The reward distribution formula

Per `infrastructure/agent-contracts/contracts/PWMRewardERC20.sol:180-185`:

| Recipient | Share | Code constant |
|---|---|---|
| **AC** (algorithm author / SP) | `p × 55%` | `SPLIT_AC_CP = 5_500` (lines 180-181) |
| **CP** (compute provider) | `(1-p) × 55%` | `SPLIT_AC_CP = 5_500` (line 181) |
| **L3** (benchmark author) | 15% | `SPLIT_L3 = 1_500` (line 182) |
| **L2** (spec author) | 10% | `SPLIT_L2 = 1_000` (line 183) |
| **L1** (principle author) | 5% | `SPLIT_L1 = 500` (line 184) |
| **T_k** (treasury / principle-level maintenance) | 15% | `SPLIT_TREASURY = 1_500` (line 185) |
| **Total** | **100%** | |

**Ranked-draw multiplier** (applied before splits): Rank 1 → 40%; Rank 2 → 5%; Rank 3 → 2%; Ranks 4-10 → 1% each; Rank 11+ rolls over.

#### Director's phased monetization plan (corrected vocabulary + mining active Phase 1b)

| Phase | Window | Mining active? | Miners pay? | Miners earn? | Users pay? |
|---|---|---|---|---|---|
| **Phase 1a (audit)** | D9 to D9+30 | ❌ NO (`MINTING_PAUSED = true`) | ❌ NO | ❌ NO (paused) | ❌ NO (read-only) |
| **Phase 1b (mining ACTIVE)** | D9+30 to D9+180 | ✅ YES | ❌ NO (Genesis doctrine) | ✅ YES (full ranked-draw from `PWMMintingERC20` + 10K Reserve sponsor bonus for PWM-CI-1 top-3) | ❌ NO (free) |
| **Phase 2** | Months 6-12 | ✅ YES | ❌ NO (Genesis) | ✅ YES (broader rewards across 2-3 benchmarks + T_k royalty distributions) | ❌ NO (still free) |
| **Phase 3** | Months 12-24 | ✅ YES | ❌ NO for Genesis | ✅ YES (T_k royalties mature) | ✅ YES for advanced verified runs (large-scale verification, private benchmarks) |
| **Phase 4** | Year 2+ | ✅ YES | ❌ NO for Genesis | ✅ YES (full Zeno emission) | ✅ YES at scale |

**Rule:** Do NOT ask users to pay before they trust the platform. First create value. Then monetize verification. Miners never pay (Genesis doctrine).

#### What "submission" means (= mining)

A miner submits a certificate by:
1. Writing an algorithm (or using one as CP)
2. Running it against the held-out test set (`PWMCertificate.sol:171` stores `address submitter`)
3. The certificate triggers ranked-draw rewards via `PWMRewardERC20.distribute`
4. AC + CP + L1 + L2 + L3 + T_k all earn per the split formula

**This is mining.** "Submit your AI4Science method" in marketing language = "Generate a certificate as a miner" in protocol language. Same operation.

#### What "user" means (= consumer, NOT miner)

A user is someone who:
1. Queries the leaderboard for solutions (free in Phase 1-2)
2. Uses MCP server to ask AI assistants for PWM-verified results (free)
3. Pays PWM (Phase 3+) for advanced verified runs on their own data (large-scale, private benchmarks)

**Users never generate certificates.** They consume the verified outputs that miners generated.

#### User services by phase — free vs paid (CLARIFIED 2026-05-22)

Director's question: "Only until phase 3, can AI4Science demand cost of PWM tokens? Is AI4Science in phase 1-2 are free?"

**Answer: Yes, AI4Science is FREE for users in Phase 1-2.** Users only pay PWM starting Phase 3, and only for **advanced verified runs via PWM's infrastructure** (not for basic access).

| User action | Phase 1-2 | Phase 3+ | Why |
|---|---|---|---|
| Browse leaderboard | ✅ FREE | ✅ FREE | Public leaderboard data; trust-building |
| Query MCP server (ask Claude / ChatGPT "what's the best method for X?") | ✅ FREE | ✅ FREE | AI-assistant integration is always free |
| Download MIT-licensed AI4Science methods | ✅ FREE | ✅ FREE | All methods MIT — anyone can download anywhere |
| Run downloaded method on own data **locally** (your own hardware) | ✅ FREE | ✅ FREE | Local execution; PWM doesn't gate this |
| Cite verified cert hashes in papers | ✅ FREE | ✅ FREE | Citation infrastructure is free |
| Read verified PSNR / SSIM scores + benchmark results | ✅ FREE | ✅ FREE | Public verification data |
| **Run method on PWM's verified infrastructure (mine-to-use)** | ❌ NOT AVAILABLE | ✅ **PAID** | Premium tier; PWM provides verified compute |
| **Large-scale batch verification via PWM** | ❌ NOT AVAILABLE | ✅ **PAID** | Premium tier; PWM batch infrastructure |
| **Private benchmarks (your data, cryptographic verification)** | ❌ NOT AVAILABLE | ✅ **PAID** | Premium tier; PWM verifies on private data |
| **Clinical-grade verified runs (FDA/IRB-traceable)** | ❌ NOT AVAILABLE | ✅ **PAID** | Premium tier; institutional / clinical use |

#### Two important nuances

**Nuance 1: Phase 1-2 "free" isn't "free-because-you-pay-nothing." It's "free-because-PWM-doesn't-offer-paid-services-yet."**

PWM doesn't have a paid "run-on-PWM-infrastructure" service in Phase 1-2. Users can browse methods (free), download methods (free, MIT-licensed), run methods locally (free, their own compute), cite cert hashes (free). There's nothing for users to PAY for in Phase 1-2 because the only paid service (mine-to-use via PWM infrastructure) doesn't activate until Phase 3.

**Nuance 2: Basic access is ALWAYS free, even in Phase 3+.**

The paid tier in Phase 3+ is specifically **advanced verified runs via PWM's infrastructure**:
- Large-scale (running many inferences via PWM's compute)
- Private (your data; PWM provides cryptographic verification you can't get from local download)
- Clinical-grade (FDA-traceable; institutional)
- Mine-to-use (PWM hosts the compute-provider role; charges PWM per inference)

A PhD student in Phase 3+ who just wants to browse the leaderboard and download a CASSI reconstruction method still pays $0. They only pay if they want PWM's **verified-infrastructure service** (not the methods themselves).

#### The monetization model — PWM sells VERIFICATION, not METHODS

PWM's value framing (per `PWM_VALUE_FRAMING_2026-05-22.md`) is "verified AI4Science platform." Note: PWM sells **verification**, not **methods**. The methods themselves are MIT-licensed — PWM Foundation cannot and does not charge for them.

What PWM monetizes (Phase 3+):
- ✅ **Verification service at scale** (batch processing many inferences)
- ✅ **Verification on private data** (cryptographic verification of YOUR data, not just public test sets)
- ✅ **Verification for clinical/institutional use** (FDA-traceable, IRB-approved data flows)
- ✅ **Compute-provider hosting** (PWM provides the CP role; charges per inference)

What PWM does NOT monetize:
- ❌ The AI4Science methods themselves (MIT-licensed; always free to download + run locally)
- ❌ Browsing the leaderboard (always free)
- ❌ Cert hash citations (always free)
- ❌ Public verification data (always free)
- ❌ MCP server queries (always free; AI assistants don't pay)
- ❌ Basic submission to public benchmarks (Genesis miners never pay; per Director's doctrine)

**Analogy:** PWM is to AI4Science as Red Hat is to Linux — the kernel/methods are free; the verified-infrastructure service + support is paid. Or as AWS is to open source — the code is free; the hosted-running infrastructure is paid.

#### Implementation timeline — when does each thing become live? (added 2026-05-22)

Director's 2026-05-22 question: "When will token system and wallet embed into website? When AI4Science should work well for users?"

**Required-by milestones for website + wallet integration:**

| Milestone | Date | What must be live | Owner |
|---|---|---|---|
| **D9 (Phase 5 mainnet deploy)** | D9 | All 9 token contracts deployed. Website should launch with wallet integration for credibility. | Director + sub-GPU |
| **D9+7 (week-of-launch latest)** | D9+7 | Landing page LIVE at physicsworldmodel.org with wallet connect (Privy / RainbowKit / WalletConnect). Read-only leaderboard. Submission guide. | sub-GPU |
| **D9+30 (Phase 1b start) — CRITICAL** | D9+30 | **REQUIRED:** Wallet integration LIVE; submission flow tested; miners can submit certificates and receive ranked-draw rewards. Mining cannot work without this. | sub-GPU + Bounty 2 reference |
| **D9+60 (mining ramps)** | D9+60 | Mobile-responsive (Bounty 10 shipping); MCP server LIVE (Bounty 9 shipping); leaderboard auto-updating from on-chain events. | Bounty 9 + 10 winners |
| **D9+90 (Phase 1b mid)** | D9+90 | Multi-method leaderboard mature; submission flow optimized; first interim benchmark report published. | sub-GPU + Director |

**The critical date is D9+30.** When Phase 1b activates (mining unpauses), miners MUST have a working wallet+submission flow on the website. Sub-GPU's primary deliverable for the launch sprint.

**Clarification (2026-05-22): wallet integration is a Phase 1a deliverable, NOT Phase 1b.**

Director's question: "Is wallet embed in Phase 1b?" Answer: **No, wallet integration must be ready BEFORE Phase 1b starts.**

- **Phase 1a (D9 to D9+30, audit window)** = the implementation phase for wallet integration. Sub-GPU builds + tests + deploys the wallet-integrated website during this window.
- **Phase 1b (D9+30 onwards)** = the CONSUMPTION phase. Mining activates; miners USE the already-live wallet to submit certificates and receive rewards.

Phase 1b depends on Phase 1a having delivered the wallet integration. If sub-GPU does NOT have wallet integration live by D9+30, Phase 1b mining cannot start on schedule, and the entire AI4Science-works-well timeline slips proportionally.

**Practical implication for sub-GPU sprint:**

| Sub-GPU deliverable | Owner | Hard deadline | Phase |
|---|---|---|---|
| Landing page at physicsworldmodel.org | sub-GPU | D9+7 | Phase 1a |
| Wallet integration (Privy + RainbowKit) — **includes SIWE login** | sub-GPU + Bounty 2 reference | D9+30 | **Phase 1a** (critical gate to Phase 1b) |
| Submission flow end-to-end tested | sub-GPU + Director | D9+30 | Phase 1a |
| Leaderboard reading on-chain events | sub-GPU | D9+30 | Phase 1a |
| Mobile responsive (Bounty 10) | Bounty 10 winner | D9+60 | Phase 1b (post-launch enhancement) |
| MCP server (Bounty 9) | Bounty 9 winner | D9+60 | Phase 1b (post-launch enhancement) |
| User profile pages (submission history, balance) | sub-GPU + Bounty 2 winner | D9+60 to D9+90 | Phase 1b (post-launch enhancement) |
| Email notifications (cert verified, prize won) | sub-GPU | D9+90+ | Phase 1b/2 (polish) |

#### Login system clarification (2026-05-22)

Director's question: "Should login system be done with the website + wallet integration?"

**Answer: YES — login IS wallet integration in PWM. They are the same system.**

In Web3 protocols, login = **SIWE (Sign-In With Ethereum)**:
1. User clicks "Connect Wallet" on physicsworldmodel.org
2. Wallet provider (RainbowKit or Privy) opens
3. User signs a challenge message with their wallet
4. PWM verifies the signature → user is authenticated
5. **User's wallet address IS their identity** — no username/password needed

**No separate "username/password" login system is needed or wanted.** The wallet IS the identity. This is the standard Web3 pattern.

**What's included in the Phase 1a wallet integration:**

| Component | What it does |
|---|---|
| **RainbowKit integration** | Native wallet connect for MetaMask, Coinbase Wallet, Rainbow, etc. (power users) |
| **Privy integration** | Embedded wallets via email / Google login (academics without crypto wallets — Privy creates a wallet for them) |
| **SIWE challenge + signature flow** | User signs challenge → authenticated session |
| **Wallet identity on cert submissions** | `msg.sender` from wallet = submitter identity in `PWMCertificate.sol:171` |
| **Session management** | Keep user logged in across page reloads |

This is ONE integrated system. Login is not separate from wallet.

**Director's earlier decision (confirmed):** Per 2026-05-21 review, Director rejected a separate "Login + Wallet Connecting" bounty alongside the proposed AI4Science bounty. The decision: login is part of Bounty 2 (Web UI / Explorer) reference impl scope, not a separate bounty. Confirmed correct — splitting login from wallet would fragment a single integration.

**What does NOT need to be in Phase 1a:**

- User profile pages (submission history, balance, rank) — Phase 1b enhancement
- Email notifications (cert verified, prize won) — Phase 1b/2 polish
- Password reset flows — NOT NEEDED (wallet-based)
- 2FA — NOT NEEDED (wallet provides cryptographic auth)
- Social profile (Twitter handle, ORCID, etc.) — optional Phase 2 polish

**When AI4Science "works well" for users by user type:**

| User type | "Works well" milestone | What "works well" means at that date |
|---|---|---|
| **Phase 1a passive browsers (audit window)** | D9+30 | Whitelisted Director seeds visible. Mostly empty leaderboard. Not really "working well" yet. |
| **Phase 1b casual browsers** | D9+90 | 5-15 methods on leaderboard. Basic browsing + citation works. |
| **Group A miners (submitters)** | D9+30 | Mining activates; full economic incentive (ranked-draw + Reserve bonus). Submission flow works end-to-end. |
| **Group A miners (mature)** | D9+180 | 10-30 methods; meaningful peer comparison; arXiv paper published. |
| **Group B PIs (choosing methods)** | Month 6-12 | Multi-benchmark (PWM-CI-1 + PWM-CI-2 + PWM-CI-3 + PWM-MED-1). LP active; token has price. Real peer-comparison data. |
| **Group C advanced users (clinical / AI labs)** | Month 12-24 | Mine-to-use ACTIVE. Pay PWM for verified runs on own data. Clinical-grade use cases viable. |
| **All user types at scale** | Year 2-3 | 50+ benchmarks; industry-standard verification venue. Self-sustaining flywheel. |

**Honest summary:**

- **AI4Science works for MINERS (submitters) by D9+30** (when Phase 1b activates).
- **AI4Science works for CASUAL BROWSERS by D9+90** (when 5-15 methods on leaderboard).
- **AI4Science works WELL for academic users by Month 6-12** (multi-benchmark; LP active; token velocity).
- **AI4Science works WELL for advanced/clinical users by Month 12-24** (mine-to-use; paid verified runs).
- **AI4Science works at full industry scale by Year 2-3.**

**Critical dependencies on the timeline:**

1. D9 mainnet deploy on schedule (current target per Director's mainnet sprint)
2. Sub-GPU completing landing page + wallet integration before D9+7
3. Bounty 2 reference impl deployed at physicsworldmodel.org by D9
4. MCP server (Bounty 9) and Mobile UX (Bounty 10) shipping by D9+60
5. First 5-10 external miners attracted via Director's Phase 1b outreach campaign (per `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md`)

If any of these slip, the AI4Science-works-well dates slip proportionally.

### 4.1 Phase 1 (Months 1-6, D9 to D9+180) — Mining ACTIVE from D9+30; users free

**REVISED 2026-05-22 per Director's framing:** Mining activates at D9+30 (post-audit), not deferred to Phase 2. The earlier framing held mining off until Phase 2 — that was based on the incorrect "submitter ≠ miner" distinction. With the corrected vocabulary (submitter = miner) + AI4Science as the value framing, mining IS the production of AI4Science solutions and should be active as soon as security gates allow.

#### Phase 1a (D9 to D9+30) — Audit window

**Mechanism:** Mainnet deploys; soft-launch caps active (`MINTING_PAUSED = true`, `TREASURY_TRANSFERS_PAUSED = true`, `STAKING_TVL_CAP_USD = $100`, `submissionPermissionless = false`). Whitelisted submissions only. No mining rewards emitted.

| Layer | Status |
|---|---|
| L3-003: Solution (= AI4Science) | 🟡 Whitelisted only |
| L3-004: Mismatch | ✅ ACTIVE (whitelisted) |
| Token mining | 🚫 PAUSED (`MINTING_PAUSED = true`) |
| Token rewards | 🚫 PAUSED (treasury transfers paused) |
| User browsing | ✅ ACTIVE (read-only) |

**Goal:** External audit of mainnet deploy + bug bounty engagement + final security check before unpausing.

#### Phase 1b (D9+30 to D9+180) — Mining ACTIVE

**Mechanism:** Soft-launch caps lift. `MINTING_PAUSED = false`. Permissionless submissions. PhD students / postdocs / AI imaging researchers MINE by submitting AI4Science methods to PWM-CI-1 → certificates generated → ranked-draw rewards via `PWMRewardERC20.distribute` from the 17.22M Minting pool. PLUS 10K Reserve sponsor bonus on top for PWM-CI-1 top-3 (Foundation early-adopter incentive).

**Miners pay nothing. Users pay nothing. Miners receive full token-economic incentive (standard ranked-draw + Reserve sponsor bonus).**

| Layer | Status | Notes |
|---|---|---|
| L3-003: Solution (= AI4Science) | ✅ **ACTIVE** | Miners submit AI methods; mining produces AI4Science solutions |
| L3-004: Mismatch | ✅ **ACTIVE** | Verification against held-out test set |
| L3-001: Imaging Design | 🟡 Embedded in anchors | 30 v3 anchors define hardware; miners don't submit new designs |
| L3-002: Sci-Simu | 🟡 Embedded in anchors | Anchors define simulation methodology; miners don't submit new simulations |
| Token mining | ✅ **ACTIVE** | `PWMMintingERC20.mintFor` emits ranked-draw rewards |
| Reserve sponsor bonus | ✅ ACTIVE | 10K PWM bonus to PWM-CI-1 top-3 on top of standard mining |
| Token staking | ✅ AVAILABLE | Optional (not required for Genesis benchmarks) |
| Token speculation / price | 🚫 NEVER FEATURED in marketing | Pre-LP; no price discovery yet |
| User payment for solutions | 🚫 NOT YET | Free in Phase 1; users pay starting Phase 3 |

**Marketing narrative for Phase 1b:** "Verified AI4Science benchmark with leak-proof test sets. Submit your method. Earn from ranked-draw rewards. Get a citable cert hash for your paper." (Per `PWM_VALUE_FRAMING_2026-05-22.md` — lead with AI4Science as the value, verification as the moat.)

**Goals:**
- 10-30 external AI4Science method submissions
- Mechanism validated end-to-end (submit → verify → score → leaderboard → cert → reward)
- Full mining-economic loop active from D9+30
- Reserve sponsor bonus distributed (10K PWM to PWM-CI-1 top-3)
- arXiv companion paper published
- Co-founder #2 recruited

**Success criteria:** 5-10 external submissions by D9+90; 10-30 by D9+180; mining mechanism operates without major issues.

**Failure criterion:** <5 external submissions by D9+90 → mechanism is broken or outreach is failing; pause LP plans; pivot before Phase 2.

### 4.2 Phase 2 (Months 6-12, D9+180 to D9+365) — LP seeded; multi-benchmark scaling

**Mechanism:** Mining was already active in Phase 1b. Phase 2 adds: LP seeded for token price discovery; more benchmarks live (PWM-CI-2, PWM-CI-3, PWM-MED-1); token has real USD-equivalent value; T_k royalty distributions begin (per-Principle accumulations from Phase 1b mining).

**New activations:**

| Layer | Status (changed from Phase 1b) | Notes |
|---|---|---|
| LP (Uniswap v3 PWM/USDC) | ✅ **SEEDED** | 1.05M PWM + $50K USDC; token has discoverable price |
| Token staking (active) | ✅ Activated for non-Genesis benchmarks | Genesis still free; new benchmarks may add 1-10 PWM refundable stake |
| T_k royalty distributions | ✅ ACTIVATED | Per-Principle pools accumulate from Phase 1b mining; distributable now |
| Bounty 5 (smart contracts competing impl) | ✅ OPEN | Gates lift at Phase 1 sign-off |
| Bounty 6 (IPFS pinning) | ✅ OPEN | Gates lift at Phase 1 sign-off |
| PWM-CI-2 (compressed sensing) | ✅ LAUNCHED | Second computational imaging benchmark |
| PWM-CI-3 (spectral imaging) | ✅ LAUNCHED | Third computational imaging benchmark |
| PWM-MED-1 (mini low-dose CT, public data) | ✅ LAUNCHED | Track B starts (Track 9 mini-benchmark) |

**Goals:**
- 50+ AI4Science submissions across PWM-CI-1 + PWM-CI-2 + PWM-MED-1
- 20-50 active miners (Group D users)
- Token price discovers via LP
- Foundation 501(c)(3) outcome known
- 1-3 grants landed

**Success criteria:** 50+ submissions; 5+ labs involved; token price > $0.50.

**Failure criterion:** Token collapses immediately post-LP (<$0.50 within 30 days) → market-maker engagement needed; reference `prevent_copy/PWM_TOKEN_VALUE_DEFENSE_2026-05-20.md`.

### 4.3 Phase 3 (Year 2+, D9+365+) — MINE to USE (Director's proposal)

**Mechanism:** Director's full vision is activated. Users want to USE existing AI4Science methods on their own data. They pay PWM to run inference. They can earn PWM by mining (verifying submissions). Self-sustaining loop.

**New layers activated:**

| Layer | Status (changed from Phase 2) | Notes |
|---|---|---|
| L3-001: Imaging Design (active submissions) | ✅ **ACTIVATED** | Hardware designers submit new CASSI mask patterns; benchmark them |
| L3-002: Sci-Simu (active submissions) | ✅ **ACTIVATED** | Forward-model researchers submit simulations; synthetic-data benchmarks possible |
| User pay-to-run inference | ✅ **ACTIVATED** | Users pay PWM to run AI4Science methods on own data |
| Mine-to-use economics | ✅ **ACTIVATED** | Director's full mechanism design |
| Full mining infrastructure scaling | ✅ ACTIVATED | Bounty 4 reference impl scaled; specialized mining clients emerge |
| RSNA / ISBI 2028 medical flagship | ✅ LAUNCHED | Track B full clinical launch |

**Goals:**
- 100-300 mini-competition participants per benchmark
- 5-10 active benchmarks across CI + MED + (eventually) other physics domains
- Self-sustaining token velocity (mining earnings ≈ fee revenue at steady state)
- 50%+ of computational imaging community cites PWM as standard venue
- Defection becomes infeasible (Chainlink-pattern)

**This is where Director's mine-first-to-use mechanism becomes the dominant pattern.** By Year 2+:
- Real AI4Science methods exist on the platform (worth paying to use)
- Real benchmarks exist (worth mining)
- Real users exist (researchers + AI/data consumers + clinical centers)
- Token has stable USD value
- Mine-to-use is the natural economic loop

**Don't skip Phases 1-2 to get here.** Each phase de-risks the next.

---

## 5. Yes to Solution + Mismatch Only at Phase 1

Director's architectural simplification: launch with only L3-003 (Solution) + L3-004 (Mismatch). Defer L3-001 (Imaging Design) + L3-002 (Sci-Simu) to later phases.

**This is the right call.** Here's why each layer is or isn't needed:

### 5.1 The PWM L3 architecture

| Layer | What it is | Phase 1 needed? |
|---|---|---|
| **L3-001: Imaging Design** | Hardware specs (CASSI mask, optical setup, detector configuration) | ❌ NO at Phase 1 |
| **L3-002: Sci-Simu** | Physics-based forward model (light propagation, sensor noise) | ❌ NO at Phase 1 |
| **L3-003: Solution (= AI4Science)** | AI reconstruction method | ✅ YES (active) |
| **L3-004: Mismatch** | Verification scoring (PSNR/SSIM vs ground truth) | ✅ YES (active) |

### 5.2 Why Imaging Design + Sci-Simu can be deferred

**For PWM-CI-1 (CASSI Reconstruction Benchmark):**

- **Dataset has REAL measurements.** No sci-simu needed for the data — it's real experimental measurements.
- **Hardware is FIXED by the dataset.** The CASSI optical setup is defined by whichever dataset is chosen. Researchers don't submit new hardware designs.
- **Researchers submit Solution (AI method).** That's the only thing they control.
- **PWM verifies via Mismatch.** Scoring is automatic.

So Phase 1 needs exactly two active L3 layers: Solution + Mismatch. The other two are embedded in the published anchors (the 30 v3 anchors describe hardware + simulation methodology as part of the principle definition).

### 5.3 What "deferring Imaging Design + Sci-Simu" actually means

Important clarification: **deferring these layers does NOT mean removing them from the published anchors.** It means:

- ✅ The 30 published anchors (CASSI, CACTI, L1-503..L1-531) still describe imaging hardware + simulation methodology
- ✅ This metadata is visible on the landing page + leaderboard (so submitters understand the context)
- ❌ Researchers do NOT submit new imaging designs or simulations to PWM-CI-1
- ❌ The contracts do NOT need to handle Imaging Design / Sci-Simu submissions at Phase 1
- ❌ The UI does NOT need submission flows for Imaging Design / Sci-Simu

### 5.4 When Imaging Design + Sci-Simu activate

| Phase | Layer activated | Use case |
|---|---|---|
| **Phase 2 (Months 6-12)** | (nothing new architecturally; just mining + LP) | Token economics matures |
| **Phase 3 (Year 2)** | L3-001: Imaging Design submissions | Hardware designers submit CASSI mask patterns; benchmark them on simulated + real data |
| **Phase 3 (Year 2)** | L3-002: Sci-Simu submissions | Forward-model researchers submit simulations; synthetic-data benchmarks for niche scenarios |

**These layers are useful but not first-launch material.** Adding them at launch increases:
- UI complexity (3-4 submission flows instead of 1)
- Contract verification logic (multiple cert types)
- Mental load for first users (which layer do I submit to?)
- Risk of partial-completion (some layers half-built)

**Defer them. Validate the Solution + Mismatch mechanism first. Add others once Phase 1 is proven.**

---

## 6. Yes to AI4Science as the Solution Layer Name

"AI4Science" is better marketing than "Solution" because:

| Criterion | "Solution" | "AI4Science" |
|---|---|---|
| Generic / specific | Generic (what does it solve?) | Specific (recognizable buzzword) |
| Academic ML community recognition | Low — unclear what's submitted | High — implies AI methods for science problems |
| Marketing tone | Bureaucratic | Active / aspirational |
| Implied user | Anyone | AI/ML researcher (the right user!) |
| Differentiation | None | Implies PWM is infrastructure for AI4Science |

**Rename in marketing only.** Internal contracts + code still call it "Solution" (L3-003) — no on-chain rename needed.

External-facing materials use "AI4Science":
- Landing page: "Submit your **AI4Science** method to PWM-CI-1"
- arXiv companion paper title: "PWM: A Cryptographic Substrate for **AI4Science** Verification"
- Twitter/X: "Submit your **AI4Science** reconstruction to the leak-proof leaderboard"
- HackerNews: "Show HN: PWM-CI-1 — **AI4Science** benchmark with on-chain test-set commitment"
- Conference talks: "PWM enables verified **AI4Science** benchmarks for computational imaging"

### 6.1 But not as a separate bounty (clarification)

In an earlier conversation (2026-05-21), Director rejected a proposed "AI4Science bounty" alongside the existing 8 bounties. That decision stands — **AI4Science is not a separate bounty; it's the name of the Solution layer that submitters interact with.**

The bounty pool stays at 10 bounties:
1-4 (infrastructure: scoring, web, CLI, miner — OPEN)
5 (contracts competing impl — SPEC PUBLISHED)
6 (IPFS pinning — SPEC PUBLISHED)
7 (genesis principle polish — SPEC PUBLISHED)
8 (LLM matcher — SPEC PUBLISHED)
9 (MCP server — SPEC PUBLISHED)
10 (mobile UX — SPEC PUBLISHED)

AI4Science is not a bounty. It's a marketing rename of the Solution layer.

---

## 7. The PWM Token System Is Already Built

Important clarification: **the PWM token mining infrastructure already exists in the smart contracts.** It's not a build-vs-not-build question.

| Contract | Purpose | Status |
|---|---|---|
| `PWMToken.sol` | ERC20 token | ✅ Deployed (testnet); mainnet at Phase 5 |
| `PWMMintingERC20.sol` | Programmatic emission for L4 mining | ✅ Deployed |
| `PWMRewardERC20.sol` | Reward distribution per rank | ✅ Deployed |
| `PWMStakingERC20.sol` | Staking + slashing logic | ✅ Deployed |
| `PWMTreasuryERC20.sol` | Reserve + grant disbursement | ✅ Deployed |
| `PWMRegistry.sol` | On-chain artifact registry | ✅ Deployed |
| `PWMCertificate.sol` | L4 certificate issuance | ✅ Deployed |
| `PWMGovernance.sol` | 5-multisig + voting | ✅ Deployed |
| `PWMVesting.sol` | Founding team vesting | ✅ Deployed |

**At Phase 5 mainnet deploy, all of these go live.** The token system EXISTS. Mining can happen at any time post-deploy.

**The question is: what to FEATURE in the launch narrative.**

### 7.1 What to feature at Phase 1 launch

✅ **DO feature:**
- Verified benchmark platform
- Leak-proof test sets
- AI4Science method submissions
- Mismatch verification
- Leaderboard rankings
- Citable L4 certificates
- Top-3 PWM token rewards (small, secondary — frame as "Foundation Reserve sponsorship for winners")

❌ **DO NOT feature:**
- Token mining (mentioned only in roadmap / FAQ)
- Token speculation / price (never mentioned)
- "Earn PWM by mining" as a launch hook
- Staking economics (mentioned only in FAQ)
- LP / market makers (deferred until Phase 2 ready)
- Mine-to-use mechanism (deferred until Phase 3 ready)

### 7.2 Why this framing

Per `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` §8 + §12:

> "PWM is not a token mining protocol for scientific computing. PWM is a verified benchmark platform for physics-grounded AI, starting with computational imaging and medical imaging."

The token is the REWARD MECHANISM, not the product. Featuring mining at launch:
- Scares academic researchers (signals "grift")
- Attracts wrong users (token speculators, not AI4Science researchers)
- Sets wrong KPIs (token price, not submission count)
- Recreates Helium pattern

**Lead with product. Token follows.**

---

## 8. The "Why Now Mine, Why Later" Decision Tree

Director's intuition that mining could bootstrap demand is correct — but timing matters. Here's the decision tree:

```
START: Is there a published AI4Science method worth paying to use?
│
├── NO (Phase 1: Months 1-6)
│   └── Don't feature mining. Focus on attracting submissions.
│       → SUBMIT to EARN model
│
├── YES, a few methods (Phase 2: Months 6-12)
│   └── Activate mining for VERIFICATION (reproducers earn).
│       Add small entry stake for submissions (tests fee mechanism).
│       LP seeded; token has price discovery.
│       → VERIFY to EARN model
│
└── YES, many methods + clinical relevance (Phase 3: Year 2+)
    └── Activate mine-to-use mechanism.
        Users pay PWM to run inference on own data.
        Miners earn PWM for verification.
        Self-sustaining loop.
        → MINE to USE model (Director's full vision)
```

**Each phase requires the previous phase to be proven.** Skipping = Helium pattern.

---

## 9. The Subtle Architectural Question

Director's message included: "we can first only the solution and mismatch without imaging design and sci-simu"

There's a subtle question about what this means for the 30 v3 anchors.

### 9.1 The 30 v3 anchors are not affected

The 30 v3 anchors (CASSI, CACTI, L1-503..L1-531) include Imaging Design + Sci-Simu data embedded in the Principle definitions. They aren't separate L3 layers in those anchors — they're embedded in the principle itself.

**"Deferring Imaging Design + Sci-Simu" does NOT mean:**
- ❌ Removing imaging hardware specs from CASSI anchor
- ❌ Removing simulation methodology from CACTI anchor
- ❌ Modifying any of the 30 v3 anchors

**"Deferring Imaging Design + Sci-Simu" DOES mean:**
- ✅ The 30 anchors still describe hardware + simulation methodology (as part of the principle)
- ✅ Researchers DON'T need to submit new Imaging Designs or Simulations to PWM-CI-1
- ✅ Researchers ONLY submit their AI4Science method (Solution) + get scored via Mismatch
- ✅ The UI does NOT show submission flows for Imaging Design or Sci-Simu at Phase 1
- ✅ The contracts do NOT activate Imaging Design / Sci-Simu submission paths at Phase 1

### 9.2 What appears on the landing page

The landing page for PWM-CI-1 should show:

```
Benchmark: PWM-CI-1 — CASSI Reconstruction

Principle:        CASSI (compressive sensing imaging spectrometer)
                   Imaging method: [link to L1-CASSI anchor]
                   Hardware: [embedded in anchor; visible but not submittable]
                   Forward model: [embedded in anchor; visible but not submittable]

Test set:         [hash committed on-chain; sealed]

What you submit:  Your AI4Science reconstruction method (Solution layer)
What gets scored: Your method vs. held-out ground truth (Mismatch layer)
What you get:     Verified L4 certificate + leaderboard rank + (top-3) PWM reward
```

**Researchers see the FULL context (hardware, simulation methodology) but only SUBMIT to the Solution layer.** This is the cleanest Phase 1 UX.

---

## 10. Updates Required to Existing Docs

This phased architecture decision triggers minor updates to two existing canonical docs:

### 10.1 `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` (sub-GPU version on main)

Add section: "Phase 1 architectural scope: Solution + Mismatch only" (per this doc §5)

Rename "Solution" → "AI4Science" in marketing-facing copy (per this doc §6).

Owner: sub-GPU (since they wrote the canonical version of that doc).

### 10.2 `PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md`

Update the following sections to use "AI4Science" terminology:

- Section 2 (PWM-CI-1): "Submit your AI4Science method" instead of "Submit your method"
- Section 3 (How it works): Clarify "AI4Science method = the Solution layer (L3-003)"
- Section 6 (Roadmap): Add explicit phasing — Phase 1 = Solution+Mismatch; Phase 2 = + Mining; Phase 3 = + Imaging Design + Sci-Simu + Mine-to-Use
- Section 8 (FAQ): Add Q "Why don't I submit hardware designs?" with answer about phased rollout

Owner: Director or sub-GPU (the landing-page draft was committed by Claude as commit `aa8028e6`).

### 10.3 `PWM_DEVELOPER_COMPENSATION_2026-05-22.md` — minor update

Section §4.4 (Bug Bounty tier card) is unchanged. Section §6 (Mining Pool) should clarify mining ACTIVATES at Phase 2, not Phase 1.

Owner: Director (this is a clarifying note, not a substantive change).

### 10.4 `PWM_API_VS_WEBSITE_2026-05-20.md` — already aligned

This doc already establishes Year 1 = website (Phase 1), Year 2 = API (Phase 2 maturing), Year 3+ = agent infrastructure (Phase 3). No changes needed.

---

## 11. Decision Points for Director (Next 14 Days)

| # | Decision | Default | Notes |
|---|---|---|---|
| 1 | Approve three-phase deployment sequence (Phase 1: Submit-to-Earn; Phase 2: Verify-to-Earn; Phase 3: Mine-to-Use)? | YES (recommended) | Aligns with Helium-pattern avoidance |
| 2 | Approve Phase 1 architectural scope = Solution + Mismatch only (defer Imaging Design + Sci-Simu)? | YES (recommended) | Director's stated preference |
| 3 | Approve "AI4Science" as marketing rename of Solution layer (no on-chain change)? | YES (recommended) | Easy win; better marketing |
| 4 | Approve deferring token-mining narrative to Phase 2 (Months 6-12)? | YES (recommended) | Don't feature mining at launch |
| 5 | Approve deferring Mine-to-Use mechanism to Phase 3 (Year 2+)? | YES (recommended) | Director's mechanism; right timing |
| 6 | Approve landing page updates (per §10.2)? | YES (recommended) | sub-GPU or Claude can execute |
| 7 | Approve clarification update to `PWM_DEVELOPER_COMPENSATION` §6 (mining activates Phase 2)? | YES (recommended) | Minor update |
| 8 | Approve adding FAQ Q "Why don't I submit hardware designs?" to landing page? | YES (recommended) | Anticipated user question |

---

## 12. Cross-References

- `pwm-team/coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` — Two-track strategy (Track A + Track B)
- `pwm-team/coordination/PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` — D9 launch landing page copy
- `pwm-team/coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` — Layer 1/2/3 compensation framework
- `pwm-team/coordination/prevent_copy/PWM_TOKEN_VALUE_DEFENSE_2026-05-20.md` §5.3 — Helium pattern analysis
- `pwm-team/coordination/prevent_copy/PWM_REALISTIC_VALUATION_2026-05-20.md` — Realistic token value framing
- `pwm-team/coordination/PWM_API_VS_WEBSITE_2026-05-20.md` — Year 1/2/3 phasing
- `pwm-team/coordination/CRISIS_COMMS_PLAYBOOK_2026-05-21.md` — Crisis comms for token-speculation discourse
- `pwm-team/coordination/prevent_copy/PWM_COMPETITIVE_DEFENSE_2026-05-20.md` — Structural defenses
- `pwm-team/bounties/INDEX.md` — Master bounty list (10 bounties, ~1.3M PWM)
- `pwm-team/plan/track_9/PWM_TRACK_9_LOW_DOSE_CT_2026-05-16.md` — Track B (medical flagship) plan
- `pwm-team/plan.md` — Master multi-track plan
- `infrastructure/agent-contracts/contracts/PWMMintingERC20.sol` — Mining contract (built; deployed; not featured at Phase 1)
- `infrastructure/agent-contracts/contracts/PWMStakingERC20.sol` — Staking contract (built; deployed; not featured at Phase 1)
- `infrastructure/agent-contracts/contracts/PWMRewardERC20.sol` — Reward distribution (built; deployed; activates for top-3 prizes at Phase 1)

---

## 13. The Single Most Important Framing

**The PWM token mining infrastructure IS built. The question is when to FEATURE it.**

**Phase 1: Don't feature mining.** Lead with benchmark platform + AI4Science + Mismatch + leaderboard + verified certs. Mining is in the roadmap, not the headline.

**Phase 2: Activate mining quietly.** Reproducers earn PWM; LP discovers price; entry stakes test fee mechanism. Mining becomes part of the story but not the lead.

**Phase 3: Mine-to-Use becomes the dominant pattern** (Director's mechanism). Users pay PWM to run inference; miners earn PWM for verification; self-sustaining economic loop.

**Each phase de-risks the next. Skipping = Helium pattern. Following = Chainlink / Arweave pattern.**

---

## 14. The Single Sentence

**PWM ships with the full token system at D9, but the launch narrative features only AI4Science submissions + Mismatch verification — token mining and mine-to-use mechanics activate in Phase 2 and Phase 3 once Phase 1 has validated the demand-side mechanism.**

This is the answer to Director's 2026-05-22 question.

---

*This doc is the canonical reference for PWM phased architecture deployment. Update when Director's decisions in §11 land. Update at Phase 1 → Phase 2 transition (~D9+180) and Phase 2 → Phase 3 transition (~D9+365) with actual vs expected activation timing.*
