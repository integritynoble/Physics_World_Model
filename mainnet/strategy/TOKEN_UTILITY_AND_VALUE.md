# PWM Token Utility and Value — Canonical Reference

**Date:** 2026-05-22
**Audience:** Director + co-founders + grant reviewers + token holders + future PWM users + securities counsel
**Status:** Canonical reference for "Why does PWM have value? Why use PWM tokens?"
**Purpose:** Resolves Director's 2026-05-22 question: *"If there is no token system, how can users treasure the PWM token in the future? Why do users use PWM token in the future?"*

This doc complements:
- `PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` — WHEN each feature activates
- `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` — WHO uses PWM
- `PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` — HOW PWM is marketed
- `PWM_DEVELOPER_COMPENSATION_2026-05-22.md` — Layer 1/2/3 compensation framework
- `prevent_copy/PWM_REALISTIC_VALUATION_2026-05-20.md` — Honest token value estimates
- `prevent_copy/PWM_TOKEN_VALUE_DEFENSE_2026-05-20.md` — 12-month execution sprint

This doc answers: **WHY does PWM token have value, and WHAT do users do with it?**

---

## TL;DR

1. **The token system EXISTS from D9.** All 9 token contracts (`PWMToken`, `PWMMintingERC20`, `PWMRewardERC20`, `PWMStakingERC20`, `PWMTreasuryERC20`, `PWMRegistry`, `PWMCertificate`, `PWMGovernance`, `PWMVesting`) deploy at Phase 5 mainnet. **The phased architecture deployment doc was about LAUNCH NARRATIVE, not contract deployment.**

2. **PWM has 6 distinct value drivers:** (1) fixed supply scarcity, (2) utility-fee revenue, (3) governance rights, (4) royalty accumulation per Principle (T_k pools), (5) mining rewards (compute-for-tokens), (6) network effects from real adoption.

3. **PWM has 8 distinct USE cases:** (1) submit benchmarks (entry stake), (2) earn rewards (top-3 prizes), (3) pay-to-run inference (Phase 3), (4) verify submissions (mining), (5) accumulate Principle royalties, (6) vote on governance, (7) hold for appreciation, (8) liquidity-provide on Uniswap.

4. **Value DOES NOT depend on speculation.** Even if zero speculation occurs, PWM has utility-anchored value from real fee revenue + governance + royalties. This is the Chainlink / Arweave pattern, not the Helium / UMA pattern.

5. **Long-term realistic value: $1-5B sustained / $10-15B upside** at Chainlink-tier success (per `prevent_copy/PWM_REALISTIC_VALUATION_2026-05-20.md`). This is achievable IF the protocol successfully bootstraps demand-side users — which is the entire focus of the user-acquisition strategy.

---

## 1. Critical Clarification: The Token System IS Deployed at D9

Director's question begins with "If there is no token system..." — this premise is incorrect for PWM. Let me clarify:

### 1.1 What deploys at Phase 5 mainnet (D9)

ALL of these contracts are deployed at Phase 5 mainnet:

| Contract | What it does | Active at D9? |
|---|---|---|
| `PWMToken.sol` | ERC20 token; 21M fixed supply | ✅ YES |
| `PWMRegistry.sol` | On-chain artifact registry (Principles, Specs, Benchmarks, Certs) | ✅ YES |
| `PWMCertificate.sol` | L4 certificate issuance | ✅ YES |
| `PWMGovernance.sol` | 5-multisig + voting | ✅ YES |
| `PWMVesting.sol` | Founding team vesting (immutable beneficiary) | ✅ YES |
| `PWMMintingERC20.sol` | Programmatic emission for L4 mining | ✅ YES (capped) |
| `PWMRewardERC20.sol` | Reward distribution per rank | ✅ YES |
| `PWMStakingERC20.sol` | Staking + slashing logic | ✅ YES (capped) |
| `PWMTreasuryERC20.sol` | Reserve + grant disbursement | ✅ YES |

**The complete token system is functional from D9 day 1.** Mining can technically happen from D9 day 1.

### 1.2 What "phased deployment" actually means

The `PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` doc was about **what to FEATURE in the launch narrative**, not what to deploy:

| Aspect | Status at D9 |
|---|---|
| Token contracts deployed | ✅ ALL deployed (`PWM_DISPATCH_PLAYBOOK_IMPROVEMENTS` Phase 5) |
| Token supply minted | ✅ 21M cap reached at deploy |
| Mining mechanism live | ✅ Code-functional from D9 |
| Staking mechanism live | ✅ Code-functional from D9 |
| Soft-launch caps | ✅ ACTIVE — `STAKING_TVL_CAP_USD = $100`, `MINTING_PAUSED = true`, `TREASURY_TRANSFERS_PAUSED = true` |
| Launch narrative feature mining? | ✅ YES from Phase 1b (D9+30) — AI4Science as value lead; mining = production of AI4Science (REVISED 2026-05-22) |
| Launch narrative feature mine-to-use? | ❌ NO (deferred until Phase 3 / Year 2+) |
| **Launch narrative feature benchmark + AI4Science + verified leaderboard** | ✅ YES |

**Critical:** The soft-launch caps (`MINTING_PAUSED = true`, `TREASURY_TRANSFERS_PAUSED = true`) at D9 mean the token system is deployed but bounded. After 30-day audit window, Director unpauses minting + treasury transfers; full token economics becomes active.

**So: token system deploys at D9. Caps protect users for 30 days. Then mining + transfers activate.**

---

## 2. Why PWM Has Value — The 6 Value Drivers

PWM token value emerges from 6 distinct mechanisms. Each contributes independently; the combination is the value moat.

### 2.1 Fixed Supply Scarcity (21M cap)

PWM has a **fixed maximum supply of 21,000,000 tokens.** No inflation mechanism exists. No central party can mint more tokens.

| Allocation | Amount | Mechanism |
|---|---|---|
| Mining pool (programmatic emission) | 17,220,000 PWM (82%) | Emission via L4 reproduction certificates; decays over time per Zeno formula |
| Reserve (Foundation discretionary) | 2,100,000 PWM (10%) | Bounty pool + grants + Director's discretion under governance |
| Liquidity (Uniswap PWM/USDC LP) | 1,050,000 PWM (5%) | Seeded at Phase 2 launch |
| Founding team (PWMVesting) | 630,000 PWM (3%) | 4-year linear, 1-year cliff, immutable beneficiary |

**Why this matters:** unlike inflationary tokens (which dilute value over time), PWM's fixed cap means demand growth translates directly to value growth. Comparable to Bitcoin's cap; differentiates from Ethereum's annual issuance.

**Investor-recognizable framing:** "21M fixed supply. No inflation. Scarcity-driven appreciation."

### 2.2 Utility-Fee Revenue (Pay-to-Use)

Users pay PWM to access protocol services. This creates direct demand-side revenue for the token.

**Fee sources:**

| Action | Fee | Phase available |
|---|---|---|
| Submit benchmark entry (skin-in-the-game stake) | 1-10 PWM, refundable on scope-pass | Phase 2+ |
| Run inference on existing AI4Science method (mine-to-use) | 0.1-1 PWM per inference | Phase 3+ |
| Premium benchmark hosting (private test sets) | 100-1,000 PWM/benchmark | Phase 3+ |
| Custom MCP server integrations | 10-100 PWM/integration | Phase 3+ |
| Foundation grants (PWM-denominated) | Variable | All phases |

**Revenue flow:**

```
User pays PWM fee
   ↓ split per PWMRewardERC20.distribute formula
       ↓ p×55% to AC (algorithm contributor; the submitter)
       ↓ (1-p)×55% to CP (compute provider; the verifier)
       ↓ 15% to L3 (Solution / AI4Science layer royalty)
       ↓ 10% to L2 (Spec author royalty)
       ↓ 5% to L1 (Principle author royalty)
       ↓ 15% to T_k pool (Principle-specific royalty accumulation)
```

**Why this matters:** every protocol interaction generates real PWM revenue flowing through the system. Token velocity is sustained by actual utility, not speculation.

### 2.3 Governance Rights

PWM holders vote on Foundation decisions. Governance rights have value beyond speculation.

**Governance voting weight:**

- **Phase 1 (Months 1-6):** 5-multisig (Path A bootstrap; Director-controlled). No public voting yet.
- **Phase 2 (Months 6-12):** Co-founder rotations begin; multisig diversifies; public proposal feedback period.
- **Phase 3 (Year 2+):** DAO activates. PWM holders vote on:
  - Foundation Reserve grants (>50K PWM)
  - Bounty additions / removals
  - Protocol parameter changes
  - Benchmark approval
  - Treasury management
  - Major roadmap decisions

**Voting weight ≈ token holdings.** A holder with 10K PWM has 10x the voting power of a 1K PWM holder.

**Why this matters:** institutions + research labs that want a voice in PWM's evolution will accumulate PWM. Universities, AI companies, foundations may all hold PWM for governance reasons.

### 2.4 Royalty Accumulation per Principle (T_k Pools)

This is unique to PWM. Each Principle has a dedicated T_k pool that accumulates 15% of all fees from benchmarks under that Principle.

**T_k pool mechanism:**

- Every benchmark under Principle k contributes 15% of its fee flow to T_k
- T_k accumulates over time (years, decades)
- Authors of Principle k can withdraw from T_k per governance-defined rules
- Long-running Principles (CASSI, low-dose CT, etc.) accumulate substantial T_k pools

**Worked example:**

Imagine CASSI Principle has 5 active benchmarks under it. Each benchmark generates ~1,000 PWM/year in fees. Then:
- Annual T_k contribution = 5 × 1,000 × 0.15 = 750 PWM/year
- Over 5 years = 3,750 PWM accumulated
- Over 10 years = 7,500 PWM
- Long-tail accumulation can reach tens of thousands of PWM per major Principle

**Why this matters:** **Principle authors have a long-term economic stake in their Principle's success.** This is unprecedented — researchers who establish a foundational Principle benefit from all future work under that Principle.

This is analogous to **academic citation royalty** (which doesn't exist financially today). PWM creates a financial mechanism for what citations historically rewarded only socially.

### 2.5 Mining Rewards (Compute-for-Tokens)

Researchers reproduce benchmark submissions and earn PWM. This is the L4 mining mechanism.

**Mining economics (Phase 2+):**

- Submit a verified L4 reproduction certificate
- Earn PWM via `PWMMintingERC20.mintFor`
- Distribution per draw: 40% rank-1, 5% rank-2, 2% rank-3, 1% rank-4-10
- Decay via Zeno formula (mining rewards decline as supply approaches 17.22M cap)

**Realistic Year 1 mining-pool emission:** Director-targeted strong-success scenario is 2.7M PWM/year initial emission, declining via Zeno decay. A top miner earning rank-1 on 5 benchmarks/year could earn 100K-1M PWM/year.

**Why this matters:** miners need PWM staked to participate (stake-to-mine). This creates buy-side pressure on the token from anyone who wants to mine.

### 2.6 Network Effects from Real Adoption

This is the highest-leverage value driver but emerges last (Year 2+).

**Network effects:**

- Each new Principle attracts more benchmarks under it
- Each new Benchmark attracts more AI4Science submissions
- Each new submission generates more L4 reproduction certificates
- Each new certificate generates more T_k pool contributions
- Each T_k pool contribution attracts more Principle authors

**This is the AlphaFold / Chainlink pattern:** once a critical mass of researchers cites PWM cert hashes in papers, defection becomes infeasible. The protocol becomes embedded in academic infrastructure.

**Realistic timeline to network effects:** Year 3-5 for full Chainlink-tier value emergence.

---

## 3. Why Users Use PWM Tokens — The 8 Use Cases

PWM has 8 distinct utility paths. Each addresses a different user motivation.

### 3.0 Canonical PWM role terminology (CORRECTED 2026-05-22)

**An earlier version of this doc distinguished "Submit-to-Earn" from "Mine-to-Earn." That distinction was incorrect — they're the same operation.** Director's 2026-05-22 correction:

> "There is no submitter. Here submitter is like the miner. The original plan, the miner in Genesis will never pay."

**Submitter = Miner.** Same entity. Submitting a certificate (L4) IS mining. The certificate triggers ranked-draw rewards via `PWMRewardERC20.distribute`.

#### The two PWM roles

| Role | What they do | Earns? | Pays? |
|---|---|---|---|
| **MINER** (= "Submitter") with sub-roles: SP/AC (algorithm author, `p × 55%`) and CP (compute provider, `(1-p) × 55%`) | Generates L4 certificates by running algorithms against benchmark test sets | ✅ YES from ranked-draw rewards | ❌ NO in Genesis (per Director doctrine) |
| **USER** (researcher / clinician / AI agent) | Queries leaderboard; consumes verified outputs; uses MCP for AI-assistant queries | ❌ NO (not a token-earner) | ❌ Free in Phase 1-2; ✅ pays for advanced verified runs in Phase 3+ |

#### The reward distribution formula (verified in `PWMRewardERC20.sol:180-185`)

```
AC (algorithm author / SP):     p × 55%
CP (compute provider):          (1-p) × 55%
L3 (benchmark author):                15%
L2 (spec author):                     10%
L1 (principle author):                 5%
T_k (treasury / principle):           15%
                                   ─────
                                   100%
```

Ranked-draw multiplier: Rank 1 → 40%; Rank 2 → 5%; Rank 3 → 2%; Ranks 4-10 → 1% each.

#### Who pays in PWM

**Genesis miners NEVER pay.** Only USERS pay (Phase 3+ for advanced verified runs like large-scale verification or private benchmarks).

Director's phased monetization (corrected 2026-05-22 — mining ACTIVE from Phase 1b):

| Phase | Window | Mining active? | Miners pay? | Miners earn? | Users pay? |
|---|---|---|---|---|---|
| **Phase 1a** | D9 to D9+30 | ❌ NO (`MINTING_PAUSED = true` during audit) | ❌ NO | ❌ NO (paused) | ❌ NO (read-only) |
| **Phase 1b** | D9+30 to D9+180 | ✅ **YES** | ❌ NO (Genesis doctrine) | ✅ YES (full ranked-draw from `PWMMintingERC20` + 10K Reserve sponsor bonus for PWM-CI-1 top-3) | ❌ NO (free) |
| **Phase 2** | Months 6-12 | ✅ YES | ❌ NO (Genesis) | ✅ YES (broader rewards across 2-3 benchmarks + T_k royalty distributions) | ❌ NO (still free) |
| **Phase 3** | Months 12-24 | ✅ YES | ❌ NO for Genesis | ✅ YES (T_k royalties mature) | ✅ YES for advanced verified runs (large-scale verification, private benchmarks) |
| **Phase 4** | Year 2+ | ✅ YES | ❌ NO for Genesis | ✅ YES (full Zeno emission) | ✅ YES at scale |

**Key revision (2026-05-22):** Mining is ACTIVE from D9+30 (post-audit), not deferred to Phase 2. The earlier framing held mining off until Phase 2 — that was based on the incorrect "submitter ≠ miner" distinction. With miner = submitter + AI4Science as the value framing (per `PWM_VALUE_FRAMING_2026-05-22.md`), mining IS the production of AI4Science solutions and should activate as soon as security gates allow (the 30-day audit window).

#### Implications for use cases below

The 8 use cases below describe what miners AND users do across phases. **"Submit benchmarks" = mine = generate a certificate.** This is the same operation, just called by different names in different contexts:

- **Marketing language (academic-friendly):** "Submit your AI4Science method to PWM-CI-1"
- **Protocol language (canonical):** "Generate a certificate as a miner via SP/AC or CP role"
- **Same operation.** Both produce an on-chain L4 certificate that triggers `PWMRewardERC20.distribute`.

See `PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` §4.0 for the full canonical vocabulary + reward formula.

### 3.1 Submit Benchmarks (Entry Stake)

**Phase:** 2+ (Months 6-12)

**Who:** AI4Science method submitters (PhD students, postdocs, junior PIs)

**What they do:** Stake 1-10 PWM (refundable on scope-pass) when submitting a method to a benchmark.

**Why:** Skin-in-the-game prevents spam submissions. Failed submissions forfeit stake; serious submissions get stake back. Tests the fee mechanism without driving away users.

**Token velocity:** ~10-30 submissions per benchmark × ~5 PWM stake = 50-150 PWM circulating per benchmark.

### 3.2 Earn Top-3 Prizes (Reward)

**Phase:** 1+ (D9 to D9+90 first round)

**Who:** Top-ranked AI4Science method submitters

**What they do:** Win 1,000-5,000 PWM per top-ranked submission.

**Why:** Compensates winners; creates concrete demand for PWM tokens.

**Token velocity:** ~10K PWM per benchmark distributed to top winners. Across 5 active benchmarks Year 1 = 50K PWM in prize circulation.

### 3.3 Pay to Run Inference (Mine-to-Use)

**Phase:** 3+ (Year 2+)

**Who:** Clinical labs, AI companies, academic researchers wanting to run existing methods on own data

**What they do:** Pay 0.1-1 PWM per inference to run a published AI4Science method on their own data via PWM's verified infrastructure.

**Why:** Get the verified, reproducible output of a well-known method without rebuilding it. Especially valuable for clinical data + AI training data.

**Token velocity:** If 1,000 inferences/month × 0.5 PWM average × 12 months = 6,000 PWM/year per active method. Across 50 active methods = 300,000 PWM/year demand.

**This is the largest demand-side revenue source.**

### 3.4 Verify Submissions (Mining)

**Phase:** 2+ (Months 6-12)

**Who:** Researchers with compute who reproduce submissions

**What they do:** Stake PWM, reproduce submissions, earn PWM rewards.

**Why:** Earn PWM for computational work. Build reputation as a reliable reproducer. Compute-for-tokens economics.

**Token velocity:** ~$50-100K/year per top miner; ~$5-20K/year per active miner. Across 30 active miners = $0.5-1M/year in PWM rewards.

### 3.5 Accumulate Principle Royalties (T_k Pool)

**Phase:** All (T_k accumulates from any fee activity)

**Who:** Principle authors (Director + co-founders + future contributors who publish foundational Principles)

**What they do:** Authors of a Principle receive 5% of all fees generated by benchmarks under their Principle, plus 15% accumulates in the Principle's T_k pool for long-term royalty distribution.

**Why:** Direct economic stake in their Principle's adoption. Academic citation revenue, made financial.

**Token velocity:** ~750-7,500 PWM/year per major Principle (per §2.4 worked example).

### 3.6 Vote on Governance

**Phase:** 3+ (DAO activation Year 2+)

**Who:** PWM holders with significant stake (institutions, foundations, major researchers)

**What they do:** Vote on Foundation Reserve grants, bounty additions, protocol parameters, benchmark approval, treasury management.

**Why:** Have a voice in PWM's evolution. Influence Foundation decisions. Defend against capture by hostile actors.

**Token velocity:** Governance staking ≈ long-term holdings. Reduces circulating supply.

### 3.7 Hold for Appreciation

**Phase:** All (speculation present from D9; intensifies post-LP)

**Who:** Long-term believers in the protocol (Foundation supporters, academic backers, early users, crypto investors)

**What they do:** Buy + hold PWM, expecting value appreciation as protocol matures.

**Why:** Speculation on Chainlink / Arweave-tier outcomes ($1-15B sustained). Reference `prevent_copy/PWM_REALISTIC_VALUATION_2026-05-20.md` for realistic appreciation scenarios.

**Risk:** Speculation can damage academic credibility if it dominates discourse. Crisis Comms playbook addresses this.

### 3.8 Liquidity-Provide on Uniswap

**Phase:** 2+ (LP seeded Month 6-12)

**Who:** DeFi LPs, market makers, sophisticated traders

**What they do:** Provide PWM/USDC liquidity on Uniswap v3; earn LP fees + protocol-incentive PWM (if Foundation provides).

**Why:** Earn yield from PWM/USDC trading. Support price discovery and protocol health.

**Token velocity:** LP commitment locks PWM. Improves price stability.

---

## 4. The Value Equation (How These Combine)

PWM's value at any phase is approximately:

```
Token Value ≈ Σ_phases [
  Utility-fee revenue          (§2.2)
  + Governance premium         (§2.3)
  + Network-effect amplifier   (§2.6)
  + Speculation discount       (negative if speculation excessive)
] × Scarcity multiplier (§2.1)
```

**Phase 1 (Months 1-6):** Mostly speculation + minimal real utility. Value modest.

**Phase 2 (Months 6-12):** Utility-fee revenue + mining starts to dominate speculation.

**Phase 3 (Year 2+):** Network effects + mature utility-fee revenue + governance premium drive long-term value.

### 4.1 Realistic value at each phase (cross-reference `PWM_REALISTIC_VALUATION_2026-05-20.md`)

| Phase | Timeline | Realistic PWM/USDC price | Total market cap |
|---|---|---|---|
| **Phase 1** | Months 1-6 (pre-LP) | $0 (no LP) | N/A |
| **Phase 1 end → Phase 2 LP seeded** | Month 6-7 | $1-5 (LP initial discovery) | $21-105M |
| **Phase 2 mature** | Months 6-12 | $5-25 | $105-525M |
| **Phase 3 mature (3-5 years)** | Year 3-5 | $25-150 ("Strong success") | $525M-3.15B |
| **Massive success** | Year 5-7 | $150-700 (Chainlink-tier) | $3.15-14.7B |
| **AlphaFold-tier** | Year 7+ | $700-2500 (rare upside) | $14.7B-52.5B |

**Probability-weighted ceiling: $1-5B sustained / $10-20B upside.** Honest assessment per `prevent_copy/PWM_REALISTIC_VALUATION_2026-05-20.md`.

---

## 5. How PWM Avoids the Helium / Filecoin / UMA Failure Pattern

The 3 failed comparable tokens (Helium, Filecoin, UMA) collapsed because supply-side mining inflated indefinitely while demand-side revenue never materialized.

**PWM avoids this with 6 specific mechanisms:**

### 5.1 Soft-launch caps prevent early collapse

`STAKING_TVL_CAP_USD = $100`, `MINTING_PAUSED = true`, `TREASURY_TRANSFERS_PAUSED = true` for first 30 days. Prevents catastrophic early launch failure.

### 5.2 No mining at Phase 1

`MINTING_PAUSED = true` at D9 means mining literally cannot happen. Eliminates Phase 1 supply-side inflation. Activates only after audit window + Director unpause.

### 5.3 Submit-to-earn precedes mine-to-earn

Phase 1 = submitters earn, not miners earn. Forces real submission activity before mining starts. Tests demand-side before supply-side.

### 5.4 Demand-side validation gate before LP

LP seeded only when 30+ submissions hit Track A. Forces real demand-side activity before token has discoverable price. Prevents Helium-pattern price collapse.

### 5.5 Two-track diversification

Track A (computational imaging) + Track B (medical imaging flagship) diversifies failure modes. If one track stalls, the other compensates.

### 5.6 Mine-to-use comes last

Director's mine-to-use mechanism is the FINAL phase, not the first. Comes after Phase 1 demand validation + Phase 2 LP discovery. Operates on top of established utility, not as a bootstrap mechanism.

**Result: PWM's tokenomics structurally avoids the failure pattern of comparable tokens.**

---

## 6. Direct Answers to Director's Question

> "How can users treasure the PWM token in the future?"

**6 ways users treasure PWM:**

1. **As a scarce store of value** (21M fixed cap; deflationary if any tokens are lost / burned)
2. **As a productive asset** (stake to earn LP fees, mining rewards, governance votes)
3. **As a utility token** (pay to run inferences, submit benchmarks)
4. **As governance equity** (vote on Foundation decisions; influence protocol)
5. **As a citation royalty stake** (Principle authors hold tokens for T_k pool participation)
6. **As speculation on Chainlink / Arweave-tier outcomes** (with realistic upside per `PWM_REALISTIC_VALUATION_2026-05-20.md`)

> "Why do users use PWM token in the future?"

**8 reasons users actively USE PWM:**

1. **Submit benchmarks** — stake PWM to enter (1-10 PWM)
2. **Win prizes** — earn PWM for top-3 ranked submissions (1K-5K PWM)
3. **Pay-to-run inference** — pay PWM to run AI4Science methods on own data (0.1-1 PWM/inference)
4. **Mine via reproduction** — earn PWM by verifying others' submissions
5. **Earn Principle royalties** — Principle authors receive 5% of fees + 15% to T_k pool
6. **Govern via voting** — vote on Reserve grants, bounty additions, protocol parameters
7. **Hold for appreciation** — speculation on protocol success
8. **LP on Uniswap** — earn LP fees + protocol-incentive PWM

---

## 7. The Phasing of Token Utility (When Each Use Case Activates)

| Use case | Phase 1 (Months 1-6) | Phase 2 (Months 6-12) | Phase 3 (Year 2+) |
|---|---|---|---|
| 1. Submit benchmarks | 🚫 Free Round 1 | ✅ Small stake (1-10 PWM refundable) | ✅ Full fee mechanism |
| 2. Earn prizes | ✅ Top-3 prizes | ✅ Active | ✅ Active |
| 3. Pay-to-run inference | 🚫 Not enabled | 🚫 Not enabled | ✅ ACTIVATED (Director's mine-to-use) |
| 4. Mine via reproduction | 🚫 Minting paused | ✅ ACTIVATED | ✅ Active + scaled |
| 5. Earn Principle royalties | 🟡 Accumulates in T_k pool | ✅ Distributable | ✅ Distributable |
| 6. Vote on governance | 🟡 5-multisig (Path A) | 🟡 Multisig + public feedback | ✅ DAO activated |
| 7. Hold for appreciation | ✅ Speculation present | ✅ Speculation + utility | ✅ All factors active |
| 8. LP on Uniswap | 🚫 No LP yet | ✅ LP seeded | ✅ Mature LP |

**Phase 1 has limited utility — but that's intentional.** Soft-launch caps prevent early collapse. Phase 2 onwards enables the full token economic structure.

---

## 8. Real Worked Example — A PhD Student's Journey with PWM

To make this concrete, here's how a PhD student in computational imaging interacts with PWM across 18 months:

### 8.1 Month 0 (D9 launch)

- Hears about PWM-CI-1 (CASSI Reconstruction) via Twitter/X post from Director
- Visits physicsworldmodel.org/benchmarks/pwm-ci-1
- Reads the spec + technical report on arXiv
- Sets up Ethereum-compatible wallet on Base mainnet (5 minutes)

**Token interaction:** None yet.

### 8.2 Month 1 (D9+30)

- Clones github.com/integritynoble/pwm-ci-1
- Implements their novel CASSI reconstruction method (their PhD work)
- Tests locally against the public mini-test-set
- Submits via `./scripts/submit.sh --wallet 0xPhDStudent`

**Token interaction:** First L4 reproduction certificate generated. PSNR = 32.5 dB.

### 8.3 Month 2 (D9+60)

- PhD student's method ranks 4 on PWM-CI-1 leaderboard
- Misses top-3 by 0.8 dB
- BUT receives citable L4 certificate hash for their paper
- Cites cert hash in their PhD thesis chapter

**Token interaction:** 0 PWM earned (not top-3); 0 PWM paid (Round 1 sponsored). Receives certificate.

### 8.4 Month 3-6 (Phase 2 transitions)

- PhD student iterates their method
- Resubmits with PSNR = 33.8 dB
- Now ranks #1 on PWM-CI-1
- Receives **5,000 PWM prize**

**Token interaction:** PhD student now holds 5K PWM.

### 8.5 Month 7 (LP seeds)

- LP active on Uniswap. PWM/USDC = $2.50
- PhD student's 5K PWM = $12,500 USD equivalent
- They have options:
  - **Sell:** Convert to USDC for grant supplementation
  - **Hold:** Wait for appreciation
  - **Stake:** Earn governance voting weight + mining rewards
  - **LP:** Provide liquidity on Uniswap, earn LP fees

**Token interaction:** PhD student becomes a token holder with options.

### 8.6 Month 8-12 (Phase 2 active)

- Their CASSI Principle (which they helped author) starts accumulating T_k pool fees
- 5% of all fees from CASSI-related benchmarks flow to them
- They co-author 2 more benchmarks (PWM-CI-2 compressed sensing, PWM-CI-3 spectral imaging)

**Token interaction:** PhD student earns ~500-2,000 PWM/year from Principle royalties.

### 8.7 Month 12-24 (Phase 3 active)

- Their published method (top-ranked CASSI reconstruction) is now widely used
- Other researchers + clinical labs run their method on own data via PWM mine-to-use mechanism
- Each inference generates 0.1-1 PWM in fees
- 15% flows to their L3 Solution royalty: ~50-500 PWM/month

**Token interaction:** PhD student now earns passive PWM income from their method's usage.

### 8.8 Year 2+ (DAO activation)

- PhD student holds ~10K PWM total
- DAO activates; their 10K PWM = significant voting weight
- They vote on PWM-CI-4 spec approval, on Reserve grants, on benchmark deprecation
- Their voice in governance matters because they hold PWM AND have contribution history

**Token interaction:** PhD student is now a full participant in Foundation governance.

### 8.9 What did the PhD student gain?

By Year 2:
- 1 PhD chapter using PWM cert hash citation
- ~10-30K PWM held (= ~$25K-$750K depending on price trajectory)
- Public reputation as CASSI Principle author
- Long-term royalty income from their method
- Voting weight in PWM Foundation governance
- Citable proof their method works (L4 certificates)
- Co-authorship on 3-5 benchmark papers

**This is what "treasuring" PWM means.** Not speculation — a real economic + reputational stake in their domain.

---

## 9. Securities Law Caveat (Important)

PWM is structured as a **utility token for protocol participation, not as an investment contract.**

### 9.1 What this means

- Foundation does NOT market PWM as an investment
- All PWM communications emphasize utility (use, govern, participate) not speculation (price, returns, profit expectations)
- No promise of price appreciation
- No guaranteed returns
- No Foundation-driven token price management

### 9.2 What users should understand

PWM has *expected* utility value (described in this doc) but no guaranteed value. Users acquiring PWM should:

1. Acquire for protocol participation, not investment
2. Understand the utility paths (this doc §3)
3. Consult a crypto-aware attorney for tax + securities considerations
4. Not rely on Foundation for price stability

### 9.3 Foundation policy

Per `coordination/PRE_DEPLOY_RISK_AUDIT_2026-05-21.md` §6 (legal/regulatory):
- Foundation does NOT engage in price-management activity
- LP seeding is for liquidity, not price support
- Market maker engagement (if any) is for spread, not price
- Token rewards are for protocol contributions, not for promoting token value

---

## 10. Cross-References

- `pwm-team/coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` — Phasing of features (this doc complements that one)
- `pwm-team/coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` — Demand-side strategy (Track A + Track B)
- `pwm-team/coordination/PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` — Marketing copy
- `pwm-team/coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` — Layer 1/2/3 compensation
- `pwm-team/coordination/prevent_copy/PWM_REALISTIC_VALUATION_2026-05-20.md` — Honest token value estimates
- `pwm-team/coordination/prevent_copy/PWM_TOKEN_VALUE_DEFENSE_2026-05-20.md` — 12-month execution sprint
- `pwm-team/coordination/prevent_copy/PWM_COMPETITIVE_DEFENSE_2026-05-20.md` — Structural defenses
- `pwm-team/coordination/PRE_DEPLOY_RISK_AUDIT_2026-05-21.md` §6 — Legal / regulatory
- `pwm-team/coordination/CRISIS_COMMS_PLAYBOOK_2026-05-21.md` — Token-volatility crisis response
- `infrastructure/agent-contracts/contracts/PWMToken.sol` — 21M cap ERC20
- `infrastructure/agent-contracts/contracts/PWMMintingERC20.sol` — Mining mechanism
- `infrastructure/agent-contracts/contracts/PWMRewardERC20.sol` — Reward distribution per rank
- `infrastructure/agent-contracts/contracts/PWMStakingERC20.sol` — Staking + slashing
- `infrastructure/agent-contracts/contracts/PWMTreasuryERC20.sol` — Reserve disbursement
- `infrastructure/agent-contracts/contracts/PWMVesting.sol` — Founding team vesting
- `infrastructure/agent-contracts/contracts/PWMGovernance.sol` — Multisig + DAO
- `pwm-team/plan.md` — Master multi-track plan

---

## 11. The Single Most Important Framing

**PWM is not a token mining protocol that adds science later. PWM is the verified AI4Science platform for physics-grounded problems, with a sophisticated token economy supporting verified solution creation + consumption.** (See `PWM_VALUE_FRAMING_2026-05-22.md` for the canonical reasoning behind this framing.)

The token economy:
- Provides 6 distinct value drivers (scarcity, utility-fee revenue, governance, T_k royalties, mining, network effects)
- Provides 8 distinct use cases (submit, earn prizes, pay-to-run, verify, accumulate royalties, govern, hold, LP)
- Activates in 3 phases (Submit-to-Earn → Verify-to-Earn → Mine-to-Use)
- Avoids the Helium / Filecoin / UMA failure pattern through structural safeguards

**Users "treasure" PWM because it's both a productive asset (earns through participation) AND a governance instrument (votes on Foundation).** Users "use" PWM across 8 distinct paths matching their motivation (researcher / institution / miner / investor).

---

## 12. The Single Sentence

**PWM token value emerges from 6 mechanisms (scarcity + utility-fee revenue + governance + Principle royalties + mining rewards + network effects) and is captured through 8 use cases (submit / earn / pay-to-run / verify / accumulate royalties / govern / hold / LP) phased over 3 stages (Submit-to-Earn → Verify-to-Earn → Mine-to-Use) — not from speculation alone.**

This is why users will treasure PWM and use it in the future.

---

*This doc is the canonical reference for PWM token utility and value. Update when major utility paths change. Update when value drivers shift (e.g., new use case added, mining cap reached). Update at Phase 1 → Phase 2 transition (~D9+180) and Phase 2 → Phase 3 transition (~D9+365) with actual vs expected activation.*
