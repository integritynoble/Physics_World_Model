# Base Builder Grants — Physics World Model (PWM) Evidence Document

**Applicant:** Chengshuai (Abraham) Yang — Founder & Lead Engineer
**Project name:** PWM (Physics World Model) — Verified AI4Science Benchmark Platform
**Target grant amount:** **5 ETH (~$15K USD)** — Base Builder Grants max tier per `docs.base.org/get-started/get-funded`
**Submission date:** 2026-05-23
**Project website:** https://physicsworldmodel.org
**GitHub:** https://github.com/integritynoble/Physics_World_Model
**Mainnet:** **Already deployed 2026-05-22T18:52:09Z on Base mainnet (chainId 8453)** — see deploy log + verification below

## How this document is used

Base Builder Grants is a **retroactive, nomination-based** program — not a portal application. Per https://docs.base.org/get-started/get-funded:
- Grant size: **1-5 ETH** per recipient
- Eligibility: shipped projects with real ecosystem value
- Process: Base team identifies projects via ecosystem activity + community nominations; recipients are contacted directly

This doc is the **comprehensive evidence package** Director attaches to / references from the short nomination message at:
- Nomination form: https://docs.google.com/forms/d/e/1FAIpQLSfXuEzmiAzRhie_z9raFCF1BXweXgVt18o-DvBuRRgyTygL2A/viewform
- Public announcement: https://paragraph.com/@grants.base.eth/calling-based-builders

For the short nomination message, see the companion doc `funds/applications/BASE_NOMINATION_PARAGRAPH_2026-05-22.md`.

---

## 1. One-paragraph summary

PWM (Physics World Model) is the first **verified AI4Science platform** for physics-grounded problems, deployed on Base mainnet. Researchers submit AI methods (reconstruction algorithms, inverse-problem solvers) to **leak-proof benchmarks** whose test sets are cryptographically committed on-chain. Submissions produce verified PSNR/SSIM scores citable as L4 reproduction certificates — solving the test-set-leakage problem that plagues ML benchmarks today. PWM **already deployed to Base mainnet on 2026-05-22** with the founding two anchors: CASSI (Coded Aperture Snapshot Spectral Imaging) and CACTI (Coded Aperture Compressive Temporal Imaging). Independent on-chain verification: **10/10 PASS** (post-deploy verifier 25/25 GREEN ×2). **We are requesting 5 ETH (~$15K) — the Base Builder Grants max tier — to seed a public bug bounty + partial audit-firm engagement.** Combined with our parallel Ethereum Foundation ESP application ($50K), the total $65K funds the formal third-party smart-contract audit so we can lift the 30-day soft-launch caps and open the first PWM-CI-1 CASSI reconstruction benchmark on 2026-06-21 (D9+30).

---

## 2. What we're building

PWM is **public-goods infrastructure for verified AI4Science**. Three on-chain primitives + one off-chain audience:

- **Registry (immutable, append-only):** Principles (L1: physics-grounded models), Specifications (L2: executable problem definitions), Benchmarks (L3: leak-proof test sets), Certificates (L4: verified solutions). Hashes committed to PWMRegistry contract on Base.
- **Verification (cryptographic):** every submission runs against the **on-chain-committed test set hash** (sealed at benchmark creation). No central party can leak the test, including PWM Foundation itself. PSNR + SSIM computed deterministically.
- **Reward distribution (programmatic):** AC p×55%, CP (1-p)×55%, L3 author 15%, L2 author 10%, L1 author 5%, per-Principle treasury T_k 15%. Ranked draws (40% rank 1, 5% rank 2, 2% rank 3, 1% each ranks 4-10). All formulas baked into PWMRewardERC20.sol; no discretion.
- **Target audience:** PhD students, postdocs, and AI imaging researchers who need a credible leaderboard for their methods — papers, citations, comparison against baselines.

**Why this matters:** Test-set leakage is the #1 problem in ML benchmarks today. ImageNet leaked. SQuAD leaked. Most "SOTA" claims include unconscious test-set contamination through hyperparameter tuning. Existing mitigations (Kaggle, MLPerf, Hugging Face) depend on **trust in the host organization**. PWM removes that trust assumption: the test set hash is committed to a smart contract; even we can't reveal it. **The product is the benchmark. The blockchain is the cryptographic substrate that makes the benchmark trustworthy.**

---

## 3. Why on Base

Base specifically — not just "an L2" — for three reasons:

1. **Cost.** PWM artifacts are written frequently (CASSI + CACTI at deploy; ongoing miner submissions thereafter). Base's ~$0.05 average transaction cost makes per-artifact economics viable. Ethereum mainnet at ~$5+ per tx would make small-bounty research infeasible.

2. **Reach.** Base's Coinbase-fronted onramp lets non-crypto-native researchers (PhD students, postdocs, faculty) participate without spinning up a separate crypto stack. "Open Coinbase Wallet → sign in with Privy email → stake PWM" is the lowest-friction path we found across L1s/L2s.

3. **Public-goods alignment.** Base has explicitly funded **open-source nonprofit infrastructure** at the scientific-research intersection. PWM is structurally identical: a public good operated by a 501(c)(3)-trajectory nonprofit (NumFOCUS sponsorship pending Round 4, Oct 15), not a for-profit DAO. Founding team's 630K PWM is locked in PWMVesting (4-year linear, 1-year cliff, immutable beneficiary).

**Concrete Base-ecosystem benefit:**
- CASSI + CACTI live on Base block explorers from day 1 (explorer.physicsworldmodel.org + Basescan).
- Adds a **non-DeFi use case** to Base's portfolio — Base's protocols are 90% finance-adjacent; PWM is scientific infrastructure.
- Cross-promotes Base to ~5,000 active academic/research Twitter handles (AI4Science community) when we launch PWM-CI-1 announcements on 2026-06-21.

---

## 4. The grant ask: lift the soft-launch caps via formal audit

PWM deployed to mainnet **before** receiving grant funding because we chose a **soft-launch posture**:

- **MINTING_PAUSED=true, TREASURY_TRANSFERS_PAUSED=true, submissionPermissionless=false, STAKING_TVL_CAP=$100** on chain for the first 30 days.
- **Maximum at-risk capital is ~$100 USD-equivalent** during the soft-launch window.
- This bounds risk while audit funding is sourced; the protocol is operationally live but transactionally bounded.

Before deploy, a **multi-agent security review** ran: 10 specialized AI agents (per-contract, cross-contract, economic-attack modeling, spec consistency, deploy-script audit, aggregator) + Slither + Mythril + 199 Hardhat property tests. **Result: 2 CRITICAL + 4 HIGH + 6 MEDIUM caught and fixed; 0 issues across 9 contracts under Mythril symbolic execution; 199/199 tests passing.**

Full review documentation: https://github.com/integritynoble/Physics_World_Model/blob/master/grants/deploy/findings/SECURITY_REVIEW_2026-05-18.md

**The grant funds a complementary tier of assurance: an independent formal audit** by a credentialed firm (Spearbit / Cantina / Halborn / OpenZeppelin / Zellic). After the audit clears, **governance proposes 3 executeCall proposals** (3-of-5 multisig + 48h timelock each) to:
1. Unpause minting (`PWMMintingERC20.setMintingPaused(false)`)
2. Unpause treasury transfers
3. Raise staking cap from $100 → permissionless (`submissionPermissionless=true`)

After Phase 1b activation, the protocol operates at production scale — and **PWM-CI-1, the first CASSI reconstruction benchmark, opens for permissionless submissions**.

### Budget breakdown (5 ETH ≈ $15K)

| Item | Cost (USD) | Purpose |
|---|---|---|
| Bug bounty pool seed (Immunefi initial USDC tier) | $5,000 | HIGH tier $2K + CRITICAL tier $3K; live within 30 days of grant |
| Audit-firm partial engagement (paired with EF ESP $50K) | $8,000 | Contribution toward Cantina / Spearbit / Cyfrin engagement |
| F-3 creator-address cross-check implementation | $2,000 | A5 finding fix; ~20 LoC + tests + re-Slither/Mythril |
| **Total request** | **5 ETH (~$15,000 USD)** | |

The full $65K audit + remediation package combines this $15K with the parallel EF ESP $50K application. Any unspent Base Builder grant funds will be returned or re-allocated to additional audit scope coverage.

### Why this scope (not larger)?

PWM has a **smaller risk surface** than typical DeFi:

- **No AMM** (Uniswap LP is external; managed separately Phase 2+)
- **No oracles** (deliberately avoid price oracles)
- **No flash loans, no upgrade proxies, no cross-chain bridges**
- **Per-principle treasury isolation** — a bug in one treasury doesn't drain the whole system
- **Fixed-PWM staking floors** (10/2/1 PWM, governance-tunable) — no oracle-priced collateral
- **Curated genesis** — only 6 founder-vetted artifacts on mainnet (CASSI L1/L2/L3 + CACTI L1/L2/L3); not the 1,591 stub-tier metadata files (those stay on testnet)

A typical $50K audit budget targets protocols with $10M+ TVL. PWM's worst-case bug impact is bounded by the soft-launch caps at construction. We're in the **$15-25K audit tier**.

---

## 5. Team

**Chengshuai (Abraham) Yang** — Founder, sole engineer
- Research Associate, UT Southwestern Medical Center (computational imaging)
- ORCID: 0000-0003-2840-5344
- 11 years computational imaging research; multiple peer-reviewed publications
- Built PWM solo over 2024-2026; this is the first deployment
- Independent of UTSW for PWM (no institutional endorsement, no IP shared, no funds flow)

**Path A bootstrap:** All 5 founder hardware wallet slots in PWMGovernance currently controlled by the Founder under documented Path A bootstrap. Co-founder recruitment is active (Track 4a of master plan); rotations to genuine 3-of-5 multi-party governance happen Months 1-6 post-mainnet.

**Why this matters for the grant:** The audit firm works with a single technical contact (the Founder), simplifying communication and remediation. Post-audit, Track 4a co-founder onboarding spreads operational risk and prepares for NumFOCUS Round 4 institutional credibility.

---

## 6. Traction at application time (2026-05-22 — deploy day + 0)

### Verified state (independent on-chain verification 2026-05-22)

| Item | Value | Evidence |
|---|---|---|
| Mainnet deployed | **2026-05-22T18:52:09Z** on Base (chainId 8453) | `deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md` |
| Contracts deployed | **9 / 9** (PWMToken, PWMGovernance, PWMRegistry, PWMTreasuryERC20, PWMRewardERC20, PWMStakingERC20, PWMCertificate, PWMMintingERC20, PWMVesting) | Basescan |
| Basescan-verified source | **9 / 9** | Public auditability |
| Total supply | **21,000,000 PWM** (immutable cap) | `PWMToken.totalSupply()` |
| Genesis distribution | 17.22M to Minting pool, 2.1M to Reserve Safe, 1.05M to Liquidity, 630K to PWMVesting | `balanceOf()` verified |
| Genesis artifacts on mainnet | **6 / 6** (CASSI L1-025/L2-025-001/L3-025-001-001 + CACTI L1-027/L2-027-001/L3-027-001-001) | PWMRegistry `exists()=true` |
| PWMRegistry owner | **PWMGovernance** (3-of-5 multisig) — handed off post-genesis | Verified `owner()` call |
| Soft-launch caps active | `mintingPaused=true, transfersPaused=true, submissionPermissionless=false` | On-chain |
| Post-deploy verifier | **25/25 GREEN × 2** (Phase 5 + Phase 5.6) | `scripts/post_deploy_verify.js` |
| Independent verification (2026-05-22 evening) | **10/10 PASS** | This application |

### Pre-deploy security work

| Item | Status |
|---|---|
| Smart contract code | 9 contracts, ~1,400 LoC, immutable (no upgrade proxies) |
| Hardhat test suite | **199 / 199 passing** |
| Slither static analysis | 58 raw findings → **0 deploy-blocking after triage** |
| Mythril symbolic execution | **0 issues across all 9 contracts** (5h 18min wall, all detectors clean) |
| Multi-agent security review | 10 agent passes (A1-A10); **2 CRITICAL + 4 HIGH + 6 MEDIUM caught and fixed** |
| Sepolia governance rehearsals | Full propose → 3-of-5 approve → 48h timelock → execute on Base Sepolia |
| Sepolia testnet artifacts | 1,591 genesis artifacts; 6 founder-vetted; 0 new since 2026-05-15 |

### Websites (production live)

| URL | Backend | Description |
|---|---|---|
| https://physicsworldmodel.org | FastAPI product app | Primary product website — benchmarks, leaderboards, submit |
| https://test.physicsworldmodel.org | FastAPI product app | Test/staging entry — identical product site |
| https://explorer.physicsworldmodel.org | Next.js read-only explorer | Chain explorer — principles, benchmarks, certificates, all 3 chains |
| Next.js explorer indexers | ✅ Base mainnet + Base Sepolia + Eth Sepolia | All 3 chains indexed live |
| SSL valid until | 2026-08-20 (Let's Encrypt; auto-renewal scheduled) |

Direct evidence:
- Multi-agent review: https://github.com/integritynoble/Physics_World_Model/blob/master/grants/deploy/findings/SECURITY_REVIEW_2026-05-18.md
- Mythril clean: https://github.com/integritynoble/Physics_World_Model/blob/master/grants/deploy/findings/A4_v2_mythril_triage_2026-05-18.md
- Deploy log: https://github.com/integritynoble/Physics_World_Model/blob/master/grants/deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md

---

## 7. Timeline

```
Phase                              Date              Action
─────────────────────────────────────────────────────────────────────────────
Phase 5 mainnet deploy             2026-05-22        ✅ COMPLETE — 25/25 verifier GREEN ×2,
                                                     10/10 independent verification

Phase 1a (soft-launch audit)       2026-05-22 to     Caps active; audit window;
                                   2026-06-21        no mining; bounded blast radius

Phase 1b activation                2026-06-21        Mining ACTIVATES (governance proposals
                                                     unpause minting/treasury/submissions);
                                                     PWM-CI-1 CASSI benchmark OPEN

Grant decision (Base reviewer)     2026-06-15±       (typical 4-week turnaround)

Audit firm engagement              2026-06-19+       Within 1 week of grant disbursement;
                                                     RFP to 3-5 firms; select; sign SoW

Audit fieldwork                    2026-06-26 to     4-6 week standard scope
                                   2026-08-07

Audit remediation                  2026-08-07 to     Fix any HIGH/CRITICAL findings
                                   2026-08-21

Audit firm sign-off                2026-08-21        Tagged release: pwm-v1.0-audited

PWM-CI-1 first benchmark report    2026-08-30        arXiv preprint + workshop submission
                                                     (NeurIPS Datasets & Benchmarks)

Phase 2 (LP seeding)               2026-Q4           Uniswap v3 PWM/USDC; token price discovers
```

**Worst case if grant denied:** soft-launch continues at conservative caps. We apply to EF ESP, Sloan OSS, CZI in parallel. The protocol still functions for low-volume miners under soft-launch caps.

---

## 8. Public-goods commitments

- **Open source.** MIT-licensed. All code (contracts, web explorer, indexer, deploy scripts, documentation) public from day 1. Repository: https://github.com/integritynoble/Physics_World_Model
- **501(c)(3) trajectory.** NumFOCUS fiscal-sponsorship application targeted for Round 4 (October 15, 2026 deadline). Eventually graduating to independent PWM Foundation (years 2-3).
- **No founder PWM premine accessible at deploy.** Founding-team allocation (3% of supply = 630K PWM) is **locked in `PWMVesting.sol` with a 1-year cliff + 4-year linear release** to an immutable beneficiary address. No team tokens accessible at deploy.
- **Reserve allocation governance.** 10% of supply (2.1M PWM) held in a **separate 3-of-5 Gnosis Safe** for ecosystem grants (research bounties, infrastructure costs, conference workshops). Spending > 50,000 PWM requires DAO vote (≥⅔ weight, 14-day window) once DAO activates (post-D9+12 months).
- **Tokens are utility for protocol participation, NOT investment contracts.** Foundation does NOT engage in price-management activity, marketing tokens as investments, or speculation discourse. PWM tokens reward verified contributions; the product is the benchmark.

---

## 9. Risks we acknowledge

| Risk | Mitigation |
|---|---|
| Bug discovered during soft-launch window before audit completes | Soft-launch caps cap maximum loss at ~$100. Pause flags freeze every state-changing path within 48h via governance. |
| Audit firm finds CRITICAL issues post-funding | Re-engineer + re-audit; cap raise delayed proportionally. Worst case: grant funds extended to a v2 audit. |
| Coinbase / CEX off-ramp friction reduces miner participation | Bridge from L1 / L2 / centralized exchanges all viable; Coinbase Wallet integration removes one common friction point |
| Single-founder operational risk during Path A bootstrap | Documented in `pwm-team/coordination/wallet/PWM_DECISION_RECORD_PATH_A_2026-05-12.md`. Mitigated by 48h timelock (no instant action even with all 5 keys), key rotation drills, post-deploy co-founder recruitment. |
| PWM-CI-1 launch attracts <5 external submitters | Multiple fallbacks: extend submission window, lower friction (longer deadline, simpler scope), co-author personal recruitment from Founder's network |
| Token-speculation discourse damages academic credibility | Crisis Comms playbook (`coordination/CRISIS_COMMS_PLAYBOOK_2026-05-21.md`) defines tier-based response; lead always with AI4Science framing, never token discourse |

---

## 10. Long-term sustainability beyond this grant

This grant funds **one audit**. PWM's long-term operating budget comes from a portfolio:

| Source | Timeline | Amount |
|---|---|---|
| Base Builder Grant (this application) | Q2 2026 | $25K (audit) |
| Ethereum Foundation ESP (parallel application) | Q3 2026 | $30-50K (extended audit + operating) |
| Sloan Foundation OSS | Q4 2026 | $200K-1M (institutional, requires NumFOCUS sponsorship first) |
| CZI Essential Open Source | Q4 2026 - Q1 2027 | $250K-1M (same gating) |
| Per-principle treasury (T_k) accrual | Year 1+ | 15% of every L4 reward; per-principle isolated |
| Reserve grants (community-directed) | Year 1+ | 2.1M PWM cap; distributed by 3-of-5 → DAO governance |
| Mine-to-use usage fees (Phase 3, Year 2+) | Year 2+ | USD-denominated, paid in PWM at market rate |

**By Month 18 post-launch**, we expect operating costs (audit refresh, hosting, community workshops) to be self-funded from on-chain fee revenue, with grants becoming supplementary rather than essential.

---

## 11. Three things that make us different

1. **We deployed first, audit second — with bounded risk.** Most protocols ask grants to *plan* a launch. PWM has already shipped (verified 10/10 on-chain) with bounded risk via soft-launch caps ($100 cap). The grant funds *removing the caps*, not *enabling the launch*. **This makes the deliverable concrete: audit complete + caps raised = success criterion clear and measurable.**

2. **Multi-agent pre-audit at $200 cost.** The 7-hour Claude Opus multi-agent review caught 2 CRITICAL + 4 HIGH + 6 MEDIUM issues that would have permanently bricked the protocol on mainnet deploy. Full audit trail in `pwm-team/deploy/findings/` (16 review docs, 4 sub-GPU verification docs, 5 implementation commits). When the formal auditor opens our SoW, they'll start from a Slither-clean + Mythril-clean + multi-agent-clean codebase — meaning shorter fieldwork and lower cost. **We've already done the auditor's week-1 work.**

3. **Scientific public good with active research backing.** PWM is not a speculative protocol. Founder is an active computational imaging researcher; the protocol's first benchmarks (CASSI calibration, CACTI temporal reconstruction, planned low-dose CT) are PWM-authored peer-review-targeted papers. **The cryptographic verification mechanism solves a real, widely-documented ML problem (test-set leakage) that no existing platform (Kaggle, MLPerf, Hugging Face, Papers with Code) addresses cryptographically.** The token is the reward mechanism; the benchmark is the product.

---

## 12. Links

| Resource | URL |
|---|---|
| Project website | https://physicsworldmodel.org |
| Chain explorer | https://explorer.physicsworldmodel.org |
| Test site | https://test.physicsworldmodel.org |
| GitHub | https://github.com/integritynoble/Physics_World_Model |
| **Mainnet deploy log** | https://github.com/integritynoble/Physics_World_Model/blob/master/grants/deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md |
| Security review (multi-agent) | https://github.com/integritynoble/Physics_World_Model/blob/master/grants/deploy/findings/SECURITY_REVIEW_2026-05-18.md |
| Mythril triage (0 issues) | https://github.com/integritynoble/Physics_World_Model/blob/master/grants/deploy/findings/A4_v2_mythril_triage_2026-05-18.md |
| Slither triage | https://github.com/integritynoble/Physics_World_Model/blob/master/grants/deploy/findings/A4_slither_triage_2026-05-18.md |
| Value framing (AI4Science) | https://github.com/integritynoble/Physics_World_Model/blob/master/grants/coordination/PWM_VALUE_FRAMING_2026-05-22.md |
| User acquisition strategy | https://github.com/integritynoble/Physics_World_Model/blob/master/grants/coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md |
| Phased architecture deployment | https://github.com/integritynoble/Physics_World_Model/blob/master/grants/coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md |
| Master plan (Tracks 1-8) | https://github.com/integritynoble/Physics_World_Model/blob/master/grants/plan/PLAN.md |
| Funding strategy | https://github.com/integritynoble/Physics_World_Model/blob/master/grants/funds/PWM_PRE_DEPLOY_AUDIT_FUNDING_OPTIONS_2026-05-17.md |
| ORCID | https://orcid.org/0000-0003-2840-5344 |

### Deployed contract addresses (Base mainnet, chainId 8453)

| Contract | Address (Basescan) |
|---|---|
| PWMToken | https://basescan.org/address/0x7326781182b9cDc1eF9Fa147fB689862f893dA14 |
| PWMGovernance (3-of-5 multisig) | https://basescan.org/address/0x83F210b9A8E5F0FAfE133c700F888b3A303f9b15 |
| PWMRegistry | https://basescan.org/address/0x9F91784c2fa884A79473304050C581424E006fbd |
| PWMTreasuryERC20 | https://basescan.org/address/0xe0FE4A050a926da763907dFA872fA51ba359b061 |
| PWMRewardERC20 | https://basescan.org/address/0x06B341BBFB3435561986f7C1821551E56D909b3D |
| PWMStakingERC20 | https://basescan.org/address/0x88D7860d800Cc68d905751696C3c0B4875Af950b |
| PWMCertificate | https://basescan.org/address/0x014492dEfc66D5b58b86027cEB636d4c84289eAe |
| PWMMintingERC20 | https://basescan.org/address/0x629190D88cdB0C4a2cFEe00Dd7EdD490c465B235 |
| PWMVesting | https://basescan.org/address/0x9c57BA6f844dAAecB050D83f31A8279E04a441a9 |

---

## 13. Contact

**Chengshuai (Abraham) Yang**
- Email: integrityyang@gmail.com / platformaiyang@gmail.com
- GitHub: integritynoble
- Affiliation (research): Research Associate, UT Southwestern Medical Center (computational imaging) — independent of UTSW for PWM
- Affiliation (legal entity for grant receipt): NextGen PlatformAI C Corp (Founder); receiving entity may shift to PWM Foundation post-NumFOCUS

I am available for a 30-minute call with the Base grants team at any time during US business hours (Central). I can demonstrate the protocol live on Base mainnet (visit https://physicsworldmodel.org now) within minutes of request.

---

**Thank you for your consideration.** PWM is a scientific public good that fits Base's mission of expanding the universe of useful onchain applications beyond DeFi. The audit this grant funds is the last technical gate between the soft-launch protocol and full-scale operation. Every dollar in this grant has a direct measurable deliverable: **audit complete → 3 governance proposals execute → caps lift → PWM-CI-1 CASSI benchmark opens → first verified AI4Science submissions flow to the leaderboard.**

— Chengshuai (Abraham) Yang
2026-05-22 (Day 1 of mainnet)

---

# Appendix A — Portal form responses (copy/paste ready)

The Base Builder Grant portal at https://base.org/grants typically presents a Typeform with ~10-15 free-text fields. Below is a copy-paste-ready answer set tuned for the typical question prompts. Adjust per actual form labels.

## Project name
PWM (Physics World Model) — Verified AI4Science Platform

## Project category / type
Public-goods infrastructure / scientific research / non-DeFi

## One-sentence pitch
PWM is a verified AI4Science platform on Base where researchers submit AI methods to leak-proof benchmarks whose test sets are cryptographically committed on-chain — producing verified PSNR/SSIM scores citable as L4 reproduction certificates.

## Project URL
https://physicsworldmodel.org

## GitHub
https://github.com/integritynoble/Physics_World_Model

## Brief description (~250 words)
PWM is a verified AI4Science platform deployed on Base mainnet 2026-05-22T18:52:09Z. We solve test-set leakage in ML benchmarks: every benchmark's test set hash is committed on-chain to PWMRegistry; submissions are scored against the sealed test set deterministically; PSNR/SSIM verdicts become citable L4 reproduction certificates. Existing benchmarks (Kaggle, MLPerf, Hugging Face) depend on trust in the host; PWM removes that trust — even we can't reveal the test set.

Nine smart contracts deployed and source-verified on Basescan (~1,400 LoC). Six founder-curated CASSI + CACTI genesis artifacts on-chain. Soft-launch posture (mintingPaused, transfersPaused, submissionPermissionless=false, 100 PWM TVL cap) bounds maximum loss at ~$130-260 USD-equivalent for the first 30 days while we secure formal audit funding.

A pre-deploy 7-hour multi-agent security review (10 AI agents + Slither + Mythril + property tests) caught and fixed 2 CRITICAL + 4 HIGH + 6 MEDIUM issues; 199/199 tests pass; Mythril clean across all 9 contracts. We've done the auditor's week-1 work — formal audit fieldwork starts from a triple-clean codebase.

PWM is non-DeFi public-goods infrastructure: Apache 2.0 licensed, 501(c)(3)-aspiring nonprofit (NumFOCUS Round 4 Oct 2026), no founder premine accessible at deploy (12-mo cliff + 48-mo linear vest in deployed PWMVesting). Director is an active computational imaging researcher at UT Southwestern.

## Why Base specifically?
Base's ~$0.05 per-tx cost makes per-artifact research economics work (our entire mainnet deploy + 6 genesis registrations + governance handoff cost ~$7.50 in gas). Base's Coinbase-fronted onramp lets non-crypto-native researchers participate. Base explicitly funds public-goods nonprofit infrastructure aligned with PWM's 501(c)(3) trajectory.

## Grant amount requested
$25,000 USD (or ETH equivalent on Base)

## What will the grant fund? (specific budget)
$15,000-22,000 — formal third-party audit (Spearbit / Cantina / Halborn / OpenZeppelin standard-tier, 4-6 week fieldwork)
$2,000-4,000 — audit remediation engineering (fix any HIGH/CRITICAL findings; re-engage auditor for sign-off)
$1,000-2,000 — Immunefi bug bounty pool seed (USDC tiers for HIGH+CRITICAL bug reports)
Total: $25,000. Any unspent funds returned or extended to scoping the on-chain rank verification oracle (H-5) and 90-day rolling activity window (H-6) needed before cap-raise.

## Timeline / milestones
T+0 (grant disbursement) → 1 week to RFP + sign SoW with audit firm
T+1-6 weeks: audit fieldwork
T+6-8 weeks: audit remediation
T+~9 weeks: tagged release pwm-v1.0-audited
T+9-10 weeks: governance cap-raise via 3 executeCall proposals (48h timelock each)
T+~10 weeks: full-scale launch; PWM-CI-1 CASSI benchmark opens

## Team
Single founder at deploy: Chengshuai (Abraham) Yang. Research Associate, UT Southwestern Medical Center (computational imaging). 11 years research. ORCID 0000-0003-2840-5344. Built PWM solo over 2024-2026.

Path A bootstrap: all 5 founder HW wallets currently Director-controlled. Co-founder recruitment Track 4a active; rotations Months 1-6 post-mainnet via documented governance with 48h timelock.

## Mainnet deploy proof
Deployed 2026-05-22T18:52:09Z on Base (chainId 8453). 9 contracts, 33 transactions, all Basescan-verified. 25/25 post-deploy verifier GREEN ×2 (deployer-owned + governance-owned registry states).

PWMToken: 0x7326781182b9cDc1eF9Fa147fB689862f893dA14
PWMGovernance: 0x83F210b9A8E5F0FAfE133c700F888b3A303f9b15
PWMRegistry: 0x9F91784c2fa884A79473304050C581424E006fbd
PWMTreasuryERC20: 0xe0FE4A050a926da763907dFA872fA51ba359b061
PWMRewardERC20: 0x06B341BBFB3435561986f7C1821551E56D909b3D
PWMStakingERC20: 0x88D7860d800Cc68d905751696C3c0B4875Af950b
PWMCertificate: 0x014492dEfc66D5b58b86027cEB636d4c84289eAe
PWMMintingERC20: 0x629190D88cdB0C4a2cFEe00Dd7EdD490c465B235
PWMVesting: 0x9c57BA6f844dAAecB050D83f31A8279E04a441a9
Reserve Safe (3-of-5): 0x76ff267239Be560fB93d4E3e97076B2A7c49FdA7
Deployer (retired post-handoff): 0xA5349f9E42CeC9612E10648609F6E29d0BA0f325

Full deploy log: github.com/integritynoble/Physics_World_Model/blob/master/grants/deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md

## Contact
Chengshuai (Abraham) Yang — integrityyang@gmail.com / platformaiyang@gmail.com — GitHub: integritynoble
