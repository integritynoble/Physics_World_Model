# Bounty 5 — Smart Contract Security Program (Tiered)

- **Total amount:** 500,000 PWM (largest single bounty)
- **Filename note:** retains `05-contracts-competing.md` for link stability; the "competing impl" framing is superseded by this tiered security program per the 2026-05-22 reframe (see §"Framing history" below)
- **Opens:** D9+0 for 5A and 5C; D9+30 for 5B; D9+0 for 5D (Immunefi pool seeded at deploy)
- **Reference target:** the single deployed PWM contract suite at `infrastructure/agent-contracts/contracts/` (9 contracts; 199 tests; audit-v3 baseline)
- **Acceptance harness:** existing Hardhat suite + multi-agent review artifacts under `pwm-team/deploy/findings/`

---

## TL;DR

Bounty #5 funds the **post-deploy security hardening** of the canonical PWM smart contract suite. It is split into four independently-claimable tiers so multiple specialist teams can each win the portion that matches their expertise.

| Tier | What | Amount | Opens |
|---|---|---|---|
| **5A** | Formal audit firm engagement | **200,000 PWM** | D9+0 |
| **5B** | Formal verification / symbolic execution | **100,000 PWM** | D9+30 (after audit firm engaged) |
| **5C** | Continuous fuzzing infrastructure | **100,000 PWM** | D9+0 |
| **5D** | Immunefi bug-bounty pool (USDC over 12 months) | **100,000 PWM** | D9+0 (seeded at deploy) |
| **Total** | | **500,000 PWM** | |

Each tier has its own acceptance gate. A single team may claim multiple tiers if their submission satisfies the gate for each.

---

## Framing history

The original Bounty #5 framing was "Smart Contract Suite (Competing Implementation)" — a second deployable PWM contract suite at a different address, modeled after Ethereum L1 client diversity (Geth / Nethermind / Besu / Erigon / Reth).

That framing was retired on **2026-05-22** because:

1. **L2 application-layer mismatch.** Ethereum's client diversity is for L1 consensus resilience — multiple clients implementing the same consensus protocol against the same chain state. PWM contracts on Base are application-layer; a "competing implementation" at a different address would create **two distinct token supplies + leaderboards + benchmark pools**, splitting the protocol rather than diversifying it. There is no equivalent of "consensus resilience" to gain.
2. **Pre-funded audit burden** on submitters ($40-120K out of pocket before claim) excluded all but mega-funded teams from competing.
3. **Double-counting with Track 7 audit funding** (`funds/PWM_PRE_DEPLOY_AUDIT_FUNDING_OPTIONS_2026-05-17.md`) which already plans Foundation-funded competitive audits at D9+30-60 via Base Builder Grants + EF ESP.
4. **Reverted to canonical wording.** `long-term-vision/pwm_overview1.md` line 1579 actually says *"Smart contract suite — 500,000 PWM — **Best audited implementation**"* — the original spec was always about audit quality, not parallel deployments. The "competing impl" framing was a downstream interpretation that drifted from the canon.

The reframe preserves the 500K PWM commitment and returns to the canonical "best audited implementation" intent — restated as a tiered security program because in 2026 the right way to spend 500K PWM on smart-contract security is not one mega-audit, but four parallel security workstreams.

---

## Why this is the largest single bounty

The 199 multi-agent-reviewed tests + audit-v3 baseline + soft-launch caps got the protocol to **deploy-ready** state at ~$200 in API costs (per `deploy/findings/SECURITY_REVIEW_2026-05-18.md`). That is sufficient for the 30-day soft-launch posture with ≤$130-260 USD-equivalent blast radius.

It is **NOT** sufficient for:
- Removing soft-launch caps (per `SECURITY_REVIEW` §2.2, **6 cap-raise-blocker issues** remain open: H-5, H-6, M-12, M-13, M-14, M-16 — all bounded by current caps but blocking cap-raise)
- Engaging large research labs, government grants, or institutional treasury allocations
- Demonstrating "credibly neutral verified scientific computing protocol" to AI labs evaluating PWM for query integration
- Surviving novel-exploit risk over the 5-10 year horizon the protocol needs

The 500K PWM funds the work that closes those gaps. At ~$1.30/PWM = **~$650K USD-equivalent** at deploy-day prices, comparable to the security budgets of similar L2-deployed protocols at their cap-raise stage (Aave V3 audit budget ~$1.2M across firms; Compound V3 ~$500K; Uniswap V4 ~$1.5M).

---

## Tier 5A — Formal audit firm engagement (200,000 PWM)

### What

Engage **one reputable smart-contract audit firm** to deliver a full audit of all 9 deployed PWM contracts at the post-deploy state (post Phase 5.4c registry handoff).

### Acceptable firms (the Foundation publishes an allowlist 14 days before this tier opens)

Initial allowlist (open to additions on Foundation board approval):

| Firm | Model | Typical scope cost (USD) |
|---|---|---|
| Code4rena | Competitive audit pool (~30-100 auditors) | $30-100K bounty pool |
| Spearbit | Boutique firm, named senior auditors | $50-150K fixed |
| Sherlock | Watson-based competitive pool + judge layer | $40-120K |
| Cantina | Spearbit's competitive arm | $40-100K |
| Trail of Bits | Premier US-based, formal-methods-aware | $80-200K |
| OpenZeppelin | Established, founder-of-EIPs lineage | $80-180K |

Any of these qualifies. Foundation may add or remove firms by governance vote (≥2/3 weight, 14-day window).

### Scope

All 9 deployed PWM contracts at the audit-v3 tag (or successor stable tag at engagement time):
1. PWMToken
2. PWMGovernance (including the `proposeExec / approveExec / executeExec / cancelExec` family + the `proposeFounderChange` family + threshold-reached timelock semantics)
3. PWMRegistry
4. PWMTreasuryERC20
5. PWMRewardERC20
6. PWMStakingERC20
7. PWMCertificate
8. PWMMintingERC20
9. PWMVesting

Plus cross-contract wiring + the deployment script `deploy/erc20.js` + the registry handoff scripts (Path C hybrid architecture per `multisig/README.md`).

The audit must explicitly address the **6 cap-raise-blocker findings** from the 2026-05-18 multi-agent review (see `deploy/findings/SECURITY_REVIEW_2026-05-18.md` §2.2):

| ID | Issue | Required outcome |
|---|---|---|
| H-5 | `rank` field in `SubmitArgs` is caller-supplied | Confirm fix; design + verify oracle or ECDSA-verifier signature |
| H-6 | 90-day rolling activity window not implemented | Confirm fix; verify per-epoch bucket arithmetic |
| M-12 | Stuck stake when pool at cap | Confirm fix; verify partial-graduate path |
| M-13 | Rollover pool sequential drain | Confirm fix; verify inter-draw delay or minimum-competitors gate |
| M-14 | `l1/l2/l3Creator` not registry cross-checked | Confirm fix |
| M-16 | Native-ETH `depositBounty` access control | Confirm normalization to `onlyGovernance` |

### Acceptance

- Firm's final audit report published in `pwm-team/deploy/findings/<firm-name>_audit_<YYYY-MM-DD>.md`
- Zero unmitigated CRITICAL or HIGH findings (mitigation = code-fixed-and-re-audited, or governance-accepted with explicit rationale)
- All findings classified per Code4rena severity rubric (or firm's standard equivalent)
- Firm signs off on the cap-raise readiness (i.e., the residuals listed in §2.2 of SECURITY_REVIEW are closed or explicitly accepted)

### Payout structure

- **150,000 PWM** released to the firm's designated wallet on report publication
- **50,000 PWM** released to a remediation-implementer (which may be the same firm, the Foundation, an external bounty hunter who submits the fix PRs, or the Director — whoever ships the merged-and-tested code that closes the audit findings). The Director may NOT claim this 50K under Path A bootstrap (self-dealing rule); after Path A → Path B transition, Director eligibility is governance-decided.

### Funding flow

- Tier 5A funds may be paid in **USDC equivalent at point of engagement** (firms typically don't accept PWM directly during the cap-raise window). The Foundation swaps PWM→USDC via governance proposal at engagement; the bounty pool sits in PWM until that swap fires.
- Alternative: pay in PWM at engagement-day exchange rate if the firm accepts.

---

## Tier 5B — Formal verification / symbolic execution (100,000 PWM)

### What

Deliver a **formal verification artifact** for the critical economic + governance invariants of PWM. This is distinct from Tier 5A's manual audit — formal verification proves mathematical properties about all reachable states, not just "we couldn't find a bug in 4 weeks of review."

### Acceptable frameworks

- **Certora Prover** — first-class formal-methods toolchain; CVL spec language; ~$100-300/hr typical
- **Halmos** — symbolic execution from a16z; Foundry-native; open source
- **hevm + SMTChecker** — Solidity-native SMT path enumeration
- **Echidna with `assert_` mode** — invariant-style fuzzing with proof guarantees on bounded depths

A submission may use one framework or combine multiple.

### Required invariants

The submission must mechanically verify at minimum:

| Invariant family | Specific properties to prove |
|---|---|
| **Token conservation** | `sum(balances) + sum(escrowed_in_staking) + sum(escrowed_in_certificate) + sum(in_vesting) == TOTAL_SUPPLY` for all reachable states |
| **Minting cap** | `PWMMintingERC20.M_emitted ≤ M_POOL` always; monotonic non-decreasing |
| **Governance timelock** | For any executed proposal, `block.timestamp ≥ proposal.thresholdReachedAt + TIME_LOCK` (catches the H-3-class bugs) |
| **Multisig threshold** | Cannot execute any governance action with < REQUIRED_APPROVALS approvals |
| **Staking cap enforcement** | `totalActiveStakeWei ≤ maxTotalStakeWei` for all reachable states (when cap is set) |
| **Reward split totals** | Sum of all rank payouts + AC + CP + L3 + L2 + L1 + T_k = total draw (no PWM created or destroyed) |
| **Registry append-only** | No reachable code path mutates or deletes a registered artifact |
| **Founder change atomicity** | No state where `founders[]` has < NUM_FOUNDERS entries or duplicates |

### Acceptance

- Submission published as a public repository (MIT-licensed) with:
  - The spec files (CVL / Halmos / hevm / Echidna invariant files)
  - The proof artifacts (Certora job IDs, Halmos run logs, etc.)
  - A README mapping each invariant to its proof
- Re-runnable: another engineer can clone the repo + reproduce the proofs with public toolchains within 24 hours
- Foundation board (or designated reviewer) confirms the spec captures the invariants listed above + any additional invariants the submission proposes

### Payout

100,000 PWM released to submitter wallet on acceptance.

---

## Tier 5C — Continuous fuzzing infrastructure (100,000 PWM)

### What

Productionize the **fuzzing harness** that the 2026-05-18 multi-agent review left as `test/property_tests/PWMInvariants.sol` + the Hardhat invariant tests. Today these are runnable manually; this tier funds the CI pipeline, alert infrastructure, and 12 months of operations.

### Required deliverables

| Deliverable | Detail |
|---|---|
| **Echidna run as scheduled CI** | GitHub Actions workflow that runs `echidna-test test/property_tests/PWMInvariants.sol --test-limit 100000` on a schedule (e.g., weekly + on-PR-touching-contracts), uploads results to a public dashboard |
| **Foundry invariant test suite** | At least 25 additional invariant tests in `forge-std` style covering all 9 contracts; runs in CI on every PR |
| **Medusa coverage report** | At least one Medusa run per release tag, results published to `deploy/findings/medusa_<version>.md` |
| **Slither + Mythril cron** | Weekly Slither sweep + monthly Mythril re-run; diffs flagged as alerts |
| **Alert pipeline** | Foundation-controlled webhook (Slack / Discord / email) fires on any new HIGH/CRITICAL finding from any of the above |
| **Public dashboard** | Read-only web page at e.g. `security.physicsworldmodel.org` showing last run, pass/fail status, finding counts |
| **Runbook** | `pwm-team/infrastructure/agent-contracts/security/RUNBOOK.md` documenting how to triage findings, escalate, and patch |
| **12-month operations** | Submitter maintains the infrastructure for 12 months after acceptance; covers tool updates, CI cost, dashboard hosting |

### Acceptance

- Infrastructure live and triggering on the public PWM repo for at least 30 days before claim
- At least 4 weekly Echidna runs completed with public results
- Foundation security reviewer (or Director under Path A) signs off on dashboard + runbook quality

### Payout

100,000 PWM released on acceptance. Submitter is on the hook for 12-month operations; if the infrastructure goes dark in months 1-12, the Foundation may invoke clawback for unfulfilled operations (returned to Reserve).

---

## Tier 5D — Immunefi bug-bounty pool seed (100,000 PWM, paid in USDC over 12 months)

### What

Seed a **public bug-bounty program** on Immunefi (or equivalent platform: Hats Finance, Cantina Reserve) covering all 9 PWM contracts. This is the ongoing-discovery defense layer that catches novel exploits the audit + formal verification + fuzzing layers miss.

### Tier structure (USDC equivalents at seed time)

Standard Immunefi tier rubric:

| Severity | Payout (USDC) | Examples |
|---|---|---|
| CRITICAL | $25,000 | Direct theft of user funds; permanent freezing of >$10K user funds; manipulation of governance enabling theft |
| HIGH | $10,000 | Theft of unclaimed rewards; griefing >$5K user funds; cap-bypass attacks |
| MEDIUM | $2,000 | DoS attacks; griefing that costs <$5K; oracle manipulation with bounded impact |
| LOW | $500 | Off-chain attack surfaces; precision-loss attacks <$100; informational issues |

### How the 100K PWM funds this

- At seed time, Foundation swaps 100K PWM → USDC via governance proposal
- USDC deposited into Immunefi escrow account
- Released to bug reporters as bounties are accepted by Foundation triage committee
- Any unclaimed balance at the end of 12 months: returns to Reserve (or rolled forward to Year 2 program by Foundation vote)

### Acceptance & ongoing operations

- Tier 5D is not "claimed" by a single submitter; it's an ongoing Foundation-run program
- 100K PWM moves from Reserve → USDC → Immunefi escrow at deploy
- The PWM-side bounty is the **seed**; the 12-month operating cost (triage time, Immunefi platform fees) is a Foundation expense, not part of the 100K seed

### Why Immunefi (not in-house bounty)

Immunefi has the infrastructure for KYC/anti-fraud, severity adjudication, and reporter privacy that running a bounty in-house would take 6+ months to build. Same reason every comparable protocol (Aave, Compound, Uniswap, Curve) uses Immunefi.

---

## What this tier reframe replaces from the old spec

| Old spec section | Reason for removal |
|---|---|
| "Independent reimplementation of the 7 PWM core contracts" | L2 mismatch with Ethereum-client framing (see §"Framing history" above) |
| "Same ABI, same economic semantics, different codebase" | Splits the protocol; no consensus-resilience gain |
| "53-test suite" | Stale; actual is 199 tests after multi-agent-review expansion |
| "Two independent audits commissioned by submitter" | Excluded all but mega-funded teams; conflicted with Track 7 grant-funded audit plan |
| "90-day shadow run" | Not applicable to security work; replaced with audit-published-report + on-chain monitoring |
| "Gas parity ±10%" | Not relevant to security work; relevant if competing impl, which is removed |

---

## What stays the same from the canonical spec

- **500,000 PWM total** — canonical commitment from `pwm_overview1.md` line 1579 preserved
- **"Best audited implementation"** intent — restored to the canonical wording
- **MIT-licensed** — all submissions remain MIT-licensed
- **Public audit reports** — all work products remain publicly readable
- **No KYC / equity / personal data** from bounty participants — only payout wallet required
- **Multisig releases** — payouts flow through 3-of-5 Path A multisig today (Path B post-rotations); >50K disbursements may require DAO vote per `pwm_overview1.md` §10 ("Spending above 50,000 PWM requires a DAO governance vote (≥2/3 weight, 14-day window)"). Foundation policy at time of payout governs.

---

## Founder claim rules (consistent with INDEX.md doctrine)

Director may NOT claim any tier of Bounty 5 under Path A bootstrap (self-dealing rule). Specifically:

- **Tier 5A** — Director cannot be the audit firm or the remediation implementer
- **Tier 5B** — Director cannot be the formal-verification submitter
- **Tier 5C** — Director cannot operate the continuous fuzzing infrastructure under bounty payout (Director may operate it as part of normal protocol ops without claiming, but not for the 100K)
- **Tier 5D** — Director cannot submit bug reports for payout from the Immunefi pool

After the Path A → Path B transition (Months 1-6 post-mainnet per Track 1d), Director eligibility for each tier becomes a governance decision per the `coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` §3 framework.

---

## Coordination with Track 7 audit funding

This bounty tier is the **Reserve-side** of the audit funding strategy. The **grant-side** (Track 7) provides external USD funding to reduce Reserve burn:

- If **Base Builder Grant** lands ($25K): supplements Tier 5A; reduces PWM swap-to-USDC requirement
- If **EF ESP** lands ($50K): supplements Tier 5A + 5D; reduces PWM swap requirement
- If grants do not land in 8 weeks per Q6 fallback in `funds/PWM_PRE_DEPLOY_AUDIT_FUNDING_OPTIONS_2026-05-17.md`: Reserve PWM self-funds via governance proposal as planned

The 500K PWM commitment in this bounty stands regardless of grant outcomes. Grants reduce the PWM-to-USDC swap pressure, not the total bounty size.

---

## Timeline (relative to D9 mainnet deploy)

| Date | Milestone |
|---|---|
| D9+0 | Tiers 5A, 5C, 5D open. Foundation publishes audit-firm allowlist for 5A. 5D Immunefi pool seeded. |
| D9+0-30 | 5A: Foundation engages an audit firm from allowlist. 5C: First Echidna CI runs scheduled. |
| D9+30 | Tier 5B opens (formal verification work begins; can run parallel to 5A audit) |
| D9+30-90 | 5A audit underway; 5A report draft circulated D9+60-90. 5C 30-day operations gate passes. |
| D9+90 | 5A final report published; 50K remediation tier triggers. 5C tier eligible for claim. |
| D9+90-180 | 5B formal verification work submitted + reviewed; tier eligible for claim. |
| D9+0 to D9+365 | 5D Immunefi pool active for full first year; unclaimed balance decision at D9+365. |

---

## Submission process (per tier)

### Tier 5A
1. Foundation publishes audit-firm allowlist 14 days before D9
2. Foundation issues RFP to allowlist firms at D9+0
3. Firms bid; Foundation board selects (governance vote if >50K PWM commitment)
4. Engagement contract signed; firm begins audit
5. Audit report published; payout disbursed per §"Payout structure"
6. Remediation tier (50K) opens; first qualifying PR claims

### Tier 5B
1. Submitter opens GitHub Discussion `[BOUNTY-5B] formal verification claim — <team>` with framework + scope
2. Submitter delivers proof artifacts in public repo
3. Foundation reviewer (or designated formal-methods consultant) confirms invariant coverage + reproducibility
4. Payout disbursed

### Tier 5C
1. Submitter opens GitHub Discussion `[BOUNTY-5C] continuous fuzzing claim — <team>` with infrastructure plan
2. Submitter ships infrastructure + 30-day proof-of-operation
3. Foundation security reviewer signs off
4. Payout disbursed; submitter on hook for 12-month operations

### Tier 5D
Not a discrete-submitter tier. The Foundation operates this on behalf of PWM. Bug reporters claim individual bounties through Immunefi's standard flow.

---

## Open questions for Foundation board (post-501(c)(3))

The reframe defers several decisions to the Foundation board once it exists (target: NumFOCUS Round 4 approval Q1 2027 per Track 4b):

1. **Audit-firm allowlist additions** — initial list above; board approves additions
2. **USDC-swap mechanics** — at what PWM/USDC rate triggers the swap; how to time-tranche
3. **Tier 5D extension** — at end of Year 1, extend the Immunefi pool or sunset?
4. **Clawback enforcement** — if Tier 5C operator goes dark in months 1-12, who triggers clawback?

Under Path A, Director makes these calls per `multisig/README.md` Path C hybrid; under Foundation governance, the board does.

---

## Cross-references

- `pwm-team/long-term-vision/pwm_overview1.md` §10 "Pool Allocation" — canonical 500K commitment + "best audited implementation" wording (line 1579)
- `pwm-team/deploy/findings/SECURITY_REVIEW_2026-05-18.md` — multi-agent review baseline; §2.2 cap-raise blockers explicitly addressed by Tier 5A
- `pwm-team/funds/PWM_PRE_DEPLOY_AUDIT_FUNDING_OPTIONS_2026-05-17.md` — Track 7 grant-funding strategy that complements (does not replace) this Reserve-side bounty
- `pwm-team/coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` — Layer 2 (Reserve bounty) framework that this bounty fits within
- `pwm-team/infrastructure/agent-contracts/multisig/README.md` — Path C hybrid admin (multisig flow for governance-routed disbursements)
- `pwm-team/bounties/INDEX.md` — top-level bounty roster + founder-vs-external doctrine
