# PWM Security Review — A5 Economic Attack Modeling
**Date:** 2026-05-18
**Agent:** A5 (economic / game-theoretic)
**Contracts reviewed:** PWMReward.sol, PWMStaking.sol, PWMMinting.sol, PWMTreasury.sol, PWMCertificate.sol
**Status:** COMPLETE

---

## Executive Summary

| Severity | Count |
|---|---|
| HIGH | 2 |
| MEDIUM | 5 |
| LOW / INFO | 3 |

**All attacks are bounded to trivial amounts (≤100 PWM) under current soft-launch caps.**

Three prerequisites are mandatory before caps are lifted:
1. On-chain rank verification (vs. caller-supplied rank field)
2. 90-day rolling activity window in PWMMinting (cumulative is currently implemented)
3. Creator-address registry cross-check in PWMCertificate.submit

---

## Finding 1 — Rank Manipulation / Caller-Supplied Rank (HIGH)

**Attack scenario:**
The `rank` field in `SubmitArgs` (PWMCertificate.submit) is fully caller-supplied and stored verbatim. There is zero on-chain verification of rank against any external leaderboard or benchmark result. Once `submissionPermissionless = true`, any address can submit a certificate with `rank=1` and drain 40% of a benchmark pool per finalization cycle. No per-submitter rate limit exists. A single attacker with one approved-submitter address can monopolize Rank 1 across every benchmark.

**Current mitigations:**
- `submissionPermissionless = false` under soft-launch (only governance-approved submitters)
- `maxBenchmarkPoolWei = 100 PWM` caps per-pool drain to 40 PWM per rank-1 cert

**Residual risk under soft-launch caps:** 40 PWM per finalization — trivial.

**Recommended fix (mandatory before opening submissionPermissionless):**
Introduce an on-chain oracle or ECDSA signature from a trusted verifier attesting to the submitted rank. Alternatively, implement a challenge mechanism where the rank can be disputed within the challenge window (the existing challenge flow only disputes the cert hash, not the rank value).

---

## Finding 2 — Rollover Pool Sequential Drain (MEDIUM)

**Attack scenario:**
No minimum time gap between draws on the same benchmark. Two colluding approved submitters can finalize back-to-back certs at ranks 1 and 2, extracting 40% + 5% = 45% in the first two draws, then 40% of the remaining 55% + 5% of 55% = ~24.75% in the next round. Over ~5 sequential finalizations, the two colluders capture ~85% of the original pool. The rollover accumulates perpetually if no legitimate submissions compete.

**Current mitigations:**
- Challenge period (7–14 days) introduces minimum per-cert latency
- `submissionPermissionless = false` limits who can submit
- Pool capped at 100 PWM

**Residual risk under soft-launch caps:** ≤100 PWM total pool — trivial.

**Recommended fix:** Consider a minimum inter-draw delay or a minimum number of competing submitters before a pool draw is eligible. Not urgent under soft-launch.

---

## Finding 3 — p-Parameter Siphoning + Creator Address Spoofing (MEDIUM)

**Attack scenario (two parts):**

*Part A — p siphoning:* The Stake Producer sets `p` at submission time in range [0.10, 0.90]. Setting `p = 0.90` legally routes 49.5% of draw proceeds to AC (agent controller) and only 6.05% to CP (compute provider). A colluding SP+AC pair can maximize their combined take to 55% of the draw legally.

*Part B — creator address spoofing:* `l1Creator`, `l2Creator`, and `l3Creator` in `SubmitArgs` are **caller-supplied and not cross-checked against PWMRegistry**. A colluding SP can set all three creator fields to their own address, concentrating the L1 (5%), L2 (10%), and L3 (15%) buckets as well. Combined with `p=0.90`, a single actor can legally capture (49.5% AC) + (5% L1) + (10% L2) + (15% L3) = **79.5% of a draw** — all within spec parameters, but clearly unintended.

**Current mitigations:**
- Governance controls who is an approved submitter
- `submissionPermissionless = false`

**Residual risk under soft-launch caps:** 79.5 PWM per 100 PWM pool — trivial but illustrates the design gap.

**Recommended fix (mandatory before cap raise):** In `PWMCertificate.submit`, cross-check `l1Creator / l2Creator / l3Creator` against `PWMRegistry.getArtifact(benchmarkHash).creator` and its parent chain. Reject submissions where creator fields don't match registry records.

---

## Finding 4 — Stake-Cycle Griefing (MEDIUM)

**Attack scenario:**
Stake L3 (1 PWM) → artificial benchmark gets promoted → graduation returns 50% (0.5 PWM) + seeds B-pool with 0.5 PWM → attacker submits rank-1 cert against their own benchmark → drains B-pool. Net cost to attacker per cycle: 0.5 PWM (lost to graduation seeding) minus draw proceeds. At low B-pool values this is negative expected value for the attacker but burdens governance with frequent cheap graduation requests and pollutes the benchmark registry.

**Current mitigations:**
- `submissionPermissionless = false` limits self-certification
- Staking floor (1 PWM L3) is non-trivially low but still low
- Governance controls graduation via `setParameter`

**Residual risk under soft-launch caps:** 1 PWM per cycle — trivial.

**Recommended fix:** Add a minimum stake duration before graduation is eligible (e.g., 30 days). Consider raising L3 floor to 10 PWM post-audit to increase attack cost.

---

## Finding 5 — M_pool Cap Bypass (LOW — No Exploit Found)

**Analysis:** The Zeno decay formula in PWMMinting ensures that cumulative emission converges asymptotically below M_POOL. The `M_emitted` accumulator is incremented only in `mintFor` and guarded by `require(!mintingPaused)`. No bypass path found.

**Recommendation (belt-and-suspenders):** Add an explicit `require(M_emitted + amount <= M_POOL, "cap exceeded")` assertion in `mintFor` as an invariant guard.

---

## Finding 6 — Adversarial-Bounty Drain (LOW — No Exploit Found)

**Analysis:** `payAdversarialBounty` in PWMTreasury enforces `amount <= treasury[principleId] / 2` (50% cap). CEI pattern is followed (state updated before transfer). `onlyGovernance` gate means direct calls are gated behind 48h timelock. No reentrancy path found. Dust-lock of 1 wei at odd balances is harmless.

**No fix required.**

---

## Finding 7 — Minting Weight Concentration / Missing Rolling Window (HIGH)

**Attack scenario:**
`pwm_overview1.md` specifies a 90-day rolling window for activity weighting. The M1.1 implementation uses **cumulative activity** — `activity[principleId]` is never decremented or windowed. An early actor who finalizes many certificates in months 1–3 permanently raises their `w_k` weight relative to all future entrants, concentrating emissions indefinitely even if they go inactive. This creates a first-mover perpetual advantage that contradicts the spec's intent of rewarding ongoing contribution.

**Current mitigations:**
- `mintingPaused = true` under soft-launch — no emission occurs at all

**Residual risk under soft-launch caps:** Zero (minting paused).

**Recommended fix (mandatory before unpausing minting):** Implement a rolling window by storing activity in epoch buckets (e.g., per-epoch deltas that expire after 90 days) or by using an exponential decay formula on `activity_k`. The spec's rolling window must be enforced before governance unpauses minting.

---

## Finding 8 — Cross-Principle Treasury Isolation Gap (LOW)

**Analysis:**
`certificate.principleId` in `SubmitArgs` is caller-supplied and not cross-checked against the benchmark's registered `principleId` in PWMRegistry. A certificate submitted for benchmark X (registered under principle 1) with `principleId = 2` in the cert payload would credit `T_k[2]` instead of `T_k[1]`. This is misdirection (redirecting the 15% T_k credit to a different principle's treasury) rather than a drain of the overall system. No funds are created or destroyed — they land in the wrong T_k bucket.

**Residual risk under soft-launch caps:** 15 PWM misrouted per 100 PWM pool — trivial.

**Recommended fix:** In `PWMCertificate.submit`, look up `PWMRegistry.getArtifact(benchmarkHash)` and assert that `args.principleId` matches the benchmark's registered principle lineage.

---

## Finding 9 — Soft-Launch Cap Bypass via Inconsistent depositBounty Access Control (MEDIUM)

**Analysis:**
`PWMReward.depositBounty()` (native ETH variant, if it exists) has inconsistent access control compared to the ERC20 sibling which is `onlyGovernance`. Any address can pre-fill a benchmark pool up to the `maxBenchmarkPoolWei` cap. While this does not bypass the cap itself, it allows griefing by filling a pool with dust from an uncontrolled address, which may complicate accounting or trigger unintended pool-full conditions before legitimate governance-sponsored deposits.

**Residual risk under soft-launch caps:** Pool capped at 100 PWM — trivial.

**Recommended fix:** Normalize access control: either both deposit paths are `onlyGovernance` or both are permissionless. Pick one and apply consistently.

---

## Finding 10 — Incentive Misalignments (INFO)

**Observations:**

1. **Delta self-reporting:** The `delta` field (quality score delta) is caller-supplied in `SubmitArgs`. Gaming a high delta extends the challenge window from 7 to 14 days — a rational submitter of a fraudulent cert would use `delta < 10` to minimize their exposure window, while a legitimate one would use true delta. No financial misalignment but creates adverse selection in challenge-window length.

2. **Gas-auction race at window close:** When a challenge period expires, any caller can trigger `finalize`. If a pool is large (post-cap-raise), a gas auction will form at window close. Consider requiring the original submitter to finalize (with a fallback after a grace period) to remove MEV extraction.

3. **L3 creator + approved-submitter consolidation (see Finding 3):** Legal under current rules but captures 79.5% of draw value in a colluding setup. Governance should consider requiring that creator addresses be distinct from the submitter address.

---

## Summary table

| ID | Severity | Issue | Status |
|---|---|---|---|
| F-1 | HIGH | Caller-supplied rank — no on-chain verification | OPEN — requires rank oracle before permissionless open |
| F-2 | MEDIUM | Rollover pool sequential drain by 2 colluders | OPEN — bounded by soft-launch; consider inter-draw delay |
| F-3 | MEDIUM | p-parameter siphoning + creator address spoofing | OPEN — registry cross-check mandatory before cap raise |
| F-4 | MEDIUM | Stake-cycle griefing | OPEN — add minimum stake duration |
| F-5 | LOW | M_pool cap bypass | NO EXPLOIT — add defensive require assertion |
| F-6 | LOW | Adversarial-bounty drain | NO EXPLOIT |
| F-7 | HIGH | Missing 90-day rolling window on activity weights | OPEN — mandatory before unpausing minting |
| F-8 | LOW | Cross-principle T_k misdirection | OPEN — add principleId registry cross-check |
| F-9 | MEDIUM | depositBounty inconsistent access control | OPEN — normalize to onlyGovernance |
| F-10 | INFO | Incentive misalignments (delta self-report, gas race, colluding roles) | Document in runbook |

## Deploy gate impact

**None of these findings block the current soft-launch deploy.** All financial impacts are bounded to ≤100 PWM by the active soft-launch caps.

**Mandatory before governance raises caps (post-audit):**
1. F-1: On-chain rank verification
2. F-7: 90-day rolling activity window in PWMMinting
3. F-3: Creator address registry cross-check in PWMCertificate.submit
